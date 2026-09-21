// Stage-2 step 1 (qwen35): build the engine's qwen35 prefill graph in VERIFY MODE (lh 4th arg from
// patch-internals.mjs (d): residual taps into `dspark.features` + all-rows argmax into `verify_tokens`) on top of a
// live cache, and prove it against the model's own greedy continuation.
//   1. prefill the prompt, take g0 (1 token) through the normal path; cache = prompt, seq = pos
//   2. capture an independent mutable-state checkpoint slot (cache.allocateCheckpointSlot(): conv + linear
//      recurrent state; the KV cache is position-indexed and only needs seqLength rewound)
//   3. take 8 more greedy tokens g1..g8 through the normal path (feeds g0..g7)
//   4. restore the slot, rewind seqLength to pos, run the 8-token verify graph on g0..g7
//   5. verify_tokens[i] must equal g[i+1] for i in 0..7; dspark.features must be [8, taps*H] and non-zero
// Serve the engine dir over http and open index-verify.html?model=/model/<gguf> in Chrome; poll window.__vs.
(() => { // hidden-tab guard: rAF never fires and short timers are throttled to 1 Hz; route both through MessageChannel
  const ch = new MessageChannel(); const q = [];
  ch.port1.onmessage = () => { const cb = q.shift(); if (cb) cb(performance.now()); };
  window.requestAnimationFrame = (cb) => { q.push(cb); ch.port2.postMessage(0); return q.length; };
  window.cancelAnimationFrame = () => {};
  const st = window.setTimeout.bind(window); const tq = new Map(); let tid = 0;
  const tch = new MessageChannel();
  tch.port1.onmessage = (e) => { const cb = tq.get(e.data); if (cb) { tq.delete(e.data); cb(); } };
  window.setTimeout = (cb, ms, ...args) => {
    if ((ms | 0) > 4 || typeof cb !== 'function') return st(cb, ms, ...args);
    const id = 1e9 + (++tid); tq.set(id, () => cb(...args)); tch.port2.postMessage(id); return id;
  };
  const ct = window.clearTimeout.bind(window);
  window.clearTimeout = (id) => { if (tq.has(id)) tq.delete(id); else ct(id); };
})();
const f16 = (h) => { const s = (h >> 15) & 1, e = (h >> 10) & 0x1f, f = h & 0x3ff; let v; if (e === 0) v = f / 1024 * 2 ** -14; else if (e === 31) v = f ? NaN : Infinity; else v = (1 + f / 1024) * 2 ** (e - 15); return s ? -v : v; };
window.__vs = { state: 'init', log: [] };
(async () => {
  const V = window.__vs; const mark = (m) => { V.log.push([Date.now(), m]); V.state = m; console.log('[vs]', m); };
  try {
    const qs = new URLSearchParams(location.search);
    const TAPS = (qs.get('taps') || '5,19,33,47,61').split(',').map(Number);
    const BLOCK = Number(qs.get('block') || 8), REPS = Number(qs.get('reps') || 6);
    // smallm=f16|f32|exact routes every lut2 projection of the verify graph and the all-rows head through
    // com.xenova.Lut2SmallMGemm (patch-internals.mjs section (e)); absent/off = the section (d) path (per-row ki heads).
    const SMALLM = qs.get('smallm'); const smallM = SMALLM && SMALLM !== 'off' ? { precision: SMALLM } : undefined;
    const mod = await import('/engine.dflash.js?v=' + Date.now());
    const Eng = mod.TernaryBonsai2; const I = Eng.__dflashInternals;
    if (!I || !I.lh || !I.ch) throw new Error('internals hook missing (lh/ch)');
    mark('imported');
    const modelUrl = qs.get('model') || '/model/Ternary-Bonsai-2-27B-PTQ1_0.gguf';
    const m = await Eng.load(modelUrl, { maxLength: 4096, onProgress: (ev) => { if (ev && ev.status) V.prog = `${ev.status} ${ev.loaded ?? ''}/${ev.total ?? ''}`; } });
    V.model = m; mark('loaded');
    if (typeof m.warmup === 'function') await m.warmup();
    const inner = m.model, cache = m.generationState.cache;
    const cfg = inner.config;
    V.head = { packsQ1: inner.packs?.q1 != null, lmHeadQ4: !!inner.lmHeadQ4, lmHeadQ8: !!inner.lmHeadQ8, lmHeadFormat: inner.lmHeadFormat ?? null, lmHeadLut: inner.lmHeadLut ?? null,
      lmHeadScanBits: inner.lmHeadScanBits ?? null, lmHeadRescoreFormat: inner.lmHeadRescoreFormat ?? null, prismHeadRotated: !!cfg.prismHadamard?.weights?.includes('output.weight'),
      hidden: cfg.hidden_size, layers: cfg.num_hidden_layers, vocab: cfg.vocab_size, f16: inner.runtime.device.features.has('shader-f16') };
    // ki's internal head route for this model (mirrors ki's own branch order): prism -> two-stage (q4+q8, !q1) -> q1 -> q4 -> q8 -> dense
    V.head.smallM = smallM ?? 'off';
    V.head.kiRoute = V.head.prismHeadRotated ? 'prism.head (LlamaPrefillMatmul M=1 + ArgMax)' : (!V.head.packsQ1 && V.head.lmHeadQ4 && V.head.lmHeadQ8) ? 'two-stage T5 (Q4 scan + Q8 rescore)'
      : V.head.packsQ1 ? 'q1 (RMSNorm + LlamaDecodeLmHeadArgmax q1_0)' : V.head.lmHeadQ4 ? `q4 LlamaDecodeLmHeadArgmax format=${V.head.lmHeadFormat}` : V.head.lmHeadQ8 ? 'q8_rows LlamaDecodeLmHeadArgmax' : 'dense LlamaDecodeLmHeadArgmax';
    if (typeof cache.allocateCheckpointSlot !== 'function') throw new Error('cache.allocateCheckpointSlot missing (no mutable-state checkpoint API)');
    const enc = (s) => m.tokenizer.encode(s, { add_special_tokens: false }).ids;
    const prompt = enc('<|im_start|>user\nWrite a short paragraph about lighthouses.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n');
    const eos = (m.eosTokenIds && m.eosTokenIds[0]) ?? 248044;
    const gen = async (ids, n) => { const out = []; for await (const t of m.streamTokens({ suffixIds: ids, maxNewTokens: n, eosTokenId: eos, stopOnEos: false }, {})) out.push(t); return out; };
    m.resetCache();
    // 1. prompt prefill -> g0. cache = prompt only.
    const [g0] = await gen(prompt, 1);
    const pos = cache.get_seq_length();
    if (pos !== prompt.length) throw new Error(`seq after prompt prefill ${pos} != prompt.length ${prompt.length}`);
    // 2. checkpoint the mutable (conv + recurrent) state at pos, in a slot of our own (the decode pipeline owns mutableStateCheckpoint()'s slot).
    const slot = await cache.allocateCheckpointSlot(); slot.checkpoint.capture();
    const rewind = () => { slot.checkpoint.restore(); cache.seqLength = pos; };
    // 3. g1..g8 through the normal path (feeds g0..g7)
    const rest = await gen([g0], BLOCK);
    const greedy = [g0, ...rest]; const seqAfter = cache.get_seq_length();
    mark(`greedy ${greedy.length} tokens: [${greedy.join(',')}], prompt ${pos}, seq ${seqAfter}`);
    if (seqAfter !== pos + BLOCK) throw new Error(`seq after ${BLOCK} greedy tokens ${seqAfter} != ${pos + BLOCK}`);
    // default-path check (graph construction only, no compile): the 4th arg absent -> no verify outputs, same node count as {}
    const gDef = I.lh(inner, cache, BLOCK), gEmpty = I.lh(inner, cache, BLOCK, {});
    const names = (g) => g.graph.tensors.filter(t => t.kind === 'output').map(t => t.name);
    V.graph = { defaultNodes: gDef.graph.nodes.length, emptyOptsNodes: gEmpty.graph.nodes.length, defaultOutputs: names(gDef), defaultHasPick: !!gDef.nextTokenPickUniform };
    // 4. verify session: ch (the qwen35 session class) with lh's 4th arg
    const opts = { tapLayers: TAPS, allRowsHead: true, ...(smallM ? { smallM } : {}) };
    class VerifySession extends I.ch { buildEmission() { return I.lh(this.model, this.cache, this.blockLen, opts); } }
    const s = new VerifySession(inner, cache, BLOCK);
    const tb = performance.now(); await s.build(); V.buildMs = +(performance.now() - tb).toFixed(0);
    V.graph.verifyNodes = s.emission.graph.nodes.length; V.graph.verifyOutputs = names(s.emission); V.graph.verifyTail = s.emission.nextTokenTailNodeCount; V.graph.defaultTail = gDef.nextTokenTailNodeCount;
    mark(`verify session built in ${V.buildMs} ms: ${V.graph.verifyNodes} nodes vs ${V.graph.defaultNodes} default; outputs ${V.graph.verifyOutputs.join(',')}`);
    const block = new Uint32Array(greedy.slice(0, BLOCK));
    const times = []; let nt = null, vt = null, feat = null, readMs = null;
    for (let r = 0; r < REPS; ++r) {
      rewind();
      const t0 = performance.now(); nt = await s.run(block, pos); times.push(+(performance.now() - t0).toFixed(1));
      if (r === 0) {
        const tr = performance.now();
        vt = Array.from(await inner.runtime.readTensor(s.compiled.tensor('verify_tokens')));
        if (smallM) { // tie-flip diagnostics from the all-rows head logits: per-row top-1 / top-2 margin
          const lg = await inner.runtime.readTensor(s.compiled.tensor('verify_logits')); const VV = cfg.vocab_size;
          V.margins = Array.from({ length: BLOCK }, (_, r2) => { let b1 = -Infinity, i1 = -1, b2 = -Infinity; for (let j = 0; j < VV; ++j) { const v = lg[r2 * VV + j]; if (v > b1) { b2 = b1; b1 = v; i1 = j; } else if (v > b2) b2 = v; } return { argmax: i1, top1: +b1.toFixed(4), margin: +(b1 - b2).toFixed(4) }; });
        }
        const fT = s.compiled.tensor('dspark.features'); const raw = await inner.runtime.readTensor(fT);
        readMs = +(performance.now() - tr).toFixed(1);
        const H = cfg.hidden_size, W = TAPS.length * H;
        feat = { dtype: fT.dtype, shape: fT.shape, length: raw.length, expectLength: BLOCK * W, shapeOk: fT.shape.length === 2 && fT.shape[0] === BLOCK && fT.shape[1] === W && raw.length === BLOCK * W };
        const val = fT.dtype === 'float16' ? (i) => f16(raw[i]) : (i) => raw[i];
        let nz = 0; for (let i = 0; i < raw.length; ++i) if (raw[i] !== 0) nz++;
        feat.nonZero = nz; feat.nonZeroFrac = +(nz / raw.length).toFixed(4);
        feat.perTap = TAPS.map((L, k) => { let sum = 0, cnt = 0, nzr = 0, mx = 0; for (let r2 = 0; r2 < BLOCK; ++r2) for (let c = 0; c < H; ++c) { const v = val(r2 * W + k * H + c); if (v !== 0) nzr++; const a = Math.abs(v); sum += a; cnt++; if (a > mx) mx = a; } return { layer: L, meanAbs: +(sum / cnt).toFixed(4), maxAbs: +mx.toFixed(3), nonZero: nzr, of: cnt, finite: Number.isFinite(sum) }; });
        feat.sampleRow0Tap0 = Array.from({ length: 6 }, (_, i) => +val(i).toFixed(4));
        feat.rowsDistinct = new Set(Array.from({ length: BLOCK }, (_, r2) => Array.from({ length: 16 }, (_, i) => raw[r2 * W + i]).join(','))).size;
      }
    }
    const expect = greedy.slice(1, BLOCK + 1);
    const match = vt.map((t, i) => t === expect[i]);
    // 5. decode reference from the same rewound state: an 8-token normal prefill of g0..g7 (its token must be g8 too) + 24 decode steps
    rewind();
    const td = performance.now(); let nd = 0, firstMs = null, prefillTok = null;
    for await (const tok of m.streamTokens({ suffixIds: greedy.slice(0, BLOCK), maxNewTokens: 25, eosTokenId: eos, stopOnEos: false }, {})) { if (firstMs === null) { firstMs = performance.now() - td; prefillTok = tok; } nd++; }
    const decodeMs = +(((performance.now() - td) - firstMs) / (nd - 1)).toFixed(1);
    const stepMin = Math.min(...times.slice(1));
    V.results = { greedy, verify_tokens: vt, expect, match, matches: match.filter(Boolean).length, next_token_from_verify: nt, next_token_expected: greedy[BLOCK], normalPrefillTok: prefillTok, normalPrefillMatch: prefillTok === greedy[BLOCK],
      stepMs: times, stepMinMs: stepMin, readbackMs: readMs, decodeMs, ratio: +(stepMin / decodeMs).toFixed(2), features: feat, head: V.head, graph: V.graph, taps: TAPS, smallM: smallM ?? 'off', margins: V.margins ?? null, minMargin: V.margins ? Math.min(...V.margins.map(x => x.margin)) : null };
    mark(`done: ${V.results.matches}/${BLOCK} rows match, verify step ${times.join('/')} ms vs decode ${decodeMs} ms/token (ratio ${V.results.ratio}); features ${feat.shapeOk ? 'shape ok' : 'SHAPE BAD'} nonzero ${feat.nonZeroFrac}`);
  } catch (e) { V.error = String(e && e.stack || e); mark('error'); console.error(e); }
})();
'armed';
