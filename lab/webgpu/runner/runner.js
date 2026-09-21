// Stage-2 step 4: the speculative generate loop against the LIVE target engine (no oracle stubs).
//   target features  : verify-mode qwen35 prefill graph (patch-internals (d)+(e): lh(model, cache, T, {tapLayers, allRowsHead, smallM}))
//                      -> dspark.features [T, 5H] f32 (entry of layers 6/20/34/48/62 = output of 5/19/33/47/61) + verify_tokens [T]
//   noise embedding  : a micro graph on the engine's own LlamaEmbed op + inverse Hadamard (the same ops lh's prologue emits)
//   lm_head          : a micro graph: Hadamard rotation of the drafter's final hidden + Lut2SmallMGemm (M=7) over the lm_head pack
//                      -> logits [7, V] read back, top-16 per slot on the CPU
//   cycle            : capture checkpoint -> draft 7 -> verify (block Lv, anchor + Lv-1 drafts) -> accept longest prefix + bonus
//                      -> context append from the verify run's feature rows 0..k -> if k+1 < Lv: restore checkpoint, replay the
//                      k+1 accepted tokens through the (k+1)-session (its next_token must equal the bonus) -> emit
// Query: ?model=<gguf> &block=5 &max=256 &prompt=code|oracle|lighthouse|<text> &smallm=f16|f32|off &selftest=1 &draft=<drafter gguf url>
//        &sink=64 &window=1024 (drafter context eviction; 0/0 = none) &eos=0 (run past EOS) &packed=0 (f16 drafter weights) &cputopk=1 (old head path)
//        &rewind=0 (tape replay through a (k+1)-row session instead of the recurrence-only RewindSession, patch section (f))
(() => { // hidden-tab guard (verify-qwen35-harness.js): rAF never fires and short timers are throttled in a hidden tab
  const ch = new MessageChannel(); const q = [];
  ch.port1.onmessage = () => { const cb = q.shift(); if (cb) cb(performance.now()); };
  window.requestAnimationFrame = (cb) => { q.push(cb); ch.port2.postMessage(0); return q.length; };
  window.cancelAnimationFrame = () => {};
  const st = window.setTimeout.bind(window); const tq = new Map(); let tid = 0; const tch = new MessageChannel();
  tch.port1.onmessage = (e) => { const cb = tq.get(e.data); if (cb) { tq.delete(e.data); cb(); } };
  window.setTimeout = (cb, ms, ...args) => { if ((ms | 0) > 4 || typeof cb !== 'function') return st(cb, ms, ...args); const id = 1e9 + (++tid); tq.set(id, () => cb(...args)); tch.port2.postMessage(id); return id; };
  const ct = window.clearTimeout.bind(window); window.clearTimeout = (id) => { if (tq.has(id)) tq.delete(id); else ct(id); };
})();
import { Drafter, CFG } from '../drafter/drafter.js';
import { fetchNpy } from '../drafter/npy.js';
const V = window.__dr = { state: 'init', log: [], results: {}, error: null };
const log = (...a) => { const s = a.join(' '); V.log.push([Date.now(), s]); console.log(s); };
const qs = new URLSearchParams(location.search);
const MODEL = qs.get('model') || '/model/Ternary-Bonsai-2-27B-PTQ1_0.gguf', DRAFT = qs.get('draft') || '/model/Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf';
const LV = Number(qs.get('block') || 5), MAX = Number(qs.get('max') || 256), PROMPT = qs.get('prompt') || 'code', SMALLM = qs.get('smallm') || 'f16', SELFTEST = qs.get('selftest') === '1', PLAIN = qs.get('plain') !== '0';
const SINK = Number(qs.get('sink') ?? 64), WINDOW = Number(qs.get('window') ?? 1024), STOP_EOS = qs.get('eos') !== '0', PACKED = qs.get('packed') !== '0', CPU_TOPK = qs.get('cputopk') === '1', REWIND = qs.get('rewind') !== '0';
const TAPS = [6, 20, 34, 48, 62];   // entry-of-layer convention == output of target_layer_ids [5,19,33,47,61]
const PROMPTS = {
  code: 'Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings, type hints, and a small pytest test file at the end.',
  oracle: 'Write a Python function that reverses a string, with a docstring.',
  lighthouse: 'Write a short paragraph about lighthouses.',
};
const chat = (u) => `<|im_start|>user\n${u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
const ms = (t) => +(performance.now() - t).toFixed(1);
(async () => {
  try {
    const mod = await import('/engine.dflash.js?v=' + Date.now()); const Eng = mod.TernaryBonsai2; const I = Eng.__dflashInternals;
    if (!I || !I.lh || !I.ch || !I.f0 || !I.ba || !I._i || !I.bi || !I.pi) throw new Error('internals hook missing');
    V.state = 'load-target';
    const m = await Eng.load(MODEL, { maxLength: 4096, onProgress: (ev) => { if (ev && ev.status) V.prog = `${ev.status} ${ev.loaded ?? ''}/${ev.total ?? ''}`; } });
    if (typeof m.warmup === 'function') await m.warmup();
    const inner = m.model, cache = m.generationState.cache, rt = inner.runtime, dev = rt.host.device, cfg = inner.config;
    const H = cfg.hidden_size, VOC = cfg.vocab_size; if (H !== CFG.H || VOC !== CFG.VOCAB) throw new Error(`target H/V ${H}/${VOC} != drafter ${CFG.H}/${CFG.VOCAB}`);
    const eos = new Set([...(m.eosTokenIds || [248046])]); const enc = (s) => m.tokenizer.encode(s, { add_special_tokens: false }).ids; const dec = (ids) => m.tokenizer.decode(ids, { skip_special_tokens: false });
    V.gpu = { f16: dev.features.has('shader-f16'), tsq: dev.features.has('timestamp-query'), arch: dev.adapterInfo?.architecture };
    log('target loaded; eos ' + [...eos].join(','));
    // ---- drafter on the engine's device ----
    V.state = 'load-drafter'; const dr = new Drafter(dev, { log, packed: PACKED });
    await dr.loadWeights(DRAFT, (p) => { V.prog = `drafter ${p.done}/${p.total}`; }); V.drafter = dr.stats;
    const ctx = dr.createContext(SINK + WINDOW > 0 ? SINK + WINDOW + 8 : 3000, { sink: SINK, window: WINDOW }); V.context = { sink: SINK, window: WINDOW, capacity: ctx.capacity };
    // ---- verify sessions, one per block length, lazily ----
    const smallM = SMALLM === 'off' ? undefined : { precision: SMALLM };
    const sessions = new Map(); const buildMs = {};
    const useRewind = REWIND && !!I.RewindSession; V.rewind = useRewind;
    const session = async (T) => { let s = sessions.get(T); if (s) return s; const opts = { tapLayers: TAPS, allRowsHead: true, ...(smallM ? { smallM } : {}), ...(useRewind && T === LV ? { teeRecurrence: true } : {}) }; class S extends I.ch { buildEmission() { return I.lh(this.model, this.cache, this.blockLen, opts); } } s = new S(inner, cache, T); const t0 = performance.now(); await s.build(); buildMs[T] = ms(t0); sessions.set(T, s); return s; };
    // run a block through the target at past_len = cache.seqLength; advances the cache; returns {next, tokens (verify_tokens), feat (binding)}
    const runBlock = async (ids) => { const T = ids.length; const s = await session(T); const pos = cache.seqLength; const next = await s.run(new Uint32Array(ids), pos); cache.seqLength = pos + T; const ft = s.compiled.tensor('dspark.features'); if (ft.dtype !== 'float32') throw new Error('features dtype ' + ft.dtype); return { next, s, pos, feat: { buffer: ft.buffer, offset: ft.byteOffset ?? 0, size: T * 5 * H * 4 }, tokens: () => rt.readTensor(s.compiled.tensor('verify_tokens')), logits: () => rt.readTensor(s.compiled.tensor('verify_logits')) }; };
    // ---- micro graphs on the engine's compile stack (pattern: wgsl-gemm-spike/bench.js) ----
    class Micro extends I.f0 { constructor(emit) { super(inner, null, 8); this._emit = emit; } buildEmission() { return this._emit(); } }
    const micro = async (name, emit) => { const s = new Micro(() => { const S = new I.ba(); const b = I._i(S); const t = emit(S, b); return { graph: S.finish({ name }), weights: b.boundWeights, states: b.states, ...t }; }); await s.build(); return s; };
    // (a) noise embedding: LlamaEmbed (the branch T0 takes for this model) + inverse Hadamard ('embed_tokens', inverse=true)
    const embedS = await micro('dflash-embed', (S, b) => {
      const re = b.w; const ids = S.stepInput('input_ids', 'uint32', [8]); const hid = S.scratch('embed.hidden', 'float32', [8, H]); let h;
      if (inner.embedBits && inner.embedScales) h = S.op('com.xenova.LlamaEmbed', { inputT: ids, bitsT: re('embed.bits', inner.embedBits), scalesT: re('embed.scales', inner.embedScales), hiddenT: hid }, { args: { hiddenSize: H, vocabSize: VOC, seqLen: 8, format: inner.embedFormat ?? 'q8_rows' } }).hiddenT;
      else { const ke = I.pi(re, inner)('top'); h = S.op('com.xenova.LlamaEmbed', { inputT: ids, weightsT: ke, hiddenT: hid }, { args: { embedOffset: inner.offsets.top.embed_tokens, hiddenSize: H, vocabSize: VOC, seqLen: 8 } }).hiddenT; }
      h = I.bi(S, inner, re)(h, 'embed_tokens', true); S.output(h, 'embed'); return {};
    });
    V.embedRoute = inner.embedBits && inner.embedScales ? `packed ${inner.embedFormat ?? 'q8_rows'}` : 'dense top weights';
    const embedT = embedS.compiled.tensor('embed'); const embedBind = { buffer: embedT.buffer, offset: embedT.byteOffset ?? 0, size: 8 * H * 4 };
    const embed = (ids) => { rt.host.writeBuffer(embedS.compiled.tensor('input_ids').buffer, 0, new Uint32Array(ids)); embedS.compiled.collector.enqueue(embedS.steps); return embedBind; };
    // (b) head: rotate (signs + Hadamard, no norm: the drafter applied its own final norm) + Lut2SmallMGemm M=7 over lm_head
    if (!(inner.lmHeadQ4 && inner.lmHeadQ4Scales && inner.lmHeadLut === 9 && cfg.prismHadamard?.weights.includes('output.weight'))) throw new Error('head route: expected prism-rotated lut2 lm_head (lmHeadLut 9)');
    const R = CFG.BLOCK - 1;
    const headS = await micro('dflash-head', (S, b) => {
      const re = b.w; const x = S.stepInput('hid', 'float32', [R, H]); const rot = I.bi(S, inner, re)(x, 'lm_head'); const lg = S.scratch('logits', 'float32', [R, VOC]);
      S.op('com.xenova.Lut2SmallMGemm', { aT: S.view(rot, 0, 'float32', [R, H], 'head.in'), bitsT: re('head.bits', inner.lmHeadQ4), scalesT: re('head.scales', inner.lmHeadQ4Scales), yT: lg }, { args: { M: R, inFeatures: H, outFeatures: VOC, blockOffset: 0, outStride: VOC, dstColStart: 0, lut: 9, precision: (smallM && smallM.precision) || 'f16', kSplits: 1 } });
      S.output(lg, 'logits'); return {};
    });
    const headIn = headS.compiled.tensor('hid'); const headLg = headS.compiled.tensor('logits'); const headLgBind = { buffer: headLg.buffer, offset: headLg.byteOffset ?? 0, size: R * VOC * 4 };
    const topK = (lg, row, K) => { const ids = new Int32Array(K).fill(-1), vals = new Float32Array(K).fill(-Infinity); const b = row * VOC; let minV = -Infinity, minI = 0; for (let j = 0; j < VOC; ++j) { const v = lg[b + j]; if (v > minV) { ids[minI] = j; vals[minI] = v; minV = vals[0]; minI = 0; for (let c = 1; c < K; ++c) if (vals[c] < minV) { minV = vals[c]; minI = c; } } } return { ids: Array.from(ids), vals: Array.from(vals) }; };
    const headTop16 = async (finBuf) => {
      const e = dev.createCommandEncoder(); e.copyBufferToBuffer(finBuf, H * 4, headIn.buffer, headIn.byteOffset ?? 0, R * H * 4); dev.queue.submit([e.finish()]); headS.compiled.collector.enqueue(headS.steps);
      const cand = [], unary = [], argmax = [];
      if (CPU_TOPK) { const lg = await rt.readTensor(headLg); for (let r = 0; r < R; ++r) { const t = topK(lg, r, CFG.TOPK); cand.push(t.ids); unary.push(t.vals); argmax.push(t.ids[t.vals.indexOf(Math.max(...t.vals))]); } return { cand, unary, argmax }; }
      const pass = dr.begin(); const t = dr.topk(pass, headLgBind, R, VOC); dr.end(pass);
      const [ov, oiF] = await Promise.all([dr.read(t.ov, R * 16), dr.read(t.oi, R * 16)]); const oi = new Uint32Array(oiF.buffer);
      for (let r = 0; r < R; ++r) { const ids = Array.from(oi.subarray(r * 16, r * 16 + 16)), vals = Array.from(ov.subarray(r * 16, r * 16 + 16)); cand.push(ids); unary.push(vals); argmax.push(ids[vals.indexOf(Math.max(...vals))]); }
      return { cand, unary, argmax };
    };
    // ---- one draft: noise -> drafter -> head -> selector; returns the 7 ids (+ timing) ----
    const draft = async (anchor) => { const t0 = performance.now(); const noise = embed([anchor, ...Array(7).fill(CFG.MASK)]); const st = dr.draftStep(ctx, noise, 1.0); const t1 = performance.now(); const hd = await headTop16(st.final); const t2 = performance.now(); const selH = await dr.read(st.selHidden, R * CFG.RANK); const sel = dr.select(anchor, hd.cand, hd.unary, selH); const t3 = performance.now(); for (const l of st.layers) for (const b of Object.values(l)) b.destroy(); st.final.destroy(); st.selHidden.destroy(); st.h0.destroy(); return { ids: sel.path, argmax: hd.argmax, cand: hd.cand, unary: hd.unary, tEnqueue: t1 - t0, tHead: t2 - t1, tSelect: t3 - t2, tDraft: t3 - t0 }; };
    // ---- prompt prefill through tapped sessions (chunks of <= 8) -> drafter context covers the whole prompt; anchor = next_token ----
    const prefill = async (ids) => { m.resetCache(); cache.seqLength = 0; dr.resetContext(ctx); let next = null; for (let i = 0; i < ids.length; i += 8) { const chunk = ids.slice(i, i + 8); const r = await runBlock(chunk); dr.appendContext(ctx, r.feat, chunk.length); next = r.next; } await rt.queueIdle(); return next; };
    const ids = enc(chat(PROMPTS[PROMPT] ?? PROMPT)); V.prompt = { name: PROMPT, tokens: ids.length };
    // ---- self-test against the MLX oracle (drafter/oracle/bf16): same prompt, cycle 1 ----
    if (SELFTEST || PROMPT === 'oracle') {
      V.state = 'selftest';
      try {
        const base = '/drafter/oracle/bf16/'; const idx = await (await fetch(base + 'index.json')).json(); const np = async (n) => (await fetchNpy(base + n + '.npy')).data;
        if (JSON.stringify(idx.prompt_ids) !== JSON.stringify(ids)) throw new Error('oracle prompt ids differ from the engine tokenizer');
        const t0 = performance.now(); const anchor = await prefill(ids); const tP = ms(t0);
        // features check: the last prompt chunk's rows vs the oracle's context_features (MLX target vs engine target)
        const cmp = (name, got, ref) => { let mx = 0, sd = 0, sr = 0; for (let i = 0; i < ref.length; ++i) { const d = Math.abs(got[i] - ref[i]); if (d > mx) mx = d; sd += d * d; sr += ref[i] * ref[i]; } const r = { n: ref.length, maxAbs: mx, rmsRel: Math.sqrt(sd / ref.length) / Math.sqrt(sr / ref.length) }; log(`selftest ${name}: maxAbs ${mx.toExponential(2)} rmsRel ${r.rmsRel.toExponential(2)}`); return r; };
        const featRef = await np('context_features'); const C = ids.length; const lastT = C % 8 || 8; const lastS = sessions.get(lastT); const gotFeat = await rt.readTensor(lastS.compiled.tensor('dspark.features'));
        const featCmp = cmp('features(last chunk, engine vs MLX target)', gotFeat.subarray(0, lastT * 5 * H), featRef.subarray((C - lastT) * 5 * H, C * 5 * H));
        // context cache: the drafter's projected K/V for the whole prompt vs the oracle's
        const kCmp = cmp('layer0_ctx_k', await dr.read(ctx.layers[0].k, C * CFG.NKV * CFG.HD), await np('layer0_ctx_k'));
        // embed check: the noise rows vs the oracle's noise_embedding (target embed of [anchor, mask x7])
        const nb = embed([idx.anchor, ...Array(7).fill(CFG.MASK)]); const gotNoise = await rt.readTensor(embedT); const noiseCmp = cmp('noise_embedding (engine LlamaEmbed+invHadamard vs MLX Packed)', gotNoise, await np('noise_embedding'));
        const d = await draft(idx.anchor);
        const refArg = Array.from(await np('argmax')); const refSel = idx.selected;
        V.results.selftest = { anchor, anchorRef: idx.anchor, anchorOk: anchor === idx.anchor, prefillMs: tP, features: featCmp, ctxK: kCmp, noise: noiseCmp, argmax: d.argmax, argmaxRef: refArg, argmaxAgree: d.argmax.filter((v, i) => v === refArg[i]).length, selected: d.ids, selectedRef: refSel, selectedMatch: JSON.stringify(d.ids) === JSON.stringify(refSel), draftMs: d.tDraft, headMs: d.tHead, selText: dec(d.ids) };
        log('selftest: anchor ' + anchor + ' (ref ' + idx.anchor + ') argmax ' + JSON.stringify(d.argmax) + ' ref ' + JSON.stringify(refArg) + ' selected ' + JSON.stringify(d.ids) + ' ref ' + JSON.stringify(refSel) + ' match=' + V.results.selftest.selectedMatch + ' text ' + JSON.stringify(dec(d.ids)));
      } catch (e) { V.results.selftest = { error: String(e && e.stack || e) }; log('selftest ERROR ' + e); }
      if (PROMPT === 'oracle' && !qs.get('max')) { V.state = 'done'; return; }
    }
    // ---- plain greedy reference (the engine's own decode path), timed ----
    let plain = null;
    if (PLAIN) {
      V.state = 'plain'; m.resetCache(); const out = []; const t0 = performance.now(); let tFirst = null;
      for await (const t of m.streamTokens({ suffixIds: ids, maxNewTokens: MAX, eosTokenId: [...eos][0], stopOnEos: false }, {})) { if (tFirst === null) tFirst = performance.now(); out.push(t); }
      const tEnd = performance.now(); plain = { tokens: out, ttftMs: +(tFirst - t0).toFixed(1), decodeMsPerTok: +((tEnd - tFirst) / (out.length - 1)).toFixed(2), tokPerS: +(1000 * (out.length - 1) / (tEnd - tFirst)).toFixed(2), text: dec(out) };
      V.results.plain = { ...plain, text: plain.text.slice(0, 400) }; log(`plain greedy: ${out.length} tokens, ${plain.decodeMsPerTok} ms/token (${plain.tokPerS} tok/s), ttft ${plain.ttftMs} ms`);
    }
    // ---- speculative loop (as a function: one warm-up pass, then the measured pass) ----
    V.state = 'prebuild'; const tb = performance.now(); if (useRewind) await session(LV); else for (let T = 1; T <= LV; ++T) await session(T); let RW = null; if (useRewind) { RW = new I.RewindSession(inner, cache, LV, sessions.get(LV)); const tr = performance.now(); await RW.build(); buildMs.rewind = ms(tr); } const prebuildMs = ms(tb); log(`prebuilt sessions in ${prebuildMs} ms: ${JSON.stringify(buildMs)} rewind=${useRewind}`);
    const slot = await cache.allocateCheckpointSlot();
    const specRun = async (maxTokens, label) => {
    V.state = 'spec-' + label;
    const tS = performance.now(); let anchor = await prefill(ids); const specTtft = ms(tS);
    const emitted = [anchor]; const cycles = []; let stop = STOP_EOS && eos.has(anchor); let divergence = null;   // the first generated token is the prompt's next_token (= plain[0])
    const checkDiv = async (tok, run, row) => { if (!plain || divergence) return; const p = emitted.length - 1; if (p < plain.tokens.length && plain.tokens[p] !== tok) { divergence = { pos: p, spec: tok, plain: plain.tokens[p], cycle: cycles.length }; try { if (run && row !== null) { const lg = await run.logits(); const b = row * VOC; const a = lg[b + tok], c = lg[b + plain.tokens[p]]; let b1 = -Infinity, b2 = -Infinity, i1 = -1; for (let j = 0; j < VOC; ++j) { const v = lg[b + j]; if (v > b1) { b2 = b1; b1 = v; i1 = j; } else if (v > b2) b2 = v; } divergence.verifyLogits = { specTok: a, plainTok: c, top1: i1, top1Val: b1, margin: b1 - b2 }; } } catch (e) { divergence.logitsError = String(e); } log('DIVERGENCE ' + JSON.stringify(divergence)); } };
    await checkDiv(anchor, null, null);
    while (emitted.length < maxTokens && !stop) {
      const c0 = performance.now(); const cyc = { pos: cache.seqLength };
      const d = await draft(anchor); cyc.draftMs = +d.tDraft.toFixed(1); cyc.headMs = +d.tHead.toFixed(1);
      const drafted = d.ids.slice(0, LV - 1); const block = [anchor, ...drafted];
      slot.checkpoint.capture(); const pos = cache.seqLength;
      const t1 = performance.now(); const run = await runBlock(block); const vt = Array.from(await run.tokens()); cyc.verifyMs = ms(t1);
      let k = 0; while (k < drafted.length && vt[k] === drafted[k]) k++;
      const bonus = vt[k]; cyc.accepted = k; cyc.tokens = k + 1;
      // emit accepted drafts + bonus (stop at EOS)
      for (let i = 0; i < k && !stop; ++i) { emitted.push(drafted[i]); await checkDiv(drafted[i], run, i); if (STOP_EOS && eos.has(drafted[i])) stop = true; }
      if (!stop) { emitted.push(bonus); await checkDiv(bonus, run, k); if (STOP_EOS && eos.has(bonus)) stop = true; }
      // drafter context: rows 0..k of the verify run's features (anchor + accepted drafts), computed from the same state
      const t2 = performance.now(); dr.appendContext(ctx, { ...run.feat, size: (k + 1) * 5 * H * 4 }, k + 1); cyc.appendMs = ms(t2);
      // target state: k+1 == LV -> the verify run consumed exactly the accepted tokens; else restore + replay the k+1 accepted tokens
      const t3 = performance.now();
      if (k + 1 === LV) { cyc.replay = 'none'; }
      else if (useRewind) { slot.checkpoint.restore(); RW.run(k + 1); cache.seqLength = pos + k + 1; cyc.replay = k + 1; cyc.replayOk = true; cyc.rewind = true; }
      else { slot.checkpoint.restore(); cache.seqLength = pos; const rp = await runBlock(block.slice(0, k + 1)); cyc.replay = k + 1; cyc.replayNext = rp.next; cyc.replayOk = rp.next === bonus; if (!cyc.replayOk) log(`replay next_token ${rp.next} != bonus ${bonus} at cycle ${cycles.length}`); }
      cyc.replayMs = ms(t3); cyc.cycleMs = ms(c0); cycles.push(cyc); anchor = bonus;
      if (cycles.length % 10 === 0) { V.prog = `cycle ${cycles.length}: ${emitted.length} tokens`; V.partial = { cycles: cycles.length, tokens: emitted.length, text: dec(emitted).slice(-200) }; }
    }
    await rt.queueIdle(); const specTotal = ms(tS);
    const out = emitted.slice(0, maxTokens); const n = cycles.length; const sum = (f) => cycles.reduce((a, c) => a + f(c), 0);
    const genMs = sum(c => c.cycleMs);
    const stats = { cycles: n, tokens: emitted.length, tokensPerCycle: +(emitted.length / n).toFixed(3), acceptedPerCycle: +(sum(c => c.accepted) / n).toFixed(3), msPerCycle: +(genMs / n).toFixed(1), msPerToken: +(genMs / emitted.length).toFixed(2), tokPerS: +(1000 * emitted.length / genMs).toFixed(2), ttftMs: specTtft, totalMs: specTotal,
      breakdownMs: { draft: +(sum(c => c.draftMs) / n).toFixed(1), head: +(sum(c => c.headMs) / n).toFixed(1), verify: +(sum(c => c.verifyMs) / n).toFixed(1), append: +(sum(c => c.appendMs) / n).toFixed(1), replay: +(sum(c => c.replayMs) / n).toFixed(1) },
      replays: cycles.filter(c => c.replay !== 'none').length, rewind: useRewind, replayMsByLen: Object.fromEntries(Array.from({ length: LV - 1 }, (_, i) => { const l = cycles.filter(c => c.replay === i + 1).map(c => c.replayMs).sort((a, b) => a - b); return [i + 1, l.length ? { n: l.length, median: l[l.length >> 1], min: l[0] } : null]; })), verifyMsMedian: [...cycles.map(c => c.verifyMs)].sort((a, b) => a - b)[n >> 1], draftMsMedian: [...cycles.map(c => c.draftMs)].sort((a, b) => a - b)[n >> 1], cycleMsMedian: [...cycles.map(c => c.cycleMs)].sort((a, b) => a - b)[n >> 1], prebuildMs, replayMismatches: cycles.filter(c => c.replay !== 'none' && !c.replayOk).length, context: { rows: ctx.C, total: ctx.total, evictions: ctx.evictions, sink: SINK, window: WINDOW }, packed: PACKED, cpuTopk: CPU_TOPK, acceptHist: Object.fromEntries(Array.from({ length: LV }, (_, i) => [i, cycles.filter(c => c.accepted === i).length])), buildMs, block: LV, smallM: SMALLM };
    let match = null; if (plain) { const L = Math.min(out.length, plain.tokens.length); let first = -1; for (let i = 0; i < L; ++i) if (out[i] !== plain.tokens[i]) { first = i; break; } match = { compared: L, firstDivergence: first, identical: first === -1 && (out.length === plain.tokens.length || stop), stoppedAtEos: stop, specLen: out.length, plainLen: plain.tokens.length, speedup: plain ? +(stats.tokPerS / plain.tokPerS).toFixed(3) : null }; }
    V.results['spec-' + label] = { stats, match, divergence, text: dec(out).slice(0, 400), cyclesSample: cycles.slice(0, 12) };
    log(`spec[${label}]: ${stats.tokens} tokens in ${n} cycles, ${stats.tokensPerCycle} tok/cycle, ${stats.msPerCycle} ms/cycle (median ${stats.cycleMsMedian}), ${stats.msPerToken} ms/token (${stats.tokPerS} tok/s); breakdown ${JSON.stringify(stats.breakdownMs)}; replay by len ${JSON.stringify(stats.replayMsByLen)}; match ${JSON.stringify(match)}`);
    };
    await specRun(Math.min(32, MAX), 'warmup');
    await specRun(MAX, 'measured');
    V.state = 'done';
  } catch (e) { V.error = String(e && e.stack || e); V.state = 'error'; log('ERROR ' + V.error); }
})();
