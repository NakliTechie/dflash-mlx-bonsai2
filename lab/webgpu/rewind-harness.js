// Stage-2 step 3: recurrence-only rewind (patch-internals.mjs section (f)) vs the tape replay.
//   1. prompt prefill (code prompt) -> g0; checkpoint slot at pos; 16 greedy tokens g1..g16 via the normal path
//   2. verify session T=8 in verify mode + teeRecurrence; RewindSession on top of it; replay sessions of length n
//   3. one verify(8) of [g0..g7] from the checkpoint (parity check), then for each n in {1,2,4,8}:
//        A: restore -> rewind.run(n)            -> read all linear-layer conv + recurrent states
//        B: restore -> replay session n of [g0..g_{n-1}] -> read the same states
//      compare A vs B bitwise (Uint32) and by max |diff|; then, from each, set seqLength = pos+n and verify(8) the
//      next 8 greedy tokens [g_n..g_{n+7}]: verify_tokens and next_token must agree (and match g_{n+1}..g_{n+8})
//   4. timing: rewind(n) (restore + run + queueIdle) vs replay(n) (restore + s.run incl. next_token readback)
// Query: ?model=<gguf> &prompt=code|lighthouse &smallm=f16 &reps=8 &ns=1,2,4,8 &taps=6,20,34,48,62
(() => { // hidden-tab guard (as in verify-qwen35-harness.js)
  const ch = new MessageChannel(); const q = [];
  ch.port1.onmessage = () => { const cb = q.shift(); if (cb) cb(performance.now()); };
  window.requestAnimationFrame = (cb) => { q.push(cb); ch.port2.postMessage(0); return q.length; };
  window.cancelAnimationFrame = () => {};
  const st = window.setTimeout.bind(window); const tq = new Map(); let tid = 0;
  const tch = new MessageChannel();
  tch.port1.onmessage = (e) => { const cb = tq.get(e.data); if (cb) { tq.delete(e.data); cb(); } };
  window.setTimeout = (cb, ms, ...args) => { if ((ms | 0) > 4 || typeof cb !== 'function') return st(cb, ms, ...args); const id = 1e9 + (++tid); tq.set(id, () => cb(...args)); tch.port2.postMessage(id); return id; };
  const ct = window.clearTimeout.bind(window);
  window.clearTimeout = (id) => { if (tq.has(id)) tq.delete(id); else ct(id); };
})();
window.__vs = { state: 'init', log: [] };
(async () => {
  const V = window.__vs; const mark = (m) => { V.log.push([Date.now(), m]); V.state = m; console.log('[vs]', m); };
  try {
    const qs = new URLSearchParams(location.search);
    const TAPS = (qs.get('taps') || '6,20,34,48,62').split(',').map(Number);
    const T = 8, REPS = Number(qs.get('reps') || 8), NS = (qs.get('ns') || '1,2,4,8').split(',').map(Number);
    const SMALLM = qs.get('smallm') || 'f16'; const smallM = SMALLM !== 'off' ? { precision: SMALLM } : undefined;
    const PROMPTS = { code: 'Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings, type hints, and a small pytest test file at the end.', lighthouse: 'Write a short paragraph about lighthouses.' };
    const chat = (u) => `<|im_start|>user\n${u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
    const mod = await import('/engine.dflash.js?v=' + Date.now()); const Eng = mod.TernaryBonsai2; const I = Eng.__dflashInternals;
    if (!I || !I.RewindSession || !I.ht) throw new Error('internals hook missing (RewindSession/ht): re-run patch-internals.mjs');
    mark('imported');
    const m = await Eng.load(qs.get('model') || '/model/Ternary-Bonsai-2-27B-PTQ1_0.gguf', { maxLength: 4096, onProgress: (ev) => { if (ev && ev.status) V.prog = `${ev.status} ${ev.loaded ?? ''}/${ev.total ?? ''}`; } });
    if (typeof m.warmup === 'function') await m.warmup();
    mark('loaded');
    const inner = m.model, cache = m.generationState.cache, rt = inner.runtime, cfg = inner.config;
    const enc = (s) => m.tokenizer.encode(s, { add_special_tokens: false }).ids;
    const eos = (m.eosTokenIds && m.eosTokenIds[0]) ?? 248046;
    const gen = async (ids, n) => { const out = []; for await (const t of m.streamTokens({ suffixIds: ids, maxNewTokens: n, eosTokenId: eos, stopOnEos: false }, {})) out.push(t); return out; };
    const prompt = enc(chat(PROMPTS[qs.get('prompt') || 'code']));
    m.resetCache();
    const [g0] = await gen(prompt, 1); const pos = cache.get_seq_length();
    const slot = await cache.allocateCheckpointSlot(); slot.checkpoint.capture();
    const rewindTo = () => { slot.checkpoint.restore(); cache.seqLength = pos; };
    const greedy = [g0, ...await gen([g0], 16)];   // g0..g16
    mark(`prompt ${pos} tokens, greedy [${greedy.join(',')}]`);
    // sessions
    const opts = { tapLayers: TAPS, allRowsHead: true, ...(smallM ? { smallM } : {}) };
    const mk = async (n, tee) => { const o = { ...opts, ...(tee ? { teeRecurrence: true } : {}) }; class S extends I.ch { buildEmission() { return I.lh(this.model, this.cache, this.blockLen, o); } } const s = new S(inner, cache, n); const t0 = performance.now(); await s.build(); return { s, buildMs: +(performance.now() - t0).toFixed(0) }; };
    const { s: s8, buildMs: b8 } = await mk(T, true);
    const replay = new Map(); const buildMs = { verify8: b8 };
    const DIAG = qs.get('diag') === '1';
    for (const n of NS) if (n < T) { const r = await mk(n, DIAG); replay.set(n, r.s); buildMs[`replay${n}`] = r.buildMs; } else replay.set(n, s8);
    const t0r = performance.now(); const R = new I.RewindSession(inner, cache, T, s8); await R.build(); buildMs.rewind = +(performance.now() - t0r).toFixed(0);
    V.dims = { ...R.dims, linearLayers: R.layers.length, rewindNodes: R.emission.graph.nodes.length, rewindSteps: R.steps.length };
    mark(`sessions built ${JSON.stringify(buildMs)}; rewind graph ${V.dims.rewindNodes} nodes / ${V.dims.rewindSteps} steps over ${V.dims.linearLayers} linear layers`);
    // states readback (all linear layers; conv window + recurrent state), as Uint32 for bitwise compare + Float32 view
    const convLen = R.dims.convDim * R.dims.convState, recLen = R.dims.numHeads * R.dims.headDimK * R.dims.headDimV;
    const readStates = async () => { await rt.queueIdle(); const out = []; for (const te of R.layers) { const c = await rt.readTensor(I.ht(cache.linearConvStates[te], 0, convLen)); const r = await rt.readTensor(I.ht(cache.linearRecurrentStates[te], 0, recLen)); out.push({ te, conv: c, rec: r }); } return out; };
    const cmp = (A, B) => { let bitsEq = 0, bitsN = 0, maxAbs = 0, maxRel = 0, maxAt = null; for (let i = 0; i < A.length; ++i) { const a = A[i].conv, b = B[i].conv, ar = A[i].rec, br = B[i].rec; for (const [x, y, kind] of [[a, b, 'conv'], [ar, br, 'rec']]) { const ux = new Uint32Array(x.buffer, x.byteOffset, x.length), uy = new Uint32Array(y.buffer, y.byteOffset, y.length); for (let j = 0; j < x.length; ++j) { bitsN++; if (ux[j] === uy[j]) bitsEq++; const d = Math.abs(x[j] - y[j]); if (d > maxAbs) { maxAbs = d; maxAt = { layer: A[i].te, kind, j, a: x[j], b: y[j] }; } const s = Math.max(Math.abs(x[j]), Math.abs(y[j])); if (s > 1e-6) maxRel = Math.max(maxRel, d / s); } } } return { elements: bitsN, bitwiseEqual: bitsEq, bitwiseFrac: +(bitsEq / bitsN).toFixed(6), maxAbs, maxRel, maxAt }; };
    const readTok = async (s) => Array.from(await rt.readTensor(s.compiled.tensor('verify_tokens')));
    V.dims.stateElementsPerSnapshot = R.layers.length * (convLen + recLen);
    // one verify(8) from the checkpoint: parity, then the T=8 final state as a third reference for n=8
    rewindTo(); const block = new Uint32Array(greedy.slice(0, T)); const nt8 = await s8.run(block, pos); const vt8 = await readTok(s8);
    const S8 = await readStates();
    V.verify8 = { next_token: nt8, expect: greedy[T], verify_tokens: vt8, expectRows: greedy.slice(1, T + 1), matches: vt8.filter((t, i) => t === greedy[i + 1]).length };
    mark(`verify(8): ${V.verify8.matches}/8 rows match, next ${nt8} (expect ${greedy[T]})`);
    const results = {};
    const verify8 = async () => { rewindTo(); return s8.run(block, pos); };   // refills the tee with rows of [g0..g7] from the checkpoint
    const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
    for (const n of NS) {
      const r = { n }; const sr = replay.get(n); const firstN = new Uint32Array(greedy.slice(0, n)); const nextIds = new Uint32Array(greedy.slice(n, n + T));
      // A: verify(8) then restore + rewind(n)  (the runner's sequence)
      await verify8(); rewindTo(); R.run(n); const SA = await readStates();
      // C: the 8-row session itself with only n real rows (real_len = n): same kernels, so the rewind should match it bitwise
      rewindTo(); const ntC = await s8.run(firstN, pos); const SC = await readStates();
      // B: the n-row replay session (what the runner does today)
      rewindTo(); const ntR = await sr.run(firstN, pos); const SB = await readStates();
      r.replayNext = { got: ntR, expect: greedy[n], ok: ntR === greedy[n], truncated8: ntC };
      r.state = { rewindVsReplayN: cmp(SA, SB), rewindVsVerify8TruncN: cmp(SA, SC), replayNVsVerify8TruncN: cmp(SB, SC) }; if (n === T) r.state.rewindVsVerify8Full = cmp(SA, S8);
      // timing (tee intact: R does not write it; sr writes its own graph's buffers)
      const tA = [], tB = [];
      for (let k = 0; k < REPS; ++k) { rewindTo(); await rt.queueIdle(); const t0 = performance.now(); slot.checkpoint.restore(); R.run(n); await rt.queueIdle(); tA.push(+(performance.now() - t0).toFixed(2)); }
      for (let k = 0; k < REPS; ++k) { rewindTo(); await rt.queueIdle(); const t0 = performance.now(); slot.checkpoint.restore(); await sr.run(firstN, pos); tB.push(+(performance.now() - t0).toFixed(2)); }
      r.timing = { rewindMs: tA, rewindMedian: med(tA), replayMs: tB, replayMedian: med(tB), speedup: +(med(tB) / med(tA)).toFixed(1) };
      // next cycle from each state: verify(8) of [g_n..g_{n+7}] at pos+n
      await verify8(); rewindTo(); R.run(n); cache.seqLength = pos + n; const ntA = await s8.run(nextIds, pos + n); const vtA = await readTok(s8);
      rewindTo(); await sr.run(firstN, pos); cache.seqLength = pos + n; const ntB = await s8.run(nextIds, pos + n); const vtB = await readTok(s8);
      rewindTo(); await s8.run(firstN, pos); cache.seqLength = pos + n; const ntCc = await s8.run(nextIds, pos + n); const vtC = await readTok(s8);
      r.nextCycle = { rewind: { next: ntA, tokens: vtA }, replayN: { next: ntB, tokens: vtB }, verify8TruncN: { next: ntCc, tokens: vtC },
        rewindEqReplayN: ntA === ntB && vtA.every((t, i) => t === vtB[i]), rewindEqTruncN: ntA === ntCc && vtA.every((t, i) => t === vtC[i]),
        greedyMatches: vtA.filter((t, i) => t === greedy[n + i + 1]).length, nextOk: ntA === greedy[n + T] };
      results[n] = r;
      mark(`n=${n}: rewind vs verify8(real_len=${n}) bitwise ${r.state.rewindVsVerify8TruncN.bitwiseEqual}/${r.state.rewindVsVerify8TruncN.elements} maxAbs ${r.state.rewindVsVerify8TruncN.maxAbs}; rewind vs replay(${n}) maxAbs ${r.state.rewindVsReplayN.maxAbs} maxRel ${r.state.rewindVsReplayN.maxRel}; next cycle: rewind==replay ${r.nextCycle.rewindEqReplayN}, rewind==trunc ${r.nextCycle.rewindEqTruncN}, greedy ${r.nextCycle.greedyMatches}/8; rewind ${r.timing.rewindMedian} ms vs replay ${r.timing.replayMedian} ms`);
    }
    V.results = { prompt: qs.get('prompt') || 'code', promptTokens: pos, greedy, smallM: SMALLM, buildMs, dims: V.dims, verify8: V.verify8, byN: results };
    mark('done');
  } catch (e) { V.error = String(e && e.stack || e); mark('error'); console.error(e); }
})();
'armed';
