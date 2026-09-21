// Stage-2 step 1 harness: build the engine's own verify graph (I0 with the dspark config: all-rows head +
// feature taps) on top of a live cache, run one 8-token verify, and check it against the model's own greedy
// continuation. Serve the scratchpad engine dir over http (python3 -m http.server) so /engine.dflash.js loads
// from the SAME origin as the app's weight cache is NOT required: this harness loads weights through the
// engine's own IndexedDB cache (gguf-cache-v1) of whatever origin it runs on. Run it on the deployed origin
// (localmind.naklitechie.com) after injecting engine.dflash.js there is not possible cross-origin, so instead:
//   1. serve the LocalMind checkout with engine.dflash.js copied next to index.html (python3 -m http.server 8765),
//   2. open http://127.0.0.1:8765/ in real Chrome, pick Bonsai 2 once so the weights land in that origin's IDB,
//   3. inject this file in the main thread with the tab visible; poll window.__vs.state / .results.
// Hidden-tab guard (from the stage-0 harness): rAF never fires and short timers are throttled to 1 Hz in a
// hidden tab; route both through MessageChannel hops.
(() => {
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
window.__vs = { state: 'init', log: [] };
(async () => {
  const V = window.__vs; const mark = (m) => { V.log.push([Date.now(), m]); V.state = m; };
  try {
    const mod = await import('/engine.dflash.js?v=' + Date.now());
    const Eng = mod.TernaryBonsai2; const I = Eng.__dflashInternals;
    if (!I) throw new Error('internals hook missing');
    mark('imported');
    const modelUrl = new URLSearchParams(location.search).get('model') || null;   // e.g. a local range-capable GGUF URL
    const m = await Eng.load(modelUrl, { maxLength: 4096, onProgress: (ev) => { if (ev && ev.status) V.prog = `${ev.status} ${ev.loaded ?? ''}/${ev.total ?? ''}`; } });
    V.model = m; mark('loaded');
    if (typeof m.warmup === 'function') await m.warmup();
    const enc = (s) => m.tokenizer.encode(s, { add_special_tokens: false }).ids;
    const prompt = enc('<|im_start|>user\nWrite a short paragraph about lighthouses.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n');
    const eos = (m.eosTokenIds && m.eosTokenIds[0]) ?? 248044;
    m.resetCache();
    // 1. prefill the prompt and take 9 greedy tokens through the normal path: t0 (from the prompt) + 8 more.
    const greedy = [];
    for await (const tok of m.streamTokens({ suffixIds: prompt, maxNewTokens: 9, eosTokenId: eos, stopOnEos: false }, {})) greedy.push(tok);
    const cache = m.generationState.cache; const seqAfter = cache.get_seq_length();
    mark(`greedy ${greedy.length} tokens, seq ${seqAfter}`);
    // The cache now holds prompt + 8 committed tokens (the 9th is only predicted). Roll back 8 so the verify
    // block re-feeds them: verify_tokens[i] must equal greedy[i+1] for i in 0..7.
    const pos = seqAfter - 8;
    if (typeof cache.set_seq_length === 'function') cache.set_seq_length(pos); else cache.seqLength = pos;
    const block = new Uint32Array(greedy.slice(0, 8));
    // 2. The engine's target-side verify scaffolding (taps + all-rows head) exists only in the Llama prefill
    //    builder (I0); Bonsai 2 is a qwen35 hybrid whose prefill builder (lh) has neither. So step 1 here measures
    //    what stage 0 could not: the cost of the engine's OWN 8-token qwen35 prefill-graph step (class ch) on top
    //    of the live cache, versus a decode step, and checks its last-row next token against the greedy path.
    const inner = m.model;
    const s = new I.ch(inner, cache, 8); const tb = performance.now(); await s.build(); V.buildMs = +(performance.now() - tb).toFixed(0);
    mark(`8-token qwen35 prefill session built in ${V.buildMs} ms`);
    const times = []; let nt = null, feat = null, vt = null;
    for (let r = 0; r < 6; ++r) {
      if (typeof cache.set_seq_length === 'function') cache.set_seq_length(pos); else cache.seqLength = pos;
      const t0 = performance.now();
      nt = await s.run(block, pos);
      times.push(+(performance.now() - t0).toFixed(1));
    }
    // decode-step reference: 24 tokens streamed from the same state
    if (typeof cache.set_seq_length === 'function') cache.set_seq_length(seqAfter); else cache.seqLength = seqAfter;
    const td = performance.now(); let nd = 0, firstMs = null;
    for await (const tok of m.streamTokens({ suffixIds: [greedy[8]], maxNewTokens: 24, eosTokenId: eos, stopOnEos: false }, {})) { if (firstMs === null) firstMs = performance.now() - td; nd++; }
    const decodeMs = +(((performance.now() - td) - firstMs) / (nd - 1)).toFixed(1);
    V.results = { greedy, next_token_from_8row_step: nt, expect_last: greedy[8], lastRowMatch: nt === greedy[8], stepMs: times, decodeMs, ratio: +(Math.min(...times.slice(1)) / decodeMs).toFixed(2), featError: 'n/a (qwen35 builder has no taps)' };
    mark(`done: last-row ${nt === greedy[8] ? 'MATCH' : 'MISMATCH'}, 8-row step ${times.join('/')} ms vs decode ${decodeMs} ms/token (ratio ${V.results.ratio})`);
    return;
    const expect = greedy.slice(1, 9);
    V.results = { greedy, verify_tokens: vt, expect, match: vt.map((t, i) => t === expect[i]), verifyMs: times, featLen: feat ? feat.length : null,
      featSample: feat ? Array.from(feat.slice(0, 8)).map(Number) : null, featNonZero: feat ? Array.from(feat.slice(0, 4096)).some(x => x !== 0) : null };
    mark(`done: ${V.results.match.filter(Boolean).length}/8 rows match, verify ${times.join('/')} ms`);
  } catch (e) { V.error = String(e && e.stack || e); mark('error'); }
})();
'armed';
