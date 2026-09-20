// Stage-0 harness: verify-row cost on the vendored Bonsai 2 WebGPU engine.
// Inject on https://localmind.naklitechie.com/ (main thread, tab foregrounded) AFTER the app has
// cached the weights and been reloaded onto its default model (so the worker holds no GPU copy).
// Poll window.__bm.state / .prog / .results.
window.__bm = { state: 'init', log: [], prog: '' };
// Hidden-tab guard: the engine yields via requestAnimationFrame, which never fires in a hidden
// tab, and chained setTimeout is throttled to 1 Hz there. A MessageChannel hop is neither.
(() => {
  const ch = new MessageChannel(); const q = [];
  ch.port1.onmessage = () => { const cb = q.shift(); if (cb) cb(performance.now()); };
  window.requestAnimationFrame = (cb) => { q.push(cb); ch.port2.postMessage(0); return q.length; };
  window.cancelAnimationFrame = () => {};
  // Zero/near-zero-delay timers are throttled to 1 Hz in a hidden tab (every engine yield read
  // exactly ~1000 ms in the first run). Route delays <= 4 ms through a MessageChannel too.
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
(async () => {
  const B = window.__bm;
  const mark = (m) => { B.log.push([Date.now(), m]); B.state = m; };
  try {
    const mod = await import('/ternary_bonsai_2_27b.js');
    mark('imported');
    const m = await mod.TernaryBonsai2.load(null, {
      maxLength: 4096,
      onProgress: (ev) => { if (ev && ev.status) B.prog = `${ev.status} ${ev.kind ?? ''} ${ev.loaded ?? ''}/${ev.total ?? ''}`; },
    });
    B.model = m;
    mark('loaded');
    if (typeof m.warmup === 'function') await m.warmup();
    mark('warm');
    const enc = (s) => m.tokenizer.encode(s, { add_special_tokens: false }).ids;
    const base = enc(('The Roman Empire was the post-Republican period of ancient Rome. As a polity it included large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia, ruled by emperors. ').repeat(8));
    const prefix = base.slice(0, 256);
    const eos = (m.eosTokenIds && m.eosTokenIds[0]) ?? 248044;
    const seqLen = () => m.generationState.cache.get_seq_length();
    const run = async (suffix, maxNew) => {
      const t0 = performance.now(); let first = null, n = 0;
      for await (const tok of m.streamTokens({ suffixIds: suffix, maxNewTokens: maxNew, eosTokenId: eos, stopOnEos: false }, {})) {
        if (first === null) first = performance.now() - t0; n++;
      }
      return { first: +first.toFixed(1), total: +(performance.now() - t0).toFixed(1), n, seq: seqLen() };
    };
    m.resetCache();
    B.results = { seed: await run(prefix, 1), steps: {}, decode: null };
    mark('seeded');
    // warm each path once, then 3 measured reps
    const rnd = (n, k) => Array.from({ length: n }, (_, i) => 1000 + ((k * 131 + i * 17) % 20000));
    for (const N of [1, 8, 16, 32, 1, 8, 16, 32, 1, 8, 16, 32]) {
      const r = await run(rnd(N, N), 1);
      (B.results.steps[N] ??= []).push(r);
      mark(`N=${N} first=${r.first}ms seq=${r.seq}`);
    }
    B.results.decode = await run(rnd(1, 7), 33);
    mark('done');
    B.results.summary = Object.fromEntries(Object.entries(B.results.steps).map(([N, rs]) => {
      const f = rs.map(r => r.first).sort((a, b) => a - b); return [N, { median_first_ms: f[Math.floor(f.length / 2)], all: f }];
    }));
    B.results.decodePerTokMs = +(((B.results.decode.total - B.results.decode.first) / (B.results.decode.n - 1)).toFixed(1));
  } catch (e) { B.error = String(e && e.stack || e); mark('error'); }
})();

// On-page banner so the run can be followed without DevTools.
(() => { const d = document.createElement('div'); d.id = 'bmOverlay'; d.style.cssText = 'position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:99999;background:#111;color:#0f0;font:14px/1.4 monospace;padding:10px 16px;border-radius:8px;max-width:90vw;white-space:pre-wrap'; document.body.appendChild(d); setInterval(() => { const B = window.__bm || {}; const s = B.results && B.results.summary; d.textContent = 'WebGPU verify-cost harness - keep this tab in front\nstate: ' + (B.state||'?') + '  ' + (B.prog||'') + (B.error ? '\nERROR ' + B.error.slice(0,80) : '') + (s ? '\nDONE  median ms  N=1 ' + s[1].median_first_ms + '  N=8 ' + s[8].median_first_ms + '  N=16 ' + s[16].median_first_ms + '  N=32 ' + s[32].median_first_ms + '  decode/tok ' + B.results.decodePerTokMs : ''); }, 500); })();
'armed';
