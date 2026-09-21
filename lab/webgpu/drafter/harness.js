// Drafter harness: load the Q4_K_M drafter into WebGPU, feed the MLX oracle's inputs for one cycle, compare every stage.
// State is published on window.__dr for cdp-drive.mjs. Query params: ?oracle=bf16|gguf (default bf16) &model=<gguf url> &reps=N
import { Drafter, CFG } from './drafter.js';
import { fetchNpy } from './npy.js';
const V = window.__dr = { state: 'init', prog: null, results: {}, error: null, log: [] };
const log = (...a) => { const s = a.join(' '); V.log.push([Date.now(), s]); console.log(s); };
const qs = new URLSearchParams(location.search); const ORACLE = qs.get('oracle') || 'bf16'; const MODEL = qs.get('model') || '../model/Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf'; const REPS = Number(qs.get('reps') || 5);
function cmp(name, got, ref) {
  if (got.length !== ref.length) throw new Error(`${name}: length ${got.length} vs ${ref.length}`);
  let maxAbs = 0, maxAt = -1, sd = 0, sr = 0, refMax = 0, nan = 0;
  for (let i = 0; i < ref.length; ++i) { const d = Math.abs(got[i] - ref[i]); if (!Number.isFinite(got[i])) nan++; if (d > maxAbs) { maxAbs = d; maxAt = i; } sd += d * d; sr += ref[i] * ref[i]; refMax = Math.max(refMax, Math.abs(ref[i])); }
  const r = { n: ref.length, maxAbs, at: maxAt, got: got[maxAt], ref: ref[maxAt], rmsRel: Math.sqrt(sd / ref.length) / (Math.sqrt(sr / ref.length) || 1), maxAbsOverRefMax: maxAbs / (refMax || 1), refMax, nonFinite: nan };
  V.results[name] = r; log(`${name}: maxAbs ${maxAbs.toExponential(3)} (got ${r.got?.toFixed(5)} ref ${r.ref?.toFixed(5)} @${maxAt}) rmsRel ${r.rmsRel.toExponential(3)} maxAbs/refMax ${r.maxAbsOverRefMax.toExponential(3)}${nan ? ' NONFINITE ' + nan : ''}`);
  return r;
}
(async () => {
  try {
    V.state = 'oracle'; const base = `./oracle/${ORACLE}/`;
    const idx = await (await fetch(base + 'index.json')).json(); V.oracle = { weights: idx.weights, C: idx.C, anchor: idx.anchor, embed_scale: idx.embed_scale, selected: idx.selected };
    const np = async (n) => (await fetchNpy(base + n + '.npy')).data;
    const C = idx.C; const feats = await np('context_features'); const noise = await np('noise_embedding');
    log(`oracle ${ORACLE}: C=${C} anchor=${idx.anchor} embed_scale=${idx.embed_scale} selected=${JSON.stringify(idx.selected)}`);
    V.state = 'gpu';
    if (!navigator.gpu) throw new Error('no navigator.gpu');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); if (!adapter) throw new Error('no WebGPU adapter');
    const lim = adapter.limits; const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [], requiredLimits: { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize, maxStorageBuffersPerShaderStage: Math.max(8, lim.maxStorageBuffersPerShaderStage >= 8 ? 8 : lim.maxStorageBuffersPerShaderStage) } });
    device.addEventListener('uncapturederror', (e) => { log('GPU uncaptured error: ' + e.error.message); V.error = V.error || e.error.message; });
    device.lost.then(i => { log('device lost: ' + i.message); V.error = V.error || 'device lost: ' + i.message; V.state = 'error'; });
    V.gpu = { vendor: adapter.info?.vendor, arch: adapter.info?.architecture, maxBuf: lim.maxBufferSize, maxBind: lim.maxStorageBufferBindingSize, f16: adapter.features.has('shader-f16') };
    log('adapter ' + JSON.stringify(V.gpu));
    const dr = new Drafter(device, { log, profile: qs.get('profile') === '1' });
    V.gpu.timestampQuery = adapter.features.has('timestamp-query'); V.gpu.profiling = dr.profile;
    V.state = 'weights'; await dr.loadWeights(MODEL, (p) => { V.prog = p; }); V.weights = dr.stats;
    // ---- stage A: projected context ----
    V.state = 'context'; let t0 = performance.now();
    const cache = dr.projectContext(feats, C, 0); await device.queue.onSubmittedWorkDone(); V.timing = { contextMs: performance.now() - t0 };
    cmp('draft_context', await dr.read(cache.ctx, C * CFG.H), await np('draft_context'));
    for (let i = 0; i < CFG.L; ++i) { cmp(`layer${i}_ctx_k`, await dr.read(cache.layers[i].k, C * CFG.NKV * CFG.HD), await np(`layer${i}_ctx_k`)); cmp(`layer${i}_ctx_v`, await dr.read(cache.layers[i].v, C * CFG.NKV * CFG.HD), await np(`layer${i}_ctx_v`)); }
    // ---- stage B: draft step (cold) ----
    V.state = 'draft'; t0 = performance.now();
    const st = dr.draftStep(cache, noise, idx.embed_scale); await device.queue.onSubmittedWorkDone(); V.timing.coldStepMs = performance.now() - t0;
    for (let i = 0; i < CFG.L; ++i) { for (const s of ['attn_in', 'attn_out', 'out']) cmp(`layer${i}_${s}`, await dr.read(st.layers[i][s], CFG.BLOCK * CFG.H), await np(`layer${i}_${s}`)); }
    const fin = await dr.read(st.final, CFG.BLOCK * CFG.H); cmp('final_hidden', fin, await np('final_hidden'));
    const selH = await dr.read(st.selHidden, (CFG.BLOCK - 1) * CFG.RANK); cmp('sel_hidden', selH, await np('sel_hidden'));
    // ---- stage C: head (stubbed from the oracle) + selector ----
    V.state = 'select';
    const candFlat = await np('cand_ids'); const unaryFlat = await np('logits_top16'); const headRows = await np('head_rows'); const edgesRef = await np('sel_edges'); const argmaxRef = await np('argmax');
    const S = CFG.BLOCK - 1, K = CFG.TOPK, H = CFG.H;
    const cand = [], unaryOracle = [], unaryOwn = [];
    for (let s = 0; s < S; ++s) { cand.push(Array.from(candFlat.subarray(s * K, (s + 1) * K))); unaryOracle.push(Array.from(unaryFlat.subarray(s * K, (s + 1) * K))); const row = []; for (let c = 0; c < K; ++c) { let a = 0; const hb = (s + 1) * H, rb = (s * K + c) * H; for (let j = 0; j < H; ++j) a += fin[hb + j] * headRows[rb + j]; row.push(a); } unaryOwn.push(row); }
    cmp('unary_logits_from_own_hidden', Float32Array.from(unaryOwn.flat()), Float32Array.from(unaryOracle.flat()));
    // argmax within the candidate set, oracle vs own-hidden logits
    const amOracle = unaryOracle.map((r, s) => cand[s][r.indexOf(Math.max(...r))]), amOwn = unaryOwn.map((r, s) => cand[s][r.indexOf(Math.max(...r))]);
    V.results.argmax = { oracle_global: Array.from(argmaxRef), oracle_in_top16: amOracle, own_hidden_in_top16: amOwn, agree: amOwn.map((v, s) => v === argmaxRef[s]) };
    log('argmax per slot: oracle ' + JSON.stringify(Array.from(argmaxRef)) + ' own-hidden(top16 stub) ' + JSON.stringify(amOwn) + ' agree ' + V.results.argmax.agree.filter(Boolean).length + '/' + S);
    const selA = dr.select(idx.anchor, cand, unaryOracle, selH); const selB = dr.select(idx.anchor, cand, unaryOwn, selH);
    cmp('sel_edges', Float32Array.from(selA.edges.flatMap(e => Array.from(e))), edgesRef);
    V.results.selected = { oracle: idx.selected, stub_unary: selA.path, own_unary: selB.path, match_stub: JSON.stringify(selA.path) === JSON.stringify(idx.selected), match_own: JSON.stringify(selB.path) === JSON.stringify(idx.selected) };
    log('selected: oracle ' + JSON.stringify(idx.selected) + ' | browser (oracle unary) ' + JSON.stringify(selA.path) + ' match=' + V.results.selected.match_stub + ' | browser (own-hidden unary) ' + JSON.stringify(selB.path) + ' match=' + V.results.selected.match_own);
    // ---- warm timings ----
    V.state = 'timing'; const release = (s) => { for (const l of s.layers) for (const b of Object.values(l)) b.destroy(); s.final.destroy(); s.selHidden.destroy(); s.h0.destroy(); };
    release(st); const times = [];
    for (let r = 0; r < REPS; ++r) { const t1 = performance.now(); const s2 = dr.draftStep(cache, noise, idx.embed_scale); await device.queue.onSubmittedWorkDone(); times.push(performance.now() - t1); release(s2); }
    const ctxTimes = []; for (let r = 0; r < 3; ++r) { const t1 = performance.now(); const c2 = dr.projectContext(feats, C, 0); await device.queue.onSubmittedWorkDone(); ctxTimes.push(performance.now() - t1); c2.ctx.destroy(); c2.fcOut.destroy(); c2.F.destroy(); for (const l of c2.layers) { l.k.destroy(); l.v.destroy(); } }
    // selector CPU time
    const t2 = performance.now(); for (let r = 0; r < 10; ++r) dr.select(idx.anchor, cand, unaryOracle, selH); V.timing.selectorCpuMs = (performance.now() - t2) / 10;
    if (dr.profile) { const s3 = dr.draftStep(cache, noise, idx.embed_scale); await device.queue.onSubmittedWorkDone(); const prof = await dr.profileTimes(); release(s3); const agg = {}; for (const { op, ms } of prof) { agg[op] = (agg[op] || 0) + ms; } V.timing.profileMs = Object.fromEntries(Object.entries(agg).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, +v.toFixed(3)])); V.timing.profileTotalMs = +prof.reduce((a, b) => a + b.ms, 0).toFixed(2); log('profile (gpu ms per op class, one warm step): ' + JSON.stringify(V.timing.profileMs) + ' total ' + V.timing.profileTotalMs); }
    V.timing.warmStepMs = times; V.timing.warmStepMedianMs = [...times].sort((a, b) => a - b)[times.length >> 1]; V.timing.contextWarmMs = ctxTimes;
    log('timing ' + JSON.stringify(V.timing));
    V.state = 'done';
  } catch (e) { V.error = String(e && e.stack || e); V.state = 'error'; log('ERROR ' + V.error); }
})();
