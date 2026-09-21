// Kernel unit tests on tiny random inputs vs JS references (no model load). window.__dr mirrors harness.js.
import { Drafter, CFG } from './drafter.js';
import { f32ArrayToF16, f16ToF32, f32ToF16, dequantize, GGML } from './gguf.js';
const V = window.__dr = { state: 'init', results: {}, error: null, log: [] };
const log = (...a) => { const s = a.join(' '); V.log.push([Date.now(), s]); console.log(s); };
let seed = 12345; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
const randf = (n, s = 1) => Float32Array.from({ length: n }, () => rnd() * s);
const f16round = (a) => { const h = f32ArrayToF16(a); return Float32Array.from(h, f16ToF32); };
function cmp(name, got, ref, tol) { let m = 0, at = -1; for (let i = 0; i < ref.length; ++i) { const d = Math.abs(got[i] - ref[i]); if (d > m) { m = d; at = i; } } const ok = m <= tol; V.results[name] = { maxAbs: m, at, got: got[at], ref: ref[at], ok }; log(`${ok ? 'OK ' : 'BAD'} ${name} maxAbs ${m.toExponential(2)} (got ${got[at]} ref ${ref[at]} @${at})`); return ok; }
(async () => {
  try {
    const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error('no adapter');
    const device = await adapter.requestDevice(); device.addEventListener('uncapturederror', e => { log('GPU error ' + e.error.message); V.error = e.error.message; });
    const dr = new Drafter(device, { log }); const H = CFG.H;
    const uploadF16 = (f32) => { const h = f32ArrayToF16(f32); const b = device.createBuffer({ size: h.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, h.buffer); return b; };
    const runPass = (fn) => { const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); fn(pass); pass.end(); device.queue.submit([enc.finish()]); };
    // gemm: M=27 (fc-like multi-tile), K=64, N=13 (odd N to test bounds)
    { const M = 27, K = 72, N = 13; const X = randf(M * K), W = randf(N * K); const Wr = f16round(W); const ref = new Float32Array(M * N); for (let m = 0; m < M; ++m) for (let n = 0; n < N; ++n) { let s = 0; for (let k = 0; k < K; ++k) s += X[m * K + k] * Wr[n * K + k]; ref[m * N + n] = s; }
      const Xb = dr.upload(X), Wb = uploadF16(W), Yb = dr.buf(M * N); runPass(p => dr.gemm(p, Xb, Wb, Yb, M, K, N)); cmp('gemm', await dr.read(Yb, M * N), ref, 1e-4); }
    // packed gemms on random blocks: Q4_K (K=512 -> 2 blocks/row) and Q6_K, M=11 (2 row tiles), N=7
    for (const [type, bpb, name] of [[GGML.Q4_K, 144, 'gemm_q4k'], [GGML.Q6_K, 210, 'gemm_q6k']]) {
      const M = 11, K = 512, N = 7, nb = N * K / 256; const raw = new Uint8Array(nb * bpb); for (let i = 0; i < raw.length; ++i) raw[i] = (rnd() + 0.5) * 256 | 0;
      const h16 = (v) => { const b = f32ToF16(v); return [b & 0xff, b >> 8]; };
      for (let b = 0; b < nb; ++b) { if (type === GGML.Q4_K) { raw.set(h16(0.01 + 0.05 * (rnd() + 0.5)), b * 144); raw.set(h16(0.002 + 0.01 * (rnd() + 0.5)), b * 144 + 2); } else raw.set(h16(0.01 + 0.05 * (rnd() + 0.5)), b * 210 + 208); }
      const Wf = dequantize(type, raw, N * K); const X = randf(M * K); const ref = new Float32Array(M * N); for (let m = 0; m < M; ++m) for (let n = 0; n < N; ++n) { let a = 0; for (let k = 0; k < K; ++k) a += X[m * K + k] * Wf[n * K + k]; ref[m * N + n] = a; }
      const Wp = dr.uploadPacked(type, raw, N * K, name); const Xb = dr.upload(X), Yb = dr.buf(M * N); runPass(p => dr.gemm(p, Xb, Wp, Yb, M, K, N)); cmp(name, await dr.read(Yb, M * N), ref, 2e-4);
    }
    // rmsnorm rows=3, D=H
    { const rows = 3, D = H; const X = randf(rows * D, 4), W = randf(D, 2); const ref = new Float32Array(rows * D); for (let r = 0; r < rows; ++r) { let s = 0; for (let i = 0; i < D; ++i) s += X[r * D + i] ** 2; const inv = 1 / Math.sqrt(s / D + CFG.EPS); for (let i = 0; i < D; ++i) ref[r * D + i] = X[r * D + i] * inv * W[i]; }
      const Xb = dr.upload(X), Wb = dr.upload(W), Yb = dr.buf(rows * D); runPass(p => dr.rmsnorm(p, Xb, Wb, Yb, rows, D)); cmp('rmsnorm', await dr.read(Yb, rows * D), ref, 1e-5); }
    // rope rows=3 heads=2 pos0=27
    { const rows = 3, heads = 2, pos0 = 27; const X = randf(rows * heads * 128); const ref = new Float32Array(X); for (let r = 0; r < rows; ++r) for (let h = 0; h < heads; ++h) { const b = (r * heads + h) * 128; for (let i = 0; i < 64; ++i) { const f = Math.pow(CFG.THETA, -2 * i / 128); const a = (pos0 + r) * f; const c = Math.cos(a), s = Math.sin(a); const x1 = X[b + i], x2 = X[b + i + 64]; ref[b + i] = x1 * c - x2 * s; ref[b + i + 64] = x1 * s + x2 * c; } }
      const Xb = dr.upload(X); runPass(p => dr.rope(p, Xb, rows, heads, pos0)); cmp('rope', await dr.read(Xb, X.length), ref, 1e-4); }
    // conv L=8, a=1, with residual
    { const L = 8, G = H / 16; const X = randf(L * H), DYN = randf(L * 4 * G), BASE = randf(2 * 2 * H), RES = randf(L * H); const a = 1; const ref = new Float32Array(L * H); for (let l = 0; l < L; ++l) for (let h = 0; h < H; ++h) { const g = (h / 16) | 0; let acc = RES[l * H + h]; for (let off = 0; off < 2; ++off) if (l >= off) acc += (BASE[a * 2 * H + off * H + h] + DYN[l * 4 * G + a * 2 * G + off * G + g]) * X[(l - off) * H + h]; ref[l * H + h] = acc; }
      const Xb = dr.upload(X), Db = dr.upload(DYN), Bb = dr.upload(BASE), Rb = dr.upload(RES), Yb = dr.buf(L * H); runPass(p => dr.conv(p, Xb, Db, Bb, Rb, Yb, L, a, true)); cmp('conv_a1_res', await dr.read(Yb, L * H), ref, 1e-5);
      const ref0 = new Float32Array(L * H); for (let l = 0; l < L; ++l) for (let h = 0; h < H; ++h) { const g = (h / 16) | 0; let acc = 0; for (let off = 0; off < 2; ++off) if (l >= off) acc += (BASE[off * H + h] + DYN[l * 4 * G + off * G + g]) * X[(l - off) * H + h]; ref0[l * H + h] = acc; }
      runPass(p => dr.conv(p, Xb, Db, Bb, Rb, Yb, L, 0, false)); cmp('conv_a0', await dr.read(Yb, L * H), ref0, 1e-5); }
    // attention: C=27, L=8, NH=32, NKV=8, window 2048 (all allowed) and a small-window variant
    { const C = 27, L = 8, NH = CFG.NH, NKV = CFG.NKV, D = 128; const Q = randf(L * NH * D, 2), Kc = randf(C * NKV * D, 2), Vc = randf(C * NKV * D), Kb = randf(L * NKV * D, 2), Vb = randf(L * NKV * D); const scale = 1 / Math.sqrt(D);
      const refAttn = (window) => { const O = new Float32Array(L * NH * D); for (let m = 0; m < L; ++m) for (let h = 0; h < NH; ++h) { const kvh = (h / (NH / NKV)) | 0; const qb = (m * NH + h) * D; const sc = []; for (let j = 0; j < C + L; ++j) { let s = 0; const kb = j < C ? (j * NKV + kvh) * D : ((j - C) * NKV + kvh) * D; const K = j < C ? Kc : Kb; for (let d = 0; d < D; ++d) s += Q[qb + d] * K[kb + d]; const ok = j < C ? ((C + m) - j) < window : true; sc.push(ok ? s * scale : -1e30); } const mx = Math.max(...sc); const e = sc.map(s => Math.exp(s - mx)); const sum = e.reduce((a, b) => a + b, 0); for (let d = 0; d < D; ++d) { let a = 0; for (let j = 0; j < C + L; ++j) { const vb = j < C ? (j * NKV + kvh) * D : ((j - C) * NKV + kvh) * D; a += e[j] * (j < C ? Vc : Vb)[vb + d]; } O[qb + d] = a / sum; } } return O; };
      const Qb = dr.upload(Q), Kcb = dr.upload(Kc), Vcb = dr.upload(Vc), Kbb = dr.upload(Kb), Vbb = dr.upload(Vb), Ob = dr.buf(L * NH * D); const posB = dr.buf(C); device.queue.writeBuffer(posB, 0, Uint32Array.from({ length: C }, (_, j) => j));
      runPass(p => dr.attn(p, Qb, Kcb, Vcb, Kbb, Vbb, Ob, C, L, C, posB)); cmp('attn_window2048', await dr.read(Ob, L * NH * D), refAttn(CFG.WINDOW), 1e-4);
      const saved = CFG.WINDOW; CFG.WINDOW = 20; runPass(p => dr.attn(p, Qb, Kcb, Vcb, Kbb, Vbb, Ob, C, L, C, posB)); cmp('attn_window20', await dr.read(Ob, L * NH * D), refAttn(20), 1e-4); CFG.WINDOW = saved; }
    // eviction (evict() directly, no weights): 72 rows in a sink-16/window-32 context, row r of K/V = r, pos = r -> keep [0..15] ++ [40..71]
    { const ctxE = dr.createContext(80, { sink: 16, window: 32 }); const RW = CFG.NKV * CFG.HD; const rowsK = new Float32Array(72 * RW); for (let r = 0; r < 72; ++r) rowsK.fill(r, r * RW, (r + 1) * RW);
      for (const l of ctxE.layers) { device.queue.writeBuffer(l.k, 0, rowsK); device.queue.writeBuffer(l.v, 0, rowsK); } device.queue.writeBuffer(ctxE.pos, 0, Uint32Array.from({ length: 72 }, (_, i) => i)); ctxE.C = 72; ctxE.total = 72;
      dr.evict(ctxE); const pos = new Uint32Array((await dr.read(ctxE.pos, ctxE.C)).buffer); const exp = [...Array.from({ length: 16 }, (_, i) => i), ...Array.from({ length: 32 }, (_, i) => 40 + i)];
      const kE = await dr.read(ctxE.layers[4].v, 48 * RW); let bad = 0; for (let i = 0; i < 48; ++i) for (let j = 0; j < RW; ++j) if (kE[i * RW + j] !== exp[i]) bad++;
      V.results.evict = { ok: ctxE.C === 48 && exp.every((v, i) => pos[i] === v) && bad === 0, C: ctxE.C, evictions: ctxE.evictions, badRows: bad, pos: Array.from(pos.slice(12, 20)) }; log((V.results.evict.ok ? 'OK ' : 'BAD') + ' evict ' + JSON.stringify(V.results.evict)); }
    // topk: R=3 rows, V=248320 random logits, vs a CPU sort
    { const R = 3, VV = 248320; const L = randf(R * VV, 30); const Lb = dr.upload(L); let t; runPass(p => { t = dr.topk(p, Lb, R, VV); }); const ov = await dr.read(t.ov, R * 16); const oi = new Uint32Array((await dr.read(t.oi, R * 16)).buffer);
      let ok = true; for (let r = 0; r < R; ++r) { const ref = Array.from({ length: VV }, (_, j) => j).sort((a, b) => L[r * VV + b] - L[r * VV + a]).slice(0, 16); const got = Array.from(oi.subarray(r * 16, r * 16 + 16)); const refSet = new Set(ref); if (!got.every(j => refSet.has(j)) || new Set(got).size !== 16) ok = false; for (let c = 0; c < 16; ++c) if (ov[r * 16 + c] !== L[r * VV + got[c]]) ok = false; }
      V.results.topk = { ok }; log((ok ? 'OK ' : 'BAD') + ' topk (3 rows x 248320: id set == CPU top-16, values consistent)'); }
    // silu_mul + scale
    { const n = 1000; const G = randf(n, 6), U = randf(n, 3); const ref = Float32Array.from(G, (g, i) => g / (1 + Math.exp(-g)) * U[i]); const Gb = dr.upload(G), Ub = dr.upload(U), Yb = dr.buf(n); runPass(p => dr.silu(p, Gb, Ub, Yb, n)); cmp('silu_mul', await dr.read(Yb, n), ref, 1e-5);
      runPass(p => dr.scale(p, Gb, Yb, n, 0.25)); cmp('scale', await dr.read(Yb, n), Float32Array.from(G, g => g * 0.25), 1e-7); }
    // selector on synthetic codebooks is exercised by the full harness; here only the API shape
    V.state = Object.values(V.results).every(r => r.ok) && !V.error ? 'done' : 'error';
  } catch (e) { V.error = String(e && e.stack || e); V.state = 'error'; log('ERROR ' + V.error); }
})();
