// Kernel unit tests on tiny random inputs vs JS references (no model load). window.__dr mirrors harness.js.
import { Drafter, CFG } from './drafter.js';
import { f32ArrayToF16, f16ToF32 } from './gguf.js';
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
    { const M = 27, K = 64, N = 13; const X = randf(M * K), W = randf(N * K); const Wr = f16round(W); const ref = new Float32Array(M * N); for (let m = 0; m < M; ++m) for (let n = 0; n < N; ++n) { let s = 0; for (let k = 0; k < K; ++k) s += X[m * K + k] * Wr[n * K + k]; ref[m * N + n] = s; }
      const Xb = dr.upload(X), Wb = uploadF16(W), Yb = dr.buf(M * N); runPass(p => dr.gemm(p, Xb, Wb, Yb, M, K, N)); cmp('gemm', await dr.read(Yb, M * N), ref, 1e-4); }
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
      const Qb = dr.upload(Q), Kcb = dr.upload(Kc), Vcb = dr.upload(Vc), Kbb = dr.upload(Kb), Vbb = dr.upload(Vb), Ob = dr.buf(L * NH * D);
      runPass(p => dr.attn(p, Qb, Kcb, Vcb, Kbb, Vbb, Ob, C, L, C, 0)); cmp('attn_window2048', await dr.read(Ob, L * NH * D), refAttn(CFG.WINDOW), 1e-4);
      const saved = CFG.WINDOW; CFG.WINDOW = 20; runPass(p => dr.attn(p, Qb, Kcb, Vcb, Kbb, Vbb, Ob, C, L, C, 0)); cmp('attn_window20', await dr.read(Ob, L * NH * D), refAttn(20), 1e-4); CFG.WINDOW = saved; }
    // silu_mul + scale
    { const n = 1000; const G = randf(n, 6), U = randf(n, 3); const ref = Float32Array.from(G, (g, i) => g / (1 + Math.exp(-g)) * U[i]); const Gb = dr.upload(G), Ub = dr.upload(U), Yb = dr.buf(n); runPass(p => dr.silu(p, Gb, Ub, Yb, n)); cmp('silu_mul', await dr.read(Yb, n), ref, 1e-5);
      runPass(p => dr.scale(p, Gb, Yb, n, 0.25)); cmp('scale', await dr.read(Yb, n), Float32Array.from(G, g => g * 0.25), 1e-7); }
    // selector on synthetic codebooks is exercised by the full harness; here only the API shape
    V.state = Object.values(V.results).every(r => r.ok) && !V.error ? 'done' : 'error';
  } catch (e) { V.error = String(e && e.stack || e); V.state = 'error'; log('ERROR ' + V.error); }
})();
