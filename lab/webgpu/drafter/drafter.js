// DFlash2 drafter forward in WebGPU (WGSL compute), weights from the Q4_K_M GGUF dequantized on the CPU into f16
// (packed two per u32, unpacked in-shader with unpack2x16float, so no shader-f16 feature is required); activations f32.
// Semantics follow the MLX oracle (dflash_mlx/draft/dflash2.py + dflash_mlx/model.py DFlashAttention, cache=ContextOnly,
// first cycle: context positions 0..C-1, block positions C..C+7, non-causal block, sliding window over the context).
import { readGGUF, dequantize, dequantRow, f32ArrayToF16, fetchSource, GGML } from './gguf.js';

export const CFG = { H: 5120, I: 17408, L: 5, NH: 32, NKV: 8, HD: 128, BLOCK: 8, GROUP: 16, KCONV: 2, RANK: 256, TOPK: 16, VOCAB: 248320, EPS: 1e-6, THETA: 1e7, WINDOW: 2048, MASK: 248070 };

// ---------------- WGSL ----------------
const WG_GEMM = /* wgsl */`
struct P { M: u32, K: u32, N: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 512>;
// Y[m,n] = sum_k X[m,k] * W[n,k]; W is f16 pairs packed in u32, read 8 at a time (vec4<u32>); K % 8 == 0.
// 64 threads = 4 n-columns x 16 k-lanes (16 lanes x 16 B = 256 B contiguous per iteration); 8-row tile per wg.y.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let nl = tid >> 4u; let ks = tid & 15u;
  let n = wg.x * 4u + nl; let m0 = wg.y * 8u;
  let K8 = p.K >> 3u; let K4 = p.K >> 2u;
  var acc: array<f32, 8>;
  for (var m = 0u; m < 8u; m++) { acc[m] = 0.0; }
  if (n < p.N) {
    let wb = n * K8;
    let rows = min(8u, p.M - m0);
    for (var k8 = ks; k8 < K8; k8 += 16u) {
      let wv = W[wb + k8];
      let w0 = unpack2x16float(wv.x); let w1 = unpack2x16float(wv.y); let w2 = unpack2x16float(wv.z); let w3 = unpack2x16float(wv.w);
      let wa = vec4<f32>(w0.x, w0.y, w1.x, w1.y); let wb4 = vec4<f32>(w2.x, w2.y, w3.x, w3.y);
      let x4 = 2u * k8;
      for (var m = 0u; m < rows; m++) {
        let xb = (m0 + m) * K4 + x4;
        acc[m] += dot(wa, X[xb]) + dot(wb4, X[xb + 1u]);
      }
    }
  }
  for (var m = 0u; m < 8u; m++) { red[tid * 8u + m] = acc[m]; }
  workgroupBarrier();
  if (ks == 0u && n < p.N) {
    for (var m = 0u; m < 8u; m++) {
      if (m0 + m < p.M) {
        var s = 0.0;
        for (var j = 0u; j < 16u; j++) { s += red[(nl * 16u + j) * 8u + m]; }
        Y[(m0 + m) * p.N + n] = s;
      }
    }
  }
}`;

// ---- packed-weight GEMMs: the GGUF blocks stay on the GPU as raw bytes and are dequantized in-shader ----
// Q4_K: block = 256 values, 144 B = 36 u32: [d|dmin] f16 pair, scales[12] (3 u32), qs[128] (32 u32). Lane w of 32 owns qs word w:
// pair j = w>>3 (sub-blocks 2j: low nibbles, k = 256b+64j+4i.. ; 2j+1: high nibbles, k+32), i = w&7. y = d*sc*sum(q*x) - dmin*mn*sum(x).
const WG_GEMM_Q4K = /* wgsl */`
struct P { M: u32, K: u32, N: u32, ks: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> xs: array<vec4<f32>, 512>;   // X tile: 8 rows x 256 k (one block) as vec4
var<workgroup> red: array<f32, 1024>;
fn sb(s0: u32, s1: u32, s2: u32, i: u32) -> u32 { var w = s0; if (i >= 8u) { w = s2; } else if (i >= 4u) { w = s1; } return (w >> (8u * (i & 3u))) & 0xffu; }
fn scmin(s0: u32, s1: u32, s2: u32, j: u32) -> vec2<f32> {
  if (j < 4u) { return vec2<f32>(f32(sb(s0, s1, s2, j) & 63u), f32(sb(s0, s1, s2, j + 4u) & 63u)); }
  let a = sb(s0, s1, s2, j + 4u); let b = sb(s0, s1, s2, j - 4u); let c = sb(s0, s1, s2, j);
  return vec2<f32>(f32((a & 0xfu) | ((b >> 6u) << 4u)), f32((a >> 4u) | ((c >> 6u) << 4u)));
}
fn nib_lo(w: u32) -> vec4<f32> { return vec4<f32>(f32(w & 0xfu), f32((w >> 8u) & 0xfu), f32((w >> 16u) & 0xfu), f32((w >> 24u) & 0xfu)); }
fn nib_hi(w: u32) -> vec4<f32> { return vec4<f32>(f32((w >> 4u) & 0xfu), f32((w >> 12u) & 0xfu), f32((w >> 20u) & 0xfu), f32((w >> 28u) & 0xfu)); }
// 64 threads = 16 column PAIRS x 4 lanes; lane q owns pair j = q of every block (qs words 8q..8q+7 = sub-blocks 2q, 2q+1).
// Each thread computes 2 adjacent output columns so every staged X load feeds 16 FMAs. X tile staged per block in workgroup memory.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let col = tid >> 2u; let q = tid & 3u;
  let n0 = wg.x * 32u + col * 2u; let n1 = n0 + 1u; let m0 = wg.y * 8u;
  let nb = p.K >> 8u; let K4 = p.K >> 2u;
  let rows = min(8u, p.M - m0);
  let ok0 = n0 < p.N; let ok1 = n1 < p.N;
  let bStart = (wg.z * nb) / p.ks; let bEnd = ((wg.z + 1u) * nb) / p.ks;
  var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0; var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
  var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0; var b4 = 0.0; var b5 = 0.0; var b6 = 0.0; var b7 = 0.0;
  for (var b = bStart; b < bEnd; b++) {
    workgroupBarrier();
    for (var idx = tid; idx < 512u; idx += 64u) { let r = idx >> 6u; let c = idx & 63u; if (r < rows) { xs[idx] = X[(m0 + r) * K4 + b * 64u + c]; } else { xs[idx] = vec4<f32>(0.0); } }
    workgroupBarrier();
    if (ok0) {
      let bb0 = (n0 * nb + b) * 36u; let bb1 = (select(n0, n1, ok1) * nb + b) * 36u;
      let dd0 = unpack2x16float(W[bb0]); let t0 = W[bb0 + 1u]; let t1 = W[bb0 + 2u]; let t2 = W[bb0 + 3u];
      let dd1 = unpack2x16float(W[bb1]); let u0 = W[bb1 + 1u]; let u1 = W[bb1 + 2u]; let u2 = W[bb1 + 3u];
      let lo0 = scmin(t0, t1, t2, 2u * q); let hi0 = scmin(t0, t1, t2, 2u * q + 1u);
      let lo1 = scmin(u0, u1, u2, 2u * q); let hi1 = scmin(u0, u1, u2, 2u * q + 1u);
      let dlo0 = dd0.x * lo0.x; let mlo0 = dd0.y * lo0.y; let dhi0 = dd0.x * hi0.x; let mhi0 = dd0.y * hi0.y;
      let dlo1 = dd1.x * lo1.x; let mlo1 = dd1.y * lo1.y; let dhi1 = dd1.x * hi1.x; let mhi1 = dd1.y * hi1.y;
      for (var i = 0u; i < 8u; i++) {
        let w0 = W[bb0 + 4u + 8u * q + i]; let w1 = W[bb1 + 4u + 8u * q + i];
        let ql0 = nib_lo(w0); let qh0 = nib_hi(w0); let ql1 = nib_lo(w1); let qh1 = nib_hi(w1);
        let c4 = 16u * q + i;
        { let xl = xs[0u + c4]; let xh = xs[8u + c4]; let sl = xl.x + xl.y + xl.z + xl.w; let sh = xh.x + xh.y + xh.z + xh.w; a0 += dlo0 * dot(ql0, xl) - mlo0 * sl + dhi0 * dot(qh0, xh) - mhi0 * sh; b0 += dlo1 * dot(ql1, xl) - mlo1 * sl + dhi1 * dot(qh1, xh) - mhi1 * sh; }
        { let xl = xs[64u + c4]; let xh = xs[72u + c4]; let sl = xl.x + xl.y + xl.z + xl.w; let sh = xh.x + xh.y + xh.z + xh.w; a1 += dlo0 * dot(ql0, xl) - mlo0 * sl + dhi0 * dot(qh0, xh) - mhi0 * sh; b1 += dlo1 * dot(ql1, xl) - mlo1 * sl + dhi1 * dot(qh1, xh) - mhi1 * sh; }
        { let xl = xs[128u + c4]; let xh = xs[136u + c4]; let sl = xl.x + xl.y + xl.z + xl.w; let sh = xh.x + xh.y + xh.z + xh.w; a2 += dlo0 * dot(ql0, xl) - mlo0 * sl + dhi0 * dot(qh0, xh) - mhi0 * sh; b2 += dlo1 * dot(ql1, xl) - mlo1 * sl + dhi1 * dot(qh1, xh) - mhi1 * sh; }
        { let xl = xs[192u + c4]; let xh = xs[200u + c4]; let sl = xl.x + xl.y + xl.z + xl.w; let sh = xh.x + xh.y + xh.z + xh.w; a3 += dlo0 * dot(ql0, xl) - mlo0 * sl + dhi0 * dot(qh0, xh) - mhi0 * sh; b3 += dlo1 * dot(ql1, xl) - mlo1 * sl + dhi1 * dot(qh1, xh) - mhi1 * sh; }
        { let xl = xs[256u + c4]; let xh = xs[264u + c4]; let sl = xl.x + xl.y + xl.z + xl.w; let sh = xh.x + xh.y + xh.z + xh.w; a4 += dlo0 * dot(ql0, xl) - mlo0 * sl + dhi0 * dot(qh0, xh) - mhi0 * sh; b4 += dlo1 * dot(ql1, xl) - mlo1 * sl + dhi1 * dot(qh1, xh) - mhi1 * sh; }
        { let xl = xs[320u + c4]; let xh = xs[328u + c4]; let sl = xl.x + xl.y + xl.z + xl.w; let sh = xh.x + xh.y + xh.z + xh.w; a5 += dlo0 * dot(ql0, xl) - mlo0 * sl + dhi0 * dot(qh0, xh) - mhi0 * sh; b5 += dlo1 * dot(ql1, xl) - mlo1 * sl + dhi1 * dot(qh1, xh) - mhi1 * sh; }
        { let xl = xs[384u + c4]; let xh = xs[392u + c4]; let sl = xl.x + xl.y + xl.z + xl.w; let sh = xh.x + xh.y + xh.z + xh.w; a6 += dlo0 * dot(ql0, xl) - mlo0 * sl + dhi0 * dot(qh0, xh) - mhi0 * sh; b6 += dlo1 * dot(ql1, xl) - mlo1 * sl + dhi1 * dot(qh1, xh) - mhi1 * sh; }
        { let xl = xs[448u + c4]; let xh = xs[456u + c4]; let sl = xl.x + xl.y + xl.z + xl.w; let sh = xh.x + xh.y + xh.z + xh.w; a7 += dlo0 * dot(ql0, xl) - mlo0 * sl + dhi0 * dot(qh0, xh) - mhi0 * sh; b7 += dlo1 * dot(ql1, xl) - mlo1 * sl + dhi1 * dot(qh1, xh) - mhi1 * sh; }
      }
    }
  }
  let rb = tid * 16u;
  red[rb] = a0; red[rb + 1u] = a1; red[rb + 2u] = a2; red[rb + 3u] = a3; red[rb + 4u] = a4; red[rb + 5u] = a5; red[rb + 6u] = a6; red[rb + 7u] = a7;
  red[rb + 8u] = b0; red[rb + 9u] = b1; red[rb + 10u] = b2; red[rb + 11u] = b3; red[rb + 12u] = b4; red[rb + 13u] = b5; red[rb + 14u] = b6; red[rb + 15u] = b7;
  workgroupBarrier();
  if (q == 0u && ok0) {
    let yb = wg.z * p.M * p.N;
    for (var m = 0u; m < 8u; m++) {
      if (m0 + m < p.M) {
        Y[yb + (m0 + m) * p.N + n0] = red[rb + m] + red[rb + 16u + m] + red[rb + 32u + m] + red[rb + 48u + m];
        if (ok1) { Y[yb + (m0 + m) * p.N + n1] = red[rb + 8u + m] + red[rb + 24u + m] + red[rb + 40u + m] + red[rb + 56u + m]; }
      }
    }
  }
}`;
// Q6_K (212 B = 53 u32 stride per block): lane q of 4 owns half = q>>1, words t = 4(q&1)..+3 of that half (64 values); 2 columns per thread.
const WG_GEMM_Q6K = /* wgsl */`
struct P { M: u32, K: u32, N: u32, ks: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> xs: array<vec4<f32>, 512>;
var<workgroup> red: array<f32, 1024>;
fn i8at(w: u32, i: u32) -> f32 { return f32((i32(w << (8u * (3u - i))) >> 24)); }
fn q6(ql0: u32, ql1: u32, qh: u32, sh: u32) -> vec4<f32> {   // (q1, q2, q3, q4) for one byte position sh = 8i
  return vec4<f32>(f32(((ql0 >> sh) & 0xfu) | (((qh >> sh) & 3u) << 4u)), f32(((ql1 >> sh) & 0xfu) | (((qh >> (sh + 2u)) & 3u) << 4u)), f32(((ql0 >> (sh + 4u)) & 0xfu) | (((qh >> (sh + 4u)) & 3u) << 4u)), f32(((ql1 >> (sh + 4u)) & 0xfu) | (((qh >> (sh + 6u)) & 3u) << 4u))) - vec4<f32>(32.0);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let col = tid >> 2u; let q = tid & 3u;
  let n0 = wg.x * 32u + col * 2u; let n1 = n0 + 1u; let m0 = wg.y * 8u;
  let nb = p.K >> 8u; let K4 = p.K >> 2u;
  let bStart = (wg.z * nb) / p.ks; let bEnd = ((wg.z + 1u) * nb) / p.ks;
  let rows = min(8u, p.M - m0);
  let ok0 = n0 < p.N; let ok1 = n1 < p.N;
  let half = q >> 1u; let t0 = 4u * (q & 1u); let is = t0 >> 2u; let sbase = 8u * half + is;
  var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0; var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
  var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0; var b4 = 0.0; var b5 = 0.0; var b6 = 0.0; var b7 = 0.0;
  for (var b = bStart; b < bEnd; b++) {
    workgroupBarrier();
    for (var idx = tid; idx < 512u; idx += 64u) { let r = idx >> 6u; let c = idx & 63u; if (r < rows) { xs[idx] = X[(m0 + r) * K4 + b * 64u + c]; } else { xs[idx] = vec4<f32>(0.0); } }
    workgroupBarrier();
    if (ok0) {
      let bb0 = (n0 * nb + b) * 53u; let bb1 = (select(n0, n1, ok1) * nb + b) * 53u;
      let d0 = unpack2x16float(W[bb0 + 52u]).x; let d1 = unpack2x16float(W[bb1 + 52u]).x;
      let s1 = d0 * i8at(W[bb0 + 48u + (sbase >> 2u)], sbase & 3u); let s2 = d0 * i8at(W[bb0 + 48u + ((sbase + 2u) >> 2u)], (sbase + 2u) & 3u); let s3 = d0 * i8at(W[bb0 + 48u + ((sbase + 4u) >> 2u)], (sbase + 4u) & 3u); let s4 = d0 * i8at(W[bb0 + 48u + ((sbase + 6u) >> 2u)], (sbase + 6u) & 3u);
      let r1 = d1 * i8at(W[bb1 + 48u + (sbase >> 2u)], sbase & 3u); let r2 = d1 * i8at(W[bb1 + 48u + ((sbase + 2u) >> 2u)], (sbase + 2u) & 3u); let r3 = d1 * i8at(W[bb1 + 48u + ((sbase + 4u) >> 2u)], (sbase + 4u) & 3u); let r4 = d1 * i8at(W[bb1 + 48u + ((sbase + 6u) >> 2u)], (sbase + 6u) & 3u);
      for (var tt = 0u; tt < 4u; tt++) {
        let t = t0 + tt;
        let qh0 = W[bb0 + 32u + half * 8u + t]; let ql00 = W[bb0 + half * 16u + t]; let ql01 = W[bb0 + half * 16u + t + 8u];
        let qh1 = W[bb1 + 32u + half * 8u + t]; let ql10 = W[bb1 + half * 16u + t]; let ql11 = W[bb1 + half * 16u + t + 8u];
        // per byte position i: q1..q4 of column 0 -> (q1,q2,q3,q4) vectors over i
        var q1: vec4<f32>; var q2: vec4<f32>; var q3: vec4<f32>; var q4: vec4<f32>; var p1: vec4<f32>; var p2: vec4<f32>; var p3: vec4<f32>; var p4: vec4<f32>;
        for (var i = 0u; i < 4u; i++) { let v = q6(ql00, ql01, qh0, 8u * i); q1[i] = v.x; q2[i] = v.y; q3[i] = v.z; q4[i] = v.w; let u = q6(ql10, ql11, qh1, 8u * i); p1[i] = u.x; p2[i] = u.y; p3[i] = u.z; p4[i] = u.w; }
        let c4 = 32u * half + t;
        { let xb = 0u + c4; let x0 = xs[xb]; let x1 = xs[xb + 8u]; let x2 = xs[xb + 16u]; let x3 = xs[xb + 24u]; a0 += s1 * dot(q1, x0) + s2 * dot(q2, x1) + s3 * dot(q3, x2) + s4 * dot(q4, x3); b0 += r1 * dot(p1, x0) + r2 * dot(p2, x1) + r3 * dot(p3, x2) + r4 * dot(p4, x3); }
        { let xb = 64u + c4; let x0 = xs[xb]; let x1 = xs[xb + 8u]; let x2 = xs[xb + 16u]; let x3 = xs[xb + 24u]; a1 += s1 * dot(q1, x0) + s2 * dot(q2, x1) + s3 * dot(q3, x2) + s4 * dot(q4, x3); b1 += r1 * dot(p1, x0) + r2 * dot(p2, x1) + r3 * dot(p3, x2) + r4 * dot(p4, x3); }
        { let xb = 128u + c4; let x0 = xs[xb]; let x1 = xs[xb + 8u]; let x2 = xs[xb + 16u]; let x3 = xs[xb + 24u]; a2 += s1 * dot(q1, x0) + s2 * dot(q2, x1) + s3 * dot(q3, x2) + s4 * dot(q4, x3); b2 += r1 * dot(p1, x0) + r2 * dot(p2, x1) + r3 * dot(p3, x2) + r4 * dot(p4, x3); }
        { let xb = 192u + c4; let x0 = xs[xb]; let x1 = xs[xb + 8u]; let x2 = xs[xb + 16u]; let x3 = xs[xb + 24u]; a3 += s1 * dot(q1, x0) + s2 * dot(q2, x1) + s3 * dot(q3, x2) + s4 * dot(q4, x3); b3 += r1 * dot(p1, x0) + r2 * dot(p2, x1) + r3 * dot(p3, x2) + r4 * dot(p4, x3); }
        { let xb = 256u + c4; let x0 = xs[xb]; let x1 = xs[xb + 8u]; let x2 = xs[xb + 16u]; let x3 = xs[xb + 24u]; a4 += s1 * dot(q1, x0) + s2 * dot(q2, x1) + s3 * dot(q3, x2) + s4 * dot(q4, x3); b4 += r1 * dot(p1, x0) + r2 * dot(p2, x1) + r3 * dot(p3, x2) + r4 * dot(p4, x3); }
        { let xb = 320u + c4; let x0 = xs[xb]; let x1 = xs[xb + 8u]; let x2 = xs[xb + 16u]; let x3 = xs[xb + 24u]; a5 += s1 * dot(q1, x0) + s2 * dot(q2, x1) + s3 * dot(q3, x2) + s4 * dot(q4, x3); b5 += r1 * dot(p1, x0) + r2 * dot(p2, x1) + r3 * dot(p3, x2) + r4 * dot(p4, x3); }
        { let xb = 384u + c4; let x0 = xs[xb]; let x1 = xs[xb + 8u]; let x2 = xs[xb + 16u]; let x3 = xs[xb + 24u]; a6 += s1 * dot(q1, x0) + s2 * dot(q2, x1) + s3 * dot(q3, x2) + s4 * dot(q4, x3); b6 += r1 * dot(p1, x0) + r2 * dot(p2, x1) + r3 * dot(p3, x2) + r4 * dot(p4, x3); }
        { let xb = 448u + c4; let x0 = xs[xb]; let x1 = xs[xb + 8u]; let x2 = xs[xb + 16u]; let x3 = xs[xb + 24u]; a7 += s1 * dot(q1, x0) + s2 * dot(q2, x1) + s3 * dot(q3, x2) + s4 * dot(q4, x3); b7 += r1 * dot(p1, x0) + r2 * dot(p2, x1) + r3 * dot(p3, x2) + r4 * dot(p4, x3); }
      }
    }
  }
  let rb = tid * 16u;
  red[rb] = a0; red[rb + 1u] = a1; red[rb + 2u] = a2; red[rb + 3u] = a3; red[rb + 4u] = a4; red[rb + 5u] = a5; red[rb + 6u] = a6; red[rb + 7u] = a7;
  red[rb + 8u] = b0; red[rb + 9u] = b1; red[rb + 10u] = b2; red[rb + 11u] = b3; red[rb + 12u] = b4; red[rb + 13u] = b5; red[rb + 14u] = b6; red[rb + 15u] = b7;
  workgroupBarrier();
  if (q == 0u && ok0) {
    let yb = wg.z * p.M * p.N;
    for (var m = 0u; m < 8u; m++) {
      if (m0 + m < p.M) {
        Y[yb + (m0 + m) * p.N + n0] = red[rb + m] + red[rb + 16u + m] + red[rb + 32u + m] + red[rb + 48u + m];
        if (ok1) { Y[yb + (m0 + m) * p.N + n1] = red[rb + 8u + m] + red[rb + 24u + m] + red[rb + 40u + m] + red[rb + 56u + m]; }
      }
    }
  }
}`;
// sum the ks partial products [ks, M, N] -> Y [M, N]
const WG_KSUM = /* wgsl */`
struct P { n: u32, ks: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pt: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i = gid.x; if (i >= p.n) { return; } var s = 0.0; for (var k = 0u; k < p.ks; k++) { s += Pt[k * p.n + i]; } Y[i] = s; }`;

const WG_RMSNORM = /* wgsl */`
struct P { rows: u32, D: u32, eps: f32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x; let tid = lid.x; let base = r * p.D;
  var s = 0.0;
  for (var i = tid; i < p.D; i += 256u) { let v = X[base + i]; s += v * v; }
  red[tid] = s; workgroupBarrier();
  for (var o = 128u; o > 0u; o >>= 1u) { if (tid < o) { red[tid] += red[tid + o]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = tid; i < p.D; i += 256u) { Y[base + i] = X[base + i] * inv * W[i]; }
}`;

// Top-16 per row over logits [R, V]: one workgroup (256 threads) per row; each thread keeps a private top-16 over its strided
// slice -> storage scratch [R, 256, 16]; 16 threads merge 256 candidates each -> workgroup memory; thread 0 merges the 256.
const WG_TOPK = /* wgsl */`
struct P { R: u32, V: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> L: array<f32>;
@group(0) @binding(2) var<storage, read_write> SV: array<f32>;
@group(0) @binding(3) var<storage, read_write> SI: array<u32>;
@group(0) @binding(4) var<storage, read_write> OV: array<f32>;
@group(0) @binding(5) var<storage, read_write> OI: array<u32>;
var<workgroup> mv: array<f32, 256>;
var<workgroup> mi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x; let tid = lid.x; let base = r * p.V;
  var vals: array<f32, 16>; var ids: array<u32, 16>;
  for (var c = 0u; c < 16u; c++) { vals[c] = -3.0e38; ids[c] = 0u; }
  var minv = -3.0e38; var mini = 0u;
  for (var j = tid; j < p.V; j += 256u) {
    let v = L[base + j];
    if (v > minv) { vals[mini] = v; ids[mini] = j; minv = vals[0]; mini = 0u; for (var c = 1u; c < 16u; c++) { if (vals[c] < minv) { minv = vals[c]; mini = c; } } }
  }
  let sb = (r * 256u + tid) * 16u;
  for (var c = 0u; c < 16u; c++) { SV[sb + c] = vals[c]; SI[sb + c] = ids[c]; }
  storageBarrier(); workgroupBarrier();
  if (tid < 16u) {
    for (var c = 0u; c < 16u; c++) { vals[c] = -3.0e38; ids[c] = 0u; }
    minv = -3.0e38; mini = 0u;
    let s0 = (r * 256u + tid * 16u) * 16u;
    for (var e = 0u; e < 256u; e++) {
      let v = SV[s0 + e];
      if (v > minv) { vals[mini] = v; ids[mini] = SI[s0 + e]; minv = vals[0]; mini = 0u; for (var c = 1u; c < 16u; c++) { if (vals[c] < minv) { minv = vals[c]; mini = c; } } }
    }
    for (var c = 0u; c < 16u; c++) { mv[tid * 16u + c] = vals[c]; mi[tid * 16u + c] = ids[c]; }
  }
  workgroupBarrier();
  if (tid == 0u) {
    for (var c = 0u; c < 16u; c++) { vals[c] = -3.0e38; ids[c] = 0u; }
    minv = -3.0e38; mini = 0u;
    for (var e = 0u; e < 256u; e++) {
      let v = mv[e];
      if (v > minv) { vals[mini] = v; ids[mini] = mi[e]; minv = vals[0]; mini = 0u; for (var c = 1u; c < 16u; c++) { if (vals[c] < minv) { minv = vals[c]; mini = c; } } }
    }
    for (var c = 0u; c < 16u; c++) { OV[r * 16u + c] = vals[c]; OI[r * 16u + c] = ids[c]; }
  }
}`;

// Non-traditional (half-split) RoPE, MLX nn.RoPE(traditional=False): pairs (i, i+HD/2), freq_i = theta^(-2i/HD), position = pos0 + row.
const WG_ROPE = /* wgsl */`
struct P { rows: u32, heads: u32, pos0: u32, pad: u32, theta: f32, pad1: f32, pad2: f32, pad3: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x & 63u; let rh = gid.x >> 6u;
  if (rh >= p.rows * p.heads) { return; }
  let row = rh / p.heads;
  let pos = f32(p.pos0 + row);
  let freq = exp(-f32(2u * i) / 128.0 * log(p.theta));
  let ang = pos * freq; let c = cos(ang); let s = sin(ang);
  let b = rh * 128u;
  let x1 = X[b + i]; let x2 = X[b + i + 64u];
  X[b + i] = x1 * c - x2 * s;
  X[b + i + 64u] = x1 * s + x2 * c;
}`;

// Attention over [context ; block] keys. Q [L, NH, HD]; Kc/Vc [C, NKV, HD]; Kb/Vb [L, NKV, HD]; O [L, NH, HD].
// Context key j sits at absolute position Kpos[j] (allowed iff qpos - kpos < window); block keys are always allowed (non-causal block).
// Capacity: C + L <= 3072 keys per query (sliding window 2048 + sink 64 + block fits); the caller checks.
const WG_ATTN = /* wgsl */`
struct P { C: u32, L: u32, qPos0: u32, pad0: u32, window: u32, NH: u32, NKV: u32, pad: u32, scale: f32, pad1: f32, pad2: f32, pad3: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(7) var<storage, read> Kpos: array<u32>;
@group(0) @binding(2) var<storage, read> Kc: array<f32>;
@group(0) @binding(3) var<storage, read> Vc: array<f32>;
@group(0) @binding(4) var<storage, read> Kb: array<f32>;
@group(0) @binding(5) var<storage, read> Vb: array<f32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
const MAXK: u32 = 3072u;   // 12 KB of the 16 KB default workgroup storage (q + red take 768 B)
var<workgroup> sc: array<f32, 3072>;
var<workgroup> q: array<f32, 128>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = wg.x; let h = wg.y; let tid = lid.x;
  let kvh = h / (p.NH / p.NKV);
  let qb = (m * p.NH + h) * 128u;
  q[tid * 2u] = Q[qb + tid * 2u]; q[tid * 2u + 1u] = Q[qb + tid * 2u + 1u];
  workgroupBarrier();
  let nk = p.C + p.L; let qpos = p.qPos0 + m;
  var lmax = -1e30;
  for (var j = tid; j < nk; j += 64u) {
    var s = 0.0; var ok = true; var kb = 0u;
    if (j < p.C) { kb = (j * p.NKV + kvh) * 128u; ok = (qpos - Kpos[j]) < p.window; for (var d = 0u; d < 128u; d++) { s += q[d] * Kc[kb + d]; } }
    else { kb = ((j - p.C) * p.NKV + kvh) * 128u; for (var d = 0u; d < 128u; d++) { s += q[d] * Kb[kb + d]; } }
    s = select(-1e30, s * p.scale, ok);
    sc[j] = s; lmax = max(lmax, s);
  }
  red[tid] = lmax; workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) { if (tid < o) { red[tid] = max(red[tid], red[tid + o]); } workgroupBarrier(); }
  let gmax = red[0]; workgroupBarrier();
  var lsum = 0.0;
  for (var j = tid; j < nk; j += 64u) { let e = exp(sc[j] - gmax); sc[j] = e; lsum += e; }
  red[tid] = lsum; workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) { if (tid < o) { red[tid] += red[tid + o]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  let d0 = tid * 2u; let d1 = d0 + 1u;
  var a0 = 0.0; var a1 = 0.0;
  for (var j = 0u; j < p.C; j++) { let vb = (j * p.NKV + kvh) * 128u; let pj = sc[j]; a0 += pj * Vc[vb + d0]; a1 += pj * Vc[vb + d1]; }
  for (var j = 0u; j < p.L; j++) { let vb = (j * p.NKV + kvh) * 128u; let pj = sc[p.C + j]; a0 += pj * Vb[vb + d0]; a1 += pj * Vb[vb + d1]; }
  O[qb + d0] = a0 * inv; O[qb + d1] = a1 * inv;
}`;

// Grouped dynamic causal conv (kernel 2, group 16) over the L block rows:
// Y[l,h] = (RES[l,h] if useRes) + sum_off (BASE[a][off][h] + DYN[l, a*2*G + off*G + g]) * X[l-off, h], g = h / 16, X[-1] = 0.
const WG_CONV = /* wgsl */`
struct P { L: u32, H: u32, G: u32, a: u32, useRes: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> DYN: array<f32>;
@group(0) @binding(3) var<storage, read> BASE: array<f32>;
@group(0) @binding(4) var<storage, read> RES: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= p.L * p.H) { return; }
  let l = idx / p.H; let h = idx % p.H; let g = h / 16u;
  let dynRow = l * (4u * p.G) + p.a * (2u * p.G);
  var acc = 0.0;
  for (var off = 0u; off < 2u; off++) {
    if (l >= off) {
      let k = BASE[p.a * (2u * p.H) + off * p.H + h] + DYN[dynRow + off * p.G + g];
      acc += k * X[(l - off) * p.H + h];
    }
  }
  if (p.useRes == 1u) { acc += RES[idx]; }
  Y[idx] = acc;
}`;

const WG_SILU_MUL = /* wgsl */`
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.n) { return; }
  let g = G[i]; Y[i] = g / (1.0 + exp(-g)) * U[i];
}`;

const WG_SCALE = /* wgsl */`
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32, s: f32, p1: f32, p2: f32, p3: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i = gid.x; if (i >= p.n) { return; } Y[i] = X[i] * p.s; }`;

// ---------------- runtime ----------------
export class Drafter {
  constructor(device, opts = {}) {
    this.device = device; this.log = opts.log || (() => {}); this.pipes = {}; this.uniforms = new Map(); this.w = {}; this.stats = { weightBytes: 0, tensors: 0 };
    this.packed = opts.packed !== false;   // default: Q4_K/Q6_K stay packed on the GPU, dequantized in-shader; packed:false = f16 dequant at load (step 3)
    this.stats.packed = this.packed;
    // Optional per-op GPU timing: with the 'timestamp-query' feature each op runs in its own pass bracketed by timestamps.
    this.profile = !!opts.profile && device.features.has('timestamp-query');
    if (this.profile) { this.qs = device.createQuerySet({ type: 'timestamp', count: 1024 }); this.qbuf = device.createBuffer({ size: 1024 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }); this.qread = device.createBuffer({ size: 1024 * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); }
    for (const [k, src] of Object.entries({ gemm: WG_GEMM, gemm_q4k: WG_GEMM_Q4K, gemm_q6k: WG_GEMM_Q6K, ksum: WG_KSUM, rmsnorm: WG_RMSNORM, topk: WG_TOPK, rope: WG_ROPE, attn: WG_ATTN, conv: WG_CONV, silu: WG_SILU_MUL, scale: WG_SCALE })) {
      const module = device.createShaderModule({ code: src, label: k });
      this.pipes[k] = device.createComputePipeline({ label: k, layout: 'auto', compute: { module, entryPoint: 'main' } });
    }
  }
  buf(n, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label = '') { return this.device.createBuffer({ size: Math.max(16, Math.ceil(n * 4 / 16) * 16), usage, label }); }
  upload(f32, label) { const b = this.buf(f32.length, undefined, label); this.device.queue.writeBuffer(b, 0, f32.buffer, f32.byteOffset, f32.byteLength); return b; }
  uni(vals) {   // vals: array of {u: n} | {f: x}; 8 slots of 4 bytes
    const key = JSON.stringify(vals); let b = this.uniforms.get(key); if (b) return b;
    const ab = new ArrayBuffer(48); const dv = new DataView(ab);
    vals.forEach((v, i) => { if ('f' in v) dv.setFloat32(i * 4, v.f, true); else dv.setUint32(i * 4, v.u, true); });
    b = this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.device.queue.writeBuffer(b, 0, ab); this.uniforms.set(key, b); return b;
  }
  bind(pipe, entries) { return this.device.createBindGroup({ layout: this.pipes[pipe].getBindGroupLayout(0), entries: entries.map((e, i) => ({ binding: i, resource: e.buffer ? e : { buffer: e } })) }); }
  run(pass, pipe, entries, x, y = 1, z = 1) {
    if (this.profile && this.enc) { const i = this.marks.length; pass = this.enc.beginComputePass({ timestampWrites: { querySet: this.qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } }); this.marks.push(this.opLabel || pipe); }
    pass.setPipeline(this.pipes[pipe]); pass.setBindGroup(0, this.bind(pipe, entries)); pass.dispatchWorkgroups(x, y, z);
    if (this.profile && this.enc) pass.end();
  }
  // profiling bracket: begin() returns a (possibly dummy) pass; end() submits and, when profiling, resolves the timestamps
  begin() { this.enc = this.device.createCommandEncoder(); this.marks = []; return this.profile ? { end() {} } : this.enc.beginComputePass(); }
  end(pass) { pass.end(); if (this.profile) { this.enc.resolveQuerySet(this.qs, 0, 2 * this.marks.length, this.qbuf, 0); this.enc.copyBufferToBuffer(this.qbuf, 0, this.qread, 0, 2 * this.marks.length * 8); } this.device.queue.submit([this.enc.finish()]); this.enc = null; }
  async profileTimes() {   // -> [{op, ms}] for the last end() when profiling
    if (!this.profile || !this.marks) return null; await this.qread.mapAsync(GPUMapMode.READ); const t = new BigInt64Array(this.qread.getMappedRange().slice(0)); this.qread.unmap();
    return this.marks.map((op, i) => ({ op, ms: Number(t[2 * i + 1] - t[2 * i]) / 1e6 }));
  }

  // ---- ops (record into a compute pass) ----
  gemm(pass, X, W, Y, M, K, N) {
    if (W && W.q) {
      if (K % 256) throw new Error('gemm_q: K % 256 != 0');
      const nb = K / 256, wgx = Math.ceil(N / 32), wgy = Math.ceil(M / 8);
      const ks = Math.max(1, Math.min(nb, Math.floor(512 / (wgx * wgy))));   // K-split so that >= ~512 workgroups run for narrow N
      if (ks === 1) { this.run(pass, W.q === 'q4k' ? 'gemm_q4k' : 'gemm_q6k', [this.uni([{ u: M }, { u: K }, { u: N }, { u: 1 }]), X, W.buffer, Y], wgx, wgy, 1); return; }
      const key = ks * M * N; if (!this._part || this._part.n < key) this._part = { n: key, buf: this.buf(key, undefined, 'gemm_partials') };
      this.run(pass, W.q === 'q4k' ? 'gemm_q4k' : 'gemm_q6k', [this.uni([{ u: M }, { u: K }, { u: N }, { u: ks }]), X, W.buffer, this._part.buf], wgx, wgy, ks);
      this.run(pass, 'ksum', [this.uni([{ u: M * N }, { u: ks }, { u: 0 }, { u: 0 }]), this._part.buf, Y], Math.ceil(M * N / 256)); return;
    }
    if (K % 8) throw new Error('gemm: K % 8 != 0'); this.run(pass, 'gemm', [this.uni([{ u: M }, { u: K }, { u: N }, { u: 0 }]), X, W, Y], Math.ceil(N / 4), Math.ceil(M / 8));
  }
  // upload a raw GGUF tensor as a packed weight: Q4_K as-is (36 u32 per block), Q6_K repacked to a 53-u32 stride
  uploadPacked(type, raw, n, label) {
    let bytes;
    if (type === GGML.Q4_K) bytes = raw;
    else if (type === GGML.Q6_K) { const nb = n / 256; bytes = new Uint8Array(nb * 212); for (let b = 0; b < nb; ++b) bytes.set(raw.subarray(b * 210, b * 210 + 210), b * 212); }
    else throw new Error('uploadPacked: ' + type);
    const buf = this.device.createBuffer({ size: Math.ceil(bytes.byteLength / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label });
    this.device.queue.writeBuffer(buf, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return { q: type === GGML.Q4_K ? 'q4k' : 'q6k', buffer: buf, bytes: bytes.byteLength };
  }
  rmsnorm(pass, X, W, Y, rows, D) { this.run(pass, 'rmsnorm', [this.uni([{ u: rows }, { u: D }, { f: CFG.EPS }, { u: 0 }]), X, W, Y], rows); }
  rope(pass, X, rows, heads, pos0) { this.run(pass, 'rope', [this.uni([{ u: rows }, { u: heads }, { u: pos0 }, { u: 0 }, { f: CFG.THETA }, { f: 0 }, { f: 0 }, { f: 0 }]), X], rows * heads); }
  attn(pass, Q, Kc, Vc, Kb, Vb, O, C, L, qPos0, Kpos) { if (C + L > 3072) throw new Error(`attn: ${C + L} keys > 3072 capacity`); this.run(pass, 'attn', [this.uni([{ u: C }, { u: L }, { u: qPos0 }, { u: 0 }, { u: CFG.WINDOW }, { u: CFG.NH }, { u: CFG.NKV }, { u: 0 }, { f: 1 / Math.sqrt(CFG.HD) }, { f: 0 }, { f: 0 }, { f: 0 }]), Q, Kc, Vc, Kb, Vb, O, Kpos], L, CFG.NH); }
  conv(pass, X, DYN, BASE, RES, Y, L, a, useRes) { const G = CFG.H / CFG.GROUP; this.run(pass, 'conv', [this.uni([{ u: L }, { u: CFG.H }, { u: G }, { u: a }, { u: useRes ? 1 : 0 }, { u: 0 }, { u: 0 }, { u: 0 }]), X, DYN, BASE, RES, Y], Math.ceil(L * CFG.H / 64)); }
  silu(pass, G, U, Y, n) { this.run(pass, 'silu', [this.uni([{ u: n }, { u: 0 }, { u: 0 }, { u: 0 }]), G, U, Y], Math.ceil(n / 256)); }
  // top-16 per row of logits L (binding or buffer, f32 [R, V]); returns {ov (f32 [R,16]), oi (u32 [R,16])} (allocated once per R)
  topk(pass, L, R, V) {
    if (!this._topk || this._topk.R !== R) this._topk = { R, sv: this.buf(R * 256 * 16, GPUBufferUsage.STORAGE), si: this.buf(R * 256 * 16, GPUBufferUsage.STORAGE), ov: this.buf(R * 16), oi: this.buf(R * 16) };
    const t = this._topk; this.run(pass, 'topk', [this.uni([{ u: R }, { u: V }, { u: 0 }, { u: 0 }]), L, t.sv, t.si, t.ov, t.oi], R); return t;
  }
  scale(pass, X, Y, n, s) { this.run(pass, 'scale', [this.uni([{ u: n }, { u: 0 }, { u: 0 }, { u: 0 }, { f: s }, { f: 0 }, { f: 0 }, { f: 0 }]), X, Y], Math.ceil(n / 256)); }

  // ---- weights ----
  // ggufSource: a URL string (Range-fetched) OR a byte-source function (offset, length) => Promise<Uint8Array> (e.g. slices of a cached Blob).
  async loadWeights(ggufSource, onProgress = () => {}) {
    const src = typeof ggufSource === 'function' ? ggufSource : fetchSource(ggufSource); const g = await readGGUF(src); this.gguf = g;
    const need = ['fc.weight', 'enc.output_norm.weight', 'output_norm.weight', 'selector_hidden.weight'];
    for (let i = 0; i < CFG.L; ++i) for (const s of ['attn_norm.weight', 'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight', 'attn_q_norm.weight', 'attn_k_norm.weight', 'attn_conv_base', 'attn_conv_proj.weight', 'ffn_norm.weight', 'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight', 'ffn_conv_base', 'ffn_conv_proj.weight']) need.push(`blk.${i}.${s}`);
    let done = 0; const t0 = performance.now(); let dequantMs = 0;
    for (const name of need) {
      const t = g.byName[name]; if (!t) throw new Error('missing tensor ' + name);
      const raw = await src(t.absOffset, t.bytes);
      const td = performance.now();
      if (t.type === GGML.F32) { const f = new Float32Array(raw.buffer, raw.byteOffset, t.n); this.w[name] = this.upload(f, name); this.stats.weightBytes += t.n * 4; }
      else if (this.packed) { const pw = this.uploadPacked(t.type, raw, t.n, name); this.w[name] = pw; this.stats.weightBytes += pw.bytes; }
      else { const f32 = dequantize(t.type, raw, t.n); const h = f32ArrayToF16(f32); const b = this.device.createBuffer({ size: h.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: name }); this.device.queue.writeBuffer(b, 0, h.buffer, h.byteOffset, h.byteLength); this.w[name] = b; this.stats.weightBytes += h.byteLength; }
      dequantMs += performance.now() - td;
      this.stats.tensors++; done++; onProgress({ done, total: need.length, name, ms: performance.now() - t0 });
      await new Promise(r => setTimeout(r, 0));
    }
    // codebooks stay raw on the CPU (one Q4_K block == one 256-wide row): rows are dequantized on demand by the selector
    this.codebook = {};
    for (const n of ['selector_predecessor.weight', 'selector_successor.weight']) { const t = g.byName[n]; this.codebook[n] = { type: t.type, raw: await src(t.absOffset, t.bytes) }; }
    await this.device.queue.onSubmittedWorkDone();
    this.stats.loadMs = performance.now() - t0; this.stats.dequantMs = dequantMs;
    this.log(`weights: ${this.stats.tensors} tensors, ${(this.stats.weightBytes / 1e9).toFixed(2)} GB on GPU, ${this.stats.loadMs.toFixed(0)} ms (dequant ${dequantMs.toFixed(0)} ms)`);
  }
  codebookRow(which, id) { const cb = this.codebook[which]; return dequantRow(cb.type, cb.raw, CFG.RANK, id); }

  // ---- forward ----
  // Stage A: projected context cache. features f32 [C, 5H] -> draft_context [C, H] -> per layer ctx K (normed+roped) / V, [C, NKV, HD].
  projectContext(features, C, ctxPos0 = 0) {
    const { H, NKV, HD } = CFG; const dev = this.device;
    const F = this.upload(features, 'features'); const fcOut = this.buf(C * H, undefined, 'fc'); const ctx = this.buf(C * H, undefined, 'draft_context');
    const layers = [];
    const pass = this.begin();
    this.opLabel = 'ctx.fc'; this.gemm(pass, F, this.w['fc.weight'], fcOut, C, 5 * H, H);
    this.rmsnorm(pass, fcOut, this.w['enc.output_norm.weight'], ctx, C, H);
    for (let i = 0; i < CFG.L; ++i) {
      const kRaw = this.buf(C * NKV * HD, undefined, `ctx_k_raw${i}`), k = this.buf(C * NKV * HD, undefined, `ctx_k${i}`), v = this.buf(C * NKV * HD, undefined, `ctx_v${i}`);
      this.gemm(pass, ctx, this.w[`blk.${i}.attn_k.weight`], kRaw, C, H, NKV * HD);
      this.rmsnorm(pass, kRaw, this.w[`blk.${i}.attn_k_norm.weight`], k, C * NKV, HD);
      this.rope(pass, k, C, NKV, ctxPos0);
      this.gemm(pass, ctx, this.w[`blk.${i}.attn_v.weight`], v, C, H, NKV * HD);
      layers.push({ k, v });
    }
    this.opLabel = null; this.end(pass);
    const posArr = new Uint32Array(C); for (let i = 0; i < C; ++i) posArr[i] = ctxPos0 + i; const pos = this.buf(C, undefined, 'ctx_pos'); dev.queue.writeBuffer(pos, 0, posArr);
    return { C, total: C, ctxPos0, ctx, fcOut, layers, F, pos };
  }
  // Stage A' (incremental): a projected-context cache with a fixed capacity, appended a few rows at a time from a GPU
  // feature binding ({buffer, offset, size}: f32 [rows, 5H], e.g. the engine's dspark.features scratch). Positions are
  // absolute (row r of the context sits at position r); no eviction (capacity <= 3064 so the attention kernel fits).
  // Eviction (MLX ContextOnlyDraftKVCache, sink 64 / window 1024 upstream defaults): after an append that leaves more than
  // sink + window rows, keep the first `sink` rows and the last `window` rows (positions travel with the rows; the attention
  // mask uses them, so sink rows older than the 2048 sliding window get masked exactly as in MLX). sink+window = 0 -> no eviction.
  createContext(capacity, opts = {}) {
    const { H, NKV, HD, BLOCK } = CFG; if (capacity + BLOCK > 3072) throw new Error('context capacity + block > 3072 keys');
    const sink = opts.sink ?? 0, window = opts.window ?? 0; if (sink + window > 0 && capacity < sink + window + BLOCK) throw new Error('capacity < sink + window + block');
    const layers = []; for (let i = 0; i < CFG.L; ++i) layers.push({ k: this.buf(capacity * NKV * HD, undefined, `ctx_k${i}`), v: this.buf(capacity * NKV * HD, undefined, `ctx_v${i}`) });
    return { C: 0, total: 0, capacity, sink, window, layers, pos: this.buf(capacity, undefined, 'ctx_pos'), tmp: sink + window > 0 ? this.buf(window * NKV * HD, undefined, 'ctx_tmp') : null, evictions: 0,
      fcOut: this.buf(BLOCK * H, undefined, 'fc'), ctx: this.buf(BLOCK * H, undefined, 'draft_context'), kRaw: this.buf(BLOCK * NKV * HD, undefined, 'k_raw'), incremental: true };
  }
  resetContext(cache) { cache.C = 0; cache.total = 0; cache.evictions = 0; }
  evict(cache) {
    const { NKV, HD } = CFG; const keep = cache.sink + cache.window; if (!keep || cache.C <= keep) return;
    const rowBytes = NKV * HD * 4; const src = cache.C - cache.window; const enc = this.device.createCommandEncoder();
    for (const l of cache.layers) for (const b of [l.k, l.v]) { enc.copyBufferToBuffer(b, src * rowBytes, cache.tmp, 0, cache.window * rowBytes); enc.copyBufferToBuffer(cache.tmp, 0, b, cache.sink * rowBytes, cache.window * rowBytes); }
    enc.copyBufferToBuffer(cache.pos, src * 4, cache.tmp, 0, cache.window * 4); enc.copyBufferToBuffer(cache.tmp, 0, cache.pos, cache.sink * 4, cache.window * 4);
    this.device.queue.submit([enc.finish()]); cache.C = keep; cache.evictions++;
  }
  appendContext(cache, feat, rows) {
    const { H, NKV, HD, BLOCK } = CFG; if (rows < 1 || rows > BLOCK) throw new Error('appendContext: 1..8 rows'); if (cache.C + rows > cache.capacity) throw new Error('context capacity exceeded');
    const rowBytes = NKV * HD * 4; const at = (b) => ({ buffer: b, offset: cache.C * rowBytes, size: rows * rowBytes });
    const posArr = new Uint32Array(rows); for (let r = 0; r < rows; ++r) posArr[r] = cache.total + r; this.device.queue.writeBuffer(cache.pos, cache.C * 4, posArr);
    const pass = this.begin();
    this.opLabel = 'ctx.fc'; this.gemm(pass, feat, this.w['fc.weight'], cache.fcOut, rows, 5 * H, H);
    this.rmsnorm(pass, cache.fcOut, this.w['enc.output_norm.weight'], cache.ctx, rows, H);
    for (let i = 0; i < CFG.L; ++i) {
      this.gemm(pass, cache.ctx, this.w[`blk.${i}.attn_k.weight`], cache.kRaw, rows, H, NKV * HD);
      this.rmsnorm(pass, cache.kRaw, this.w[`blk.${i}.attn_k_norm.weight`], at(cache.layers[i].k), rows * NKV, HD);
      this.rope(pass, at(cache.layers[i].k), rows, NKV, cache.total);
      this.gemm(pass, cache.ctx, this.w[`blk.${i}.attn_v.weight`], at(cache.layers[i].v), rows, H, NKV * HD);
    }
    this.opLabel = null; this.end(pass); cache.C += rows; cache.total += rows; this.evict(cache);
  }
  // Stage B: one draft step over the 8 block rows. noise f32 [L, H] (raw target embed; scaled here by embedScale).
  // Returns the GPU buffers of every stage (for readback) + the selector-projected hidden.
  draftStep(cache, noise, embedScale = 1.0) {
    const { H, I, NH, NKV, HD, BLOCK: L } = CFG; const dev = this.device; const C = cache.C; const G = H / CFG.GROUP;
    const N = noise instanceof Float32Array ? this.upload(noise, 'noise') : noise;   // Float32Array -> upload; else a GPU binding {buffer, offset, size}
    const h0 = this.buf(L * H, undefined, 'h0');
    const st = { layers: [] };
    const pass = this.begin();
    this.scale(pass, N, h0, L * H, embedScale);
    let h = h0;
    for (let i = 0; i < CFG.L; ++i) {
      const w = (s) => { this.opLabel = s.replace('.weight', ''); return this.w[`blk.${i}.${s}`]; };
      const normed = this.buf(L * H), dynA = this.buf(L * 4 * G), xin = this.buf(L * H, undefined, `attn_in${i}`);
      this.rmsnorm(pass, h, w('attn_norm.weight'), normed, L, H);
      this.gemm(pass, normed, w('attn_conv_proj.weight'), dynA, L, H, 4 * G);
      this.conv(pass, normed, dynA, w('attn_conv_base'), normed, xin, L, 0, false);
      const q = this.buf(L * NH * HD), qn = this.buf(L * NH * HD), k = this.buf(L * NKV * HD), kn = this.buf(L * NKV * HD), v = this.buf(L * NKV * HD), o = this.buf(L * NH * HD), ao = this.buf(L * H), hA = this.buf(L * H, undefined, `attn_out${i}`);
      this.gemm(pass, xin, w('attn_q.weight'), q, L, H, NH * HD); this.rmsnorm(pass, q, w('attn_q_norm.weight'), qn, L * NH, HD); this.rope(pass, qn, L, NH, cache.total);
      this.gemm(pass, xin, w('attn_k.weight'), k, L, H, NKV * HD); this.rmsnorm(pass, k, w('attn_k_norm.weight'), kn, L * NKV, HD); this.rope(pass, kn, L, NKV, cache.total);
      this.gemm(pass, xin, w('attn_v.weight'), v, L, H, NKV * HD);
      this.opLabel = 'attention'; this.attn(pass, qn, cache.layers[i].k, cache.layers[i].v, kn, v, o, C, L, cache.total, cache.pos);
      this.gemm(pass, o, w('attn_output.weight'), ao, L, NH * HD, H);
      this.conv(pass, ao, dynA, w('attn_conv_base'), h, hA, L, 1, true);
      const normed2 = this.buf(L * H), dynM = this.buf(L * 4 * G), xm = this.buf(L * H), gate = this.buf(L * I), up = this.buf(L * I), act = this.buf(L * I), down = this.buf(L * H), hM = this.buf(L * H, undefined, `out${i}`);
      this.rmsnorm(pass, hA, w('ffn_norm.weight'), normed2, L, H);
      this.gemm(pass, normed2, w('ffn_conv_proj.weight'), dynM, L, H, 4 * G);
      this.conv(pass, normed2, dynM, w('ffn_conv_base'), normed2, xm, L, 0, false);
      this.gemm(pass, xm, w('ffn_gate.weight'), gate, L, H, I); this.gemm(pass, xm, w('ffn_up.weight'), up, L, H, I);
      this.opLabel = 'silu_mul'; this.silu(pass, gate, up, act, L * I);
      this.gemm(pass, act, w('ffn_down.weight'), down, L, I, H);
      this.conv(pass, down, dynM, w('ffn_conv_base'), hA, hM, L, 1, true);
      st.layers.push({ attn_in: xin, attn_out: hA, out: hM, q: qn, k: kn, v, o });
      h = hM;
    }
    this.opLabel = 'output_norm'; const fin = this.buf(L * H, undefined, 'final'); this.rmsnorm(pass, h, this.w['output_norm.weight'], fin, L, H);
    const selH = this.buf((L - 1) * CFG.RANK, undefined, 'sel_hidden');
    this.opLabel = 'selector_hidden'; this.gemm(pass, { buffer: fin, offset: H * 4, size: (L - 1) * H * 4 }, this.w['selector_hidden.weight'], selH, L - 1, H, CFG.RANK);
    this.opLabel = null; this.end(pass);
    st.final = fin; st.selHidden = selH; st.h0 = h0;
    return st;
  }
  // Selector (CPU, f32): candidates [7][16] ids + unary logits [7][16] (from the target head), selHidden [7*256]. Greedy path walk.
  select(anchor, candIds, unary, selHidden) {
    const R = CFG.RANK; let pred = anchor; const path = []; const edgesAll = [];
    for (let pos = 0; pos < candIds.length; ++pos) {
      const pv = this.codebookRow('selector_predecessor.weight', pred); const hp = selHidden.subarray(pos * R, (pos + 1) * R);
      const edges = new Float32Array(candIds[pos].length); let best = -Infinity, bi = 0;
      for (let c = 0; c < candIds[pos].length; ++c) {
        const sv = this.codebookRow('selector_successor.weight', candIds[pos][c]); let e = 0; for (let r = 0; r < R; ++r) e += pv[r] * hp[r] * sv[r];
        edges[c] = e; const s = unary[pos][c] + e; if (s > best) { best = s; bi = c; }
      }
      pred = candIds[pos][bi]; path.push(pred); edgesAll.push(edges);
    }
    return { path, edges: edgesAll };
  }
  async read(buf, n, byteOffset = 0) {
    const s = this.device.createBuffer({ size: Math.ceil(n * 4 / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buf, byteOffset, s, 0, n * 4); this.device.queue.submit([enc.finish()]);
    await s.mapAsync(GPUMapMode.READ); const out = new Float32Array(s.getMappedRange().slice(0)); s.unmap(); s.destroy(); return out;
  }
}
