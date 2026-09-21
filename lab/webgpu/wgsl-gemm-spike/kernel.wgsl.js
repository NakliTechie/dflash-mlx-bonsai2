// WGSL small-M ternary GEMM for the engine's lut2_128 (lutId 9) GPU-resident layout of PTQ1_0 weights.
//
// Layout (established from the engine's transcode spec `prism_ptq1_0_to_lut2` and the decode block-dot template):
//   - a weight row of K inputs is K/32 blocks; relative block index blk = row * (K/32) + kb; absolute = blockOffset + blk
//   - bits[2*blk + 0] holds elements 0..15 of the block, bits[2*blk + 1] elements 16..31; element e sits at bit
//     2*(e & 15) of its word as code = trit + 1 (0 -> -1, 1 -> 0, 2 -> +1)
//   - scales are f16 per 128-element group packed two per u32: scale(blk) = unpack2x16float(scales[blk >> 3])[(blk >> 2) & 1]
//   - the pack's blockOffset is a multiple of 8, so wordBase = 2 * blockOffset and scaleBase = blockOffset / 8 are exact
//
// Design: one workgroup owns TN output rows; each lane owns RN rows and one 32-element block slot (KS = 8 slots) of a
// 256-element K chunk; the 8 activation rows of that chunk sit in workgroup memory (loaded once per chunk, transposed
// so the 8 block-slot lanes hit consecutive vec4s); each weight word is read once and dotted against all 8 activation
// rows; the KS partials are reduced with subgroupShuffleXor (the 8 slot lanes are contiguous in a subgroup).
//
// Variants:
//   useF16A  - activation tile stored as f16 in workgroup memory (math stays f32)
//   f16Math  - activation tile f16, trits f16, per-block partial sums f16; the scaled accumulation across blocks is f32
export function gemmWgsl({ TN = 64, RN = 2, useF16A = false, f16Math = false } = {}) {
  const KS = 8, M = 8, CH = 256, WG = (TN / RN) * KS;
  if (WG > 1024 || WG % 32 !== 0) throw new Error(`bad tile: WG=${WG}`);
  const f16 = useF16A || f16Math;
  const aElem = f16 ? 'vec4<f16>' : 'vec4<f32>';
  const mathV = f16Math ? 'vec4<f16>' : 'vec4<f32>';
  const mathS = f16Math ? 'f16' : 'f32';
  return `${f16 ? 'enable f16;\n' : ''}enable subgroups;
struct Params { N: u32, K: u32, wordBase: u32, scaleBase: u32 }
@group(0) @binding(0) var<storage, read> bits: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;

const M = ${M}u; const KS = ${KS}u; const CH = ${CH}u; const TN = ${TN}u; const RN = ${RN}u; const WG = ${WG}u;
const CHV = CH / 4u;            // vec4s per activation row per chunk (64)
var<workgroup> As: array<${aElem}, M * CHV>;   // transposed: index = m*CHV + g*KS + kg (g = vec4 group in block 0..7)

fn trits(word: u32, sh: u32) -> vec4<f32> {
  let codes = (vec4<u32>(word >> sh) >> vec4<u32>(0u, 2u, 4u, 6u)) & vec4<u32>(3u);
  return bitcast<vec4<f32>>(codes | vec4<u32>(0x4b000000u)) - vec4<f32>(8388609.0);   // code - 1
}

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let kg = lid & (KS - 1u);
  let rg = lid / KS;
  let row0 = wid.x * TN + rg * RN;
  let BPR = P.K / 32u;
  let KV = P.K / 4u;
  let nChunks = P.K / CH;
  var acc: array<array<f32, ${M}>, ${RN}>;
  for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { acc[r][m] = 0.0; } }
  for (var c = 0u; c < nChunks; c++) {
    let kc0v = c * CHV;
    for (var i = lid; i < M * CHV; i += WG) {
      let m = i / CHV; let v = i - m * CHV;
      As[m * CHV + (v & 7u) * KS + (v >> 3u)] = ${aElem}(A[m * KV + kc0v + v]);
    }
    workgroupBarrier();
    let kb = c * KS + kg;
    var wa: array<u32, ${RN}>; var wb: array<u32, ${RN}>; var sc: array<f32, ${RN}>;
    for (var r = 0u; r < RN; r++) {
      let blk = (row0 + r) * BPR + kb;
      wa[r] = bits[P.wordBase + blk * 2u];
      wb[r] = bits[P.wordBase + blk * 2u + 1u];
      sc[r] = unpack2x16float(scales[P.scaleBase + (blk >> 3u)])[(blk >> 2u) & 1u];
    }
    var part: array<array<${mathS}, ${M}>, ${RN}>;
    for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { part[r][m] = ${mathS}(0.0); } }
    for (var g = 0u; g < 8u; g++) {
      let base = g * KS + kg;
      var av: array<${mathV}, ${M}>;
      for (var m = 0u; m < M; m++) { av[m] = ${mathV}(As[m * CHV + base]); }
      for (var r = 0u; r < RN; r++) {
        let w = ${mathV}(trits(select(wa[r], wb[r], g >= 4u), (g & 3u) * 8u));
        for (var m = 0u; m < M; m++) { part[r][m] += dot(w, av[m]); }
      }
    }
    for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { acc[r][m] += sc[r] * f32(part[r][m]); } }
    workgroupBarrier();
  }
  for (var r = 0u; r < RN; r++) {
    for (var m = 0u; m < M; m++) {
      var v = acc[r][m];
      v += subgroupShuffleXor(v, 1u);
      v += subgroupShuffleXor(v, 2u);
      v += subgroupShuffleXor(v, 4u);
      if (kg == 0u) { Y[m * P.N + row0 + r] = v; }
    }
  }
}`;
}

// v2: fully unrolled (no dynamically indexed private arrays), trits pre-scaled by the group scale (exact: the scale is
// an f16 value and +-scale/0 are representable in both f16 and f32), so each lane keeps only RN*8 f32 accumulators and
// 8 activation vec4s; dot runs in `math` precision (f16 or f32) and the block sums accumulate in f32.
export function gemmWgslV2({ TN = 64, RN = 4, math = 'f16', aStore = 'f16' } = {}) {
  const KS = 8, M = 8, CH = 256, WG = (TN / RN) * KS;
  if (WG > 1024 || WG % 32 !== 0) throw new Error(`bad tile: WG=${WG}`);
  const f16 = math === 'f16' || aStore === 'f16';
  const aElem = aStore === 'f16' ? 'vec4<f16>' : 'vec4<f32>';
  const mv = math === 'f16' ? 'vec4<f16>' : 'vec4<f32>';
  const ms = math === 'f16' ? 'f16' : 'f32';
  const L = [];
  const rows = [...Array(RN).keys()], ms8 = [...Array(M).keys()];
  L.push(`${f16 ? 'enable f16;\n' : ''}enable subgroups;
struct Params { N: u32, K: u32, wordBase: u32, scaleBase: u32 }
@group(0) @binding(0) var<storage, read> bits: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;
const M = ${M}u; const KS = ${KS}u; const CH = ${CH}u; const TN = ${TN}u; const RN = ${RN}u; const WG = ${WG}u; const CHV = CH / 4u;
var<workgroup> As: array<${aElem}, M * CHV>;
fn trits(word: u32, sh: u32) -> vec4<f32> {
  let codes = (vec4<u32>(word >> sh) >> vec4<u32>(0u, 2u, 4u, 6u)) & vec4<u32>(3u);
  return bitcast<vec4<f32>>(codes | vec4<u32>(0x4b000000u)) - vec4<f32>(8388609.0);
}
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x; let kg = lid & (KS - 1u); let rg = lid / KS;
  let row0 = wid.x * TN + rg * RN; let BPR = P.K / 32u; let KV = P.K / 4u; let nChunks = P.K / CH;`);
  for (const r of rows) for (const m of ms8) L.push(`  var acc${r}_${m} = 0.0;`);
  L.push(`  for (var c = 0u; c < nChunks; c++) {
    let kc0v = c * CHV;
    for (var i = lid; i < M * CHV; i += WG) { let m = i / CHV; let v = i - m * CHV; As[m * CHV + (v & 7u) * KS + (v >> 3u)] = ${aElem}(A[m * KV + kc0v + v]); }
    workgroupBarrier();
    let kb = c * KS + kg;`);
  for (const r of rows) L.push(`    let blk${r} = (row0 + ${r}u) * BPR + kb; let wa${r} = bits[P.wordBase + blk${r} * 2u]; let wb${r} = bits[P.wordBase + blk${r} * 2u + 1u]; let sc${r} = ${ms}(unpack2x16float(scales[P.scaleBase + (blk${r} >> 3u)])[(blk${r} >> 2u) & 1u]);`);
  for (let g = 0; g < 8; ++g) {
    L.push(`    {`);
    for (const m of ms8) L.push(`      let a${m} = ${mv}(As[${m * (CH / 4)}u + ${g * KS}u + kg]);`);
    for (const r of rows) {
      L.push(`      let w${r} = ${mv}(trits(w${g < 4 ? 'a' : 'b'}${r}, ${(g & 3) * 8}u)) * sc${r};`);
      for (const m of ms8) L.push(`      acc${r}_${m} += f32(dot(w${r}, a${m}));`);
    }
    L.push(`    }`);
  }
  L.push(`    workgroupBarrier();
  }`);
  for (const r of rows) for (const m of ms8) L.push(`  { var v = acc${r}_${m}; v += subgroupShuffleXor(v, 1u); v += subgroupShuffleXor(v, 2u); v += subgroupShuffleXor(v, 4u); if (kg == 0u) { Y[${m}u * P.N + row0 + ${r}u] = v; } }`);
  L.push(`}`);
  return L.join('\n');
}

// v3: 1024-element chunks (each lane owns one full 128-element scale group per chunk, so the group scale is folded once
// per 128 elements), the bias trick sum(code*a) - sum(a) (codes 0..2 go straight through u32->f32 conversion, no -1
// per element; sum(a) per (m, lane) is computed cooperatively by the tile loader), RN rows per lane with scalar dot
// partials, and per-(ksplit, kg) partial outputs written to global memory (no persistent accumulators) reduced by a
// second tiny pass. aStore f32 avoids the f16->f32 converts; f16 halves workgroup-memory traffic.
export function gemmWgslV3({ TN = 128, RN = 8, aStore = 'f32', chunksPerWG = 0, CH = 0 } = {}) {
  const KS = 8, M = 8; CH = CH || (aStore === 'f16' ? 1024 : 512);
  const CHV = CH / 4, BPL = CH / 32 / KS, WG = (TN / RN) * KS;   // BPL = blocks per lane per chunk (2 or 4)
  if (WG > 1024 || WG % 32 !== 0 || WG < 64) throw new Error(`bad tile: WG=${WG}`);
  if (BPL !== 2 && BPL !== 4) throw new Error(`bad CH=${CH}`);
  const f16 = aStore === 'f16';
  const aElem = f16 ? 'vec4<f16>' : 'vec4<f32>';
  const LPP = WG / 64;                       // loader lanes per (m, kg) pair
  const JPL = (BPL * 8) / LPP;               // vec4 j-slots per loader lane
  const rows = [...Array(RN).keys()], ms8 = [...Array(M).keys()];
  const L = [];
  L.push(`${f16 ? 'enable f16;\n' : ''}enable subgroups;
struct Params { N: u32, K: u32, wordBase: u32, scaleBase: u32, chunksPerWG: u32, nSplits: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> bits: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Ypart: array<f32>;   // [nSplits*KS][M][N]
@group(0) @binding(4) var<uniform> P: Params;
const M = ${M}u; const KS = ${KS}u; const CH = ${CH}u; const CHV = ${CHV}u; const BPL = ${BPL}u; const TN = ${TN}u; const RN = ${RN}u; const WG = ${WG}u;
var<workgroup> As: array<${aElem}, M * CHV>;      // index = m*CHV + j*KS + kg  (j = vec4 slot inside the lane's BPL blocks)
var<workgroup> Asum4: array<f32, 64 * ${LPP}>;   // partial sums of the stored activation per (m, kg) pair
fn codes4(word: u32, sh: u32) -> vec4<f32> {
  return vec4<f32>((vec4<u32>(word >> sh) >> vec4<u32>(0u, 2u, 4u, 6u)) & vec4<u32>(3u));
}
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x; let kg = lid & (KS - 1u); let rg = lid / KS;
  let row0 = wid.x * TN + rg * RN; let BPR = P.K / 32u; let KV = P.K / 4u;
  let c0 = wid.y * P.chunksPerWG; let c1 = c0 + P.chunksPerWG;
  let pair = lid / ${LPP}u; let sub = lid - pair * ${LPP}u;   // loader role: pair = m*8 + kgL
  let lm = pair >> 3u; let lkg = pair & 7u;
  let split = wid.y * KS + kg;
  let outBase = split * (M * P.N) + row0;
  for (var c = c0; c < c1; c++) {
    {
      var s4 = vec4<f32>(0.0);
      let srcBase = lm * KV + c * CHV + lkg * ${BPL * 8}u + sub * ${JPL}u;
      let dstBase = lm * CHV + lkg;
      for (var jj = 0u; jj < ${JPL}u; jj++) {
        let x = ${aElem}(A[srcBase + jj]);
        As[dstBase + (sub * ${JPL}u + jj) * KS] = x;
        s4 += vec4<f32>(x);
      }
      Asum4[pair * ${LPP}u + sub] = dot(s4, vec4<f32>(1.0));
    }
    workgroupBarrier();`);
  for (const r of rows) for (const m of ms8) L.push(`    var p${r}_${m} = 0.0;`);
  L.push(`    let blkBase = row0 * BPR + c * ${CH / 32}u + kg * BPL;
    for (var b = 0u; b < BPL; b++) {`);
  for (const r of rows) L.push(`      let wa${r} = bits[P.wordBase + (blkBase + ${r}u * BPR + b) * 2u]; let wb${r} = bits[P.wordBase + (blkBase + ${r}u * BPR + b) * 2u + 1u];`);
  L.push(`      for (var g = 0u; g < 8u; g++) {
        let sh = (g & 3u) * 8u; let hi = g >= 4u; let ai = (b * 8u + g) * KS + kg;`);
  for (const m of ms8) L.push(`        let a${m} = vec4<f32>(As[${m * CHV}u + ai]);`);
  for (const r of rows) {
    L.push(`        { let w = codes4(select(wa${r}, wb${r}, hi), sh);`);
    for (const m of ms8) L.push(`          p${r}_${m} += dot(w, a${m});`);
    L.push(`        }`);
  }
  L.push(`      }
    }`);
  for (const m of ms8) L.push(`    let asum${m} = ${[...Array(LPP).keys()].map(s => `Asum4[${(m * 8) * LPP + s}u + kg * ${LPP}u]`).join(' + ')};`);
  for (const r of rows) {
    L.push(`    { let blk = blkBase + ${r}u * BPR; let sc = unpack2x16float(scales[P.scaleBase + (blk >> 3u)])[(blk >> 2u) & 1u];`);
    for (const m of ms8) L.push(`      { let v = sc * (p${r}_${m} - asum${m}); let o = outBase + ${m}u * P.N + ${r}u; if (c == c0) { Ypart[o] = v; } else { Ypart[o] += v; } }`);
    L.push(`    }`);
  }
  L.push(`    workgroupBarrier();
  }
}`);
  return L.join('\n');
}
export function reduceWgsl() {
  return `struct Params { N: u32, nParts: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> Ypart: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let total = 8u * P.N; if (i >= total) { return; }
  var s = 0.0; for (var p = 0u; p < P.nParts; p++) { s += Ypart[p * total + i]; }
  Y[i] = s;
}`;
}

// v4: v1's structure (dynamic loops with const bounds, per-block partials folded with the group scale into f32), plus
//   - spread dequant: per 16-element word, s_q = (word >> 2q) & 0x03030303 puts the codes of elements {q, q+4, q+8, q+12}
//     into byte lanes (7 ops per word); per 4 elements the codes are then unpack4xU8 (or a manual byte unpack) + one
//     convert + one subtract, instead of the per-4 shift/mask/bitcast sequence;
//   - the activation therefore uses a fixed permutation: inside every 16-element half, vec4 q holds elements
//     {q, q+4, q+8, q+12} (a 4x4 transpose per 16 elements) -- `permuteA()` below produces it on the host;
//   - vec4 partial accumulators in `math` precision (pure FMAs, no per-dot horizontal add), folded per 32-element
//     block: acc += scale * (p.x + p.y + p.z + p.w) in f32.
export function permuteA(A, M, K) {
  const out = new Float32Array(A.length);
  for (let m = 0; m < M; ++m) for (let h = 0; h < K / 16; ++h) for (let q = 0; q < 4; ++q) for (let i = 0; i < 4; ++i)
    out[m * K + h * 16 + q * 4 + i] = A[m * K + h * 16 + q + 4 * i];
  return out;
}
export function gemmWgslV4({ TN = 64, RN = 4, math = 'f16', aStore = 'f16', unpack = 'builtin', vecAcc = true } = {}) {
  const KS = 8, M = 8, CH = 256, WG = (TN / RN) * KS;
  if (WG > 1024 || WG % 32 !== 0) throw new Error(`bad tile: WG=${WG}`);
  const f16 = math === 'f16' || aStore === 'f16';
  const aElem = aStore === 'f16' ? 'vec4<f16>' : 'vec4<f32>';
  const mv = math === 'f16' ? 'vec4<f16>' : 'vec4<f32>';
  const one = math === 'f16' ? '1.0h' : '1.0';
  const unpackFn = unpack === 'builtin' ? 'unpack4xU8(x)' : '(vec4<u32>(x) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(0xffu)';
  return `${f16 ? 'enable f16;\n' : ''}enable subgroups;
${unpack === 'builtin' ? 'requires packed_4x8_integer_dot_product;\n' : ''}struct Params { N: u32, K: u32, wordBase: u32, scaleBase: u32 }
@group(0) @binding(0) var<storage, read> bits: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> A: array<vec4<f32>>;      // permuted layout (see permuteA)
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;
const M = ${M}u; const KS = ${KS}u; const CH = ${CH}u; const TN = ${TN}u; const RN = ${RN}u; const WG = ${WG}u; const CHV = CH / 4u;
var<workgroup> As: array<${aElem}, M * CHV>;
fn tritsq(x: u32) -> ${mv} { return ${mv}(${unpackFn}) - ${mv}(${one}); }
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x; let kg = lid & (KS - 1u); let rg = lid / KS;
  let row0 = wid.x * TN + rg * RN; let BPR = P.K / 32u; let KV = P.K / 4u; let nChunks = P.K / CH;
  var acc: array<array<f32, ${M}>, ${RN}>;
  for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { acc[r][m] = 0.0; } }
  for (var c = 0u; c < nChunks; c++) {
    let kc0v = c * CHV;
    for (var i = lid; i < M * CHV; i += WG) { let m = i / CHV; let v = i - m * CHV; As[m * CHV + (v & 7u) * KS + (v >> 3u)] = ${aElem}(A[m * KV + kc0v + v]); }
    workgroupBarrier();
    let kb = c * KS + kg;
    var sp: array<array<u32, 8>, ${RN}>;   // spread words: [r][h*4+q]
    var sc: array<f32, ${RN}>;
    for (var r = 0u; r < RN; r++) {
      let blk = (row0 + r) * BPR + kb;
      let wa = bits[P.wordBase + blk * 2u]; let wb = bits[P.wordBase + blk * 2u + 1u];
      sc[r] = unpack2x16float(scales[P.scaleBase + (blk >> 3u)])[(blk >> 2u) & 1u];
      for (var q = 0u; q < 4u; q++) { sp[r][q] = (wa >> (2u * q)) & 0x03030303u; sp[r][4u + q] = (wb >> (2u * q)) & 0x03030303u; }
    }
${vecAcc ? `    var part: array<array<${mv}, ${M}>, ${RN}>;
    for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { part[r][m] = ${mv}(0.0); } }
    for (var g = 0u; g < 8u; g++) {
      let base = g * KS + kg;
      var av: array<${mv}, ${M}>;
      for (var m = 0u; m < M; m++) { av[m] = ${mv}(As[m * CHV + base]); }
      for (var r = 0u; r < RN; r++) { let w = tritsq(sp[r][g]); for (var m = 0u; m < M; m++) { part[r][m] = fma(w, av[m], part[r][m]); } }
    }
    for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { let p = part[r][m]; acc[r][m] += sc[r] * f32(p.x + p.y + p.z + p.w); } }` :
`    var part: array<array<${math === 'f16' ? 'f16' : 'f32'}, ${M}>, ${RN}>;
    for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { part[r][m] = ${math === 'f16' ? '0.0h' : '0.0'}; } }
    for (var g = 0u; g < 8u; g++) {
      let base = g * KS + kg;
      var av: array<${mv}, ${M}>;
      for (var m = 0u; m < M; m++) { av[m] = ${mv}(As[m * CHV + base]); }
      for (var r = 0u; r < RN; r++) { let w = tritsq(sp[r][g]); for (var m = 0u; m < M; m++) { part[r][m] += dot(w, av[m]); } }
    }
    for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { acc[r][m] += sc[r] * f32(part[r][m]); } }`}
    workgroupBarrier();
  }
  for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) {
    var v = acc[r][m]; v += subgroupShuffleXor(v, 1u); v += subgroupShuffleXor(v, 2u); v += subgroupShuffleXor(v, 4u);
    if (kg == 0u) { Y[m * P.N + row0 + r] = v; }
  } }
}`;
}

// v5: fully unrolled scalars like v2, with vec4<f16> per-(row, m) partial accumulators over one 32-element block
// (pure fma, no per-dot horizontal add) folded into f32 with the group scale once per block.
export function gemmWgslV5({ TN = 64, RN = 2 } = {}) {
  const KS = 8, M = 8, CH = 256, WG = (TN / RN) * KS;
  if (WG > 1024 || WG % 32 !== 0) throw new Error(`bad tile: WG=${WG}`);
  const rows = [...Array(RN).keys()], ms8 = [...Array(M).keys()];
  const L = [];
  L.push(`enable f16;
enable subgroups;
struct Params { N: u32, K: u32, wordBase: u32, scaleBase: u32 }
@group(0) @binding(0) var<storage, read> bits: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;
const M = ${M}u; const KS = ${KS}u; const CH = ${CH}u; const TN = ${TN}u; const RN = ${RN}u; const WG = ${WG}u; const CHV = CH / 4u;
var<workgroup> As: array<vec4<f16>, M * CHV>;
fn trits(word: u32, sh: u32) -> vec4<f16> {
  let codes = (vec4<u32>(word >> sh) >> vec4<u32>(0u, 2u, 4u, 6u)) & vec4<u32>(3u);
  return vec4<f16>(bitcast<vec4<f32>>(codes | vec4<u32>(0x4b000000u)) - vec4<f32>(8388609.0));
}
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x; let kg = lid & (KS - 1u); let rg = lid / KS;
  let row0 = wid.x * TN + rg * RN; let BPR = P.K / 32u; let KV = P.K / 4u; let nChunks = P.K / CH;`);
  for (const r of rows) for (const m of ms8) L.push(`  var acc${r}_${m} = 0.0;`);
  L.push(`  for (var c = 0u; c < nChunks; c++) {
    let kc0v = c * CHV;
    for (var i = lid; i < M * CHV; i += WG) { let m = i / CHV; let v = i - m * CHV; As[m * CHV + (v & 7u) * KS + (v >> 3u)] = vec4<f16>(A[m * KV + kc0v + v]); }
    workgroupBarrier();
    let kb = c * KS + kg;`);
  for (const r of rows) L.push(`    let blk${r} = (row0 + ${r}u) * BPR + kb; let wa${r} = bits[P.wordBase + blk${r} * 2u]; let wb${r} = bits[P.wordBase + blk${r} * 2u + 1u]; let sc${r} = unpack2x16float(scales[P.scaleBase + (blk${r} >> 3u)])[(blk${r} >> 2u) & 1u];`);
  for (const r of rows) for (const m of ms8) L.push(`    var p${r}_${m} = vec4<f16>(0.0h);`);
  for (let g = 0; g < 8; ++g) {
    L.push(`    {`);
    for (const m of ms8) L.push(`      let a${m} = As[${m * (CH / 4) + g * KS}u + kg];`);
    for (const r of rows) {
      L.push(`      let w${r} = trits(w${g < 4 ? 'a' : 'b'}${r}, ${(g & 3) * 8}u);`);
      for (const m of ms8) L.push(`      p${r}_${m} = fma(w${r}, a${m}, p${r}_${m});`);
    }
    L.push(`    }`);
  }
  for (const r of rows) for (const m of ms8) L.push(`    acc${r}_${m} += sc${r} * f32(p${r}_${m}.x + p${r}_${m}.y + p${r}_${m}.z + p${r}_${m}.w);`);
  L.push(`    workgroupBarrier();
  }`);
  for (const r of rows) for (const m of ms8) L.push(`  { var v = acc${r}_${m}; v += subgroupShuffleXor(v, 1u); v += subgroupShuffleXor(v, 2u); v += subgroupShuffleXor(v, 4u); if (kg == 0u) { Y[${m}u * P.N + row0 + ${r}u] = v; } }`);
  L.push(`}`);
  return L.join('\n');
}

// v6: v1 f16-math structure with the activation tile stored as vec4<u32> (8 halves = two vec4<f16> for consecutive g),
// halving the number of workgroup-memory load instructions per lane (16 B per load instead of 8 B).
export function gemmWgslV6({ TN = 64, RN = 4 } = {}) {
  const KS = 8, M = 8, CH = 256, WG = (TN / RN) * KS;
  if (WG > 1024 || WG % 32 !== 0) throw new Error(`bad tile: WG=${WG}`);
  return `enable f16;
enable subgroups;
struct Params { N: u32, K: u32, wordBase: u32, scaleBase: u32 }
@group(0) @binding(0) var<storage, read> bits: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;
const M = ${M}u; const KS = ${KS}u; const CH = ${CH}u; const TN = ${TN}u; const RN = ${RN}u; const WG = ${WG}u; const CHV = CH / 4u;
var<workgroup> As: array<vec4<u32>, M * CHV / 2u>;   // index = m*(CHV/2) + gp*KS + kg holds g=2gp (xy) and g=2gp+1 (zw)
fn trits(word: u32, sh: u32) -> vec4<f16> {
  let codes = (vec4<u32>(word >> sh) >> vec4<u32>(0u, 2u, 4u, 6u)) & vec4<u32>(3u);
  return vec4<f16>(bitcast<vec4<f32>>(codes | vec4<u32>(0x4b000000u)) - vec4<f32>(8388609.0));
}
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x; let kg = lid & (KS - 1u); let rg = lid / KS;
  let row0 = wid.x * TN + rg * RN; let BPR = P.K / 32u; let KV = P.K / 4u; let nChunks = P.K / CH;
  var acc: array<array<f32, ${M}>, ${RN}>;
  for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { acc[r][m] = 0.0; } }
  for (var c = 0u; c < nChunks; c++) {
    let kc0v = c * CHV;
    for (var i = lid; i < M * CHV / 2u; i += WG) {
      let m = i / (CHV / 2u); let vp = i - m * (CHV / 2u);          // vp = pair index 0..31: block slot = vp >> 2, gp = vp & 3
      let src = m * KV + kc0v + (vp >> 2u) * 8u + (vp & 3u) * 2u;    // two consecutive vec4 of the same block slot
      let lo = vec4<f16>(A[src]); let hi = vec4<f16>(A[src + 1u]);
      As[m * (CHV / 2u) + (vp & 3u) * KS + (vp >> 2u)] = vec4<u32>(bitcast<vec2<u32>>(lo), bitcast<vec2<u32>>(hi));
    }
    workgroupBarrier();
    let kb = c * KS + kg;
    var wa: array<u32, ${RN}>; var wb: array<u32, ${RN}>; var sc: array<f32, ${RN}>;
    for (var r = 0u; r < RN; r++) {
      let blk = (row0 + r) * BPR + kb;
      wa[r] = bits[P.wordBase + blk * 2u]; wb[r] = bits[P.wordBase + blk * 2u + 1u];
      sc[r] = unpack2x16float(scales[P.scaleBase + (blk >> 3u)])[(blk >> 2u) & 1u];
    }
    var part: array<array<f16, ${M}>, ${RN}>;
    for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { part[r][m] = 0.0h; } }
    for (var gp = 0u; gp < 4u; gp++) {
      let base = gp * KS + kg;
      var av0: array<vec4<f16>, ${M}>; var av1: array<vec4<f16>, ${M}>;
      for (var m = 0u; m < M; m++) { let x = As[m * (CHV / 2u) + base]; av0[m] = bitcast<vec4<f16>>(x.xy); av1[m] = bitcast<vec4<f16>>(x.zw); }
      for (var r = 0u; r < RN; r++) {
        let w0 = trits(select(wa[r], wb[r], gp >= 2u), (gp & 1u) * 16u);
        let w1 = trits(select(wa[r], wb[r], gp >= 2u), (gp & 1u) * 16u + 8u);
        for (var m = 0u; m < M; m++) { part[r][m] += dot(w0, av0[m]) + dot(w1, av1[m]); }
      }
    }
    for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) { acc[r][m] += sc[r] * f32(part[r][m]); } }
    workgroupBarrier();
  }
  for (var r = 0u; r < RN; r++) { for (var m = 0u; m < M; m++) {
    var v = acc[r][m]; v += subgroupShuffleXor(v, 1u); v += subgroupShuffleXor(v, 2u); v += subgroupShuffleXor(v, 4u);
    if (kg == 0u) { Y[m * P.N + row0 + r] = v; }
  } }
}`;
}
