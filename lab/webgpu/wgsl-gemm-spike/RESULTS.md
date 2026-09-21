# WGSL small-M ternary GEMM spike — Bonsai 2 27B (PTQ1_0) inside Xenova's WebGPU engine

Date: 2026-09-21. Machine: Apple M4 Pro (20 GPU cores, 24 GB unified), Chrome 153 headless (`--headless=new
--enable-unsafe-webgpu`), adapter `metal-3`, device features used: `shader-f16`, `subgroups`, `timestamp-query`.
Model: `Ternary-Bonsai-2-27B-PTQ1_0.gguf` loaded by the vendored engine (`engine.dflash.js`, patched with
`patch-internals.mjs` in this dir) from its own IndexedDB weight cache in ~10 s.

Goal: an 8-row GEMM on one gate/up-sized projection at **<= 3x** the engine's single-token decode matvec cost.

**M=8 outcome: not met. Best 8-row kernel = 0.356 ms vs 0.101-0.108 ms per decode matvec -> ratio 3.3-3.5**
(a later generalized variant reached 0.323 ms, ratio 3.1 — section 7). It is 2.6x faster than the engine's own
8-row prefill matmul (0.928 ms, ratio 9.2), and the f32 variant is exact.

**Follow-up (section 7): the real DFlash 2 verify is 4-5 rows. At M=4 the ratio is 1.9 (up_proj) / 1.9-2.0
(down_proj) / 1.6 (lm_head), meeting the <= 2 target; at M=5 it is 2.2 / 2.1 / 1.9 (target missed on up_proj by
~10%).** The design note for wiring the M<=8 kernel into the qwen35 prefill builder is in section 8.

## 1. Layout facts (established from the transcode WGSL, confirmed by the exact CPU cross-check)

- Every Bonsai 2 projection incl. `down_proj` and `lm_head` goes through **`prism_ptq1_0_to_lut2`** (lutId **9**,
  pack buffer `lut2_128`, exposed as `inner.packs.lut2_32[i]` with `lut2: 1, compact: 1`). Routing:
  `lut2Suffixes: e => e.prismHadamard ? [...pl, "down_proj"] : pl` (engine.pretty.js:43107); `lmHeadLut = 9`.
  Live-confirmed: `packs.lut2_32` = 2 shards (bits 4,276,879,360 B + 1,804,861,440 B; scales 267 MB + 113 MB),
  `lmHeadQ4` = 79,462,400 u32 (= 248320 x 160 blocks x 2 words), `lmHeadQ4Scales` = 4,966,400 u32.
  `PRISM_PTQ1_0Q4` / `PRISM_PTQ1_0IQ` are not used for this model.
- PTQ1_0 source block: 128 elements in 28 bytes — 26 bytes of base-3 packed trits (16 bytes x 5 trits, 8 bytes x 5,
  2 bytes x 4) + one f16 scale at byte 26. Decode: `q = (byte * 3^k) & 255; trit = ((q*3) >> 8) - 1`.
- GPU-resident lut2_128 layout, per weight row of K inputs (K/32 blocks; relative block `blk = row*(K/32) + kb`,
  absolute block = pack `blockOffset` (a multiple of 8) + blk):
  - `bits[2*blk]` = elements 0..15, `bits[2*blk+1]` = elements 16..31; element `e` sits at bit `2*(e & 15)` as
    `code = trit + 1` (00 -> -1, 01 -> 0, 10 -> +1; code 3 never occurs — sampled histogram [308363, 301819, 308682, 0]).
  - scales: one f16 per 128-element group, two per u32: `scale(blk) = unpack2x16float(scales[blk >> 3])[(blk >> 2) & 1]`.
  - value = `(code - 1) * scale`. The decode op (`q2q` with `LUT_GRID` {-1,0,1,2} and `lut2_native_scale`) does the same.
- Layer-0 block offsets: gate 3,604,480, up 6,389,760, down 9,175,040. Shapes: hidden 5120, intermediate 17408.
- Activations must be Hadamard-rotated first: the engine's `com.xenova.BlockHadamard` (block 1024, explicit Prism
  signs for widths 5120/6144/17408), emitted through the `bi()` helper. The bench uses the engine's own op on 8
  seeded Gaussian rows (post-rotation rms 1.0017).
- Precision of the engine's own paths (measured against a float64 reference from the dequantized weights):
  the M=8 prefill matmul (`LlamaPrefillMatmul`, format lut2_32/lut 9, f32 in/out) has max abs err 6.0e-4,
  mean 1.06e-4 — identical, element for element, to a kernel that rounds the activation to f16 and does f32 math,
  so the engine's prefill route rounds activations to f16 internally.

## 2. What was measured (all GPU timestamp-query medians; ITERS = 41 in the final run, 60 warm dispatches first)

| Measurement | ms | Notes |
|---|---|---|
| engine `LlamaDecodeGateUp`, 1 token (gate+up = 2 matrices 17408x5120) | **0.203-0.206** | 0.101-0.103 per matrix; ~230 GB/s effective |
| engine `LlamaDecodeResidualProjection`, down_proj (1 matrix 5120x17408) | **0.108** | single-matrix decode reference |
| engine `LlamaDecodeGateUp` with `tokens: 4` (its pre-normalized multi-row mode) | 1.327 | 6.5x the 1-token cost: not a verify path |
| engine `LlamaPrefillMatmul`, M=8, up_proj only | **0.928** | ratio 9.2 vs per-matrix decode; matches the 8.1 whole-step ratio |
| engine `LlamaPrefillMatmul`, lm_head M=1 (decode head route) | 1.422 | |
| engine `LlamaPrefillMatmul`, lm_head M=8 | 12.82 | ratio 9.0 |
| spike v2 f32 math, f32 tile, TN=64 RN=4 (**exact**) | 0.454 | ratio 4.5 / 4.2 |
| spike v2 f32 math, f16 tile, TN=64 RN=4 | 0.391 | ratio 3.9 / 3.6; error == engine prefill |
| spike v1 **f16 math**, f16 tile, TN=64 RN=4 (**best**) | **0.356** (min 0.353) | ratio **3.5 / 3.3** |
| spike v1 f16 math on lm_head (248320 x 5120), 8 rows | 4.55 | ratio 3.2 vs M=1 head; 2.8x faster than engine M=8 |

Ratios are `kernel ms / (gate+up ms / 2)` and `kernel ms / down_proj ms`. Wall-clock per iteration (queue submit
to `onSubmittedWorkDone`, 41 back-to-back dispatches) runs 0.01-0.04 ms above the GPU time for every row.

Error vs the float64 CPU reference (8 x 17408 outputs, real layer-0 up_proj weights, rotated Gaussian activation,
|ref| max 3.02):

| Variant | max abs | mean abs | max abs / max ref | argmax per row |
|---|---|---|---|---|
| f32 math, f32 tile (v2, v1 TN=32 RN=2) | 8.6e-7 | 5.8e-8 to 1.2e-7 | 2.8e-7 | 8/8 |
| f32 math, f16 tile | 6.0e-4 | 1.06e-4 | 2.0e-4 | 8/8 (identical to the engine's M=8 prefill output error) |
| f16 math (block partials in f16, f32 across groups) | 1.8e-3 | 3.0e-4 | 5.9e-4 | 8/8 |
| f16 math on lm_head (first 2048 rows checked) | 2.4e-3 | 4.3e-4 | 6.5e-4 | 8/8 |

## 3. Kernel design (best variant; `gemm-lut2-m8.best.wgsl`, generator `gemmWgsl` in `kernel.wgsl.js`)

- Workgroup of 128 lanes owns TN=64 output rows; lane = (row group `rg` of RN=4 rows, block slot `kg` in 0..7).
- K is walked in 256-element chunks (8 blocks). The 8 activation rows of the chunk (8 x 256) are loaded once per
  chunk into workgroup memory as `vec4<f16>` (4 KB), stored transposed (`[m][g][kg]`) so the 8 slot lanes of a row
  group read consecutive vec4s (no bank conflicts).
- Each lane reads its 2 weight words per row (8 words for 4 rows) and the group scale, decodes 4 codes at a time with
  the bitcast trick `bitcast<f32>(code | 0x4b000000) - 8388609` (= code - 1), converts to f16, and dots against the 8
  activation vec4s; each weight word is read exactly once from global memory for all 8 activation rows.
- Per-block partials accumulate in f16 (32 elements), then `acc += scale * f32(partial)` in f32 per 128-element
  group; the 8 K-split partials per output are reduced with `subgroupShuffleXor(1,2,4)` (the slot lanes are
  contiguous in the 32-wide subgroup; the CPU check would catch a wrong lane mapping) and lane `kg == 0` stores
  `Y[m*N + row]`. Grid = N/64 = 272 workgroups for up_proj, 3880 for lm_head.
- The f32-exact variant (`gemm-lut2-m8.exact-f32.wgsl`, `gemmWgslV2`) is the same structure fully unrolled, f32 tile
  and math, trits pre-scaled by the (f16-exact) group scale, `acc += f32(dot(...))`.

## 4. Where the time goes, and what did not work

Effective weight bandwidth of the best kernel is 67 GB/s vs ~230 GB/s for the decode matvec, so the 8-row kernel is
not memory-bound; it is bound by ALU issue and workgroup-memory traffic. Evidence from the variant sweep
(runs `run3..run9`, logs + JSON in this dir):

- f32 tile vs f16 tile with identical f32 math: 0.643 -> 0.428 ms (v1 RN=2) and 0.454 -> 0.391 (v2 RN=4): halving
  workgroup-memory bytes is worth 15-35%, i.e. LDS bandwidth (~64 B/clk/core) caps the f32-tile variants.
- f16 math vs f32 math (same f16 tile): 0.391 -> 0.356: only ~10%, so this GPU does not run f16 FMAs at 2x.
- ALU budget per lane per 4-element step at RN=4: ~84 ops dequant (shift/mask/or/sub/convert), ~160 ops for
  32 `dot`+add, 32 f16->f32 converts when the math is f32 -> roughly 50% of issue slots are the 128 useful MACs.
  713M MACs at the M4 Pro's ~4 T FMA/s is 0.18 ms; 0.356 ms is ~50% MAC efficiency.
- Failed attempts (all correct, all slower): fully unrolled f16 math with RN=4/8 (0.82/2.2 ms: register spills);
  v3 with 1024-element chunks, the Sum(code*a) - Sum(a) bias trick, and K-split partials to global memory (0.46-1.44 ms:
  runtime inner loops + 8-row accumulators spilled); v4 spread-byte dequant via `unpack4xU8` with a permuted
  activation layout (0.43 ms with dot accumulation, 1.5 ms with vec4 partial arrays -> arrays spilled); v5 fully
  unrolled `vec4<f16>` fma partials (0.55-0.62 ms); v6 16-byte tile loads (0.44 ms). TN sweep at RN=4: 64 = 128 >
  32; RN=8 was 2-6x slower in every family (timing consistent with register spills; not verified with a profiler).
- Chrome's `unpack4xU8` (the `packed_4x8_integer_dot_product` language feature is reported present) timed the same
  as a manual 8-op byte unpack in the v4 vecAcc form (1.488 vs 1.492 ms): it behaves like a polyfill on Metal.

## 5. Bearing on the DFlash verify step

- Today's 8-token prefill-graph step is 279 ms vs 34.6 ms decode (8.1x). Its projection matmuls run at 9.0-9.2x the
  per-matrix decode cost (measured here in isolation for up_proj and lm_head). Swapping those for this kernel gives
  ~2.6x on the projections (3.3-3.5x decode instead of 9.2x). If the rest of the 8-token step scaled the same way,
  the step would land near 3.5 decode steps, not the <= 2.2 the full port needs, so the brief's gate (<= 3 in
  isolation) is the right one and it is not met.
- Levers that the measurements say could close the gap, none tried here: (a) `chromium-experimental-subgroup-matrix`
  is present on this device (the engine already probes `subgroupMatrixConfigs` for n >= 512); an f16 8x8x8 subgroup
  matrix would remove both the per-dot add and most of the dequant from the ALU budget, but it is not stable WGSL;
  (b) a hand-written Metal-friendly layout that keeps the activation chunk in registers across more rows
  (needs > 128 live registers on this GPU — every RN=8 attempt timed 2-6x slower, consistent with spills); (c) a lower-level polyfill-free dequant
  (Chrome polyfills `unpack4xU8`/`dot4I8Packed` on Metal).
- The engine's `LlamaDecodeGateUp` `tokens` mode (<= 4 rows) is not a shortcut: 4 rows cost 6.5 single-row calls.

## 6. Reproduce

```
# serve the engine dir (engine.dflash.js + /model + this dir symlinked as /spike) on 127.0.0.1:8795, then:
node cdp-drive.mjs 'http://127.0.0.1:8795/spike/bench.html?iters=41&variants=v64x4f,v64x4s,64x4m,64x2h,32x2' 560 > final.log
node summarize.mjs final.log        # table + final.json
node cdp-drive.mjs 'http://127.0.0.1:8795/spike/probe.html' 300   # engine/pack/device introspection
```
`patch-internals.mjs` is the extended version of `lab/webgpu/patch-internals.mjs` (adds `ba`, `_i`, `fi`, `bi`, `s0`,
`Xs`, `et`, `Cu`, `pi`, `a0`, `N2` to `TernaryBonsai2.__dflashInternals`); run it on the LocalMind engine file to
regenerate `engine.dflash.js` (the LocalMind repo itself was not modified). Variant syntax is in `bench.js`
(`v<TN>x<RN>[f|s]` = v2, `<TN>x<RN>[h|m]` = v1, `w…` = v3, `u…` = v4, `p…` = v5, `s…` = v6).

## 7. M = 4 / 5 (the real DFlash 2 verify width) — `bench2.js`, run `m45final.log` / `.json`

Same harness, generalized kernel `gemmWgslM` (activation rows M baked as a constant; `unroll` = fully unrolled
scalars; `ksplit` = K split across workgroups with a reduce pass; `dequant` = alu | lutw | luts). Per projection the
engine's own single-token decode op is the reference: gate/up op ÷ 2 for up_proj (0.103 ms), the decode residual
projection for down_proj (0.109 ms), the M=1 head matmul for lm_head (1.425 ms). Rotation = the engine's
`BlockHadamard` with the projection's own Prism signs (width 5120 for up/head, 17408 for down). Float64 CPU reference
on all rows (up: 17408, down: 5120) or the first 2048 rows (lm_head). 41 GPU-timestamp samples, 60 warm dispatches.

| M | up_proj (17408x5120) | down_proj (5120x17408) | lm_head (248320x5120) |
|---|---|---|---|
| 4 | **0.193 ms, ratio 1.88** | **0.213 ms, ratio 1.95** (ksplit=4: 0.217-0.221, 1.99-2.02) | **2.34 ms, ratio 1.65** |
| 5 | 0.229 ms, ratio 2.23 | 0.226 ms, ratio 2.07 (ksplit=4) | 2.75 ms, ratio 1.93 |
| 8 | 0.323 ms, ratio 3.14 | 0.324 ms, ratio 2.96 (ksplit=4) | 4.06 ms, ratio 2.85 |

Best variant everywhere: `TN=64 RN=4, f16 math, f16 tile, unrolled` (`gemm-lut2-m4.best.wgsl`); for down_proj at M=4
the un-split and ksplit=4 forms are within noise (80 vs 320 workgroups). For scale: the engine's own prefill matmul
at M=4 costs 0.693 ms (ratio 6.7) and at M=5 0.801 ms (ratio 7.8), so the kernel is 3.5-3.6x faster than the path the
prefill graph uses today at those widths.

Precision tiers at M=4 (up_proj / down_proj / lm_head; max abs err, max|ref| = 3.02 / 31.3 / 3.73; argmax matched
on every row in every run):

| Variant | ms (up / down / head) | ratio | max abs err (up / down / head) | max abs / max ref |
|---|---|---|---|---|
| f16 math, f16 tile, unrolled (best) | 0.193 / 0.213 / 2.34 | 1.88 / 1.95 / 1.65 | 1.7e-3 / 2.5e-2 / 2.4e-3 | 5.5e-4 / 8.0e-4 / 6.5e-4 |
| f32 math, f16 tile, unrolled (== engine prefill precision) | 0.217 / 0.219 / 2.72 | 2.11 / 2.00 / 1.91 | 5.9e-4 / 5.4e-3 / 7.8e-4 | 2.0e-4 / 1.7e-4 / 2.1e-4 |
| f32 math, f32 tile, unrolled (exact) | 0.251 / 0.260 / 3.09 | 2.44 / 2.38 / 2.17 | 3.9e-7 / 2.1e-6 / 6.2e-7 | 1.3e-7 / 6.7e-8 / 1.7e-7 |

At M=5 the same three tiers on up_proj: 0.229 / 0.265 / 0.309 ms (ratio 2.23 / 2.57 / 3.01), errors as at M=4.

"Two output-row tiles per workgroup" (amortizing the activation tile): tried as RN=8 rows per lane (each tile read
serves 8 weight rows) and as TN=128 (two 64-row tiles share one workgroup). At M=4 neither helps: RN=8 0.231 vs
RN=4 0.221 ms (loop form), 1.03 ms unrolled (register spill); TN=128 0.223-0.247. At M=4 the kernel is not
LDS-bound — per 4 activation elements a lane does 4 LDS loads against 4 dequants + 16 dots, so ALU issue dominates
and extra rows per lane only add register pressure. Two further ALU-side attempts also lost: a 256-entry
`vec4<f16>` trit lookup table in workgroup memory (0.248 ms unrolled, 0.198 loop form) or in a storage buffer
(0.259) instead of the 6-op shift/mask/bitcast decode, and K-splits for more workgroups on up_proj (no change).
Full sweeps: `m45.log`, `m45b.log`, `m45c.log` (summaries via `node summarize2.mjs <log>`).

Why M=5 misses: the kernel cost grows ~linearly in M above the fixed per-word dequant (0.193 -> 0.229 -> 0.323 ms
for M = 4 / 5 / 8, i.e. ~0.032 ms per extra row on top of ~0.065 ms fixed), while the reference stays 0.103 ms, so
M=4 is the last width under 2.0 on gate/up-sized matrices; M=5 needs either a ~10% cheaper inner loop or the
reference op to be counted as the gate+up pair the graph actually issues.

## 8. Design note — wiring the M<=8 kernel into the qwen35 prefill builder `lh` as the verify path

Applies because the M=4 ratio is <= 2 on gate/up-sized matrices (section 7). No implementation here.

1. **One choke point.** Every packed projection in `lh` (engine.pretty.js:42463) is emitted through `G2()`
   (engine.pretty.js:39567): `projInto`/`proj` for q/k/v, `linear_in_proj_qkv`, `linear_in_proj_z`, `o_proj`,
   `linear_out_proj`, `down_proj`, and `mlpInto` for gate+up (two `R()` calls into one `[T, 2*inter]` scratch with
   `dstColStart` 0 and `inter`). For Bonsai 2 they all take the `et(t, name) && W(K, l, o)` branch and call `R()`,
   which emits `com.xenova.LlamaPrefillMatmul` with `format: "lut2_32", lut: 9, M: T, blockOffset, outStride,
   dstColStart` and an f32 activation (`actGemm` is float32 for this model: `Wt(e)` is false and `e.weights.dtype`
   is float32). The lm_head is emitted by `E0()` right after the layer loop (engine.pretty.js:42796) and today
   computes only the last row (`lastRowUniform`); DFlash verify needs all T rows of logits — that is the second
   change, independent of the kernel.
2. **New op, not a new route inside `LlamaPrefillMatmul`.** Register `com.xenova.Lut2SmallMGemm` in the op table next
   to `LlamaPrefillMatmul` (engine.pretty.js:25672) with the same tensor contract (`aT [M, inFeatures]` f32, `bitsT`
   u32, `scalesT` u32, `yT [M, outStride]` f32) and args `M (1..8)`, `inFeatures`, `outFeatures`, `blockOffset`
   (multiple of 8), `outStride`, `dstColStart`, `lut` (9 only), `precision` (f16 | f32; default f16 per section 7),
   `kSplits`. `buildWgsl` = `gemmWgslM({M, TN: 64, RN: 4, unroll: true, math: precision, aStore: 'f16', ksplit})`
   with `dstColStart`/`outStride` folded into the store (`Y[m*outStride + dstColStart + row]`) and the reduce pass
   emitted as a second program when `kSplits > 1` (partials scratch `[kSplits, M, outFeatures]`). `workgroupSize`
   128, dispatch `(ceil(outFeatures/64), kSplits)`. Contract: `inFeatures % 256 == 0` (all Bonsai 2 widths are),
   `outFeatures % 64 == 0` (17408, 5120, 12288, 1024, 10240, 6144, 248320 all are), `blockOffset % 8 == 0`.
3. **Routing rule in `G2.R()`**: when `x(b)` (lut2, lut 8/9) and `n <= 8` (the session's `blockLen`), emit
   `Lut2SmallMGemm` instead of `LlamaPrefillMatmul`, keeping `realLenT` semantics by ignoring it (masked rows are
   computed and discarded; cost is baked by M, not by the live length). `kSplits` = smallest power of two that
   brings `outFeatures/64 * kSplits >= 256` workgroups, capped at `inFeatures/256`: 1 for gate/up (272 WGs) and
   lm_head, 4 for down_proj / o_proj / linear_out_proj (N = 5120), 8-16 for k/v (N = 1024, K = 5120 -> 20 chunks;
   use 4 or 5 since ksplit must divide 20). Leave the Hadamard input transform (`transformInput: Se` = `bi()`) as is:
   the kernel consumes the rotated f32 activation exactly like `LlamaPrefillMatmul` does.
4. **Head for all rows**: give `E0()` an `allRows` mode for the verify session that runs the final norm + rotate on
   all T rows and emits `Lut2SmallMGemm` with `M: T, outFeatures: vocab` on `lmHeadQ4`/`lmHeadQ4Scales` (measured
   2.34 ms at M=4 vs 1.42 ms for the single-row head the decode graph uses), then a per-row argmax (the decode graph's
   `verify_tokens` scaffold in `I0` shows the shape). Rank the accepted length on the host from the T argmaxes.
5. **Expected step cost** (inferred from section 7, not measured end to end): with every projection at ratio
   1.9-2.0 and the head at 1.65, a 4-row verify step should land near 2.0-2.2 decode steps (today 8-token step =
   8.1 decode steps; 4-row projections through the current route = 6.7x). Attention / GDN / norm ops at T=4 are the
   remaining unknown; measure the whole `ch` session at `blockLen` 4 and 5 after the swap before deciding M=5.
6. **Precision choice**: default `precision: "f16"` (max abs 1.7e-3 on up_proj, 2.5e-2 on down_proj at |ref| up to 31;
   argmax preserved in every run); switch to `"f32"` math (+12% time, error identical to the engine's current
   prefill route) if verify-vs-greedy mismatches appear in the live check.

## 9. Wired in (2026-09-21, later): `com.xenova.Lut2SmallMGemm` + route + all-rows head

Section 8 is implemented as marker-guarded patches in `../patch-internals.mjs` section (e) with the op package in
`smallm-op.mjs` (manifest + jinja templates generated from the same kernel structure as `gemmWgslM(unroll: true)`).
Through the engine's own compile path the op reproduces the standalone numbers (`op2.log`, 21 samples): M=4 up_proj
0.189 ms (ratio 1.86), down_proj 0.192 ms with kSplits=4 (1.81), lm_head 2.34 ms (1.70); errors identical to §7.
Full verify-session results (block 4 / 5 / 8 at ratio 2.07 / 2.34 / 3.29 vs decode, parity 4/4, 5/5, 8/8) are in
`../RESULTS-verify-qwen35.md`, "Stage-2 step 2".
