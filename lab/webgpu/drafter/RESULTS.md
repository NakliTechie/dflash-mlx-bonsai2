# DFlash2 drafter in WebGPU — stage-2 step 3 (2026-09-21)

The round-3 DFlash2 drafter (`Qwen3.8-27B-DFlash2-r3`, 1.9 B params) runs as WGSL compute in headless Chrome 153, loaded
from the Q4_K_M GGUF, and reproduces the MLX oracle stage by stage for one speculation cycle: with the oracle holding the
same GGUF weights every stage agrees to rms-relative ≤ 2.4e-6, the logits argmax agrees on 7/7 slots, and the candidate
selector picks the same 7 draft ids (`python\ndef reverse_string(s:` after the anchor `` ``` ``). Against the exact bf16
drafter the differences are the Q4_K_M quantization deltas (1–13 % rms on the residual stream) and the 7 ids still match.
Warm draft step 27 ms, context projection 10.7 ms per 27 tokens, weight load 4.9 s (CPU dequant to f16, 3.6 GB on the GPU).

Files (all under `lab/webgpu/drafter/`):

| file | what |
|---|---|
| `oracle/dump.py` | MLX oracle: one cycle, every stage to `.npy` + `index.json`. `--weights bf16` (exact drafter) and `--weights gguf` (drafter weights replaced by the GGUF's dequantized, f16-rounded values = the browser's copy). Runs in ~5 s / ~15 s with 4.5 GB, because the target side loads only the two modules the drafter borrows (`model.embed_tokens`, `lm_head` Packed modules, straight from the pack safetensors) and reuses the tapped features from `lab/webgpu/oracle/context_features.npy` (a full-target prefill; regenerated in-script when the cache is absent, 8 GB). |
| `gguf.js` | GGUF v3 reader (browser Range fetch or Node fs) + Q4_K / Q6_K / F16 / F32 dequant; `test-gguf.mjs` matches gguf-py on 6 tensors to 1e-7 relative. |
| `drafter.js` | the WGSL kernels + the forward (`projectContext`, `draftStep`, `select`) + optional per-op timestamp profiling. |
| `unit.js` / `unit.html` | 9 kernel unit tests on random inputs vs JS references (max abs ≤ 5e-7). |
| `harness.js` / `harness.html` | loads the drafter, feeds the oracle inputs, compares every stage, times cold/warm steps. `?oracle=bf16|gguf&profile=1&reps=N`. |
| `cdp-drive.mjs` | headless Chrome over CDP; prints `window.__dr` transitions and a `RESULTS` json line. |
| `run-*.log` | the runs quoted below (`run-gguf-vec4.log`, `run-bf16-vec4.log`, `run-gguf-vec4-profile.log`; `run-gguf-1.log` / `run-bf16-1.log` are the first kernel before the vec4 gemm). |

Reproduce: serve the scratchpad engine dir (`npx http-server <engine dir> -p 8795 -a 127.0.0.1 --cors -c-1`; it holds
symlinks `drafter -> lab/webgpu/drafter` and `model/Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf -> ~/Code/models/...`), run
`.venv/bin/python lab/webgpu/drafter/oracle/dump.py --weights gguf` (and `bf16`) from the repo root, then
`node drafter/cdp-drive.mjs "http://127.0.0.1:8795/drafter/harness.html?oracle=gguf"` from the engine dir.

## Weight layout facts (verified by the reader + gguf-py cross-check)

- 81 tensors, `general.architecture = dflash`, `general.file_type = 15` (Q4_K_M). The Q4_K_M recipe is a MIX: 74 matrices are
  Q4_K, but `blk.2.attn_v`, `blk.2.ffn_down`, `blk.4.attn_v`, `blk.4.ffn_down` are **Q6_K**. Norms, conv bases, q/k norms are
  F32. A loader that only implements Q4_K fails on 4 tensors.
- GGUF `ne` is fastest-dimension-first; every projection is stored PyTorch-style `W[out][in]` (row-major), so `y = x · Wᵀ`
  with each output column's `in`-vector contiguous. Sizes: `fc` ne [25600, 5120] (in 25600 = 5 taps × H); `attn_q` [5120,
  4096]; `attn_k/v` [5120, 1024]; `attn_output` [4096, 5120]; `ffn_gate/up` [5120, 17408]; `ffn_down` [17408, 5120];
  `attn_conv_proj` / `ffn_conv_proj` [5120, 1280]; `selector_hidden` [5120, 256]; `selector_predecessor/successor` [256,
  248320] (= 248320 rows of 256).
- `attn_conv_base` / `ffn_conv_base` F32 ne [5120, 2, 2] = row-major `[a][offset][h]`, a = 0 for `prepare`, 1 for `finish`
  (MLX `base_kernel[a][offset]`). The dynamic kernel `kernel_projection(x)` [.., 1280] reshapes to `[a][offset][group]`
  (index `a*640 + offset*320 + g`), 320 groups of 16 channels.
- **A codebook row is exactly one Q4_K block** (256 elements = 144 bytes at `row*144`): the selector dequantizes only the rows
  it touches (1 predecessor + 16 successors per slot) on the CPU; the two codebooks (35.8 MB each, raw) never go to the GPU.
- `dflash.rope.dimension_sections = [64, 0, 0, 0]`: llama.cpp's m-rope sections, all 64 rotation pairs on the temporal
  axis = plain neox-style RoPE over the full head_dim 128 (pairs (i, i+64), θ = 1e7). The MLX oracle uses `nn.RoPE(128,
  traditional=False, base=1e7)` — the same thing; nothing is "partial".
- Block layouts implemented: `block_q4_K` = d f16, dmin f16, scales[12] (6-bit packed, `get_scale_min_k4`), qs[128]
  (low nibbles → sub-block 2j, high → 2j+1); `block_q6_K` = ql[128], qh[64], scales[16] i8, d f16.
- Dequant throughput in the browser: 1.9 B params in 4.2 s (Float16Array conversion included); GGUF read 1.14 GB by 81
  Range requests from the local server.

## The forward, as implemented (one cycle, fresh context cache)

Activations are f32 throughout; weights are f16 packed two per u32 and unpacked in-shader (`unpack2x16float`), so the
`shader-f16` feature is not required. Semantics follow `dflash_mlx/model.py::DFlashAttention` (ContextOnly cache path)
and `dflash_mlx/draft/dflash2.py`.

Stage A, `projectContext(features [C, 5H])` — the projected context cache, once per new context token:
`fc` gemm (M = C, K = 25600, N = 5120) → `hidden_norm` RMSNorm → `draft_context [C, H]`; per layer: `k_proj` gemm →
`k_norm` (per head, 128) → RoPE at positions 0..C-1; `v_proj` gemm. Kept as `ctx_k / ctx_v [C, 8, 128]` per layer. The
context is NOT passed through `input_layernorm` (the oracle feeds `draft_context` straight into k/v_proj).

Stage B, `draftStep(cache, noise [8, H])` — per cycle:
`h = noise · embed_scale` (embed_scale = 1.0 for this target); for each of the 5 layers:
`input_layernorm` → `attn_conv_proj` gemm (dyn [8, 1280]) → `conv(a=0)` (prepare) → q/k/v gemms on the conv'd rows → q_norm,
k_norm → RoPE at positions C..C+7 → attention (32 q heads, 8 kv heads, scale 1/√128) over `[ctx_k ; block_k]` with the
mask `context: qpos − kpos < 2048; block: all` (the block is non-causal, `is_causal=false`) → `o_proj` gemm →
`conv(a=1)` (finish) + residual; then `post_attention_layernorm` → `ffn_conv_proj` → `conv(a=0)` → gate/up gemms → SiLU·up →
down gemm → `conv(a=1)` + residual. Finally `output_norm` and `selector_hidden` gemm on rows 1..7 (bound at a 20480-byte
buffer offset).

Stage C, `select(anchor, cand_ids [7][16], unary [7][16], sel_hidden [7, 256])` — CPU: greedy path walk
`score = unary + Σ_r pred[r]·hp[r]·succ[r]`, predecessor = the chosen id of the previous slot, starting at the anchor.

Kernels: `gemm` (64 threads = 4 output columns × 16 k-lanes, vec4<u32> weight loads = 8 f16 per lane per iteration, 8-row
tile, workgroup reduction), `rmsnorm` (one workgroup per row), `rope`, `attn` (one workgroup per query row × head, scores
in 12 KB of workgroup memory → ≤ 3072 keys), `conv`, `silu_mul`, `scale`. Q/K/V/O are laid out `[row, head, 128]`.

## Per-stage errors (headless Chrome 153, Apple M4 Pro, `metal-3` adapter, shader-f16 + timestamp-query available)

Columns 3–4: the browser vs the MLX oracle holding the SAME GGUF weights (f16-rounded) = kernel/accumulation error.
Columns 5–6: the browser vs the exact bf16 drafter = kernel error + Q4_K_M quantization. `ref max` = max |oracle value|.

| stage | n | vs GGUF-weight oracle: max abs | rms rel | vs bf16 oracle: max abs | rms rel | ref max |
|---|---|---|---|---|---|---|
| draft_context | 138240 | 5.25e-06 | 5.7e-07 | 1.03e-01 | 5.8e-02 | 7.74 |
| layer0_ctx_k | 27648 | 4.17e-06 | 2.9e-07 | 3.75e-01 | 3.1e-02 | 14.1 |
| layer0_ctx_v | 27648 | 1.14e-05 | 5.7e-07 | 1.19e+00 | 7.7e-02 | 33.5 |
| layer1_ctx_k | 27648 | 5.72e-06 | 3.0e-07 | 3.50e-01 | 3.7e-02 | 11.9 |
| layer1_ctx_v | 27648 | 1.53e-05 | 4.8e-07 | 1.37e+00 | 6.0e-02 | 27.4 |
| layer2_ctx_k | 27648 | 3.81e-06 | 2.7e-07 | 4.45e-01 | 3.3e-02 | 12.5 |
| layer2_ctx_v | 27648 | 1.91e-05 | 4.5e-07 | 9.81e-01 | 3.7e-02 | 41.5 |
| layer3_ctx_k | 27648 | 3.81e-06 | 2.7e-07 | 3.50e-01 | 3.4e-02 | 11.2 |
| layer3_ctx_v | 27648 | 1.91e-05 | 5.8e-07 | 1.74e+00 | 8.0e-02 | 37.4 |
| layer4_ctx_k | 27648 | 3.34e-06 | 2.6e-07 | 3.40e-01 | 3.2e-02 | 11.1 |
| layer4_ctx_v | 27648 | 1.76e-05 | 7.4e-07 | 1.61e+00 | 6.9e-02 | 38.2 |
| layer0_attn_in | 40960 | 3.81e-06 | 3.0e-07 | 6.77e-01 | 4.4e-02 | 22.8 |
| layer0_attn_out | 40960 | 2.54e-02 | 7.8e-07 | 7.80e+02 | 3.2e-02 | 1.91e+04 |
| layer0_out | 40960 | 1.56e-02 | 3.9e-07 | 1.19e+03 | 1.5e-02 | 5.64e+04 |
| layer1_attn_in | 40960 | 1.43e-05 | 1.9e-06 | 2.23e-01 | 9.3e-02 | 6.78 |
| layer1_attn_out | 40960 | 1.95e-02 | 3.4e-07 | 1.32e+03 | 1.8e-02 | 6.26e+04 |
| layer1_out | 40960 | 7.03e-02 | 4.9e-07 | 2.09e+03 | 2.1e-02 | 1.08e+05 |
| layer2_attn_in | 40960 | 1.24e-05 | 2.0e-06 | 1.49e+00 | 1.2e-01 | 10 |
| layer2_attn_out | 40960 | 9.38e-02 | 4.6e-07 | 2.15e+03 | 1.6e-02 | 1.16e+05 |
| layer2_out | 40960 | 3.91e-02 | 4.0e-07 | 2.14e+03 | 2.0e-02 | 1.25e+05 |
| layer3_attn_in | 40960 | 3.53e-05 | 2.2e-06 | 1.57e+00 | 1.0e-01 | 22.8 |
| layer3_attn_out | 40960 | 3.91e-02 | 4.2e-07 | 2.22e+03 | 2.1e-02 | 1.31e+05 |
| layer3_out | 40960 | 4.69e-02 | 5.3e-07 | 2.45e+03 | 2.7e-02 | 1.49e+05 |
| layer4_attn_in | 40960 | 4.39e-05 | 2.4e-06 | 5.90e+00 | 1.3e-01 | 33.4 |
| layer4_attn_out | 40960 | 5.47e-02 | 5.9e-07 | 2.55e+03 | 3.1e-02 | 1.48e+05 |
| layer4_out | 40960 | 8.59e-02 | 2.2e-06 | 5.09e+03 | 1.1e-01 | 4.81e+04 |
| final_hidden | 40960 | 4.55e-05 | 2.4e-06 | 1.66e+00 | 1.2e-01 | 22.1 |
| sel_hidden | 1792 | 2.00e-05 | 1.9e-06 | 1.06e+00 | 1.1e-01 | 14.4 |
| unary logits (own hidden · oracle head rows) | 112 | 3.15e-05 | 6.4e-07 | 1.20e+00 | 2.8e-02 | 27.2 |
| sel_edges | 112 | 3.99e-03 | 3.0e-04 | 1.02e+00 | 1.0e-01 | 12.9 |

Discrete outcomes (both oracles): logits argmax per slot 7/7 (`[12305, 198, 727, 9637, 3773, 1104, 25]`), selected ids
7/7 with the oracle's unary logits, and 7/7 with unary logits recomputed from the browser's own final hidden.

Reading the table:
- The GGUF-oracle column is f32 accumulation-order noise (2.6e-7 … 2.4e-6 rms), growing from the residual scale — the
  residual stream reaches |h| ≈ 1.5e5 by layer 3, so a 9e-2 max-abs there is 6e-7 relative. Every kernel is right.
- `sel_edges` 3e-4 rms is the one stage where the browser is MORE exact than the GGUF oracle: the oracle's codebooks were
  f16-rounded before the MLX matmul, while the browser dequantizes codebook rows to f32; the bf16-oracle column confirms
  the selector is otherwise identical (same 1.0e-1 as the surrounding stages).
- The bf16 column equals the MLX-vs-MLX bf16-vs-GGUF deltas to the digit (e.g. `draft_context` 1.030e-1 / 5.75e-2 in both
  runs), i.e. the entire gap to the exact drafter is Q4_K_M quantization, none of it the port. Quantization moves the
  residual stream by 1.5–13 % rms and the final hidden by 12 %, yet leaves the argmax and the drafted path intact here
  (one prompt, one cycle; acceptance over many cycles is the real measure — item 6 below).
- **Magnitude warning for the engine wiring:** the drafter's residual stream is f32 in the oracle and peaks at 1.49e5 (layer 3
  out) — above f16's 65504. The engine's prefill activations are f16 on the q1 path (inferred from `patch-internals.mjs`: `normedDtype: O&&S ? "float16" : "float32"`, features scratch in the `actGemm` dtype); the drafter's residual
  stream must stay f32 (or the layer inputs be pre-scaled) when the port moves onto the engine's op set, or it overflows.

## Timings (`run-gguf-vec4.log`, `run-bf16-vec4.log`, `run-gguf-vec4-profile.log`)

| what | ms |
|---|---|
| weight load (81 Range fetches + CPU dequant + upload, 3.60 GB f16 on the GPU) | 4949 (dequant 4218) |
| context projection, C = 27 tokens, cold / warm | 26.1 / 10.7 (median of 3: 10.6, 10.7, 10.9) |
| draft step, cold (first submit after load: pipeline warm-up + first-touch) | 96.1 (41.9 in the bf16 run) |
| draft step, warm (5 reps) | 40.2, 27.0, 26.9, 27.2, 27.0 → median **27.0** |
| selector walk on the CPU (7 slots × 16 candidates, row dequant included) | 0.15 |
| gemm kernel v1 (scalar u32 loads), for the record | warm step 78.1, context 29.9 |

Per-op GPU time in one warm step (timestamp queries, summed over the 5 layers): `ffn_down` 6.83, `ffn_up` 6.20, `ffn_gate`
6.07, `attn_q` 1.62, `attn_output` 1.57, `attn_conv_proj` 0.59, `attn_k` 0.58, `ffn_conv_proj` 0.58, `attn_v` 0.57, attention
0.25, `selector_hidden` 0.06, all norms/convs/silu < 0.08 each; total ≈ 25 ms. The step is the weight stream: 3.6 GB of f16
at ~140 GB/s. Against the engine's own numbers (decode 34.6 ms/token, 8-token verify 279 ms today; a real cycle draws
2.4–4.6 accepted tokens): the drafter costs 0.8 decode steps per cycle in this form.

## What is stubbed (taken from the oracle dumps, not computed in the browser)

1. **Context features** `[C, 5·H]` — the target's residual stream at layers 5/19/33/47/61 for the 27 prompt tokens, from
   the MLX target (`lab/webgpu/oracle/context_features.npy`). In the product these come from the engine's `dspark.features`
   tap (verify-graph agent).
2. **Noise embedding** `[8, H]` — the target's `embed_tokens` of `[anchor, mask×7]` (Hadamard-inverse-rotated 2-bit rows,
   MLX Packed module). In the product: the engine's `LlamaEmbed` on `token_embd` (rotation handled by the engine).
3. **The target lm_head on rows 1..7** — two forms, both from the oracle: (a) the top-16 candidate ids and their logits per
   slot (`cand_ids.npy`, `logits_top16.npy`, the plain stub); (b) the effective lm_head rows for those 112 candidates
   (`head_rows.npy`, 7×16×5120, obtained by pushing the identity through the pack's linear Hadamard+ternary head), so the
   browser recomputes the unary logits from its OWN final hidden and the selection is exercised end to end from the
   browser's hidden. The top-16 candidate SET is still the oracle's: the browser cannot rank the full 248320-row vocab
   without the head weights. In the product: the engine's two-stage head scan (`LlamaDecodeLmHeadArgmax topK` +
   `LlamaDecodeLmHeadRescore`) already produces per-row top-K candidates + scores — exactly this interface.
4. `embed_scale` (1.0), `output_multiplier` (1.0), `final_logit_softcapping` (none) read from the oracle index.
5. One cycle only: context positions 0..C-1 from a fresh cache; the ContextOnly cache's sink-64 / window-1024 eviction and
   the per-cycle append of new context rows (`append_projected_context_cache`) are not implemented — `projectContext` is a
   whole-context rebuild. The attention kernel's key capacity is 3072 (sink + window + block fit).

## What is next

1. **Wire to the live target buffers** (with the verify-graph agent's `dspark.features` + all-rows head): replace stubs 1–3
   with GPU buffers from the engine's graph session — features need a dtype/layout adapter (engine `act` is f16 `[T, 5H]`
   strided copy; the drafter reads f32 `[C, 5H]`), the noise rows come from `LlamaEmbed` on `[anchor, mask×7]`, and the
   head-scan top-K per row replaces `cand_ids`/`logits_top16`. Keep the drafter residual stream f32 (magnitude warning).
2. **Incremental context cache**: append K/V for the k newly accepted tokens each cycle (M = k ≤ 8 through `fc` + k/v per
   layer, ≈ 0.3 ms per token at the measured rate) instead of the whole-context rebuild; implement sink/window eviction
   (or the FullContext variant) and pass real key positions (the kernel takes `ctxPos0`; a positions buffer is needed once
   eviction starts).
3. **Runner**: one cycle = drafter step (27 ms) → verify graph (target, other agent) → accept → append context. The first
   version can re-project the context from the tap after each verify.
4. **Speed levers, in order:** (a) the ffn gemms are 19 of 25 ms — a fused gate+up kernel (one weight stream, two
   accumulators) and 2-row-per-thread tiles should reach 200+ GB/s (≈ 18 ms/step); (b) in-shader Q4_K dequant to keep
   the weights at 1.1 GB instead of 3.6 GB f16 (3.3× less traffic, ≈ 8 ms/step, and no 4 s CPU dequant, and 2.5 GB less
   unified memory next to the 5.9 GB target); (c) with acceptance ≈ 4 tokens per cycle and the verify at 279 ms today the
   drafter is not the bottleneck — the small-M ternary verify kernel (other agent) is.
5. **Bind-group / buffer churn**: `draftStep` allocates ~40 buffers and ~70 bind groups per call; pre-allocate once per
   (C-capacity) and reuse — required before a runner loops thousands of cycles.
6. Fold `dump.py`'s MLX-vs-MLX bf16-vs-GGUF comparison into a multi-cycle acceptance check once the runner exists (does
   Q4_K_M cost acceptance vs the bf16 drafter on real prompts?).
