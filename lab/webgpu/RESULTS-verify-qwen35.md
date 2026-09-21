# Stage-2 step 1 — qwen35 prefill graph "verify mode" (Ternary Bonsai 2 27B, WebGPU engine)

Date: 2026-09-21. Scope: give the vendored LocalMind engine's qwen35 prefill builder (`lh`) the same optional
verify-mode surface the Llama builder (`I0`) already has — per-position argmax for ALL rows of a <=8-token block
plus a copy of the residual stream at the drafter's tap layers — and prove it against the model's own greedy
continuation. The small-M kernel is out of scope (see `wgsl-gemm-spike/`).

## What the patch touches (`patch-internals.mjs`, section (d))

All replacements are marker-guarded (`once(...)`: each marker must match exactly once in the minified source) and
re-applied by `node patch-internals.mjs && node --check engine.dflash.js` against the untouched
`~/Code/naklios-universe/LocalMind/ternary_bonsai_2_27b.js`. Three markers, all inside `function lh(e,t,r)`
(graph name `qwen35-prefill`):

| marker | change |
|---|---|
| `function lh(e,t,r){` | signature gains a 4th argument `$v` = `{tapLayers?: number[], allRowsHead?: boolean}` |
| `Je=Se(T0({...}),"embed_tokens",!0);for(let te=0;te<i;++te){let ge=e.offsets.layers[te],` | after the embedding: `$f = J.scratch("dspark.features", k, [T, taps*H])` when `tapLayers` is non-empty (`k` = actGemm dtype, as `I0` uses `me = y`); at the ENTRY of each layer `te` in `tapLayers`, `StridedCopy` of the residual `Je` into column block `indexOf(te)` of `$f` |
| `Ee(te,ge,Je)}let Ze=J.nodeCount,st=E0({...});return{...}` | after the layer loop: `J.output($f,"dspark.features")`; when `allRowsHead`: scratch `verify_tokens [T]` (uint32) filled by one `ki(...)` per row (`hidden: J.view(Je, row*H, "float32", [H])`, `q1: Wt(e)`, `nameSuffix: V.r<row>.`, `outputOffset: row`), declared output; `E0` then receives `nextTokenPick` so `next_token` = `verify_tokens[real_len-1]`; the return carries `nextTokenPickUniform` (`K2.writeRunInputs` already writes it) |

Default path: with no 4th argument every addition is gated on `$v` / `$f` / `$p`, so the emitted graph is the same
node sequence as before (inferred from the gating; runtime check below compares node counts and outputs of
`lh(model, cache, 8)` vs `lh(model, cache, 8, {})`).

Tap convention (mirrors `I0`): `tapLayers` entry `Z` copies the residual at the ENTRY of layer `Z`, i.e. the output
of layer `Z-1`. The MLX drafter (`dflash_mlx/engine/target_qwen_gdn.py:842`, `captured[layer_id + 1]`) uses the
OUTPUT of `target_layer_ids[k]`. To feed the drafter from this graph pass `tapLayers = target_layer_ids.map(x => x + 1)`.
The harness default `[5,19,33,47,61]` follows the task statement (I0 semantics) and only proves plumbing + shape.

## Head path used

`I0` for this model: `Wt(e)` (= `packs.q1 != null`) decides between the batched `LlamaDecodeLmHeadArgmax` paths and
the per-row `ki` fallback. The batched `I5(...)` branch requires `vocab <= 256*512 = 131072`; Bonsai 2's vocab is
248320, so `I0` itself would fall through to per-row `ki` unless a Q4+Q8 rescore pair is loaded. The patch
therefore uses per-row `ki` — the same call `E0` makes for today's decode `next_token` — and `ki` dispatches
internally (prism head / two-stage Q4+Q8 / q1 / q4 / q8 / dense) from the loaded packs. The route the loaded model
took is recorded by the harness (`head.kiRoute`), see the run section.

## Harness (`verify-qwen35-harness.js`, page `index-verify.html`, driver `cdp-drive-verify.mjs`, gate `wait-then-verify.sh`)

1. `Eng.load('/model/Ternary-Bonsai-2-27B-PTQ1_0.gguf', {maxLength: 4096})` from the same http origin (range-served GGUF).
2. Prompt prefill -> `g0` (1 token). `pos = cache.get_seq_length()` (= prompt length).
3. `slot = await cache.allocateCheckpointSlot(); slot.checkpoint.capture()` — an independent capture of the conv +
   linear-recurrent state (the shared `mutableStateCheckpoint()` slot belongs to the pipelined decode loop, which
   captures/restores it itself). `rewind()` = `slot.checkpoint.restore(); cache.seqLength = pos`. This is the fix
   for the 2026-09-21 stage-0 MISMATCH: setting `seqLength` alone does not rewind the linear-attention state.
4. 8 more greedy tokens `g1..g8` via the normal path (feeds `g0..g7`).
5. `class VerifySession extends I.ch { buildEmission() { return I.lh(model, cache, 8, {tapLayers, allRowsHead: true}) } }`;
   `rewind(); s.run(g0..g7, pos)` x 6 reps (timed); read `verify_tokens` and `dspark.features` after rep 0.
6. Check `verify_tokens[i] == g[i+1]` for i in 0..7; features shape `[8, 5*5120]`, non-zero, per-tap stats.
7. Decode reference from the same rewound state: normal 8-token prefill of `g0..g7` (its token must also be `g8`) + 24 decode steps.

GPU-sharing gate (`wait-then-verify.sh`): no `llama-server`, no real `dflash serve` process, no other Chrome on a
`chrome-profile` user-data-dir, >= 7 GB unused; polled every 60 s; one headless Chrome, killed by the driver on exit.

## Run

Log: `run-verify-2026-09-21-2237.log` (gate cleared 22:36:55 IST at 9454 MB unused after the llama-server 100k run
exited; headless Chrome `--enable-unsafe-webgpu`, hidden tab with the MessageChannel rAF/timer shim; model load
~10 s from the range-served local GGUF, weights cached in `chrome-profile/`).

Prompt: 21 tokens (`<|im_start|>user\nWrite a short paragraph about lighthouses.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`).
Greedy: `[43, 21204, 19154, 2418, 430, 50032, 17210, 314, 55394]` (identical to the stage-0 run of 20:07).

### Parity (verified: the log's RESULTS block)

| check | result |
|---|---|
| `verify_tokens[i] == greedy[i+1]`, i = 0..7 | **8/8** — `[21204, 19154, 2418, 430, 50032, 17210, 314, 55394]` |
| `next_token` from the verify graph (`nextTokenPick` of row 7) | 55394 == greedy[8] |
| normal 8-token prefill of `g0..g7` from the same rewound state | 55394 == greedy[8] (the rewind itself is sound) |
| `dspark.features` | dtype float32 (actGemm `k` is f32 for this model: no q1 pack, weights not f16), shape `[8, 25600]` = `[8, 5*5120]`, 204800/204800 non-zero, 8 distinct rows |
| per-tap mean-abs / max-abs (entry of layer 5/19/33/47/61) | 0.087/28.0, 0.371/60.5, 0.532/50.6, 0.861/54.7, 2.870/54.6 — all finite, growing with depth as a residual stream does |

### Head path taken (observed: `head` block of the log)

`packs.q1 = null` (so `Wt(e)` is false), `lmHeadQ4` set with `lmHeadLut = 9` (the lut2_128 pack), no `lmHeadQ8`,
`config.prismHadamard.weights` includes `output.weight`. `ki` therefore took its FIRST branch — the prism head:
Hadamard-rotated final norm (`s0(...)`), `LlamaPrefillMatmul` with `M: 1, format: "lut2_32", lut: 9` over the
248320-row head, `ai.onnx.ArgMax`, `StridedCopy` into `verify_tokens[row]` — the same route today's decode
`next_token` takes through `E0`. Neither `LlamaDecodeLmHeadArgmax` nor the two-stage rescore is reachable for this
checkpoint (no Q8 rescore pack; `I5` also fails on vocab > 131072), so `I0` would have used this same per-row path.

### Graph (observed)

| graph | nodes | tail nodes | outputs |
|---|---|---|---|
| `lh(model, cache, 8)` (no 4th arg) | 1351 | 5 | `next_token` |
| `lh(model, cache, 8, {})` | 1351 | 5 | `next_token` |
| `lh(model, cache, 8, {tapLayers: [5,19,33,47,61], allRowsHead: true})` | 1385 (+5 taps, +8x~3.6 head, +1 pick) | 34 | `dspark.features`, `verify_tokens`, `next_token` |

Verify session build (graph + pipeline compile, warm shader cache): 126 ms.

### Timings (observed, 6 reps, ms; `s.run()` includes the `next_token` readback)

| step | ms |
|---|---|
| 8-token verify step (taps + all-rows head) | 364.3 / 374.0 / 381.8 / 381.8 / 379.1 / 373.5 (min of reps 2-6: 373.5) |
| readback of `verify_tokens` + `dspark.features` (819 KB) | 0.7 |
| decode step (24 tokens after an 8-token prefill, same state) | 42.2 / token |
| ratio verify-step / decode-step | 8.85 |

Reference points: the stage-0 plain 8-token `qwen35-prefill` step measured 279 ms against 34.6 ms decode
(ratio 8.07) at 20:07 on an idle GPU. This run's decode is 22 % slower (42.2 vs 34.6 ms) with another agent's
WGSL microbench Chrome allowed to run concurrently by the gate (it only excludes `chrome-profile` Chromes), so the
absolute numbers carry that noise; the verify-mode delta itself (about +95 ms at equal decode speed) is the
per-row prism head: 8 x (M=1 lut2 matmul over 248k rows + argmax), versus 1 x in the plain graph. Batching those 8
rows into one M=8 head matmul is the obvious next cut and belongs with the small-M kernel spike.

## What is verified vs not

- verified (this run, log above): 8/8 row parity; next_token parity; features shape/non-zero/distinct rows;
  checkpoint rewind through `allocateCheckpointSlot` (the plain 8-token prefill from the rewound state reproduces g8).
- observed: timings (single session, GPU shared with a concurrent microbench).
- inferred: the default path emits the same graph as before — every added statement is gated on `$v`; runtime shows
  identical node count / tail count / outputs for no-arg vs `{}`; I did not diff the emitted node list against an
  unpatched build.
- not run: a real drafter consuming these features (needs `tapLayers = target_layer_ids + 1` per the convention
  note); blocks shorter than 8 (`real_len < blockLen` — `writeRunInputs` writes `real_len`, `nextTokenPick` reads
  row `real_len - 1`, but `verify_tokens` rows beyond `real_len` are computed on padding and unchecked); any prompt
  other than the one above; f16-activation checkpoints (features would come back as `Uint16Array` bits; the harness
  decodes them, untested here).

