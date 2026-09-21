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


## Stage-2 step 2 — small-M GEMM route + all-rows head (2026-09-21, later session)

The `wgsl-gemm-spike` kernel is now an engine op and the verify graph's projection + head path
(`patch-internals.mjs` section (e), building on section (d); op package in `wgsl-gemm-spike/smallm-op.mjs`):

| change | where |
|---|---|
| `com.xenova.Lut2SmallMGemm` registered in the engine's op override map (`Lf`, resolved first by `_8`) — manifest + two jinja assets (main, split-K reduce); M 1..8 baked per program, `precision` f16 / f32 / exact, `kSplits` | `Df();Lf.set(...)` before the export line |
| `G2.R()` (the single emitter for every packed projection in `lh`: q/k/v, in_proj_qkv/z, o, out_proj, gate+up, down) takes the new op for lut2 / lut 9 weights when `n <= 8` and a small-M option is active (`lh` 4th-arg `smallM: {precision, headPrecision?, kSplits?}` or `globalThis.__dflashSmallM`); K-splits auto: smallest divisor of K/256 giving >= 256 workgroups (1 for gate/up + head, 4 for N=5120, 20 for k/v) | marker `R=(k,b,T,I,$,X,Q,V)=>{let K=` + `G2({...,smallM:$smallM})` + `G2({transformInput:Se,smallM:$v?.smallM,` |
| verify head: with smallM active, section (d)'s T per-row `ki` heads become one rotated final norm (`s0`) over all T rows + one `Lut2SmallMGemm` (M = T) on `lmHeadQ4`/`lmHeadQ4Scales` + `ai.onnx.ArgMax(axis 1)` + `StridedCopy` into `verify_tokens`; logits declared as output `verify_logits` | marker = section (d)'s `for(let $r=0;$r<a;++$r)ki(...)` |

Without a small-M option nothing changes: `lh(model, cache, 8)` still emits 1351 nodes / `next_token` only, and the
section (d) verify graph still emits 1385 nodes (runs `verify-b8-off.log`, `verify-b4-off.log`, `verify-b5-off.log`).

### Run matrix (`run-verify-matrix.sh`; logs `verify-b<block>-<smallm>.log`; GPU gate cleared, idle GPU, taps [6,20,34,48,62])

Same prompt and greedy tokens as the step-1 run (`[43, 21204, 19154, 2418, 430, 50032, 17210, 314, 55394]`).
`stepMin` = min of reps 2-6 of `s.run()` (incl. `next_token` readback); decode = 24 tokens after the block.

| block | route | parity `verify_tokens[i] == g[i+1]` | next_token | step ms (min) | decode ms/tok | **ratio** | nodes |
|---|---|---|---|---|---|---|---|
| 8 | off (section d) | 8/8 | 55394 ok | 289.6 | 34.8 | 8.32 | 1385 |
| 8 | smallM f16 | **8/8** | ok | **114.1** | 34.7 | **3.29** | 1357 |
| 8 | smallM f32 | 8/8 | ok | 489.7 | 35.3 | 13.87 (M=8 f32 tier spills, as in the spike) | 1357 |
| 4 | off | 4/4 | 430 ok | 242.3 | 34.7 | 6.98 | 1369 |
| 4 | smallM f16 | **4/4** | ok | **72.1** | 34.8 | **2.07** | 1357 |
| 4 | smallM f32 | 4/4 | ok | 79.1 | 35.5 | 2.23 | 1357 |
| 5 | off | 5/5 | 50032 ok | 259.5 | 34.7 | 7.48 | 1373 |
| 5 | smallM f16 | **5/5** | ok | **82.1** | 35.1 | **2.34** | 1357 |
| 5 | smallM f32 | 5/5 | ok | 92.0 | 35.8 | 2.57 | 1357 |

Step-1's 373 ms / ratio 8.85 for block 8 (measured with a concurrent Chrome) is 289.6 / 8.32 on the idle GPU.
Verify-session build with the new op: 328-439 ms (vs 117 ms; the op compiles one pipeline per (M, K, N, kSplits)).
`dspark.features` unchanged in shape / non-zero fraction (1.0) in every run; the normal 8-token prefill from the
rewound state still reproduces g8 in every run.

### Precision / tie-flip check (observed: `verify_logits` top-1 minus top-2 per row)

| row | f16 tier margin | f32 tier margin |
|---|---|---|
| 0..7 (block 8) | 10.628, 6.355, 1.112, 6.095, 0.529, 0.354, 9.931, 1.260 | 10.620, 6.352, 1.109, 6.092, 0.525, 0.352, 9.934, 1.266 |

The two tiers move each logit by <= ~0.008 on this prompt; the smallest top-2 margin (row 5, 0.354) is ~45x the
f16 tier's max-abs logit error measured against the float64 reference on 2048 head rows (2.4e-3, spike RESULTS §7)
and ~450x the f32 tier's (7.8e-4). No row flipped in any run. The current (section d / `ki`) path has no logits
output, so its margins are not measured; its argmax agrees with both tiers on all 8 rows.

### What is verified vs not

- verified: parity 8/8, 4/4, 5/5 and next_token on every route (logs above); ratios per block length on an idle GPU;
  op numerics vs the float64 CPU reference through the engine op (`wgsl-gemm-spike/op2.log`: same errors as the
  standalone kernel, argmax preserved at M = 4/5/8 on up_proj, down_proj, lm_head).
- observed: build times; margins (one prompt).
- inferred: the ~25 ms not spent in projections at block 4 (64 layers x ~0.7 ms of projections + 2.3 ms head ≈ 47 ms
  of the 72 ms) is GDN recurrence / attention / norms / Hadamards at T=4 — not profiled per op.
- not run: the f32 tier at block 6-7; `kSplits` overrides; a second prompt; the normal (non-verify) prefill graph with
  `globalThis.__dflashSmallM` set (the route is wired for it but only the verify graph was exercised).

## Stage-2 step 3 — recurrence-only rewind for the runner (2026-09-21, late)

Goal: replace the runner's per-cycle tape replay (52–111 ms: restore + a (k+1)-row verify session) by a rewind that
only re-advances what the checkpoint restore made stale — the 48 linear-attention layers' conv windows and
recurrent states — using the rows the verify run already computed. The attention KV rows are position-indexed and
already correct. Patch: `patch-internals.mjs` section (f); harness `rewind-harness.js` / `index-rewind.html`; run
`rewind-run2.log` (`rewind-run1.log` / `rewind-diag1.log` are the first attempt, whose harness ran the rewind after the
tee had been overwritten by a later verify — superseded).

### What the patch adds

- **f1 tee** — `lh(model, cache, T, { ..., teeRecurrence: true })`: for every linear-attention layer `L`, the conv
  input rows `Ae` (in_proj_qkv output, `[T, convDim=10240]` f32) and the per-row recurrence gates `xt`
  (`[T, 2*numHeads=96]` f32: beta | decay) become graph outputs `rw.bcx.L` / `rw.gate.L` (own buffers, 48 x 8 x 10240
  x 4 B = 15.7 MB at T = 8). No copy op: the existing scratches are re-declared as outputs, so the verify step's cost
  is unchanged (the node count is the same; the buffers just leave the pool). Everything else the recurrence needs
  (q/k/v) is recomputed from `bcx` in the rewind, so the in-place q/k normalization done by the linear-attention op
  never leaks into the tee.
- **f2 `RewindSession`** (`TernaryBonsai2.__dflashInternals.RewindSession`):
  ```js
  const R = new I.RewindSession(model /* m.model */, cache /* m.generationState.cache */, T /* the verify session's blockLen */, verifySession /* built with teeRecurrence:true */);
  await R.build();      // graph "qwen35-rewind": per linear layer Qwen35PrefillConv + Qwen35LinearAttention over the
                        // teed rows, bound to the SAME conv/recurrent state tensors the verify graph updates; 96 nodes,
                        // 144 prepared steps, build 3 ms (all pipelines already cached by the verify session)
  R.run(n);             // 1 <= n <= T: enqueue only (no readback). Precondition: the verify session's last run was the
                        // block whose rows 0..n-1 are being accepted, and the checkpoint taken before that run has been
                        // restored (slot.checkpoint.restore()). Effect: conv + recurrent states of every linear layer
                        // advance by rows 0..n-1 of that verify run, exactly as the verify graph with real_len = n.
                        // The caller then sets cache.seqLength = pos + n. Rows n..T-1 of the KV cache are stale but
                        // position-indexed; the next verify overwrites them.
  R.layers, R.dims;     // the linear layer indices and {convDim, convState, numHeads, headDimK, headDimV}
  R.dispose();
  ```
  One `RewindSession` per verify session (it binds that session's tee outputs). `real_len` is a uniform, so one
  rewind graph serves every n. Also exposed: `I.ht(tensor, elementOffset, elementCount)` (the engine's tensor view)
  for reading the state slices `cache.linearConvStates[L]` / `cache.linearRecurrentStates[L]`.
- Runner integration (not done here — the runner agent owns `runner/`): replace `restore + replay session (k+1)` by
  `slot.checkpoint.restore(); R.run(k + 1); cache.seqLength = pos + k + 1;` with the verify sessions built with
  `teeRecurrence: true` and one `RewindSession` per block length. When k+1 == Lv nothing is needed (as today).

### Check (`rewind-run2.log`, code prompt = the runner's LRUCache prompt, 57 tokens, smallm f16, taps [6,20,34,48,62])

Greedy `g0..g16 = [71093, 12305, 198, 12237, 198, 75, 2585, 11198, 6971, 271, 1378, 4522, 264, 11088, 436, 34810, 318]`;
verify(8) of `g0..g7` from the checkpoint: 8/8 rows match, next_token 6971 == g8. For n = k+1 in {1, 2, 4, 8} (accept
k in {0, 1, 3, 7}), three states are compared over all 48 linear layers (39,223,296 f32 elements per snapshot:
conv 10240x3 + recurrent 48x128x128 per layer):

| n | rewind vs verify8 run with real_len = n (same kernels) | rewind vs replay through the n-row session (today's runner path) | next-cycle verify(8) tokens: rewind == replay == truncated | rewind ms (median of 8) | replay ms | speedup |
|---|---|---|---|---|---|---|
| 1 | **bitwise identical** (39,223,296 / 39,223,296) | max abs 1.07e-2 (conv, layer 60: -2.3471 vs -2.3578), 4.3 % bitwise equal | yes; 8/8 == greedy | **7.4** | 50.4 | 6.8x |
| 2 | **bitwise identical** | **bitwise identical** | yes; 8/8 == greedy | **7.8** | 54.8 | 7.0x |
| 4 | **bitwise identical** | **bitwise identical** | yes; 8/8 == greedy | **8.4** | 74.7 | 8.9x |
| 8 | **bitwise identical** (also == the untruncated verify(8) state) | **bitwise identical** | yes; 8/8 == greedy | **10.0** | 117.8 | 11.8x |

Reading: the rewind reproduces, bit for bit, the state the 8-row verify graph itself produces after n real rows —
which is the state consistent with the KV rows the verify wrote. The n-row replay sessions for n >= 2 land on the
same bits; the 1-row session does not (T = 1 selects the engine's single-token conv / decode-attention variants, so
its residual stream differs at the 1e-2 level from the 8-row graph's row 0) — the rewind is therefore *more*
consistent than today's replay at k = 0, and the next cycle's 8 verify tokens are identical either way on this
prompt. "rewind ms" = `restore + R.run(n) + queueIdle`; "replay ms" = `restore + session.run()` incl. its
`next_token` readback, as the runner does. The rewind is ~7 ms + 0.4 ms/row: 144 small dispatches (48 x (conv +
l2norm + recurrence)); dispatch count, not work, sets its floor.

Expected runner effect (inferred from `runner/RESULTS.md`'s breakdown, not run): replay mean 59.3 -> ~8 ms at
block 8 (220.6 -> ~170 ms/cycle, 49.5 -> ~38 ms/token, 0.73x -> ~0.95x); block 5: 33.9 -> ~5 ms mean
(160.5 -> ~132 ms/cycle, 43.8 -> ~36 ms/token, 0.81x -> ~1.0x).

### What is verified vs not

- verified: the four rows of the table (bitwise state identity vs the truncated verify graph for n = 1, 2, 4, 8; bitwise
  identity vs the n-row replay for n = 2, 4, 8; identical next-cycle tokens and next_token for all n; timings).
- observed: build times (rewind 3 ms; the verify session with the tee 151 ms).
- inferred: the runner-level speedup above; that the tee adds no measurable cost to the verify step (same node count;
  not re-timed).
- not run: the lighthouse prompt; block lengths other than 8 for the tee/rewind (the class takes any T); the runner
  itself with the rewind (owned by the runner agent); a rewind after a verify whose block had padding rows
  (real_len < T) — `R.run(n)` with n <= real_len is the intended use.
