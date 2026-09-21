# Stage-2 step 4 — the speculative runner against the live target engine (2026-09-21)

A full DFlash2 speculative generate loop runs in headless Chrome 153 against the vendored LocalMind engine holding
Ternary Bonsai 2 27B, with **no oracle stubs**: the target's residual taps, its embedding rows and its lm_head all come
from the engine's own graphs, the drafter is the WebGPU port of step 3, and the cycle is checkpoint → draft → verify →
accept → context append → (restore + replay) → emit. On the leg-7 code prompt (57 prompt tokens, thinking off) the
speculative output is **identical to the engine's plain greedy decode for all 256 tokens** at every verify block length
tried (8, 5, 4); the replay's `next_token` equalled the bonus token in every one of the 47 + 38 + 31 replays.

Speed today is **below plain decode**: best 43.8 ms/token (block 5, 22.8 tok/s) against 35.7 ms/token plain (28.0
tok/s) = 0.81×. The cycle is verify (80–122 ms) + tape replay (52–111 ms when fewer than all drafts are accepted) +
draft/head (39 ms); the two target-side terms are the whole story and are quantified below.

Files (`lab/webgpu/runner/`): `runner.js` (the loop; hidden-tab shim from the verify harness), `runner.html`,
`cdp-drive.mjs` (headless Chrome over CDP, profile `chrome-profile-runner`, port 9338), logs `run-selftest-1.log`,
`run-code-b8-1.log` (lazy session builds inside the timed loop — superseded), `run-code-b8-2.log`, `run-code-b5.log`,
`run-code-b4.log`, `run-lighthouse-b5.log`. Drafter changes in `lab/webgpu/drafter/drafter.js`: `createContext(cap)` +
`appendContext(cache, featBinding, rows)` (incremental projected-context cache fed from a GPU buffer binding) and
`draftStep` accepting the noise rows as a GPU binding. `patch-internals.mjs` untouched (sections (d)+(e) as committed by
the verify-graph and kernel agents, `ec050ea`).

Reproduce (engine dir served on 8795 with symlinks `runner`, `drafter`, `model/*.gguf`):
`node runner/cdp-drive.mjs "http://127.0.0.1:8795/runner/runner.html?prompt=code&block=8&max=256"`; `prompt=oracle` runs
the MLX-oracle self-test only; `plain=0` skips the reference; `smallm=f16|f32|off`.

## The pieces, as wired

| piece | source in the engine | how the runner uses it |
|---|---|---|
| target features + all-rows argmax | `lh(model, cache, T, {tapLayers:[6,20,34,48,62], allRowsHead:true, smallM:{precision:'f16'}})` (patch (d)+(e)); outputs `dspark.features [T, 5·5120]` f32, `verify_tokens [T]`, `verify_logits [T, V]`, `next_token` | one `VerifySession extends ch` per block length 1..Lv, prebuilt (8 sessions: 1.1 s, 125–187 ms each) and reused; `run(ids, past_len)` then `cache.seqLength = past_len + T` (the session does not advance it) |
| prompt prefill | the same tapped sessions, prompt in chunks of ≤ 8 (57 tokens = 7×8 + 1) | features of every chunk → `appendContext`; the last chunk's `next_token` is the first generated token (the anchor) |
| noise embedding | a micro graph (`f0` subclass + `ba` builder, pattern `wgsl-gemm-spike/bench.js`): `LlamaEmbed` on the packed embedding (`embedBits/embedScales`, format `q4_0` — the branch `T0` takes for this model) + `bi(...)(h, 'embed_tokens', true)` = the pack's inverse Hadamard with signs | `input_ids = [anchor, mask×7]` written per cycle; the output scratch is bound directly into the drafter's first op |
| lm_head on the drafter's rows 1..7 | micro graph: `bi(...)(x, 'lm_head')` (signs + forward Hadamard, NO final norm — the drafter applied its own `output_norm`) + `Lut2SmallMGemm` M=7 over `lmHeadQ4/lmHeadQ4Scales` (lut 9, f16 tier) | the drafter's final hidden rows 1..7 are copied GPU→GPU into the graph's step input; logits [7, 248320] read back (7 MB); top-16 per row on the CPU; selector walk as in step 3 |
| target state rewind | `cache.allocateCheckpointSlot()` (conv + linear-recurrent state; own slot, not the decode pipeline's) | `capture()` before each verify; `restore()` + `seqLength = pos` + replay when fewer than Lv tokens were consumed |

Cycle (block length Lv, anchor a): draft 7 → keep the first Lv−1 drafts → `capture` → verify `[a, d1..d_{Lv−1}]` →
k = longest prefix with `verify_tokens[i] == d_{i+1}` → emit `d1..dk` and the bonus `verify_tokens[k]` → append feature
rows 0..k of the verify run to the drafter context (they were computed from the same state; causal, so rows 0..k are
exactly what a replay would produce) → if k+1 == Lv the target state is already right; else `restore`, rewind, replay
`[a, d1..dk]` through the (k+1)-session (its `next_token` must equal the bonus — asserted and logged) → anchor = bonus.
Stop on EOS (`248046`) or the token cap. Drafter context positions are absolute target positions; no eviction
(capacity 3000 keys; the MLX ContextOnly cache's sink-64/window-1024 eviction is not implemented).

## Self-test against the MLX oracle (`run-selftest-1.log`, prompt = the step-3 oracle prompt, 27 tokens)

| check | result |
|---|---|
| anchor (engine `next_token` after the tapped prefill) | 71093 == oracle |
| engine features (last chunk, 3 rows) vs MLX target taps | maxAbs 9.9e-2, rmsRel 2.2e-3 (two implementations of the same ternary target) |
| engine embed rows (`LlamaEmbed q4_0` + inverse Hadamard) vs MLX Packed embed | maxAbs 1.5e-5, rmsRel 2.1e-4 |
| drafter context K, layer 0, vs the bf16 oracle | rmsRel 3.1e-2 (= the Q4_K_M quantization delta of step 3) |
| head argmax per slot (engine lut2 head on the browser drafter's hidden) | `[12305,198,727,9637,3773,1104,25]` == oracle, 7/7 |
| selected ids | `[12305,198,727,9637,3773,1104,25]` == oracle (`python\ndef reverse_string(s:`) |

## Correctness: speculative vs plain greedy, token for token

Plain reference = the engine's own `streamTokens` path (`stopOnEos:false`, 256 new tokens) on the same prompt, run
before each speculative pass; the speculative pass compares every emitted token at emission time and, on a mismatch,
reads that row's `verify_logits` for the margin.

| prompt | block Lv | compared | first divergence | replay next_token == bonus |
|---|---|---|---|---|
| code (LRUCache, 57 prompt tokens) | 8 | 256 | none | 47/47 |
| code | 5 | 256 | none | 38/38 |
| code | 4 | 256 | none | 31/31 |
| lighthouse (21 prompt tokens) | 5 | 97 (spec stopped at EOS at token 97; plain ran on past EOS) | none | 33/33 |

No fp16 tie flip occurred on these 865 compared tokens, so no margin was reported (the machinery is in `checkDiv`).
The generated text is the LRUCache module (`lru_cache.py` docstring, doubly-linked list + dict, pytest at the end).

## Speed (steady state: sessions prebuilt, one warm-up pass of 32 tokens, then the measured 256-token pass)

Idle GPU, headless Chrome, Apple M4 Pro; plain decode measured in the same page right before each pass.

| prompt | Lv | cycles | tok/cycle | ms/cycle | ms/token | tok/s | plain ms/token (tok/s) | ratio |
|---|---|---|---|---|---|---|---|---|
| code | 8 | 59 | 4.46 | 220.6 | 49.5 | 20.2 | 36.0 (27.7) | 0.73× |
| code | 5 | 71 | 3.66 | 160.5 | **43.8** | **22.8** | 35.7 (28.0) | **0.81×** |
| code | 4 | 82 | 3.16 | 142.4 | 45.1 | 22.2 | 36.5 (27.4) | 0.81× |
| lighthouse | 5 | 38 | 2.55 | 188.8 | 74.0 | 13.5 | 39.1 (25.6) | 0.53× |

Per-cycle breakdown (means over the measured pass; "replay" averages the zero of no-replay cycles in):

| Lv | draft+head | verify | replay (mean) | replay by accepted length (median ms, n) | accept histogram (k drafts) |
|---|---|---|---|---|---|
| 8 | 38.9 | 122.2 | 59.3 | 1: 53 (8) · 2: 59 (10) · 3: 66 (7) · 4: 79 (8) · 5: 89 (4) · 6: 98 (5) · 7: 111 (5) · none (12) | 0:8 1:10 2:7 3:8 4:4 5:5 6:5 7:12 |
| 5 | 38.7 | 87.8 | 33.9 | 1: 52 (9) · 2: 58 (12) · 3: 66 (7) · 4: 78 (10) · none (33) | 0:9 1:12 2:7 3:10 4:33 |
| 4 | 39.7 | 80.5 | 22.1 | 1: 53 (13) · 2: 57 (13) · 3: 67 (5) · none (51) | 0:13 1:13 2:5 3:51 |

"draft+head" 39 ms = the drafter step (27 ms GPU, step 3) + the head micro graph, the 7 MB logits readback and the CPU
top-16 (~12 ms). The verify-graph cost is linear in rows: ≈ 40 ms fixed + 12 ms per row (replay(1) 53, verify(8) 120).
Session builds are outside the timed loop (the first run, `run-code-b8-1.log`, had them inside: 275 ms/cycle, of which
64 ms/cycle was one-time shader compilation).

Reading the numbers:
- With the tape replay removed (a state rewind that costs nothing) block 8 would be 161 ms/cycle = 36 ms/token ≈ plain,
  block 5 = 127 ms/cycle = 34.6 ms/token ≈ 1.03×. **The replay is exactly the margin between a loss and parity**, and the
  verify at 12 ms/row is what stands between parity and a real gain.
- The acceptance is healthy: 3.4 drafts per cycle at block 8 on code (MLX leg 7: ~4 tokens per cycle on the same prompt
  with the same drafter), 4/4 in 33 of 71 cycles at block 5. Chat-style text (lighthouse) accepts 1.6 per cycle and is a
  clear loss on any config today, as on MLX.
- Determinism: the block-8 pass reproduced the run-1 histogram cycle for cycle (same 59 cycles, same accepts).

## What is verified / observed / not run

- verified (logs above): 256/256 token identity vs plain greedy on the code prompt at Lv = 8, 5, 4; 97/97 on lighthouse
  up to EOS; every replay's `next_token` == bonus; the self-test's 7/7 argmax and 7/7 selected ids vs the MLX oracle.
- observed: all timings (single runs per config, idle GPU, plain decode re-measured in-page each time: 35.7–39.1 ms/token).
- inferred: rows 0..k of the verify run's features equal a replay's (causality of the prefill graph) — used for the
  context append without a check against the replay's own `dspark.features`.
- not run: the ContextOnly eviction (prompt + generation ≤ 3000 tokens here); prompts longer than 8×N with the drafter
  context beyond 313 tokens; temperature > 0 (the selector is greedy); `smallm=off/f32`; a second seed of the code prompt.

## What is next (ordered by ms per cycle)

1. **Replace the tape replay by a recurrence-only rewind** (engine work, with the verify-graph agent): the KV rows 0..k
   written by the verify run are already correct (position-indexed); only the conv window + linear-attention state are
   advanced by Lv instead of k+1. A graph that re-runs just the `Qwen35LinearAttention` recurrences for rows 0..k from
   the verify run's per-layer scratch inputs and the checkpoint would cost a few ms instead of 52–111. Alternatively a
   per-row state snapshot inside the linear-attention op (T snapshots of the recurrent state, pick k+1) makes the rewind
   a copy. Either takes block 5 from 0.81× to ≈1.0× and block 8 to ≈1.0× before any kernel work.
2. **Verify at M = 4/5** with the kernel agent's M=4 route (ratio 1.9–2.2 in the spike): verify(5) 88 → ~65 ms → block 5 at
   ~105 ms/cycle = 29 ms/token ≈ 1.2× (with item 1).
3. **Drafter at Q4_K in-shader** (1.1 GB stream instead of 3.6 GB f16): 27 → ~10 ms per step; and a GPU top-16 instead of
   the 7 MB logits readback: −8 ms. Together ≈ −25 ms/cycle → block 5 ≈ 80 ms/cycle = 22 ms/token ≈ 1.6×.
4. Adaptive block length (MLX's adaptive verify): pick Lv per cycle from the selector's scores so chat prompts run at
   Lv 2–3 and code at 5–8.
5. Product wiring: `specDecodeRunner()` behind the extractor, the 1.1 GB drafter download + IDB cache, a settings toggle;
   the runner's `prefill` should reuse the engine's normal prefill (16-token chunks) with taps added to it instead of
   8-token verify sessions (observed TTFT: 888 ms through the tapped 8-token sessions vs 1783 ms for the engine's own prompt path in
   `run-code-b8-2.log` on this 57-token prompt; a 2k-token prompt needs the wide chunks).


## Update 2026-09-21 (later): packed drafter + GPU top-16 + eviction; block 5 default

Runner options now: `block` (default **5**), `smallm` (f16 default / f32 / off), `sink` / `window` (64 / 1024; 0/0 =
no eviction), `eos=0` (run past EOS, for long identity checks), `packed=0` (f16 drafter weights), `cputopk=1` (the old
7 MB readback path), `max`, `prompt`. The kernel agent's recurrence-only rewind is not landed yet; the tape replay stays.

| run | tokens | identical to plain greedy | tok/cycle | ms/cycle | ms/token (tok/s) | plain ms/token | ratio | log |
|---|---|---|---|---|---|---|---|---|
| code, block 5, packed drafter, GPU top-16 | 256 | yes, 256/256, no divergence | 3.66 | 141.0 (median 155.8) | **38.5 (26.0)** | 34.7 (28.8) | **0.90×** | `run-code-b5-packed.log` |
| lighthouse, block 5, `eos=0`, 1536 tokens | 1536 | yes, 1536/1536 | 3.46 | 190.5 (median 196.7) | 55.1 (18.1) | 36.9 (27.1) | 0.67× | `run-lighthouse-1536.log` |

Code breakdown per cycle: draft+head **22.6** (was 38.7), verify 85.5 (median 84.7), replay 32.8 mean (50 / 56 / 64 / 76 ms
for 1 / 2 / 3 / 4 replayed tokens; none in 33 of 71 cycles), append 0.1. Accept histogram 0:9 1:12 2:7 3:10 4:33 — the
same cycle-for-cycle histogram as the f16-drafter run (the packed kernels change nothing the selector sees).

Lighthouse at 1536 tokens (the model runs past its EOS into repetitive text): 445 cycles, eviction fired 118 times, the
drafter context stayed at 1088 rows (sink 64 + window 1024) for 1558 absolute positions, and every token still equals
plain greedy — the eviction + positions path is exercised and exact for this run. Speed there is the tape-replay worst
case: 294 of 445 cycles accepted exactly 3 of 4 drafts, i.e. a 4-token replay (84 ms) after a 92 ms verify.

With the replay removed the code run would be 108 ms/cycle = 29.6 ms/token = 1.17× plain; that is now the whole gap.
