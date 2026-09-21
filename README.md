<p align="center">
  <h1 align="center">dflash-mlx-bonsai2</h1>
  <p align="center">DFlash 2 speculative decoding for PrismML's Ternary-Bonsai-2-27B on Apple Silicon (MLX)</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
  <img src="https://img.shields.io/badge/status-experimental-orange" alt="Experimental">
</p>

A fork of [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) that runs
[DFlash 2](https://github.com/z-lab/dflash) block-diffusion speculative decoding against
[prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit),
PrismML's 2-bit, Hadamard-rotated quantization of Qwen3.8-27B. It ships an OpenAI-compatible
server (`dflash serve`) that any local chat client can talk to.

**Status: experimental.** One machine (Apple M4 Pro, 24 GB), one target model, one drafter.
The numbers below are from that machine; the limitations section is as load-bearing as the
numbers.

## What this fork adds

Upstream dflash-mlx already runs Qwen3.8-27B + z-lab's DFlash2 drafter on stock 4-bit MLX
models. The ternary Bonsai 2 pack is not a stock MLX model: every linear is a `Packed`
module (blockwise Hadamard rotation on the activations, then a 2-bit group-128 affine
`quantized_matmul`), and MLX's 2-bit kernel is tuned for single-row decode, not for the
multi-row verify step that speculative decoding lives on. This fork closes both gaps:

- **`dflash_mlx/runtime/prism_pack.py`** — a text-only loader for the PrismML MLX pack
  (schema 2). It builds a stock `mlx_lm` Qwen3.5 `TextModel` and installs the pack's
  `Packed` (Hadamard + 2-bit affine qmm) modules into it, skipping the vision tower, so all
  of upstream's Qwen GDN target hooks (hidden capture, tape-replay rollback, prefix cache)
  work unchanged. `runtime/loading.py` dispatches to it on `model_type ==
  prism_hadamard_qwen35`. Parity vs the pack's own loader: max |Δ logits| 3.3e-5, argmax
  identical.
- **`dflash_mlx/runtime/prism_qmm.py`** — a Metal small-M (8-row) 2-bit GEMM for the verify
  path. `v7` is a `simdgroup_matrix` 8×8 MMA kernel with register-only dequantization and
  a fused prep kernel (sign × Hadamard × transpose × per-group row sums), installed as a
  class-level `Packed.__call__` override. Real DFlash2 verify calls carry 2–8 rows (the
  selector emits variable-length paths), so 2–8-row calls are zero-padded to 8 and sliced;
  1-row decode calls get an fp16 activation cast only. Exact vs the stock kernel (argmax
  identical on 3–8-row inputs; 0 argmax flips vs the fp32 path on 1000 real tokens).
  Selected with `DFLASH_PRISM_VERIFY=v7|v4b|fp16|off`.
- **`DFLASH_TOOL_PARSER=off`** in `dflash serve` — disables the strict tool-call parser
  (and mlx-lm's in-stream `tool_call_start` detection) for clients that describe tools in
  the system prompt and parse `<tool_call>` text themselves. Without it the server rejects a
  call to an undeclared tool and cuts the stream.
- **A re-fitted drafter** —
  [naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2](https://huggingface.co/naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2):
  z-lab's Qwen3.8-27B-DFlash2 fine-tuned on 1.5 M tokens of the ternary model's own greedy
  generations, so the drafter predicts what the 2-bit target will actually say rather than
  what the bf16 base would have said. Training pipeline in `lab/aws/` (llama.cpp tap-dump
  capture of hidden states + targets, CUDA trainer). It beats the shipped z-lab drafter on
  every prompt measured (4–16 % fewer verify cycles).

Everything else — the server, CLI, prefix cache, adaptive verify, diagnostics — is
upstream's, documented in [`docs/`](docs/).

## Requirements

- Apple Silicon Mac, macOS. **24 GB unified memory recommended**: the pack is 8.6 GB on
  disk, and pack + drafter + a 2 K-token prompt peaks at ~12.5–14.8 GB of Metal memory
  (with `--prefill-step-size 512`; the default 2048 step peaks at ~17.7 GB on a 2 K-token
  prefill). Nothing else GPU-resident should be running at the same time.
- Python 3.11+ (tested on 3.12.9).
- `mlx >= 0.32.1`, `mlx-lm >= 0.31.3` (tested with mlx 0.32.2 / mlx-lm 0.31.3; the pack's
  own `runtime/requirements.txt` pins mlx 0.32.0 / mlx-lm 0.31.3).
- ~13 GB of disk for the two model downloads (pack 8.6 GB + drafter 3.85 GB bf16).
- `hf` (the `huggingface_hub` CLI) for the downloads; the setup script installs it into the
  venv if missing.

## Install

One command does the venv, the editable install and both downloads (idempotent; re-running
skips whatever is already there):

```bash
git clone https://github.com/NakliTechie/dflash-mlx-bonsai2
cd dflash-mlx-bonsai2
bash scripts/setup-bonsai2.sh
```

It ends by printing the exact `dflash serve` command for your paths. `--dry-run` shows what
it would do without touching anything; `HF_HOME` is respected for the download location.

Manual equivalent:

```bash
python3 -m venv .venv && source .venv/bin/activate      # or: uv venv && source .venv/bin/activate
pip install -e .                                        # this repo, editable
pip install "huggingface_hub>=1.0"                      # the `hf` CLI
hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit      # 8.6 GB, prints the snapshot path
hf download naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2 --exclude "*.gguf"   # 3.85 GB
```

## Run

```bash
bash scripts/serve-bonsai2.sh          # extra args are passed through to `dflash serve`
```

which runs (with the paths resolved from the HF cache):

```bash
DFLASH_TOOL_PARSER=off DFLASH_PRISM_VERIFY=v7 \
dflash serve \
  --model  "$HF_HOME/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/<rev>" \
  --draft  "$HF_HOME/hub/models--NakliTechie--Qwen3.8-27B-DFlash2-ternary-bonsai2/snapshots/<rev>" \
  --port 8790 \
  --prefill-step-size 512
```

`--model` must be a **local directory** for the pack (the prism loader dispatches on the
pack's `config.json`; a bare repo id falls through to `mlx_lm.load`, which cannot build it).
The drafter is W4-quantized at load by the registry default (`--draft-quant w4`); pass
`--draft-quant none` to keep it bf16 (measured: no acceptance difference).

Then point any OpenAI-compatible client at `http://127.0.0.1:8790/v1`:

```bash
curl http://127.0.0.1:8790/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Ternary-Bonsai-2-27B-mlx-2bit",
       "messages": [{"role": "user", "content": "Write a Python LRU cache class."}],
       "max_tokens": 512, "temperature": 0, "stream": true}'
```

**Use `temperature: 0`.** Speculation only engages on greedy requests; a sampled request
runs target-only autoregressive decode (the server logs `exact target AR`).

One-shot generation and the benchmark work the same way:

```bash
DFLASH_PRISM_VERIFY=v7 dflash generate --model <pack-dir> --draft <drafter-dir> --prompt "..."

DFLASH_PRISM_VERIFY=v7 dflash benchmark --model <pack-dir> --draft <drafter-dir> \
  --only-dflash --block-tokens 8 --verify-mode dflash --max-tokens 512 --no-eos --prompt "..."
```

`--only-dflash` is required: the benchmark's baseline leg loads the target through stock
`mlx_lm.load`, which cannot build the pack. Compare against the plain-decode number below.

## Measured (Apple M4 Pro, 24 GB)

Round-3 drafter, `DFLASH_PRISM_VERIFY=v7`, fixed 8-token blocks, thinking off, quiet
machine, each prompt run twice (`lab/leg8/padded/`, 2026-09-21). Plain decode through the
same pack is **~21.5 tok/s** (46 ms/token with the fp16 activation cast; 18.1 tok/s on the
pack's own fp32 path).

| prompt | tokens per 8-token cycle | tok/s | vs plain decode |
|---|---|---|---|
| code, chat template (1024 tokens) | 4.08 | 26.9 / 29.0 | ~1.3× |
| math, raw prompt (512) | 3.61 / 3.63 | 25.4 / 26.0 | ~1.2× |
| code, raw completion (512) | 4.57 | 33.0 / 33.0 | ~1.5× |
| chat / email (512) | ~2.4 | ~17 | slight loss |

Per-cycle cost is 138–152 ms cold (isolated verify(8) 105–115 ms + ~25–35 ms
draft/commit), so throughput is `tokens per cycle / cycle time`; a prompt the drafter
predicts well (code) wins, a prompt it does not (chat, and above all thinking traces) does
not. Cycle time grows with context: ~149 ms at short context → 171–182 ms at ~2 K tokens,
so long answers run 25–30 % slower per token than short ones. Prefill is ~80–90 tok/s
(TTFT on a 2.2 K-token input is 22–24 s).

Caveats on these numbers:

- One machine, one session each. Session-to-session cycle-time variance (thermal, memory
  pressure) is of the same size as the drafter gain; the per-cycle savings are the robust
  signal, the tok/s are illustrative.
- Any memory pressure (swap, another GPU-resident process, Chrome with WebGPU content)
  multiplies the cycle time by 1.5–2×. Measure on a quiet machine or not at all.
- The table is `--verify-mode dflash` (fixed 8-token blocks). `dflash serve` defaults to
  `--verify-mode adaptive`, which probes shorter blocks on low acceptance and brought the
  chat prompt back to plain-decode parity (19.0 vs 17.4 tok/s fixed, in an earlier session
  with the shipped drafter).

## Losslessness

Every emitted token is the target's argmax at verification time. The output equals plain
greedy decoding **up to fp16 ties**: the 8-row verify kernel and the 1-row decode kernel
accumulate in different orders, so at positions where the top-2 logit margin is ~0.00–0.02
they can disagree. Measured: `v7` produced 0 argmax flips against the fp32 reference on
1000 real tokens teacher-forced in 5-row blocks (the fp16-cast stock path produced 1); an
earlier 8-row kernel showed 5 flips in 1024 positions, all at margins ≤ 0.016. Over a
~2 K-token generation that is enough for two runs to diverge at some point and both be
coherent. Upstream documents the same "MLX dispatch divergence".

## Environment flags

| Variable | Values | Meaning |
|---|---|---|
| `DFLASH_PRISM_VERIFY` | `v7` (default), `v4b`, `fp16`, `off` | verify-path matmul for `Packed` modules. `v7` = padded 8-row MMA kernel + fused prep; `v4b` = the earlier threadgroup-tile kernel (8-row calls only); `fp16` = stock kernel with an fp16 activation cast; `off` = the pack's own fp32 path (slow; parity reference). |
| `DFLASH_PRISM_VERIFY_STATS` | set | print a rows-per-call histogram at exit (diagnostic). |
| `DFLASH_TOOL_PARSER` | `on` (default), `off` | `off` disables the server's strict tool-call parser and mlx-lm's in-stream tool-call detection; tool-call text streams through as content for clients that parse it themselves. |

All upstream flags (`--verify-mode`, `--block-tokens`, `--draft-quant`, prefix cache,
diagnostics, `--prefill-step-size`, ...) apply; see [`docs/runtime-flags.md`](docs/runtime-flags.md).

## Known limitations

- **Chat is roughly break-even.** At ~2.4 accepted tokens per cycle, chat/email prompts
  land at ~17 tok/s under fixed blocks against ~21.5 plain; adaptive verify (the serve
  default) recovers parity, no more. Thinking traces are the worst case (highest-entropy
  stream; acceptance decays over long traces). The wins are code (~1.3×), math (~1.2×) and
  raw code completion (~1.5×).
- **Greedy loops on long generations.** The target's true greedy decode of open-ended
  prompts can fall into a repetition loop past ~800 tokens ("I am sorry for the life we
  didn't live…"). That is the model under greedy decoding, not the speculation, but
  speculation requires greedy. Give the client a repetition penalty or accept sampled
  (non-speculative) decoding for long creative output.
- **fp16 tie flips** (above): not bit-identical to single-token decode.
- **Memory.** 24 GB is the practical minimum; on that box nothing else GPU-resident can run
  alongside the server, and Chrome loses its WebGPU adapter while the server holds ~15 GB.
- **`/v1/models` lists every model in the HF cache**, not only the served one (upstream
  behaviour); pick the pack entry in your client.
- **`dflash benchmark` needs `--only-dflash`** (its baseline leg cannot load the pack).
- The prism loader is text-only: the pack's vision tower is skipped.
- Pre-M5 GPUs use upstream's steel fallback kernels for the W4 drafter (`dflash doctor`
  reports NAX unavailable); measured on M4 Pro only.

## Layout

```
dflash_mlx/runtime/prism_pack.py   pack loader (Packed modules into a stock TextModel)
dflash_mlx/runtime/prism_qmm.py    small-M 2-bit Metal GEMM + fused prep kernel, verify-path install
dflash_mlx/server/model_provider.py DFLASH_TOOL_PARSER seam
dflash_mlx/runtime/registry.py     ("Ternary-Bonsai-2-27B",) registry row
scripts/setup-bonsai2.sh           one-command setup; scripts/serve-bonsai2.sh runs the server
lab/                               the research trail: kernel iterations v1..v8, parity and tie-flip
                                   audits, per-leg benchmark logs, lab/aws/ drafter training pipeline
```

`lab/` is kept as evidence, not as product code; nothing in `dflash_mlx/` imports it.

## Attribution

- [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) — the runtime this fork
  builds on (Apache-2.0). The server, CLI, target hooks, rollback and cache are theirs.
- [z-lab/dflash](https://github.com/z-lab/dflash) and
  [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) — the
  DFlash 2 method and the drafter this fork's drafter was fine-tuned from (Apache-2.0).
  Paper: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036).
- [PrismML](https://huggingface.co/prism-ml) — Ternary-Bonsai-2-27B and its MLX pack runtime
  (Apache-2.0). "Created using Bonsai by Prism ML."
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — the base model (Apache-2.0).

See [`NOTICE`](NOTICE).

## Citation

```bibtex
@misc{chen2026dflash,
  title={DFlash: Block Diffusion for Flash Speculative Decoding},
  author={Jian Chen and Yesheng Liang and Zhijian Liu},
  year={2026},
  eprint={2602.06036},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2602.06036}
}
```

## License

Apache-2.0, unchanged from upstream. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
