<h1 align="center">dflash-mlx-bonsai2</h1>

<p align="center"><b>DFlash 2 speculative decoding for PrismML's Ternary-Bonsai-2-27B on Apple Silicon: the 2-bit 27B model, 1.2–1.5× faster on code and math, same greedy output.</b></p>

<p align="center">Apple Silicon, 24 GB. Runs local. No account, no server, no telemetry.</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache--2.0-4c6ef5?style=flat-square" alt="License: Apache-2.0">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-4c6ef5?style=flat-square" alt="Platform: Apple Silicon">
  <img src="https://img.shields.io/badge/output-greedy--identical-4c6ef5?style=flat-square" alt="Output: greedy-identical up to fp16 ties">
  <img src="https://img.shields.io/badge/status-experimental-4c6ef5?style=flat-square" alt="Status: experimental, one machine measured">
</p>

## Install

| Platform | Command |
|---|---|
| Apple Silicon Mac (macOS, Python 3.11+) | `git clone https://github.com/NakliTechie/dflash-mlx-bonsai2 && cd dflash-mlx-bonsai2 && bash scripts/setup-bonsai2.sh` |

The script makes a venv, installs this repo, and downloads the model pack (8.6 GB) and the drafter (3.85 GB) into the Hugging Face cache; it is idempotent and `--dry-run` shows the plan. Then start the server and point any OpenAI-compatible client at it:

```bash
bash scripts/serve-bonsai2.sh                      # dflash serve on http://127.0.0.1:8790/v1
curl http://127.0.0.1:8790/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Ternary-Bonsai-2-27B-mlx-2bit","messages":[{"role":"user","content":"Write a Python LRU cache."}],"temperature":0,"max_tokens":512,"stream":true}'
```

Use `temperature: 0`: speculation only engages on greedy requests. No config file, no account. Try it without installing anything by reading the measured tables in [docs/BONSAI2.md](docs/BONSAI2.md).

## Why

You have a 27B model that fits in 6 GB, and it answers at 21 tokens a second because every token re-reads all 6 GB. Speculative decoding fixes that in principle: a small drafter guesses a block of tokens and the big model checks them in one pass. In practice, on the 2-bit Bonsai 2 target, two things were broken, and this fork fixes both.

The drafter ([z-lab's DFlash 2](https://github.com/z-lab/dflash)) was trained on bf16 Qwen3.8 and guesses what the bf16 model would say, not what the ternary one does. And MLX's 2-bit matmul is tuned for one row, so checking 8 tokens cost as much as decoding 8 tokens one by one. This fork of [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) adds a loader for the Hadamard-rotated pack, a Metal 8-row 2-bit GEMM for the check, and a drafter re-fitted on the ternary model's own output.

**Use something else if** you want the base Qwen3.8-27B at 4 bits, where upstream [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) already works and has more memory headroom; you want a CUDA box, where [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) with a DFlash 2 drafter is the right path (our port of it is in `lab/leg9/patches`, not fast on Metal); or your traffic is chat and thinking, where this fork is break-even and plain [mlx-lm](https://github.com/ml-explore/mlx-lm) is simpler.

## What it does to the numbers

Apple M4 Pro, 24 GB, quiet machine, thinking off, greedy. Plain decode through the same pack: ~21.5 tok/s.

| prompt | tokens per 8-token cycle | tok/s | vs plain |
|---|---|---|---|
| code (chat template, 1024 tokens) | 4.08 | 27–29 | ~1.3× |
| math (raw, 512) | 3.6 | 25–26 | ~1.2× |
| code completion (raw, 512) | 4.57 | 33 | ~1.5× |
| chat / email (512) | ~2.4 | ~17 | slight loss |

Output equals plain greedy decoding up to fp16 ties (the 8-row and 1-row kernels accumulate in different orders; on 1000 real tokens the `v7` path made 0 argmax flips against the fp32 reference). The gains and the caveats behind each row are in [docs/BONSAI2.md](docs/BONSAI2.md).

## What this fork adds

The **pack loader** (`dflash_mlx/runtime/prism_pack.py`) builds a stock `mlx_lm` Qwen3.5 text model and installs the pack's Hadamard + 2-bit modules into it, so upstream's hidden capture, tape-replay rollback and prefix cache work unchanged (parity with the pack's own loader: max logit delta 3e-5). The **verify kernel** (`dflash_mlx/runtime/prism_qmm.py`, `DFLASH_PRISM_VERIFY=v7`) is a simdgroup 8×8 MMA GEMM that dequantizes 2-bit weights straight into matrix-tile registers, with a fused sign, Hadamard, transpose and row-sum prep; 2–8-row verify calls are padded to 8 because the real DFlash 2 verify is 4–5 rows wide. It cut the 8-token check from 438 ms to 105–115 ms against a 46 ms decode step.

The **drafter** ([naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2](https://huggingface.co/naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2)) is z-lab's DFlash 2 fine-tuned on 1.5 M tokens of the ternary model's own greedy generations, captured with a llama.cpp hidden-state dumper and trained on a rented GPU (`lab/aws/`). It accepts more on every prompt measured: chat 2.46 to 2.68, code 3.07 to 3.28, raw code 4.41 to 4.57 tokens per cycle against the shipped drafter. `DFLASH_TOOL_PARSER=off` lets clients that parse `<tool_call>` text themselves stream it through the server.

## Commands

```bash
bash scripts/setup-bonsai2.sh [--dry-run]                       # venv + install + both model downloads
bash scripts/serve-bonsai2.sh                                   # OpenAI-compatible server on :8790
DFLASH_PRISM_VERIFY=v7 dflash generate  --model <pack-dir> --draft <drafter-dir> --prompt "..."
DFLASH_PRISM_VERIFY=v7 dflash benchmark --model <pack-dir> --draft <drafter-dir> --only-dflash --block-tokens 8 --verify-mode dflash --max-tokens 512 --no-eos --prompt "..."
DFLASH_PRISM_VERIFY=v4b|fp16|off ...                            # older verify paths, for A/B
DFLASH_PRISM_VERIFY_STATS=1 ...                                 # rows-per-call histogram at exit
DFLASH_TOOL_PARSER=off dflash serve ...                         # stream tool-call text unparsed
```

`--model` must be the pack's local directory (a bare repo id falls through to `mlx_lm.load`, which cannot build it); `dflash benchmark` needs `--only-dflash` for the same reason.

## Verify it yourself

```bash
python lab/padded_parity.py      # 3–8-row verify path vs the stock fp32 path: argmax identical, fp16-level deltas
python lab/tie_flip_audit.py     # 1000 real tokens: argmax flips per verify path vs the fp32 reference
python lab/verify_timing_modes.py v4b v7   # isolated verify(8) vs decode(1), per kernel
```

The parity and tie-flip audits are what "greedy-identical up to fp16 ties" points at; the timing script is what the 105–115 ms figure points at. Verified end to end on one M4 Pro: `dflash serve` with this drafter driving a tool-calling chat client at 70–73 % acceptance. Not verified: other Apple chips, other memory sizes.

## License

Apache-2.0, as upstream; see NOTICE for z-lab, PrismML and Qwen attribution. Founding document and every measurement: [docs/BONSAI2.md](docs/BONSAI2.md) · runtime flags: [docs/runtime-flags.md](docs/runtime-flags.md) · research trail: `lab/`.
