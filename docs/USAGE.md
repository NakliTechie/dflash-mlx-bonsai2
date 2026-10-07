# Using the Bonsai 2 drafter: Mac, NVIDIA, browser

One drafter, [naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2](https://huggingface.co/naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2),
speeds up one target, PrismML's [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
Pick the row that matches your hardware; each section below is complete on its own.

| You have | Path | Download | Measured speedup |
|---|---|---|---|
| Apple Silicon Mac, 24 GB | [1. Mac: `dflash serve`](#1-mac-dflash-serve) | 8.6 GB pack + 3.85 GB drafter | 1.2–1.5× on code and math (M4 Pro) |
| NVIDIA GPU, 24 GB | [2. NVIDIA: llama.cpp](#2-nvidia-llamacpp) | 7.2 GB GGUF + 1.1 GB drafter GGUF | 2.2× maths and code, 3.2× code edits with prompt lookup (L4) |
| No GPU, a Google Cloud account | [3. Cloud Run L4](#3-nvidia-in-the-cloud-cloud-run-l4) | none locally | 2.1× math and code, 3.2× code edits (L4) |
| Chrome with WebGPU | [4. Browser: LocalMind](#4-browser-localmind) | 5.9 GB model + 1.1 GB drafter, once | 1.18× on code (M4 Pro) |

All of these speed up **code, math and structured text**. On chat and prose the drafter guesses less, so expect
break-even. Speculation never changes what the model says at temperature 0, apart from fp16 ties.

---

## 1. Mac: `dflash serve`

**Needs:** Apple Silicon, macOS, 24 GB unified memory, Python 3.11+, ~13 GB free disk. No Hugging Face login.

```bash
git clone https://github.com/NakliTechie/dflash-mlx-bonsai2
cd dflash-mlx-bonsai2
bash scripts/setup-bonsai2.sh          # venv + install + both downloads; --dry-run prints the plan
bash scripts/serve-bonsai2.sh          # OpenAI-compatible server on http://127.0.0.1:8790/v1
```

Call it from any OpenAI-compatible client. **Send `temperature: 0`**: the server only speculates on greedy requests
and quietly decodes plainly otherwise.

```bash
curl http://127.0.0.1:8790/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Ternary-Bonsai-2-27B-mlx-2bit","messages":[{"role":"user","content":"Write a Python LRU cache."}],"temperature":0,"max_tokens":512,"stream":true}'
```

**With a chat UI.** Open [LocalMind](https://localmind.naklitechie.com) in Chrome → Settings → Models → endpoint
presets → *Ternary Bonsai 2 — fast (Mac, dflash serve)*. The preset fills `http://127.0.0.1:8790/v1` and sets
temperature 0. Chrome asks once for *Local network access*; allow it. Any other OpenAI-compatible client works too,
with base URL `http://127.0.0.1:8790/v1`, any API key, and temperature 0.

**If it is tight on memory:** close other GPU-heavy apps. Peak Metal memory is ~12.5–14.8 GB on a 2K-token prompt
with the script's `--prefill-step-size 512`. Port, verify kernel and tool parsing are environment variables; see
`scripts/bonsai2-common.sh` and [runtime-flags.md](runtime-flags.md).

---

## 2. NVIDIA: llama.cpp

DFlash 2 support is in PrismML's llama.cpp fork (`prism` branch, merged in
[PrismML-Eng/llama.cpp#261](https://github.com/PrismML-Eng/llama.cpp/pull/261)). Stock llama.cpp cannot load
the ternary target yet. Tested on an NVIDIA L4 (24 GB, CUDA 12.8); other cards are not measured.

**Build** (CUDA toolkit, CMake and a C++ compiler):

```bash
git clone -b prism https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

**Download** the target and the drafter's GGUF:

```bash
pip install -U huggingface_hub
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir models
hf download naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2 Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf --local-dir models
```

**Serve:**

```bash
./build/bin/llama-server \
  -m  models/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 999 -fa on -c 16384 --jinja \
  -md models/Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf -ngld 999 \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --spec-type ngram-mod \
  --port 8080
```

Point any OpenAI-compatible client at `http://localhost:8080/v1`, with temperature 0 for the measured numbers.

The second `--spec-type ngram-mod` turns on llama.cpp's prompt lookup. It runs before the drafter and reuses text
already in your prompt, so it helps most when the answer repeats your input. On 80 HumanEval refactors on one L4,
the drafter alone gave 2.46×, prompt lookup alone 1.57×, and both together **3.15×**. On maths and code it costs
almost nothing (2.14× vs 2.17× on GSM8K). It is the default in the Cloud Run deploy below. Drop the line if you
want the drafter alone. For
chat-heavy traffic, `--spec-draft-n-max 3` wastes less work. To answer faster, add
`"chat_template_kwargs": {"enable_thinking": false}` to a request; it turns off thinking.

On the L4, plain decoding of this target runs ~30 tok/s. With the drafter it reached 65.6 tok/s on a code prompt,
and stayed ~1× on prose. Greedy output with speculation is not byte-identical to plain decoding. It splits only
where the plain run's top two candidates were within 0.03 nats.

---

## 3. NVIDIA in the cloud: Cloud Run L4

No GPU of your own: [NakliTechie/bonsai2-run](https://github.com/NakliTechie/bonsai2-run) deploys the same
llama.cpp build, target and drafter to one Google Cloud Run L4 that scales to zero. In
[Cloud Shell](https://shell.cloud.google.com), on a project with paid billing:

```bash
curl -fsSL https://raw.githubusercontent.com/NakliTechie/bonsai2-run/main/cloudrun/bonsai2-cloudrun.sh | bash
```

Setup takes about ten minutes. You get a private OpenAI- and Anthropic-compatible URL. It costs ~$1.42 per GPU hour
while answering, $0 for the GPU when idle, and ~17¢ a month for the model files. `… | DOWN=1 bash` removes everything.
Costs, options and benchmarks are in that repo's README.

---

## 4. Browser: LocalMind

Nothing to install. [LocalMind](https://localmind.naklitechie.com) runs Ternary Bonsai 2 27B inside a Chrome tab
on WebGPU, with this drafter ported to WGSL (`lab/webgpu/`). Safari does not work yet.

1. Open [localmind.naklitechie.com](https://localmind.naklitechie.com) in Chrome (or another browser with WebGPU).
2. Open the model picker and choose **Ternary Bonsai 2 27B**. It downloads ~5.9 GB once and caches it in the browser.
3. Settings → Models → **Speculative decoding for Ternary Bonsai 2 27B** is on by default. After the model loads,
   LocalMind fetches the 1.1 GB drafter once. It keeps the drafter packed on the GPU, which uses ~1 GB more GPU memory.
4. Ask a coding question. The status line shows `· speculative` once the drafter is attached. Turns that start
   before it lands decode plainly.

Measured in the app on an M4 Pro: 34.4 vs 40.7 ms/token on a 1,486-token code answer (1.18×), output identical to
plain decoding. Prose shows no gain. It needs a GPU with room for the 27B model plus ~1 GB; other GPUs are not
measured. Everything runs in the tab, so nothing you type leaves the machine.

---

## Which one?

- **Fastest you can run at home:** NVIDIA + llama.cpp.
- **On a Mac:** `dflash serve`, optionally behind LocalMind's preset for a chat UI.
- **Try it in five minutes with no install:** LocalMind in Chrome.
- **An API for a demo or agent, without paying for an idle GPU:** Cloud Run.

Numbers, method and caveats: [BONSAI2.md](BONSAI2.md) (Mac), the
[bonsai2-run results](https://github.com/NakliTechie/bonsai2-run/tree/main/results) (L4) and
`lab/webgpu/runner/RESULTS.md` (browser).
