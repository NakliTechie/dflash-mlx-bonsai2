"""CopySpec x DFlash2 correctness gate + timing. One model load; greedy; thinking off.

Modes: plain (mlx_lm generate_step on the same target), off, auto, conservative,
forced (harness-only: copyspec_mode='on' injected past config validation, so the
engine never one-strike-disables copy drafts).
"""
import json, os, sys, time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step

import dflash_mlx
from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
from dflash_mlx.runtime.bundle import load_runtime_bundle
from dflash_mlx.runtime.context import build_offline_runtime_context
from dflash_mlx.engine.events import SummaryEvent, TokenEvent

HERE = Path(__file__).parent
PACK = os.path.expanduser("~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit")
DRAFT = os.path.expanduser("~/Code/models/Qwen3.8-27B-DFlash2-r3")
MAX_NEW = 512
REPS = int(os.environ.get("REPS", "2"))
MODES = os.environ.get("MODES", "plain,off,auto,conservative,forced").split(",")
PROMPTS = os.environ.get("PROMPTS", "code,article,math").split(",")
OUT = Path(os.environ.get("OUT", HERE / "results.jsonl"))

mx.set_cache_limit(1 << 30)
print("dflash_mlx:", dflash_mlx.__file__, "verify:", os.environ.get("DFLASH_PRISM_VERIFY"), flush=True)
base_ctx = build_offline_runtime_context(prefill_step_size=512, copyspec_mode="off")
bundle = load_runtime_bundle(model_ref=PACK, draft_ref=DRAFT, verify_config=base_ctx.verify)
tok = bundle.tokenizer
stop_ids = list(get_stop_token_ids(tok))
print("draft caps:", bundle.draft_model.capabilities, "stop:", stop_ids, flush=True)


def ids_for(name):
    text = (HERE / f"{name}_prompt.txt").read_text()
    return list(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False))


def run_plain(ids):
    out, t_first = [], None
    t0 = time.perf_counter()
    for token, _ in generate_step(mx.array(ids), bundle.target_model, max_tokens=MAX_NEW, prefill_step_size=512):
        token = int(token)
        if t_first is None:
            t_first = time.perf_counter()
        out.append(token)
        if token in stop_ids:
            break
    t_end = time.perf_counter()
    return out, dict(ttft_s=t_first - t0, decode_tps=(len(out) - 1) / (t_end - t_first))


def run_dflash(ids, mode):
    ctx = build_offline_runtime_context(prefill_step_size=512,
                                        copyspec_mode=("conservative" if mode == "forced" else mode))
    if mode == "forced":
        ctx = replace(ctx, runtime=replace(ctx.runtime, copyspec_mode="on"))
    out, t_first, summary = [], None, None
    t0 = time.perf_counter()
    for ev in stream_dflash_generate(target_model=bundle.target_model, target_ops=bundle.target_ops, tokenizer=tok,
                                     draft_model=bundle.draft_model, draft_backend=bundle.draft_backend, prompt="",
                                     max_new_tokens=MAX_NEW, stop_token_ids=stop_ids, prompt_tokens_override=ids,
                                     runtime_context=ctx):
        if isinstance(ev, TokenEvent):
            if t_first is None:
                t_first = time.perf_counter()
        elif isinstance(ev, SummaryEvent):
            summary = ev
    t_end = time.perf_counter()
    out = list(summary.generated_token_ids)
    return out, dict(ttft_s=t_first - t0, decode_tps=(len(out) - 1) / (t_end - t_first),
                     cycles=summary.cycles_completed, tokens_per_cycle=summary.tokens_per_cycle,
                     acceptance_ratio=summary.acceptance_ratio, copyspec_hits=summary.copyspec_hits,
                     copyspec_tokens=summary.copyspec_tokens, block_tokens=summary.block_tokens,
                     adaptive_reductions=summary.adaptive_block_reductions, fallback_ar=summary.fallback_ar)


# warmup (kernels, drafter) on a short prompt
wu = list(tok.apply_chat_template([{"role": "user", "content": "Say hello."}], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=False))
run_dflash(wu, "off"); run_plain(wu)

with OUT.open("a") as fh:
    for rep in range(REPS):
        for name in PROMPTS:
            ids = ids_for(name)
            for mode in MODES:
                mx.clear_cache()
                out, stats = run_plain(ids) if mode == "plain" else run_dflash(ids, mode)
                rec = dict(prompt=name, mode=mode, rep=rep, prompt_tokens=len(ids), gen_tokens=len(out),
                           token_ids=out, **stats)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(json.dumps({k: v for k, v in rec.items() if k != "token_ids"}), flush=True)
print("DONE", flush=True)
