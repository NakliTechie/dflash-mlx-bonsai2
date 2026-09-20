# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os

import sys
import time

import mlx.core as mx
import mlx_lm.server as mlx_server

from dflash_mlx.runtime.bundle import load_runtime_bundle
from dflash_mlx.runtime.registry import resolve_optional_draft_ref
from dflash_mlx.server.tool_calls import install_tool_parser

# mlx_lm 0.31.x ModelProvider / ResponseGenerator read these attrs directly.
_MLX_LM_SERVER_DEFAULTS: dict[str, object | None] = {
    "adapter_path": None,
    "pipeline": False,
    "trust_remote_code": False,
    "num_draft_tokens": 3,
    "decode_concurrency": 32,
    "prompt_concurrency": 8,
    "prompt_cache_bytes": None,
}

class DFlashModelProvider(mlx_server.ModelProvider):
    def __init__(self, cli_args):
        for attr, default in _MLX_LM_SERVER_DEFAULTS.items():
            if not hasattr(cli_args, attr):
                setattr(cli_args, attr, default)
        self.target_ops = None
        self.draft_backend = None
        self.draft_meta = None
        self.target_meta = None
        self.effective_draft_quant = None
        super().__init__(cli_args)

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        default_map = getattr(self, "_model_map", None)
        if default_map is None:
            default_map = getattr(self, "default_model_map", {})
        requested_model = default_map.get(model_path, model_path)
        if self.cli_args.model is not None:
            model_ref = self.cli_args.model
        elif requested_model == "default_model":
            raise ValueError(
                "A model path has to be given as a CLI argument or in the HTTP request"
            )
        else:
            model_ref = requested_model

        if draft_model_path == "default_model":
            draft_ref = self.cli_args.draft_model
        elif draft_model_path is not None:
            draft_ref = draft_model_path
        else:
            draft_ref = None
        resolved_draft_ref = resolve_optional_draft_ref(model_ref, draft_ref)

        if (
            self.model_key == (model_ref, None, resolved_draft_ref)
            and self.target_ops is not None
            and self.draft_backend is not None
        ):
            return self.model, self.tokenizer

        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.draft_model = None
        self.target_ops = None
        self.draft_backend = None
        self.draft_meta = None
        self.target_meta = None
        self.effective_draft_quant = None

        bundle = load_runtime_bundle(
            model_ref=model_ref,
            draft_ref=draft_ref,
            draft_quant=getattr(self.cli_args, "draft_quant", None) or None,
            verify_config=self.cli_args.runtime_context.verify,
        )
        model = bundle.target_model
        tokenizer = bundle.tokenizer
        draft_model = bundle.draft_model
        draft_backend = bundle.draft_backend
        target_ops = bundle.target_ops
        target_meta = dict(getattr(bundle, "target_meta", {}) or {})
        draft_meta = dict(getattr(bundle, "draft_meta", {}) or {})
        effective_draft_quant = getattr(bundle, "effective_draft_quant", None)

        if self.cli_args.chat_template:
            tokenizer.chat_template = self.cli_args.chat_template
        if self.cli_args.use_default_chat_template and tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template
        if os.environ.get("DFLASH_TOOL_PARSER", "on").lower() != "off":
            install_tool_parser(tokenizer)
        else:
            # LAB: clients that describe tools in the system prompt and parse <tool_call> XML from
            # content themselves (LocalMind) need the raw text; the strict parser would reject a call
            # to an undeclared tool and cut the stream.
            tokenizer.tool_parser = None

        try:
            mx.eval(model.parameters())
            if draft_model is not None:
                mx.eval(draft_model.parameters())
        except Exception as _eval_err:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] weight "
                f"materialization failed: "
                f"{type(_eval_err).__name__}: {_eval_err}\n"
            )
            sys.stderr.flush()
            raise RuntimeError("DFlash weight materialization failed") from _eval_err

        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.draft_backend = draft_backend
        self.target_meta = target_meta
        self.draft_meta = draft_meta
        self.effective_draft_quant = effective_draft_quant
        self.target_ops = target_ops
        self.model_key = (model_ref, None, bundle.resolved_draft_ref)

        return self.model, self.tokenizer

def wait_for_initial_model_load(
    model_provider: DFlashModelProvider,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.2,
) -> None:
    start = time.perf_counter()
    announced = False
    while not _runtime_components_ready(model_provider):
        if not announced:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] loading model "
                f"on generation worker thread...\n"
            )
            sys.stderr.flush()
            announced = True
        if time.perf_counter() - start > timeout_s:
            raise RuntimeError(
                f"DFlash generation worker failed to publish a complete runtime bundle within "
                f"{timeout_s}s; check earlier log lines for the underlying error."
            )
        time.sleep(poll_interval_s)


def _runtime_components_ready(model_provider: DFlashModelProvider) -> bool:
    return (
        model_provider.model_key is not None
        and model_provider.target_ops is not None
        and model_provider.draft_backend is not None
    )
