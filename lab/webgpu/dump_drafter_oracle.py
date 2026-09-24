"""Oracle dumps for the browser drafter: for a fixed prompt, run the MLX DFlash2 path on the ternary pack and save
(a) the drafter's inputs for one cycle — context features [C, 5*H] fp32 (the tapped residuals at layers 5/19/33/47/61
of the last C prompt tokens), the anchor token id, position ids — and (b) the drafter's outputs — hidden [8, H] after
the 5 layers + norm, per-slot top-16 candidates from the head scan, the selected 7 draft ids; plus per-layer
intermediates (after each decoder layer) so a WGSL port can be checked stage by stage. Uses the round-3 drafter in
bf16 (no W4) so the reference is exact. Output: lab/webgpu/oracle/*.npy + meta.json."""
import sys, os, json, numpy as np
sys.path.insert(0, 'lab')
import mlx.core as mx
from prism_pack_loader import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.runtime.prism_pack import load_pack_tokenizer
from dflash_mlx.runtime.loading import load_draft_bundle
PACK = os.path.expanduser('~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'); DRAFT = os.path.expanduser('~/Code/models/Qwen3.8-27B-DFlash2-r3')
OUT = 'lab/webgpu/oracle'; os.makedirs(OUT, exist_ok=True)
model, _ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
tok = load_pack_tokenizer(PACK); enc = lambda s: tok.encode(s, add_special_tokens=False) if hasattr(tok, 'encode') else tok(s)['input_ids']
prompt = '<|im_start|>user\nWrite a Python function that reverses a string, with a docstring.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
ids = list(enc(prompt)); CAP = {6, 20, 34, 48, 62}
cache = ops.make_cache(model, enable_speculative_linear_cache=True)
logits, captured = ops.forward_with_hidden_capture(model, input_ids=mx.array([ids]), cache=cache, capture_layer_ids=CAP); mx.eval(logits)
anchor = int(logits[0, -1].argmax()); print('prompt tokens', len(ids), 'anchor', anchor, repr(tok.decode([anchor]) if hasattr(tok, 'decode') else ''))
feats = mx.concatenate([captured[k][0] for k in sorted(captured)], axis=-1).astype(mx.float32)   # [T, 5H]
mx.eval(feats); np.save(f'{OUT}/context_features.npy', np.array(feats)); print('features', feats.shape)
bundle = load_draft_bundle(DRAFT, draft_quant=None) if 'draft_quant' in load_draft_bundle.__code__.co_varnames else load_draft_bundle(DRAFT)
draft = bundle.model if hasattr(bundle, 'model') else bundle[0]
print('draft type', type(draft).__name__, 'block', getattr(draft, 'block_size', None), 'mask', getattr(draft, 'mask_token_id', None))
meta = {'prompt': prompt, 'prompt_ids': ids, 'anchor': anchor, 'T': len(ids), 'H': 5120, 'taps': [5, 19, 33, 47, 61], 'draft_class': type(draft).__name__, 'draft_attrs': [a for a in dir(draft) if not a.startswith('_')][:80]}
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1); print('wrote meta; draft public attrs:', meta['draft_attrs'][:40])
