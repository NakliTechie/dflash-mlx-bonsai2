"""Capture residual-stream hidden states at the DFlash2 tap layers (5,19,33,47,61 -> adapter keys +1) for a fixed
token sequence, from either the ternary pack or the 4-bit reference. Usage: capture_hidden.py pack|ref out.npz"""
import sys, json, numpy as np
sys.path.insert(0, 'lab')
import mlx.core as mx
which, out = sys.argv[1], sys.argv[2]
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
REF='/Users/chiragpatnaik/Code/models/Qwen3.8-27B-4bit-mlx'
plain = json.load(open('lab/leg4/plain.json')); ids = (plain['prompt_ids'] + plain['gen_ids'])[:768]
if which == 'pack':
    from dflash_mlx.runtime.prism_pack import load_text_model
    model, _ = load_text_model(PACK)
else:
    from mlx_lm import load
    model, _ = load(REF)
from dflash_mlx.engine.target_ops import resolve_target_ops
ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
CAP = {6, 20, 34, 48, 62}
cache = ops.make_cache(model, enable_speculative_linear_cache=True)
feats = {}
# chunked prefill to keep memory flat; capture per chunk and concatenate
pos = 0; parts = {k: [] for k in CAP}; logits_last = None
while pos < len(ids):
    chunk = ids[pos:pos+256]
    lg, cap = ops.forward_with_hidden_capture(model, input_ids=mx.array([chunk]), cache=cache, capture_layer_ids=CAP); mx.eval(lg)
    for k in CAP: parts[k].append(np.array(cap[k][0].astype(mx.float16)))
    logits_last = np.array(lg[0].astype(mx.float32)).argmax(-1)
    pos += len(chunk)
np.savez_compressed(out, **{f'L{k-1}': np.concatenate(parts[k], 0) for k in CAP}, ids=np.array(ids), argmax_tail=logits_last)
print(which, 'captured', {f'L{k-1}': np.concatenate(parts[k],0).shape for k in CAP}, 'peak GB', round(mx.get_peak_memory()/1e9, 2))
