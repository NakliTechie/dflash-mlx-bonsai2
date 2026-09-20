"""Numerical divergence between the verify path (8-row v4b kernel) and the decode path (1-row qmv, fp16 cast):
teacher-force the plain generation's tokens and count positions where argmax(verify M=8) != argmax(decode M=1)."""
import sys, json
sys.path.insert(0,'lab')
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from dflash_mlx.runtime.prism_pack import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
model,_ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
plain = json.load(open('lab/leg4/plain.json')); ids = plain['prompt_ids'] + plain['gen_ids']
P = len(plain['prompt_ids']); G = len(plain['gen_ids'])
# decode-path argmax at every generated position: prefill prompt, then 1 token at a time
cache = ops.make_cache(model, enable_speculative_linear_cache=True)
lg,_ = ops.forward_with_hidden_capture(model, input_ids=mx.array([ids[:P]]), cache=cache, capture_layer_ids={6}); mx.eval(lg)
dec = [int(lg[0,-1].argmax())]
for t in ids[P:P+G-1]:
    lg,_ = ops.forward_with_hidden_capture(model, input_ids=mx.array([[t]]), cache=cache, capture_layer_ids={6}); mx.eval(lg); dec.append(int(lg[0,-1].argmax()))
# verify-path argmax: same prefix, then blocks of 8 teacher-forced tokens through verify_block
cache2 = ops.make_cache(model, enable_speculative_linear_cache=True)
lg,_ = ops.forward_with_hidden_capture(model, input_ids=mx.array([ids[:P]]), cache=cache2, capture_layer_ids={6}); mx.eval(lg)
ver = [int(lg[0,-1].argmax())]  # position P-1 predicts gen[0] via prefill (M=P path)
pos = P
while pos < P+G-1:
    blk = ids[pos:min(pos+8, P+G-1)]
    ops.arm_rollback(cache2, prefix_len=pos)
    vl,_ = ops.verify_block(target_model=model, verify_ids=mx.array([blk]), target_cache=cache2, capture_layer_ids={6}); mx.eval(vl)
    ver.extend(int(vl[0,i].argmax()) for i in range(len(blk)))
    ops.restore_after_acceptance(cache2, target_len=pos+len(blk), acceptance_length=len(blk), drafted_tokens=len(blk))
    pos += len(blk)
n = min(len(dec), len(ver)); flips = [i for i in range(n) if dec[i] != ver[i]]
dec_vs_plain = sum(1 for i in range(n) if dec[i] != plain['gen_ids'][i])
print(f'positions {n} | verify-vs-decode argmax flips {len(flips)} ({100*len(flips)/n:.2f}%) first at {flips[:8]} | decode-path vs recorded plain mismatches {dec_vs_plain}')
