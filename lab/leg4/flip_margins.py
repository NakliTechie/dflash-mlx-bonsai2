import sys, json
sys.path.insert(0,'lab')
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from dflash_mlx.runtime.prism_pack import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
model,_ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
plain = json.load(open('lab/leg4/plain.json')); ids = plain['prompt_ids'] + plain['gen_ids']; P = len(plain['prompt_ids'])
FLIPS = [100, 242, 353, 460, 939]   # generated-position indices from flip_rate.py
def margins(lg_row):
    v = lg_row.astype(mx.float32); top = mx.argpartition(-v, 2)[:2]; a, b = sorted([float(v[int(top[0])]), float(v[int(top[1])])], reverse=True)
    return int(v.argmax()), round(a - b, 4)
# decode path: prefill prompt + gen[:k-1] as one prefill (M large), then a 1-token step at position P+k-1 -> predicts gen[k]
for k in FLIPS:
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    lg,_ = ops.forward_with_hidden_capture(model, input_ids=mx.array([ids[:P+k-1]]), cache=cache, capture_layer_ids={6}); mx.eval(lg)
    lg1,_ = ops.forward_with_hidden_capture(model, input_ids=mx.array([[ids[P+k-1]]]), cache=cache, capture_layer_ids={6}); mx.eval(lg1)
    d_tok, d_m = margins(lg1[0,-1])
    # verify path: same prefix up to P+k-8, then an 8-token verify block ending at position P+k-1
    cache2 = ops.make_cache(model, enable_speculative_linear_cache=True)
    start = P+k-8
    lg,_ = ops.forward_with_hidden_capture(model, input_ids=mx.array([ids[:start]]), cache=cache2, capture_layer_ids={6}); mx.eval(lg)
    ops.arm_rollback(cache2, prefix_len=start)
    vl,_ = ops.verify_block(target_model=model, verify_ids=mx.array([ids[start:start+8]]), target_cache=cache2, capture_layer_ids={6}); mx.eval(vl)
    v_tok, v_m = margins(vl[0,7])
    print(f'pos {k:4d}: decode argmax {d_tok} margin {d_m:7.4f} | verify argmax {v_tok} margin {v_m:7.4f} | plain token {plain["gen_ids"][k]}')
