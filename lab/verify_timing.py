import sys, time, statistics as st
sys.path.insert(0,'lab')
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from prism_pack_loader import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
model, config = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
CAP = {6,20,34,48,62}
def fresh(prefill_len):
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    ids = mx.array([[248044] + [1000 + (i*17) % 20000 for i in range(prefill_len-1)]])
    lg, cap = ops.forward_with_hidden_capture(model, input_ids=ids, cache=cache, capture_layer_ids=CAP); mx.eval(lg)
    feat = ops.extract_context_feature(cap, [5,19,33,47,61]); mx.eval(feat)
    return cache, feat
cache, feat = fresh(256); print('prefill 256 ok; context feature', feat.shape, feat.dtype)
def verify(N, reps=5):
    ts=[]
    for r in range(reps):
        ops.arm_rollback(cache, prefix_len=256)
        vids = mx.array([[3000 + (r*31+i*7) % 20000 for i in range(N)]])
        t0=time.perf_counter(); vl, vcap = ops.verify_block(target_model=model, verify_ids=vids, target_cache=cache, capture_layer_ids=CAP); mx.eval(vl); ts.append(time.perf_counter()-t0)
        ns = ops.restore_after_acceptance(cache, target_len=256, acceptance_length=0, drafted_tokens=N)
        assert cache[3].offset == 256, cache[3].offset
    return ts
import inspect; print('restore_after_acceptance sig', inspect.signature(ops.restore_after_acceptance))
res={}
for N in (1,8,16,32):
    ts = verify(N); res[N]=st.median(ts[1:]); print(f'verify_block N={N:2d}: median {res[N]*1000:.1f} ms  (all {[round(t*1000,1) for t in ts]})  logits per position: yes')
print('RATIO t(16)/t(1) =', round(res[16]/res[1],2), ' t(8)/t(1) =', round(res[8]/res[1],2), ' t(32)/t(1) =', round(res[32]/res[1],2))
print('peak', round(mx.get_peak_memory()/1e9,2), 'GB')
