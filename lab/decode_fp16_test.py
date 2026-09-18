import sys, time, statistics as st
sys.path.insert(0,'lab')
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from dflash_mlx.runtime.prism_pack import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
sys.path.insert(0, PACK+'/runtime'); import runtime as prism_rt
model,_ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
cache = ops.make_cache(model, enable_speculative_linear_cache=True)
ids = mx.array([[248044]+[1000+(i*17)%20000 for i in range(255)]]); lg,_ = ops.forward_with_hidden_capture(model, input_ids=ids, cache=cache, capture_layer_ids={6}); mx.eval(lg)
cur = prism_rt.Packed.__call__; stock = prism_rt.Packed._dflash_stock_call
def cast1(self, x):
    if self.embedding: return stock(self, x)
    return stock(self, x.astype(mx.float16)).astype(x.dtype)
def step(reps=8):
    ts=[]; last=None
    for r in range(reps):
        ops.arm_rollback(cache, prefix_len=256); t0=time.perf_counter(); vl,_ = ops.verify_block(target_model=model, verify_ids=mx.array([[3000]]), target_cache=cache, capture_layer_ids={6}); mx.eval(vl); ts.append((time.perf_counter()-t0)*1000); last=int(vl[0,-1].argmax())
        ops.restore_after_acceptance(cache, target_len=256, acceptance_length=0, drafted_tokens=1)
    return round(st.median(ts[1:]),1), last
for phase in ('stock','fp16-all','stock','fp16-all'):
    prism_rt.Packed.__call__ = cur if phase=='stock' else cast1
    print(phase, 'decode-step', step())
