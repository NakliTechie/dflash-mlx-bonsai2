"""End-to-end: verify_block(8) with stock Packed qmm vs Packed patched to use the v4b MMA kernel for 8-row inputs."""
import sys, time, statistics as st
sys.path.insert(0,'lab')
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from prism_pack_loader import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
from qmm_smallm_v4b import qmm_v4b
sys.path.insert(0, PACK + '/runtime')
import runtime as prism_rt
model,_ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
CAP={6,20,34,48,62}
stock_call = prism_rt.Packed.__call__
USE_V4B = {'on': False, 'calls': 0}
def patched_call(self, x):
    if self.embedding or not USE_V4B['on']: return stock_call(self, x)
    shape = x.shape; M = 1
    for d in shape[:-1]: M *= d
    if M != 8 or self.weight.shape[0] % 64 != 0: return stock_call(self, x)
    if USE_V4B['on'] == 'stock16':
        USE_V4B['calls'] += 1
        return stock_call(self, x.astype(mx.float16)).astype(x.dtype)
    USE_V4B['calls'] += 1
    x2 = x.reshape(M, shape[-1])
    if self.block: x2 = prism_rt.fwht(x2, self.block, self.signs)
    return qmm_v4b(x2.astype(mx.float16), self.weight, self.scales, self.biases).astype(x.dtype).reshape(*shape[:-1], -1)
prism_rt.Packed.__call__ = patched_call
def prefill():
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    ids = mx.array([[248044]+[1000+(i*17)%20000 for i in range(255)]])
    lg,_ = ops.forward_with_hidden_capture(model, input_ids=ids, cache=cache, capture_layer_ids=CAP); mx.eval(lg); return cache
cache = prefill()
vids = mx.array([[3000+(i*7)%20000 for i in range(8)]])
def verify(N, reps=6):
    ts=[]; last=None
    for r in range(reps):
        ops.arm_rollback(cache, prefix_len=256)
        ids = vids if N == 8 else mx.array([[3000]])
        t0=time.perf_counter(); vl,_ = ops.verify_block(target_model=model, verify_ids=ids, target_cache=cache, capture_layer_ids=CAP); mx.eval(vl); ts.append((time.perf_counter()-t0)*1000); last=vl
        ops.restore_after_acceptance(cache, target_len=256, acceptance_length=0, drafted_tokens=N)
    return st.median(ts[1:]), [round(t,1) for t in ts], last
for phase in ('stock','stock16','v4b','stock16','v4b'):
    USE_V4B['on'] = {'stock': False, 'stock16': 'stock16', 'v4b': True}[phase]; USE_V4B['calls'] = 0
    m1,_,_ = verify(1); m8, ts8, lg = verify(8)
    am = lg[0].argmax(-1).tolist()
    print(f'{phase:7s}: verify(1) {m1:6.1f} ms   verify(8) {m8:6.1f} ms  {ts8}   ratio {m8/m1:.2f}   v4b calls {USE_V4B["calls"]}   argmax {am}')
print('swap', __import__('subprocess').check_output(['sysctl','vm.swapusage']).decode().strip()[:60])
