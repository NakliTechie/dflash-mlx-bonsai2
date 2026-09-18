import sys, time, statistics as st
sys.path.insert(0,'lab')
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from prism_pack_loader import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
model,_ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
CAP={6,20,34,48,62}
def prefill():
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    ids = mx.array([[248044]+[1000+(i*17)%20000 for i in range(255)]])
    lg,_ = ops.forward_with_hidden_capture(model, input_ids=ids, cache=cache, capture_layer_ids=CAP); mx.eval(lg); return cache
vids = lambda N: mx.array([[3000+(i*7)%20000 for i in range(N)]])
def med(fn, reps=4):
    ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    return round(st.median(ts),1), [round(t,1) for t in ts]
# A: unarmed plain forward (original_call path), fresh cache each rep (recurrent state can't roll back unarmed)
for N in (1,8,16):
    def run():
        c = prefill(); t0=time.perf_counter(); lg,_ = ops.forward_with_hidden_capture(model, input_ids=vids(N), cache=c, capture_layer_ids=CAP); mx.eval(lg); run.t=(time.perf_counter()-t0)*1000
    ts=[]
    for _ in range(4): run(); ts.append(round(run.t,1))
    print(f'A unarmed forward+capture  N={N:2d}: median {st.median(ts):.1f} ms {ts}')
# B: armed verify_block with / without capture, same cache restored between reps
cache = prefill()
for N in (1,8,16):
    for cap,label in ((CAP,'capture5'),(set(),'nocapture')):
        def run():
            ops.arm_rollback(cache, prefix_len=256); lg,_ = ops.verify_block(target_model=model, verify_ids=vids(N), target_cache=cache, capture_layer_ids=cap); mx.eval(lg); ops.restore_after_acceptance(cache, target_len=256, acceptance_length=0, drafted_tokens=N)
        run(); m, ts = med(run); print(f'B armed verify_block {label:9s} N={N:2d}: median {m} ms {ts}')
# C: GDN kernel alone, S=8/16, 48 v-heads 128, 16 k-heads 128
from mlx_lm.models import gated_delta as gd
from dflash_mlx.kernels import gated_delta_kernel_with_tape
for S in (1,8,16):
    q = mx.random.normal((1,S,16,128)).astype(mx.float16); k = mx.random.normal((1,S,16,128)).astype(mx.float16); v = mx.random.normal((1,S,48,128)).astype(mx.float16)
    g = -mx.random.uniform(shape=(1,S,48)).astype(mx.float32); beta = mx.random.uniform(shape=(1,S,48)).astype(mx.float16); state = mx.zeros((1,48,128,128), dtype=mx.float32)
    m1,_ = med(lambda: mx.eval(gd.gated_delta_kernel(q,k,v,g,beta,state,None)[0]), 8)
    m2,_ = med(lambda: mx.eval(gated_delta_kernel_with_tape(q,k,v,g,beta,state,None)[0]), 8)
    print(f'C gdn kernel S={S:2d}: plain {m1} ms  with_tape {m2} ms  (x48 layers -> plain {m1*48:.0f} ms, tape {m2*48:.0f} ms)')
