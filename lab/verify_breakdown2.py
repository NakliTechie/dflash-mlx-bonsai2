"""Attribute verify_block(8) time on the pack: per-shape v4b qmm microbench x counts, fwht, GDN tape kernel,
and the real verify with (a) v4b, (b) qmm stubbed to zeros, (c) qmm + fwht stubbed -> the remainder is
everything else incl. launch overhead."""
import sys, time, statistics as st, collections
sys.path.insert(0, 'lab')
PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from prism_pack_loader import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.runtime.prism_qmm import qmm_m8
sys.path.insert(0, PACK + '/runtime'); import runtime as prism_rt
model, _ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
CAP = {6, 20, 34, 48, 62}
def bench(fn, reps=15):
    for _ in range(3): mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1000)
    return st.median(ts)
# 1. enumerate Packed modules by shape
shapes = collections.Counter(); example = {}
def walk(m, path=''):
    for name, child in m.children().items() if hasattr(m, 'children') else []:
        p = f'{path}.{name}' if path else name
        if isinstance(child, prism_rt.Packed):
            if not child.embedding:
                key = (child.weight.shape[0], child.weight.shape[1] * 16, child.block); shapes[key] += 1; example.setdefault(key, (p, child))
        elif isinstance(child, dict):
            for k, v in child.items(): walk(v, f'{p}.{k}') if hasattr(v, 'children') else None
        elif isinstance(child, list):
            for i, v in enumerate(child): walk(v, f'{p}.{i}')
        else:
            walk(child, p)
walk(model)
print(f'{len(shapes)} distinct Packed shapes, {sum(shapes.values())} modules')
# 2. per-shape microbench: v4b M=8, stock fp16 M=8, stock fp16 M=1, fwht M=8
tot_v4b = tot_s8 = tot_s1 = tot_fw = 0.0
print(f"{'shape (N,K,block)':26s} {'count':>5s} {'v4b M8':>8s} {'stock16 M8':>10s} {'stock16 M1':>10s} {'fwht M8':>8s}  {'v4b x n':>8s} {'M1 x n':>8s}  example")
for key, n in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1] * kv[1]):
    N, K, block = key; path, m = example[key]
    x8 = mx.random.normal((8, K)).astype(mx.float16); x1 = mx.random.normal((1, K)).astype(mx.float16)
    v = bench(lambda: qmm_m8(x8, m.weight, m.scales, m.biases)) if N % 64 == 0 else float('nan')
    s8 = bench(lambda: mx.quantized_matmul(x8, m.weight, scales=m.scales, biases=m.biases, transpose=True, group_size=128, bits=2))
    s1 = bench(lambda: mx.quantized_matmul(x1, m.weight, scales=m.scales, biases=m.biases, transpose=True, group_size=128, bits=2))
    fw = bench(lambda: prism_rt.fwht(x8, block, m.signs)) if block else 0.0
    tot_v4b += (v if v == v else s8) * n; tot_s8 += s8 * n; tot_s1 += s1 * n; tot_fw += fw * n
    print(f"{str(key):26s} {n:5d} {v:8.3f} {s8:10.3f} {s1:10.3f} {fw:8.3f}  {(v if v==v else s8)*n:8.1f} {s1*n:8.1f}  {path}")
print(f"SUM: v4b M8 {tot_v4b:.1f} ms | stock16 M8 {tot_s8:.1f} ms | stock16 M1 {tot_s1:.1f} ms | fwht M8 {tot_fw:.1f} ms")
# 3. GDN tape kernel S=8 vs S=1
from mlx_lm.models import gated_delta as gd
from dflash_mlx.kernels import gated_delta_kernel_with_tape
for S in (1, 8):
    q = mx.random.normal((1, S, 16, 128)).astype(mx.float16); k = mx.random.normal((1, S, 16, 128)).astype(mx.float16); v = mx.random.normal((1, S, 48, 128)).astype(mx.float16)
    g = -mx.random.uniform(shape=(1, S, 48)).astype(mx.float32); beta = mx.random.uniform(shape=(1, S, 48)).astype(mx.float16); state = mx.zeros((1, 48, 128, 128), dtype=mx.float32)
    m1 = bench(lambda: gd.gated_delta_kernel(q, k, v, g, beta, state, None)[0]); m2 = bench(lambda: gated_delta_kernel_with_tape(q, k, v, g, beta, state, None)[0])
    print(f'GDN kernel S={S}: plain {m1:.3f} ms, with_tape {m2:.3f} ms -> x48: plain {m1*48:.1f} ms, tape {m2*48:.1f} ms')
# 4. real verify with stubs
def prefill():
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    ids = mx.array([[248044] + [1000 + (i * 17) % 20000 for i in range(255)]])
    lg, _ = ops.forward_with_hidden_capture(model, input_ids=ids, cache=cache, capture_layer_ids=CAP); mx.eval(lg); return cache
cache = prefill(); vids = mx.array([[3000 + (i * 7) % 20000 for i in range(8)]])
MODE = {'m': 'v4b'}
stock_call = prism_rt.Packed._dflash_stock_call if hasattr(prism_rt.Packed, '_dflash_stock_call') else prism_rt.Packed.__call__
def call(self, x):
    if self.embedding: return stock_call(self, x)
    shape = x.shape; rows = 1
    for d in shape[:-1]: rows *= d
    if rows != 8 or self.weight.shape[0] % 64 != 0: return stock_call(self, x.astype(mx.float16)).astype(x.dtype)
    x2 = x.reshape(rows, shape[-1])
    if MODE['m'] == 'nofwht_noqmm': return mx.zeros((*shape[:-1], self.weight.shape[0]), dtype=x.dtype)
    if self.block: x2 = prism_rt.fwht(x2, self.block, self.signs)
    if MODE['m'] == 'noqmm': return mx.zeros((*shape[:-1], self.weight.shape[0]), dtype=x.dtype) + x2[..., :1].reshape(*shape[:-1], 1)
    return qmm_m8(x2.astype(mx.float16), self.weight, self.scales, self.biases).astype(x.dtype).reshape(*shape[:-1], -1)
prism_rt.Packed.__call__ = call
def verify(N, reps=6):
    ts = []
    for r in range(reps):
        ops.arm_rollback(cache, prefix_len=256); ids = vids if N == 8 else mx.array([[3000]])
        t0 = time.perf_counter(); vl, _ = ops.verify_block(target_model=model, verify_ids=ids, target_cache=cache, capture_layer_ids=CAP); mx.eval(vl); ts.append((time.perf_counter() - t0) * 1000)
        ops.restore_after_acceptance(cache, target_len=256, acceptance_length=0, drafted_tokens=N)
    return st.median(ts[1:])
for mode in ('v4b', 'noqmm', 'nofwht_noqmm', 'v4b'):
    MODE['m'] = mode; print(f'verify(8) mode={mode:13s}: {verify(8):.1f} ms')
MODE['m'] = 'v4b'; print(f'verify(1) (stock fp16 path): {verify(1):.1f} ms')
