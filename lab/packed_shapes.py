import sys, time, statistics as st, collections
sys.path.insert(0, 'lab')
PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from prism_pack_loader import load_text_model
from dflash_mlx.runtime.prism_qmm import qmm_m8
sys.path.insert(0, PACK + '/runtime'); import runtime as prism_rt
model, _ = load_text_model(PACK)
shapes = collections.Counter(); example = {}
for name, m in model.named_modules():
    if isinstance(m, prism_rt.Packed) and not m.embedding:
        key = (m.weight.shape[0], m.weight.shape[1] * 16, m.block); shapes[key] += 1; example.setdefault(key, (name, m))
print(f'{len(shapes)} distinct shapes, {sum(shapes.values())} Packed modules')
def bench(fn, reps=15):
    for _ in range(3): mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1000)
    return st.median(ts)
tot = collections.defaultdict(float)
print(f"{'(N, K, block)':24s} {'n':>4s} {'v4b M8':>7s} {'stk16 M8':>8s} {'stk16 M1':>8s} {'fwht8':>6s} | {'v4b*n':>6s} {'M1*n':>6s} {'fw*n':>6s} {'GB':>5s}  example")
for key, n in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1] * kv[1]):
    N, K, block = key; path, m = example[key]
    x8 = mx.random.normal((8, K)).astype(mx.float16); x1 = mx.random.normal((1, K)).astype(mx.float16)
    v = bench(lambda: qmm_m8(x8, m.weight, m.scales, m.biases)) if N % 64 == 0 else float('nan')
    s8 = bench(lambda: mx.quantized_matmul(x8, m.weight, scales=m.scales, biases=m.biases, transpose=True, group_size=128, bits=2))
    s1 = bench(lambda: mx.quantized_matmul(x1, m.weight, scales=m.scales, biases=m.biases, transpose=True, group_size=128, bits=2))
    fw = bench(lambda: prism_rt.fwht(x8, block, m.signs)) if block else 0.0
    vv = v if v == v else s8; gb = N * K * 2.25 / 8 / 1e9
    tot['v4b'] += vv * n; tot['s8'] += s8 * n; tot['s1'] += s1 * n; tot['fw'] += fw * n; tot['gb'] += gb * n
    print(f"{str(key):24s} {n:4d} {v:7.3f} {s8:8.3f} {s1:8.3f} {fw:6.3f} | {vv*n:6.1f} {s1*n:6.1f} {fw*n:6.1f} {gb*n:5.2f}  {path.split('.')[-1]} ({'gdn' if 'linear_attn' in path else 'attn' if 'self_attn' in path else 'mlp' if 'mlp' in path else path})")
print(f"TOTAL: v4b M8 {tot['v4b']:.1f} ms | stock16 M8 {tot['s8']:.1f} | stock16 M1 {tot['s1']:.1f} | fwht {tot['fw']:.1f} | weights {tot['gb']:.2f} GB -> {tot['gb']/0.273*1000:.0f} ms at 273 GB/s; M=8 flops {tot['gb']/2.25*8*16:.0f} GFLOP")
