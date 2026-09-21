"""In-graph-like microbench: 16 different layers' weights per shape, one mx.eval over all 16 calls (amortises
the per-eval sync that dominates single-op timings). Reports per-call ms and achieved TFLOPS at M=8."""
import sys, time, statistics as st, importlib
import mlx.core as mx
sys.path.insert(0, 'lab')
from dflash_mlx.runtime.prism_qmm import qmm_m8 as qmm_v4b
from qmm_smallm_v5 import qmm_v5
KERNELS = {'stock16': lambda x, w, s, b: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2), 'v4b': qmm_v4b, 'v5': qmm_v5}
for extra in sys.argv[1:]:
    mod, _, fn = extra.partition(':'); m = importlib.import_module(mod)
    KERNELS[fn or mod] = getattr(m, fn) if fn else getattr(m, [a for a in dir(m) if a.startswith('qmm_')][0])
PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
W = mx.load(PACK + '/model.safetensors')
def mods(fmt, layers):
    out = []
    for l in layers:
        p = fmt.format(l)
        if p + '.weight' in W: out.append((W[p + '.weight'], W[p + '.scales'], W[p + '.biases']))
    return out
shapes = {'up_proj 17408x5120 (x128)': mods('language_model.model.layers.{}.mlp.up_proj', range(16)),
          'down_proj 5120x17408 (x64)': mods('language_model.model.layers.{}.mlp.down_proj', range(16)),
          'in_proj_qkv 10240x5120 (x48)': mods('language_model.model.layers.{}.linear_attn.in_proj_qkv', [i for i in range(24) if i % 4 != 3][:16]),
          'in_proj_z 6144x5120 (x48)': mods('language_model.model.layers.{}.linear_attn.in_proj_z', [i for i in range(24) if i % 4 != 3][:16]),
          'o_proj 5120x6144 (x64)': mods('language_model.model.layers.{}.self_attn.o_proj', range(16)),
          'q_proj 12288x5120 (x16)': mods('language_model.model.layers.{}.self_attn.q_proj', [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]),
          'v_proj 1024x5120 (x32)': mods('language_model.model.layers.{}.self_attn.v_proj', [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]),
          'lm_head 248320x5120 (x1)': mods('language_model.lm_head', [0]) or [(W['language_model.lm_head.weight'], W['language_model.lm_head.scales'], W['language_model.lm_head.biases'])]}
counts = {'up_proj': 128, 'down_proj': 64, 'in_proj_qkv': 48, 'in_proj_z': 48, 'o_proj': 64, 'q_proj': 16, 'v_proj': 32, 'lm_head': 1}
def bench_all(fns, reps=15):
    """interleave kernels per rep; return per-kernel min (uncontended estimate) and median"""
    for fn in fns.values():
        for _ in range(2): mx.eval(fn())
    ts = {k: [] for k in fns}
    for _ in range(reps):
        for k, fn in fns.items():
            t0 = time.perf_counter(); mx.eval(fn()); ts[k].append((time.perf_counter() - t0) * 1000)
    return {k: (min(v), st.median(v)) for k, v in ts.items()}
names = list(KERNELS); tot = {k: 0.0 for k in names}
print(f"{'shape':30s} " + " ".join(f"{k:>13s}" for k in names) + "   (per-call ms min/median; TFLOPS for last, min)")
for name, ws in shapes.items():
    for w, s, b in ws: mx.eval(w, s, b)
    N, K = ws[0][0].shape[0], ws[0][0].shape[1] * 16
    xs = [(mx.random.normal((8, K)) * 0.5).astype(mx.float16) for _ in ws]; mx.eval(*xs)
    fns = {k: (lambda fn=fn: [fn(x, w, s, b) for x, (w, s, b) in zip(xs, ws)]) for k, fn in KERNELS.items() if k == 'stock16' or N % 64 == 0}
    res = bench_all(fns); row = []
    for k in names:
        if k not in res: row.append((float('nan'), float('nan'))); continue
        mn, md = res[k]; row.append((mn / len(ws), md / len(ws))); tot[k] += mn / len(ws) * counts[name.split()[0]]
    gf = N * K * 16 / 1e9
    print(f"{name:30s} " + " ".join(f"{mn:6.3f}/{md:6.3f}" for mn, md in row) + f"   {gf / row[-1][0]:5.2f}")
print("TOTAL per verify, min-of-15 (ms): " + "  ".join(f"{k} {v:.1f}" for k, v in tot.items()))
