import sys, time, statistics as st
sys.path.insert(0,'lab')
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from prism_pack_loader import load_text_model
model,_ = load_text_model(PACK)
def bench(fn, reps=20):
    for _ in range(3): mx.eval(fn())
    ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter()-t0)*1000)
    return st.median(ts)
l0 = model.model.layers[0]; l3 = model.model.layers[3]
mods = {'gdn.in_proj_qkv (5120->%d)'%l0.linear_attn.in_proj_qkv.weight.shape[0]: l0.linear_attn.in_proj_qkv,
        'gdn.out_proj': l0.linear_attn.out_proj,
        'mlp.gate_proj (5120->17408)': l0.mlp.gate_proj, 'mlp.down_proj (17408->5120)': l0.mlp.down_proj,
        'attn.q_proj (layer3)': l3.self_attn.q_proj, 'lm_head (5120->248320)': model.lm_head}
print(f"{'module':34s} " + " ".join(f"M={M:>2d}" .rjust(9) for M in (1,2,4,8,16,32)))
for name, m in mods.items():
    din = m.weight.shape[1]*16 if False else None
    # infer input width from scales: weight packed 2-bit -> in_features = weight.shape[1]*16
    din = m.weight.shape[1]*16
    row=[]
    for M in (1,2,4,8,16,32):
        x = mx.random.normal((1,M,din)).astype(mx.float16)
        row.append(bench(lambda: m(x)))
    print(f"{name:34s} " + " ".join(f"{t:8.2f}ms" for t in row))
# whole-layer sanity: one hybrid layer fwd at M=1/8/16 (no cache) is not needed; per-op suffices.
print('block', l0.mlp.gate_proj.block, 'signs', None if l0.mlp.gate_proj.signs is None else l0.mlp.gate_proj.signs.shape, 'weight', l0.mlp.gate_proj.weight.shape, l0.mlp.gate_proj.weight.dtype, 'scales', l0.mlp.gate_proj.scales.shape, l0.mlp.gate_proj.scales.dtype)
