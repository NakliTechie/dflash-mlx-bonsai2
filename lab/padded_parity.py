"""Parity of the padded v7 path vs stock fp32 for the row counts a real verify uses (3,4,5,7,8)."""
import sys; sys.path.insert(0, 'lab')
PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from prism_pack_loader import load_text_model
from dflash_mlx.runtime.prism_qmm import install_prism_verify_linears
sys.path.insert(0, PACK + '/runtime'); import runtime as prism_rt
model, _ = load_text_model(PACK)
mods = {'up_proj': model.model.layers[0].mlp.up_proj, 'down_proj': model.model.layers[0].mlp.down_proj, 'in_proj_qkv': model.model.layers[0].linear_attn.in_proj_qkv, 'o_proj': model.model.layers[3].self_attn.o_proj, 'lm_head': model.lm_head}
for name, m in mods.items():
    K = m.weight.shape[1] * 16
    for rows in (3, 4, 5, 7, 8):
        x = (mx.random.normal((1, rows, K)) * 0.7).astype(mx.float32)
        prism_rt.Packed._dflash_verify_mode = None; install_prism_verify_linears(prism_rt.Packed, prism_rt.fwht, 'off'); ref = m(x).astype(mx.float32)
        prism_rt.Packed._dflash_verify_mode = None; install_prism_verify_linears(prism_rt.Packed, prism_rt.fwht, 'v7'); got = m(x).astype(mx.float32)
        mx.eval(ref, got)
        err = float(mx.abs(ref - got).max()); rel = err / float(mx.abs(ref).max()); am = int((ref.argmax(-1) == got.argmax(-1)).sum())
        print(f'{name:12s} rows={rows}: shape {got.shape} max|Δ| {err:.4f} rel {rel:.1e} argmax agree {am}/{rows}')
