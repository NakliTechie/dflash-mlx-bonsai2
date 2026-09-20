"""Exact dequantization of PrismML's MLX pack modules to plain bf16 weights, for the PyTorch trainer:
  embed_tokens: rows are stored 2-bit affine (g128) in the ROTATED basis; the runtime applies the inverse
                blockwise Hadamard (H x) * signs after lookup -> W_embed[v] = fwht_inv(dequant(row_v)).
  lm_head:      y = W_rot . fwht(signs * x)  ->  in the original basis W_orig = W_rot . (H . diag(signs)) / ...
                We keep it simple and exact: export W_rot (bf16) plus block + signs, and apply the same
                activation transform in torch (fwht(signs*x) then x @ W_rot.T). Same for embed: export the
                dequantized rotated rows and apply the inverse transform in torch — no basis-change error.
Usage: dequant_pack.py PACK_DIR OUT.safetensors   (numpy only; ~5 GB output)
"""
import sys, json, numpy as np
from safetensors.numpy import save_file

def dequant_2bit_affine(w_u32, scales, biases, group=128):
    # MLX 2-bit affine packing: 16 values per uint32 along the last dim, low bits first; w = q*scale + bias per group
    N, K16 = w_u32.shape; K = K16 * 16
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, None, :]
    q = ((w_u32[:, :, None] >> shifts) & 3).astype(np.float32).reshape(N, K)
    s = np.repeat(scales.astype(np.float32), group, axis=1); b = np.repeat(biases.astype(np.float32), group, axis=1)
    return q * s + b

if __name__ == '__main__':
    pack, out = sys.argv[1], sys.argv[2]
    cfg = json.load(open(pack + '/config.json'))
    from safetensors.numpy import load_file
    W = load_file(pack + '/model.safetensors')
    prefix = 'language_model.' if cfg.get('components', {}).get('vision') else ''
    recs = {r['path']: r for r in cfg['modules']}
    outd = {}
    for name in ('lm_head', 'model.embed_tokens'):
        r = recs[name]; key = prefix + name
        w = dequant_2bit_affine(W[key + '.weight'], W[key + '.scales'], W[key + '.biases'])
        outd[name + '.weight_rot'] = w.astype(np.float32).astype(np.float16)   # rotated-basis dequantized weights
        outd[name + '.signs'] = W[key + '.signs'].astype(np.float16) if (key + '.signs') in W else np.ones(w.shape[1], np.float16)
        outd[name + '.block'] = np.array([r['block'] or 0], dtype=np.int32)
        print(name, 'rotated weight', w.shape, 'block', r['block'], 'embedding' if r['embedding'] else 'linear')
    save_file(outd, out); print('saved', out)
