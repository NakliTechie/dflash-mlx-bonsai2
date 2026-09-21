"""How often does each verify path (stock fp32 reference, fp16-cast stock, v4b/8-only, padded v7) change the
target's greedy argmax? Teacher-force a ~1000-token real text through verify_block in 5-row chunks (the real
verify width) and count argmax disagreements vs the fp32 reference, plus mean |logit margin| at the flips."""
import sys, re; sys.path.insert(0, 'lab')
PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx, numpy as np
from prism_pack_loader import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.runtime.prism_qmm import install_prism_verify_linears
sys.path.insert(0, PACK + '/runtime'); import runtime as prism_rt
model, _ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
from dflash_mlx.runtime.prism_pack import load_pack_tokenizer
tok = load_pack_tokenizer(PACK)
enc = lambda s: tok.encode(s, add_special_tokens=False) if hasattr(tok, 'encode') else tok(s)['input_ids']
CAP = {6, 20, 34, 48, 62}; W = 5
text = open('lab/leg8/quiet/story-text-v4b.log').read(); text = re.sub(r'^\[transformers\].*$', '', text, flags=re.M).rsplit('\n', 2)[0]
prompt = '<|im_start|>user\nWrite a complete short story of about 1500 words: a lighthouse keeper on a remote island receives a letter that changes everything. Include dialogue, a turning point, and a resolution.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
ids = enc(prompt + text)
P = len(enc(prompt)); ids = list(ids)[:P + 1000]
print('tokens', len(ids), 'prompt', P)
def run_mode(mode):
    prism_rt.Packed._dflash_verify_mode = None; install_prism_verify_linears(prism_rt.Packed, prism_rt.fwht, mode)
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    lg, _ = ops.forward_with_hidden_capture(model, input_ids=mx.array([ids[:P]]), cache=cache, capture_layer_ids=CAP); mx.eval(lg)
    am, top2 = [], []
    for p0 in range(P, len(ids), W):
        chunk = ids[p0:p0 + W]
        if len(chunk) < W: break
        ops.arm_rollback(cache, prefix_len=p0)
        vl, _ = ops.verify_block(target_model=model, verify_ids=mx.array([chunk]), target_cache=cache, capture_layer_ids=CAP)
        l = vl[0].astype(mx.float32); s = mx.sort(l, axis=-1); mx.eval(l, s)
        am += l.argmax(-1).tolist(); top2 += (s[:, -1] - s[:, -2]).tolist()
        ops.restore_after_acceptance(cache, target_len=p0 + W, acceptance_length=W, drafted_tokens=W)
    return np.array(am), np.array(top2)
ref, margin = run_mode('off')
print(f'reference fp32: {len(ref)} positions, median top-2 margin {np.median(margin):.3f}')
for mode in ('fp16', 'v4b', 'v7'):
    am, _ = run_mode(mode); flips = np.nonzero(am != ref)[0]
    print(f'{mode:5s}: {len(flips)} flips / {len(ref)} ({100*len(flips)/len(ref):.2f} %) | fp32 margin at flips: {np.round(margin[flips], 4).tolist()[:12]}')
