import sys, time, numpy as np
sys.path.insert(0,'lab')
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from prism_pack_loader import load_text_model, load_pack_tokenizer
t0=time.time(); model, config = load_text_model(PACK); print(f'text-only load {time.time()-t0:.1f}s active {mx.get_active_memory()/1e9:.2f} GB')
tok = load_pack_tokenizer(PACK); s='Hello, the capital of Japan is'; e=tok.encode(s); print('tokenizer', len(e), e[:8], repr(tok.decode(e)))
from mlx_lm.models import cache as cm
ids = mx.array([[248044]+list(range(1000,1015))]); c = cm.make_prompt_cache(model)
t0=time.time(); out = model(ids, cache=c); mx.eval(out); print(f'text lm forward 16 tok {time.time()-t0:.2f}s')
ref = np.load('lab/ref_logits_vl.npy'); mine = np.array(out.astype(mx.float32))
print('parity max|Δ|', float(np.abs(ref-mine).max()), 'argmax equal', bool((ref[0].argmax(-1)==mine[0].argmax(-1)).all()))
from dflash_mlx.engine.target_ops import resolve_target_ops
ops = resolve_target_ops(model); print('target_ops', type(ops).__name__, ops.model_type(model), ops.family(model))
tm = ops.text_model(model); print('embed', type(tm.embed_tokens).__name__, 'lm_head', type(model.lm_head).__name__, 'tie', model.args.tie_word_embeddings)
la = tm.layers[0].linear_attn; print('gdn type', type(la).__name__, 'children', {k:type(v).__name__ for k,v in la.children().items()})
ops.install_speculative_hooks(model); cache = ops.make_cache(model, enable_speculative_linear_cache=True); print('cache', len(cache), type(cache[0]).__name__, type(cache[3]).__name__)
t0=time.time(); logits, cap = ops.forward_with_hidden_capture(model, input_ids=ids, cache=cache, capture_layer_ids={5,19,33,47,61}); mx.eval(logits); print(f'capture forward 16 tok {time.time()-t0:.2f}s logits {logits.shape} {logits.dtype}')
print('captured', {k:(tuple(v.shape), str(v.dtype)) for k,v in (cap.items() if isinstance(cap,dict) else enumerate(cap))})
print('capture-path parity max|Δ| vs VL', float(np.abs(ref-np.array(logits.astype(mx.float32))).max()))
feat = ops.extract_context_feature(cap, [5,19,33,47,61]); mx.eval(feat); print('context feature', feat.shape, feat.dtype)
h = mx.zeros((1,1,5120), dtype=mx.float16); lg = ops.logits_from_hidden(model, h); mx.eval(lg); print('logits_from_hidden ok', lg.shape)
# verify_block on a fresh cache: 8 tokens after 16-token prefill
cache2 = ops.make_cache(model, enable_speculative_linear_cache=True); _ = ops.forward_with_hidden_capture(model, input_ids=ids, cache=cache2, capture_layer_ids={5}); mx.eval(_[0])
ops.arm_rollback(cache2, prefix_len=16)
vids = mx.array([list(range(2000,2008))]); t0=time.time(); vl, vcap = ops.verify_block(target_model=model, verify_ids=vids, target_cache=cache2, capture_layer_ids={5,19,33,47,61}); mx.eval(vl); print(f'verify_block 8 tok {time.time()-t0:.3f}s logits {vl.shape}')
for i in range(5):
    t0=time.time(); vl, vcap = ops.verify_block(target_model=model, verify_ids=vids, target_cache=cache2, capture_layer_ids={5,19,33,47,61}); mx.eval(vl); print(f'  verify_block rep {i} {time.time()-t0:.3f}s')
print('peak', round(mx.get_peak_memory()/1e9,2), 'GB')
