import sys, time
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
sys.path.insert(0, PACK+'/runtime')
import mlx.core as mx
from vision_artifact import load_vl_model
t0=time.time(); model, processor, config = load_vl_model(PACK); print(f'load {time.time()-t0:.1f}s active_mem {mx.get_active_memory()/1e9:.2f} GB type {type(model).__name__}')
print('children', list(model.children().keys())[:10])
lm = getattr(model,'language_model',None) or model
print('lm type', type(lm).__name__, 'children', list(lm.children().keys())[:10], 'args.model_type', getattr(getattr(lm,'args',None),'model_type',None))
from dflash_mlx.engine.target_ops import resolve_target_ops
for cand,name in ((model,'model'),(lm,'language_model')):
    try:
        ops = resolve_target_ops(cand); print(name,'-> target_ops', type(ops).__name__, 'model_type', ops.model_type(cand), 'family', ops.family(cand)); target=cand; break
    except Exception as ex: print(name,'resolve FAIL', repr(ex)[:160])
tm = ops.text_model(target); print('layers', len(tm.layers), 'embed', type(tm.embed_tokens).__name__, 'as_linear', hasattr(tm.embed_tokens,'as_linear'), 'lm_head', type(ops.text_wrapper(target).lm_head).__name__, 'tie', getattr(ops.text_wrapper(target).args,'tie_word_embeddings',None))
l0=tm.layers[0]; print('layer0 attrs', [a for a in ('linear_attn','self_attn','is_linear','mlp') if hasattr(l0,a)])
la=l0.linear_attn; print('gdn children', {k:type(v).__name__ for k,v in la.children().items()})
print('caps', ops.capabilities_for(target))
t0=time.time(); ops.install_speculative_hooks(target); print(f'hooks installed {time.time()-t0:.2f}s')
cache = ops.make_cache(target, enable_speculative_linear_cache=True); print('cache entries', len(cache), type(cache[0]).__name__, type(cache[3]).__name__)
ids = mx.array([[248044]+list(range(1000,1015))])
t0=time.time(); logits, cap = ops.forward_with_hidden_capture(target, input_ids=ids, cache=cache, capture_layer_ids={5,19,33,47,61}); mx.eval(logits); print(f'capture forward 16 tok {time.time()-t0:.2f}s logits {logits.shape} {logits.dtype} captured {type(cap).__name__} {[ (k, tuple(v.shape), str(v.dtype)) for k,v in (cap.items() if isinstance(cap,dict) else enumerate(cap))][:5]}')
feat = ops.extract_context_feature(cap, [5,19,33,47,61]); mx.eval(feat); print('context feature', feat.shape, feat.dtype)
h = mx.zeros((1,1,5120), dtype=mx.float16)
try: lg = ops.logits_from_hidden(target, h); mx.eval(lg); print('logits_from_hidden ok', lg.shape)
except Exception as ex: print('logits_from_hidden FAIL', repr(ex)[:200])
# baseline plain generate via the pack's own forward for a sanity token
from mlx_lm.models import cache as cm
c2 = cm.make_prompt_cache(lm); t0=time.time(); out = lm(ids, cache=c2); mx.eval(out); print(f'plain lm forward 16 tok {time.time()-t0:.2f}s argmax_last {int(out[0,-1].argmax())} vs capture argmax_last {int(logits[0,-1].argmax())}')
print('peak', round(mx.get_peak_memory()/1e9,2), 'GB')
