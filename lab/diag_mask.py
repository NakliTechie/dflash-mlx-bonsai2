import sys; sys.path.insert(0,'lab')
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from prism_pack_loader import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
from mlx_lm.models.base import create_ssm_mask, create_attention_mask
model,_ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
inner = ops.text_model(model)
for S in (16, 64, 256):
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    print('S', S, 'armed', getattr(cache[0], '_armed', 'n/a'), 'type', type(cache[0]).__name__)
    ids = mx.array([[248044] + [1000+i for i in range(S-1)]]); h = inner.embed_tokens(ids)
    m = create_ssm_mask(h, cache[inner.ssm_idx]); fa = create_attention_mask(h, cache[inner.fa_idx])
    print('   ssm_mask', None if m is None else (m.shape, m.dtype), ' fa_mask', None if fa is None else (getattr(fa,'shape',fa)))
    try:
        lg,_ = ops.forward_with_hidden_capture(model, input_ids=ids, cache=cache, capture_layer_ids={6}); mx.eval(lg); print('   forward ok', lg.shape)
    except Exception as ex: print('   forward FAIL', str(ex)[:160])
