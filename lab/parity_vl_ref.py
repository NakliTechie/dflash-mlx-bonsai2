import sys, time, numpy as np
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
sys.path.insert(0, PACK+'/runtime'); import mlx.core as mx
from vision_artifact import load_vl_model
model,_,_ = load_vl_model(PACK, load_processor=False); lm = model.language_model
from mlx_lm.models import cache as cm
ids = mx.array([[248044]+list(range(1000,1015))]); c = cm.make_prompt_cache(lm)
t0=time.time(); out = lm(ids, cache=c); out = getattr(out, "logits", out); mx.eval(out); print(f"VL lm forward 16 tok {time.time()-t0:.2f}s", out.shape, out.dtype)
np.save('lab/ref_logits_vl.npy', np.array(out.astype(mx.float32)))
print('argmax', np.array(out.astype(mx.float32))[0].argmax(-1).tolist())
