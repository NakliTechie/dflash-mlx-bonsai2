"""256-token greedy generation parity: stock fp32 Packed path vs the fp16 verify patch (rows==1 also cast)."""
import sys, time
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from dflash_mlx.runtime.prism_pack import load_text_model, load_pack_tokenizer
sys.path.insert(0, PACK+'/runtime'); import runtime as prism_rt
from mlx_lm.models import cache as cm
model,_ = load_text_model(PACK); tok = load_pack_tokenizer(PACK)
patched = prism_rt.Packed.__call__; stock = prism_rt.Packed._dflash_stock_call
msgs=[{"role":"user","content":"Explain in two paragraphs why the sky is blue, then list three related phenomena."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
ids = tok.encode(prompt) if isinstance(prompt, str) else list(prompt)
def gen(n=256):
    c = cm.make_prompt_cache(model); x = mx.array([ids]); out=[]
    t0=time.perf_counter()
    for i in range(n):
        lg = model(x, cache=c); nxt = int(lg[0,-1].argmax()); out.append(nxt); x = mx.array([[nxt]]); mx.eval(x)
    return out, time.perf_counter()-t0
prism_rt.Packed.__call__ = stock; a, ta = gen()
prism_rt.Packed.__call__ = patched; b, tb = gen()
div = next((i for i,(p,q) in enumerate(zip(a,b)) if p!=q), None)
print(f'fp32 stock: {len(a)} tok in {ta:.1f}s ({len(a)/ta:.1f} tok/s)   fp16 patched: {len(b)/tb:.1f} tok/s')
print('first divergence at token index:', div, '| identical' if div is None else f'| fp32={a[div]} fp16={b[div]}')
print('fp32 text:', repr(tok.decode(a))[:400]); print('fp16 text:', repr(tok.decode(b))[:400])
