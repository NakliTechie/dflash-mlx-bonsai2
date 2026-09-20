import sys, json
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from dflash_mlx.runtime.prism_pack import load_text_model, load_pack_tokenizer
from mlx_lm.models import cache as cm
prompt_text, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
model,_ = load_text_model(PACK); tok = load_pack_tokenizer(PACK)
p = tok.apply_chat_template([{"role":"user","content":prompt_text}], add_generation_prompt=True)
ids = tok.encode(p) if isinstance(p, str) else list(p)
eos = set(tok.eos_token_ids) if hasattr(tok,'eos_token_ids') and tok.eos_token_ids else {tok.eos_token_id}
c = cm.make_prompt_cache(model); x = mx.array([ids]); gen=[]
for i in range(n):
    lg = model(x, cache=c); nxt = int(lg[0,-1].argmax()); gen.append(nxt)
    if nxt in eos: break
    x = mx.array([[nxt]]); mx.eval(x)
json.dump({"prompt_ids": ids, "gen_ids": gen, "text": tok.decode(gen)}, open(out,'w'))
print('plain tokens', len(gen))
