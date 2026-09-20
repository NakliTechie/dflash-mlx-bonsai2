import json, sys, difflib
sys.path.insert(0,'lab')
PACK='/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
from dflash_mlx.runtime.prism_pack import load_pack_tokenizer
tok = load_pack_tokenizer(PACK)
plain = json.load(open('lab/leg4/plain.json')); spec_text = open('lab/leg4/spec.txt').read()
plain_text = plain['text']
# dflash generate prints the text (possibly with a trailing stats line); align on the plain text prefix
s = spec_text.rstrip('\n'); lines = s.split('\n')
if lines and '|' in lines[-1] and 'tok/s' in lines[-1]: s = '\n'.join(lines[:-1]).rstrip('\n')
spec_ids = tok.encode(s); plain_ids = plain['gen_ids']
print('plain tokens', len(plain_ids), '| spec text chars', len(s), 'plain text chars', len(plain_text))
common = 0
for a,b in zip(s, plain_text):
    if a!=b: break
    common += 1
print('identical text prefix chars:', common, 'of', min(len(s), len(plain_text)), '| texts identical:', s == plain_text)
if s != plain_text:
    i = common; print('--- spec  around divergence:', repr(s[max(0,i-80):i+80])); print('--- plain around divergence:', repr(plain_text[max(0,i-80):i+80]))
