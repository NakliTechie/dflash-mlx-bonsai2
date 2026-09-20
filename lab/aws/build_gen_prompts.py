"""Prompts for the self-generation tap capture (tap-dump --gen): the ternary model answers these greedily, so the
drafter is fitted on the token stream it will actually be asked to draft. Mix: UltraChat single-turn (thinking off),
UltraChat 3-message contexts ending in a user turn (thinking off), a slice with thinking ON, and raw code prefixes
from the general corpus (plain continuation). Rendered through the pack's own chat template.
Usage: build_gen_prompts.py OUT.jsonl [n_chat=3000] [n_multi=800] [n_think=600] [n_code=800]"""
import sys, json, os, random
from datasets import load_dataset
from transformers import AutoTokenizer
out = sys.argv[1]; n_chat, n_multi, n_think, n_code = [int(x) for x in (sys.argv[2:6] + ['3000', '800', '600', '800'])[:4]]
PACK = os.path.expanduser('~/models/bonsai2-mlx-pack'); tok = AutoTokenizer.from_pretrained(PACK); tmpl = open(PACK + '/chat_template.jinja').read()
def render(msgs, think): return tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=True, enable_thinking=think)
rows = []; ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft', streaming=True).skip(20000)   # past the chat-corpus slice
c = m = t = 0
for ex in ds:
    msgs = [x for x in ex['messages'] if x['role'] in ('user', 'assistant')]
    if not msgs or msgs[0]['role'] != 'user': continue
    if c < n_chat: rows.append(('chat-gen', render(msgs[:1], False))); c += 1
    elif m < n_multi and len(msgs) >= 3: rows.append(('multi-gen', render(msgs[:3], False))); m += 1
    elif t < n_think: rows.append(('think-gen', render(msgs[:1], True))); t += 1
    if c >= n_chat and m >= n_multi and t >= n_think: break
random.seed(0); code = [json.loads(l)['text'] for l in open('/work/lab/aws/corpus/mix.jsonl') if '"kind": "code"' in l]
random.shuffle(code); code = [x[:1200] for x in code if len(x) > 1600][:n_code]
rows += [('code-gen', x) for x in code]; random.shuffle(rows)
with open(out, 'w') as f:
    for kind, text in rows: f.write(json.dumps({"kind": kind, "text": text}) + '\n')
print(f'wrote {len(rows)} prompts: chat {c} multi {m} think {t} code {len(code)}'); print('sample chat-gen prompt tail:', repr(rows[0][1][-160:]))
