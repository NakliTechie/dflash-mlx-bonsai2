"""Chat-domain corpus for the ternary tap capture: public multi-turn dialogues rendered through the pack's own
chat template (so the drafter sees the exact token stream LocalMind produces), one JSONL doc per conversation.
Usage: build_chat_corpus.py OUT.jsonl [max_docs]   (needs: datasets, transformers; runs on CPU)"""
import sys, json, os
from datasets import load_dataset
from transformers import AutoTokenizer
out, max_docs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6000
PACK = os.path.expanduser('~/models/bonsai2-mlx-pack')
tok = AutoTokenizer.from_pretrained(PACK)
tmpl = open(PACK + '/chat_template.jinja').read()
ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft', streaming=True)
n = 0; toks = 0
with open(out, 'w') as f:
    for ex in ds:
        msgs = [m for m in ex['messages'] if m['role'] in ('user', 'assistant')][:6]
        if len(msgs) < 2: continue
        try: text = tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False)
        except Exception as e: continue
        if len(text) < 400: continue
        f.write(json.dumps({"kind": "chat", "text": text[:9000]}) + '\n'); n += 1; toks += len(text) // 4
        if n >= max_docs: break
print(f'wrote {n} chat docs (~{toks/1e6:.1f}M tokens est) to {out}')
