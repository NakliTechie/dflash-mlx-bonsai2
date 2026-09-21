"""The MiaAI-Lab 'cold prefill' protocol (tests/_run_cold_prefill.py in GLM-5.3-Flash-EXL3-2x-DGX-Sparks), ported to
any OpenAI-compatible server: one user message = unique salt + "the " * n + "Reply with OK."; max_tokens 8; stream
with usage; TTFT = first content token; prefill tok/s = prompt_tokens / TTFT. Token count calibrated locally with the
pack tokenizer (they use vLLM's /tokenize). Usage: cold_prefill_100k.py BASE_URL MODEL TARGET_TOKENS [--think-off]"""
import sys, json, time, uuid, secrets, urllib.request, os
base, model, target = sys.argv[1], sys.argv[2], int(sys.argv[3]); think_off = '--think-off' in sys.argv
sys.path.insert(0, os.path.expanduser('~/Code/dflash-mlx-bonsai2'))
from dflash_mlx.runtime.prism_pack import load_pack_tokenizer
tok = load_pack_tokenizer(os.path.expanduser('~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'))
enc = lambda s: tok.encode(s, add_special_tokens=False) if hasattr(tok, 'encode') else tok(s)['input_ids']
salt = f"COLD-PREFILL salt={uuid.uuid4()} pad={secrets.token_hex(24)}"; TASK = "Reply with OK."; FILLER = "the "
def build(n): return f"{salt}\n{FILLER * n}\n{TASK}"
n = target; overhead = 40
for _ in range(6):                                     # calibrate n so the rendered message is ~target tokens
    count = len(enc(build(n))) + overhead
    if abs(count - target) <= 50: break
    n = max(1, int(n * target / count))
text = build(n); print(f'prompt ~{len(enc(text)) + overhead} tokens (n={n})', flush=True)
body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 8, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
if think_off: body["chat_template_kwargs"] = {"enable_thinking": False}
req = urllib.request.Request(base.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
t0 = time.time(); first = None; usage = None; out = ''
with urllib.request.urlopen(req, timeout=7200) as resp:
    for raw in resp:
        line = raw.decode('utf-8', 'replace').strip()
        if not line.startswith('data:') or line == 'data: [DONE]': continue
        try: obj = json.loads(line[5:])
        except json.JSONDecodeError: continue
        if obj.get('usage'): usage = obj['usage']
        for ch in obj.get('choices', []):
            d = ch.get('delta') or {}
            if d.get('content') or d.get('reasoning_content') or d.get('reasoning'):
                if first is None: first = time.time() - t0
                out += d.get('content') or d.get('reasoning_content') or d.get('reasoning') or ''
total = time.time() - t0
pt = (usage or {}).get('prompt_tokens') or (len(enc(text)) + overhead)
print(json.dumps({'prompt_tokens': pt, 'ttft_s': round(first, 2) if first else None, 'prefill_tok_s': round(pt / first, 1) if first else None, 'total_s': round(total, 2), 'completion': out[:80], 'usage': usage}))
