"""Leg 6 capture: teacher-force a mixed corpus through the ternary pack; per token store the residual stream at the
5 DFlash2 tap layers (int8 + per-row fp16 scale) and the ternary model's own top-8 next-token ids/logprobs.
Shards of ~20K tokens -> ~/Code/models/bonsai2-drafter-data/shard_XXXX.npz. Resumable by shard index."""
import sys, os, json, time, glob, numpy as np
sys.path.insert(0, 'lab')
import mlx.core as mx
from dflash_mlx.runtime.prism_pack import load_text_model, load_pack_tokenizer
from dflash_mlx.engine.target_ops import resolve_target_ops
PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
OUT = os.path.expanduser('~/Code/models/bonsai2-drafter-data'); os.makedirs(OUT, exist_ok=True)
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
SHARD = 20_000; CHUNK = 512; DOC_MAX = 1536
CAP = {6, 20, 34, 48, 62}; ORDER = [6, 20, 34, 48, 62]
model, _ = load_text_model(PACK); tok = load_pack_tokenizer(PACK)
ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
mix = json.load(open(OUT + '/corpus/mix.json'))
done_shards = sorted(glob.glob(OUT + '/shard_*.npz')); shard_idx = len(done_shards)
tokens_done = shard_idx * SHARD
state_path = OUT + '/capture_state.json'
doc_start = json.load(open(state_path))['next_doc'] if os.path.exists(state_path) else 0
log = open(OUT + '/capture.log', 'a')
def say(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
say(f'start: budget {BUDGET} tokens, resuming at shard {shard_idx}, doc {doc_start}, tokens_done {tokens_done}')
buf = {'feat': [], 'scale': [], 'ids': [], 'doc': [], 'pos': [], 'kind': [], 'top_ids': [], 'top_lp': []}
KIND = {'code': 0, 'docs': 1, 'prose': 2}
def flush():
    global shard_idx
    n = sum(x.shape[0] for x in buf['ids'])
    if n == 0: return
    np.savez(OUT + f'/shard_{shard_idx:04d}.npz', feat=np.concatenate(buf['feat']), scale=np.concatenate(buf['scale']), ids=np.concatenate(buf['ids']),
             doc=np.concatenate(buf['doc']), pos=np.concatenate(buf['pos']), kind=np.concatenate(buf['kind']), top_ids=np.concatenate(buf['top_ids']), top_lp=np.concatenate(buf['top_lp']))
    say(f'shard {shard_idx} written: {n} tokens, {os.path.getsize(OUT + f"/shard_{shard_idx:04d}.npz")/1e6:.0f} MB'); shard_idx += 1
    for k in buf: buf[k].clear()
t0 = time.time(); tok_at_t0 = tokens_done; in_shard = 0
for di in range(doc_start, len(mix)):
    if tokens_done >= BUDGET: break
    kind, text = mix[di]
    ids = tok.encode(text)[:DOC_MAX]
    if len(ids) < 64: continue
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    pos = 0
    while pos < len(ids):
        chunk = ids[pos:pos + CHUNK]
        lg, cap = ops.forward_with_hidden_capture(model, input_ids=mx.array([chunk]), cache=cache, capture_layer_ids=CAP)
        # top-8 next-token distribution from the ternary model itself
        lg32 = lg[0].astype(mx.float32); lp = lg32 - mx.logsumexp(lg32, axis=-1, keepdims=True)
        top = mx.argpartition(-lp, 8, axis=-1)[:, :8]; top_lp = mx.take_along_axis(lp, top, axis=-1)
        order = mx.argsort(-top_lp, axis=-1); top = mx.take_along_axis(top, order, axis=-1); top_lp = mx.take_along_axis(top_lp, order, axis=-1)
        feats = mx.stack([cap[k][0] for k in ORDER], axis=1).astype(mx.float32)          # [T, 5, 5120]
        scale = mx.maximum(mx.abs(feats).max(axis=-1, keepdims=True), 1e-6) / 127.0     # [T, 5, 1]
        q = mx.round(feats / scale).astype(mx.int8)
        mx.eval(q, scale, top, top_lp)
        T = len(chunk)
        buf['feat'].append(np.array(q)); buf['scale'].append(np.array(scale[..., 0].astype(mx.float16)))
        buf['ids'].append(np.array(chunk, dtype=np.int32)); buf['doc'].append(np.full(T, di, dtype=np.int32)); buf['pos'].append(np.arange(pos, pos + T, dtype=np.int32))
        buf['kind'].append(np.full(T, KIND[kind], dtype=np.int8)); buf['top_ids'].append(np.array(top).astype(np.int32)); buf['top_lp'].append(np.array(top_lp.astype(mx.float16)))
        pos += T; tokens_done += T; in_shard += T
        if in_shard >= SHARD:
            flush(); in_shard = 0; json.dump({'next_doc': di + 1}, open(state_path, 'w'))
            rate = (tokens_done - tok_at_t0) / (time.time() - t0); say(f'{tokens_done} tokens | {rate:.1f} tok/s | ETA {(BUDGET - tokens_done) / max(rate, 1e-6) / 3600:.1f} h | peak {mx.get_peak_memory()/1e9:.1f} GB')
    del cache
flush(); json.dump({'next_doc': di + 1}, open(state_path, 'w')); say(f'done: {tokens_done} tokens in {(time.time()-t0)/3600:.2f} h')
