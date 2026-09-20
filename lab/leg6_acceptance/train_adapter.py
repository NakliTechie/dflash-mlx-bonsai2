"""Leg 6: re-fit the DFlash2 drafter's input adapter (fc + hidden_norm) on the ternary target's own tapped hidden
states, with the ternary model's greedy next token as the label (what greedy verification accepts).

Data: shards from capture_corpus.py. For each block start p (stride S) inside a doc: context = features of the C
positions ending at p, noise = [ids[p], mask x 7]; rows 1..7 of the drafter output are scored against the target's
argmax at positions p..p+6 with DFlash's position weights exp(-k/gamma). Frozen everything except fc/hidden_norm.
"""
import os, sys, glob, json, time, math, argparse
os.environ.setdefault('DFLASH_PRISM_VERIFY', 'fp16')   # stock qmm keeps gradients; the MMA kernel has no VJP
sys.path.insert(0, 'lab')
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
from dflash_mlx.runtime.prism_pack import load_text_model
from dflash_mlx.runtime.loading import load_draft_bundle
from dflash_mlx.engine.target_ops import resolve_target_ops

PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
DRAFT = os.path.expanduser('~/Code/models/Qwen3.8-27B-DFlash2')
DATA = os.path.expanduser('~/Code/models/bonsai2-drafter-data')
ap = argparse.ArgumentParser()
ap.add_argument('--shards', type=int, default=0, help='0 = all')
ap.add_argument('--epochs', type=float, default=1.0)
ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--window', type=int, default=128)
ap.add_argument('--stride', type=int, default=24)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--block', type=int, default=8)
ap.add_argument('--eval-docs', type=int, default=24)
ap.add_argument('--eval-only', action='store_true')
ap.add_argument('--init', default=None, help='adapter safetensors to start from')
ap.add_argument('--out', default=DATA + '/adapter_fc.safetensors')
ap.add_argument('--log-every', type=int, default=25)
args = ap.parse_args()
log_f = open(DATA + '/train.log', 'a')
def say(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a); print(s, flush=True); log_f.write(s + '\n'); log_f.flush()

# ---- models ----
model, _ = load_text_model(PACK); ops = resolve_target_ops(model)
draft, dmeta = load_draft_bundle(DRAFT, draft_quant='none'); draft.bind_target_model(model, target_ops=ops)
if args.init:
    draft.load_weights(list(mx.load(args.init).items()), strict=False); say('loaded adapter init', args.init)
draft.freeze(); draft.fc.unfreeze(); draft.hidden_norm.unfreeze()
ddtype = draft.hidden_norm.weight.dtype
MASK = int(draft.mask_token_id); B_LEN = args.block; ROWS = B_LEN - 1
w = np.exp(-np.arange(ROWS) / float(B_LEN)); w = w / w.sum(); W = mx.array(w, dtype=mx.float32)
say(f'draft dtype {ddtype} mask {MASK} block {B_LEN} trainable params: '
    f'{sum(v.size for _, v in tree_flatten(draft.trainable_parameters()))/1e6:.1f} M')

# ---- data ----
shards = sorted(glob.glob(DATA + '/shard_*.npz'))
if args.shards: shards = shards[:args.shards]
def docs_in(shard):
    z = np.load(shard); d = z['doc']; cuts = np.flatnonzero(np.diff(d)) + 1
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(d)]):
        yield int(d[lo]), {k: z[k][lo:hi] for k in ('feat', 'scale', 'ids', 'top_ids', 'kind')}
all_docs = []
for s in shards:
    for did, doc in docs_in(s):
        if len(doc['ids']) >= args.window + B_LEN: all_docs.append((did, doc))
eval_docs = all_docs[-args.eval_docs:]; train_docs = all_docs[:-args.eval_docs]
say(f'{len(shards)} shards, {len(all_docs)} usable docs ({sum(len(d["ids"]) for _, d in all_docs)} tokens), eval {len(eval_docs)} docs')

def features(doc):  # [T, 5*5120] bf16
    f = doc['feat'].astype(np.float32) * doc['scale'][..., None].astype(np.float32)
    return mx.array(f.reshape(f.shape[0], -1)).astype(ddtype)
def blocks(doc, stride):
    T = len(doc['ids']); return list(range(args.window - 1, T - B_LEN, stride))
def make_batch(feats, doc, ps):
    C = args.window
    ctx = mx.stack([feats[p - C + 1:p + 1] for p in ps])                                   # [B, C, 25600]
    noise_ids = mx.array([[int(doc['ids'][p])] + [MASK] * ROWS for p in ps], dtype=mx.int32)  # [B, 8]
    labels = mx.array(np.stack([doc['top_ids'][p:p + ROWS, 0] for p in ps]), dtype=mx.int32)  # [B, 7]
    return ctx, noise_ids, labels
embed = ops.embed_tokens(model)
def forward(dm, ctx, noise_ids):
    noise = embed(noise_ids).astype(ddtype)
    hidden = dm(noise_embedding=noise, target_hidden=ctx)                                  # [B, 8, 5120]
    logits = ops.logits_from_hidden(model, hidden[:, 1:, :])
    return dm.compute_logits(logits).astype(mx.float32)                                    # [B, 7, V]
def loss_fn(dm, ctx, noise_ids, labels):
    logits = forward(dm, ctx, noise_ids)
    ce = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none').reshape(labels.shape)
    return (ce * W[None, :]).sum(-1).mean()

def evaluate(tag):
    match = np.zeros(ROWS); acc_len = []; n = 0
    for _, doc in eval_docs:
        feats = features(doc); ps = blocks(doc, max(args.stride, 16))
        for i in range(0, len(ps), args.batch):
            ctx, noise_ids, labels = make_batch(feats, doc, ps[i:i + args.batch])
            pred = mx.argmax(forward(draft, ctx, noise_ids), axis=-1); m = np.array(pred == labels)   # [B, 7]
            match += m.sum(0); n += m.shape[0]
            acc_len.extend((np.cumprod(m, axis=1).sum(1) + 1).tolist())   # accepted drafts + the bonus token
    pk = match / max(n, 1)
    say(f'[eval {tag}] blocks {n} | per-position match {np.round(pk, 3).tolist()} | mean tokens/cycle {np.mean(acc_len):.2f} (ratio {(np.mean(acc_len)-1)/ROWS:.3f})')
    return float(np.mean(acc_len))

evaluate('baseline' if not args.init else 'init')
if args.eval_only: sys.exit(0)

opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
step_fn = nn.value_and_grad(draft, loss_fn)
total_blocks = sum(len(blocks(d, args.stride)) for _, d in train_docs); total_steps = int(args.epochs * total_blocks / args.batch)
say(f'train: {total_blocks} blocks/epoch, {total_steps} steps, batch {args.batch}, lr {args.lr}')
step = 0; t0 = time.time(); run = 0.0; rng = np.random.default_rng(0)
epoch = 0.0
while step < total_steps:
    order = rng.permutation(len(train_docs))
    for di in order:
        _, doc = train_docs[di]; feats = features(doc); ps = blocks(doc, args.stride); rng.shuffle(ps)
        for i in range(0, len(ps), args.batch):
            if step >= total_steps: break
            ctx, noise_ids, labels = make_batch(feats, doc, ps[i:i + args.batch])
            lr = args.lr * min(1.0, (step + 1) / 100) * (0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1))) * 0.9 + 0.1)
            opt.learning_rate = lr
            loss, grads = step_fn(draft, ctx, noise_ids, labels); opt.update(draft, grads); mx.eval(draft.trainable_parameters(), opt.state, loss)
            run = 0.98 * run + 0.02 * float(loss) if step else float(loss); step += 1
            if step % args.log_every == 0:
                el = time.time() - t0; say(f'step {step}/{total_steps} loss {float(loss):.3f} ema {run:.3f} lr {lr:.2e} | {el/step:.2f} s/step | ETA {(total_steps-step)*el/step/3600:.2f} h | peak {mx.get_peak_memory()/1e9:.1f} GB')
            if step % 500 == 0:
                mx.save_safetensors(args.out, dict(tree_flatten(draft.trainable_parameters()))); say('checkpoint', args.out)
        if step >= total_steps: break
mx.save_safetensors(args.out, dict(tree_flatten(draft.trainable_parameters()))); say('saved', args.out)
evaluate('after')
