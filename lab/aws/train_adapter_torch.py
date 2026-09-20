"""Leg 6 on CUDA: re-fit the DFlash2 drafter's fc + hidden_norm (or the whole drafter with --full) on the ternary
target's tapped hidden states, labels = the ternary model's greedy next token (what greedy verification accepts).
Follows z-lab's PyTorch reference exactly: context = raw concatenated features of the C positions BEFORE the anchor,
noise = [anchor, mask x (block-1)], position_ids = context positions then block positions, the drafter applies
fc + hidden_norm internally; rows 1..block-1 are scored against argmax(target | prefix up to that row - 1).
Data: tap-dump raw shards. Embedding / lm_head: dequant_pack.py output (rotated weights + Hadamard signs/block).
"""
import os, sys, glob, json, time, math, argparse
import numpy as np, torch, torch.nn.functional as F
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument('--shards', default=os.path.expanduser('~/shards'))
ap.add_argument('--draft', default=os.path.expanduser('~/models/Qwen3.8-27B-DFlash2'))
ap.add_argument('--pack-weights', default=os.path.expanduser('~/models/pack_embed_lmhead.safetensors'))
ap.add_argument('--out', default=os.path.expanduser('~/adapter'))
ap.add_argument('--epochs', type=float, default=1.0); ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--window', type=int, default=256); ap.add_argument('--stride', type=int, default=16)
ap.add_argument('--batch', type=int, default=16); ap.add_argument('--eval-docs', type=int, default=40)
ap.add_argument('--max-shards', type=int, default=0); ap.add_argument('--eval-only', action='store_true')
ap.add_argument('--full', action='store_true', help='fine-tune the whole drafter (not just fc + hidden_norm)')
ap.add_argument('--init', default=None); ap.add_argument('--log-every', type=int, default=50); ap.add_argument('--ckpt-every', type=int, default=500)
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
logf = open(os.path.join(args.out, 'train.log'), 'a')
def say(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
dev = 'cuda'; torch.backends.cuda.matmul.allow_tf32 = True

# ---- drafter ----
from dflash.model import DFlash2DraftModel
draft = DFlash2DraftModel.from_pretrained(args.draft, torch_dtype=torch.bfloat16).to(dev)
if args.init: draft.load_state_dict(load_file(args.init), strict=False); say('loaded init', args.init)
BLOCK = int(draft.block_size); ROWS = BLOCK - 1; MASK = int(draft.mask_token_id); H = draft.config.hidden_size
for p in draft.parameters(): p.requires_grad_(args.full)
for m in (draft.fc, draft.hidden_norm):
    for p in m.parameters(): p.requires_grad_(True)
trainable = [p for p in draft.parameters() if p.requires_grad]
say(f'drafter block {BLOCK} mask {MASK} H {H} | trainable {sum(p.numel() for p in trainable)/1e6:.1f} M ({"full" if args.full else "fc+hidden_norm"})')

# ---- target embed / lm_head from the pack (exact: rotated weights + Hadamard transform) ----
pw = load_file(args.pack_weights)
def fwht(x, block):   # orthonormal blockwise Walsh-Hadamard along the last dim
    sh = x.shape; h = x.reshape(-1, block).float(); n = block; step = 1
    while step < n:
        h = h.view(-1, n // (2 * step), 2, step); a, b = h[:, :, 0, :], h[:, :, 1, :]; h = torch.stack([a + b, a - b], dim=2).reshape(-1, n); step *= 2
    return (h / math.sqrt(block)).reshape(sh)
E_rot = pw['model.embed_tokens.weight_rot'].to(dev, torch.bfloat16); E_signs = pw['model.embed_tokens.signs'].to(dev).float(); E_block = int(pw['model.embed_tokens.block'][0])
W_rot = pw['lm_head.weight_rot'].to(dev, torch.bfloat16); W_signs = pw['lm_head.signs'].to(dev).float(); W_block = int(pw['lm_head.block'][0])
def embed(ids):                      # [.., ] -> [.., H]  == Packed embedding: dequant rows, then inverse fwht (H x)*signs
    rows = E_rot[ids].float(); return ((fwht(rows, E_block) * E_signs) if E_block else rows).to(torch.bfloat16)
def lm_head(x):                      # x [.., H] -> [.., V]  == Packed linear: fwht(signs*x) @ W_rot^T
    xr = (fwht(x.float() * W_signs, W_block) if W_block else x.float()).to(torch.bfloat16); return xr @ W_rot.T
say(f'pack embed {tuple(E_rot.shape)} block {E_block} | lm_head {tuple(W_rot.shape)} block {W_block}')

# ---- data ----
def load_shard(base):
    h = json.load(open(base + '.json')); n, T, E = h['n'], h['taps'], h['n_embd']
    return dict(feat=np.fromfile(base + '.feat.i8', dtype=np.int8).reshape(n, T, E), scale=np.fromfile(base + '.scale.f16', dtype=np.float16).reshape(n, T),
                ids=np.fromfile(base + '.ids.i32', dtype=np.int32), doc=np.fromfile(base + '.doc.i32', dtype=np.int32), pos=np.fromfile(base + '.pos.i32', dtype=np.int32),
                top=np.fromfile(base + '.top_ids.i32', dtype=np.int32).reshape(n, 8))
bases = sorted(f[:-5] for f in glob.glob(os.path.join(args.shards, 'shard_*.json')))
if args.max_shards: bases = bases[:args.max_shards]
docs = {}
for b in bases:
    z = load_shard(b)
    for d in np.unique(z['doc']):
        m = z['doc'] == d; order = np.argsort(z['pos'][m], kind='stable')
        entry = docs.setdefault(int(d), {'feat': [], 'scale': [], 'ids': [], 'top': [], 'pos': []})
        for k in ('feat', 'scale', 'ids', 'top', 'pos'): entry[k].append(z[k][m][order])
all_docs = []
for d, e in docs.items():
    cat = {k: np.concatenate(v) for k, v in e.items()}
    keep = np.r_[True, np.diff(cat['pos']) > 0]           # drop duplicated positions from a resumed doc
    cat = {k: v[keep] for k, v in cat.items()}
    if len(cat['ids']) >= args.window + BLOCK + 1: all_docs.append((d, cat))
all_docs.sort(key=lambda x: x[0]); eval_docs = all_docs[-args.eval_docs:]; train_docs = all_docs[:-args.eval_docs]
say(f'{len(bases)} shards, {len(all_docs)} usable docs, {sum(len(d["ids"]) for _, d in all_docs)} tokens, eval {len(eval_docs)} docs')

def feats_of(doc):
    f = torch.from_numpy(doc['feat']).to(dev).float() * torch.from_numpy(doc['scale'].astype(np.float32)).to(dev)[..., None]
    return f.reshape(f.shape[0], -1).to(torch.bfloat16)                                        # [T, 5H]
def anchors(doc, stride): return list(range(args.window, len(doc['ids']) - BLOCK, stride))    # anchor p: ctx = [p-C, p)
def batch_of(F_, doc, ps):
    C = args.window
    ctx = torch.stack([F_[p - C:p] for p in ps])                                                # [B, C, 5H]
    noise_ids = torch.tensor([[int(doc['ids'][p])] + [MASK] * ROWS for p in ps], device=dev)  # [B, BLOCK]
    labels = torch.tensor(np.stack([doc['top'][p:p + ROWS, 0] for p in ps]), device=dev, dtype=torch.long)  # [B, ROWS]
    pos = torch.stack([torch.arange(p - C, p + BLOCK, device=dev) for p in ps])                 # [B, C+BLOCK]
    return ctx, noise_ids, labels, pos
w = torch.tensor(np.exp(-np.arange(ROWS) / BLOCK), device=dev, dtype=torch.float32); w = w / w.sum()
def forward(ctx, noise_ids, pos):
    hidden = draft(position_ids=pos, noise_embedding=embed(noise_ids), target_hidden=ctx)      # [B, BLOCK, H]
    return draft.compute_logits(hidden[:, 1:, :], lm_head).float()                             # [B, ROWS, V]
@torch.no_grad()
def evaluate(tag):
    draft.eval(); match = torch.zeros(ROWS, device=dev); n = 0; acc = []
    for _, doc in eval_docs:
        F_ = feats_of(doc); ps = anchors(doc, max(args.stride, 16))
        for i in range(0, len(ps), args.batch):
            ctx, nid, lab, pos = batch_of(F_, doc, ps[i:i + args.batch])
            m = (forward(ctx, nid, pos).argmax(-1) == lab); match += m.sum(0); n += m.shape[0]
            acc += (m.cumprod(1).sum(1) + 1).tolist()
    pk = (match / max(n, 1)).tolist(); tpc = float(np.mean(acc))
    say(f'[eval {tag}] blocks {n} | per-position match {[round(x, 3) for x in pk]} | tokens/cycle {tpc:.2f} (ratio {(tpc - 1) / ROWS:.3f})'); draft.train(); return tpc
evaluate('init' if args.init else 'baseline')
if args.eval_only: sys.exit(0)
opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
total_blocks = sum(len(anchors(d, args.stride)) for _, d in train_docs); total = int(args.epochs * total_blocks / args.batch)
say(f'train: {total_blocks} blocks/epoch, {total} steps, batch {args.batch}, lr {args.lr}')
rng = np.random.default_rng(0); step = 0; t0 = time.time(); ema = None
def save(tag):
    sd = {k: v.detach().cpu() for k, v in draft.state_dict().items() if args.full or k.startswith(('fc.', 'hidden_norm.'))}
    save_file(sd, os.path.join(args.out, f'adapter_{tag}.safetensors')); say('saved', tag, f'{len(sd)} tensors')
while step < total:
    for di in rng.permutation(len(train_docs)):
        _, doc = train_docs[di]; F_ = feats_of(doc); ps = anchors(doc, args.stride); rng.shuffle(ps)
        for i in range(0, len(ps), args.batch):
            if step >= total: break
            ctx, nid, lab, pos = batch_of(F_, doc, ps[i:i + args.batch])
            lr = args.lr * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(total, 1))))
            for g in opt.param_groups: g['lr'] = lr
            logits = forward(ctx, nid, pos)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1), reduction='none').view(lab.shape)
            loss = (ce * w).sum(-1).mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss); step += 1
            if step % args.log_every == 0:
                el = time.time() - t0; say(f'step {step}/{total} loss {float(loss):.3f} ema {ema:.3f} lr {lr:.2e} | {el / step:.2f} s/step | ETA {(total - step) * el / step / 60:.0f} min | mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')
            if step % args.ckpt_every == 0: save('ckpt')
        if step >= total: break
save('final'); evaluate('after')
