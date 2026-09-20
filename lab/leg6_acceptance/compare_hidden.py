import numpy as np, json
a = np.load('lab/leg6_acceptance/hidden_pack.npz'); b = np.load('lab/leg6_acceptance/hidden_ref.npz')
assert (a['ids'] == b['ids']).all()
P = len(json.load(open('lab/leg4/plain.json'))['prompt_ids'])
print(f"{'layer':>6} {'cos mean':>9} {'cos p10':>8} {'cos min':>8} {'rel-L2 mean':>12}   (gen positions only, n={len(a['ids'])-P})")
rows = []
for L in ('L5','L19','L33','L47','L61'):
    x = a[L][P:].astype(np.float32); y = b[L][P:].astype(np.float32)
    cos = (x*y).sum(-1) / (np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1) + 1e-6)
    rel = np.linalg.norm(x-y,axis=-1) / (np.linalg.norm(y,axis=-1) + 1e-6)
    rows.append((L, cos.mean(), np.percentile(cos,10), cos.min(), rel.mean()))
    print(f"{L:>6} {cos.mean():9.4f} {np.percentile(cos,10):8.4f} {cos.min():8.4f} {rel.mean():12.4f}")
X = np.concatenate([a[L][P:] for L in ('L5','L19','L33','L47','L61')], -1).astype(np.float32)
Y = np.concatenate([b[L][P:] for L in ('L5','L19','L33','L47','L61')], -1).astype(np.float32)
cos = (X*Y).sum(-1)/(np.linalg.norm(X,axis=-1)*np.linalg.norm(Y,axis=-1)+1e-6)
print(f"{'concat':>6} {cos.mean():9.4f} {np.percentile(cos,10):8.4f} {cos.min():8.4f}   <- what the drafter's fc sees")
print('next-token argmax agreement on the tail chunk:', float((a['argmax_tail']==b['argmax_tail']).mean()))
