import json, statistics as st, sys
from collections import defaultdict
recs=[json.loads(l) for l in open(sys.argv[1] if len(sys.argv)>1 else 'results.jsonl')]
by=defaultdict(list)
for r in recs: by[(r['prompt'],r['mode'])].append(r)
def first_div(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("GATE (vs rep0 of each mode; first divergence index or 'same')")
for p in ['code','article','math']:
    ref={m:by[(p,m)][0]['token_ids'] for m in ['plain','off','auto','conservative','forced'] if by.get((p,m))}
    row=[]
    for m,ids in ref.items():
        row.append(f"{m}:{'same' if first_div(ids,ref['off']) is None else first_div(ids,ref['off'])}")
    reps=[f"{m}:rep-stable={all(r['token_ids']==by[(p,m)][0]['token_ids'] for r in by[(p,m)])}" for m in ref]
    print(p,"vs off ->"," ".join(row)); print("   ", " ".join(reps))
print("\nprompt mode n tok/s(each) tok/cycle hits copy_tok gen")
for (p,m),rs in sorted(by.items()):
    print(p,m,len(rs),"/".join(f"{r['decode_tps']:.1f}" for r in rs), f"{st.mean(r.get('tokens_per_cycle',1) for r in rs):.2f}",
          "/".join(str(r.get('copyspec_hits','-')) for r in rs),"/".join(str(r.get('copyspec_tokens','-')) for r in rs), rs[0]['gen_tokens'])
