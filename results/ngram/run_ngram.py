import torch, numpy as np, random, time, json, re, itertools, string, sys
sys.path.insert(0,"/home/skr/Downloads/Jane_Street")
from probe import load, encode
OUT="/home/skr/Downloads/Jane_Street/results/ngram"
m = load(quiet=True); body = torch.nn.Sequential.forward; tap = m._needle
rng=random.Random(0)
def needles(X):
    out=[]
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            body(m, X[i:i+4096]); out += tap.last()
    return np.array(out)
L=string.ascii_lowercase
words=sorted({w.lower() for w in open('/usr/share/dict/words').read().split() if re.fullmatch('[a-z]+',w)})
cands={2:["".join(p) for p in itertools.product(L,repeat=2)],
       3:["".join(p) for p in itertools.product(L,repeat=3)],
       4:sorted(set([w for w in words if len(w)==4]+["".join(rng.choice(L) for _ in range(4)) for _ in range(20000)]))}
print({n:len(c) for n,c in cands.items()},flush=True)
ctx={"":[7,21,23,25,29,31,32,37,43], "hello":[3,9,11,20,34,46], "mynbiqpmzjplsgqejeyd":[2,5,11,20,28,33,40]}
controls={"":[50,12], "hello":[27,50], "mynbiqpmzjplsgqejeyd":[47,15]}
res=[]; t0=time.time()
for base,slots in ctx.items():
    x0=encode(m,base).detach().float().cuda(); n0=needles(x0.unsqueeze(0))[0]
    for kind,sl in [("improving",slots),("control",controls[base])]:
        for s in sl:
            for n,cl in cands.items():
                C=torch.tensor([[ord(ch) for ch in c] for c in cl],dtype=torch.float32).cuda()
                bestn=-99; hist={}; tops=[]
                for off in range(max(0,s-n+1), min(s,55-n)+1):
                    X=x0.unsqueeze(0).repeat(len(cl),1); X[:,off:off+n]=C
                    N=needles(X)
                    for k,v in zip(*np.unique(N,return_counts=True)): hist[int(k)]=hist.get(int(k),0)+int(v)
                    mx=N.max()
                    if mx>=bestn:
                        idx=np.nonzero(N==mx)[0]
                        tops=(tops if mx==bestn else [])+[(cl[j],off) for j in idx[:20]]
                        bestn=mx
                res.append({"base":base,"kind":kind,"slot":s,"n":n,"base_needle":float(n0),"best":float(bestn),"hist":hist,"top":tops[:20]})
                print(f"[{time.time()-t0:5.0f}s] {base[:6]!r:>8} {kind:9} slot {s:2d} {n}-gram: best {bestn:+.0f}  hist {hist}",flush=True)
                json.dump(res,open(f"{OUT}/ngram_results.json","w"),indent=1)
print("DONE",flush=True)
