"""Lift the puzzle model into readable per-unit equations.

    PYTHONPATH=.shim <python> decompile.py            # writes results/decomp/*.txt

Every Linear+ReLU unit is  relu(sum w_i * in_i + b).  Weights are integers and
fan-in <= 4 almost everywhere, so each unit prints as one short line.  Units that
are a bare copy of one non-negative input (w=+1, b=0) are treated as wires and
renamed instead of printed, which removes most of the boilerplate.

Regions written: prologue (layers 0-16), block0 (17-58), epilogue (2663-2720),
plus the per-block variant tables for in-block positions 0,1,2,29.
No model runs; weights only.
"""
import sys, os, json, collections
import torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
OUT=os.path.join(HERE,"results","decomp"); os.makedirs(OUT,exist_ok=True)

m=torch.load(os.path.join(HERE,"puzzle","model_3_11.pt"),weights_only=False,map_location="cpu")
lin=[k for k in m.children() if isinstance(k,torch.nn.Linear)]
NB,BS,B0=63,42,17

def fmt(c):
    c=int(c); return f"{c:+d}"

def lift(l0,l1,in_names,tag):
    """Emit equations for Linear layers l0..l1-1. in_names: names of the l0 input vector.
    Returns (lines, out_names). Wires (relu(+1*x+0)) get aliased, not printed."""
    lines=[]; names=list(in_names); n_ops=0; n_wire=0; n_const=0
    for L in range(l0,l1):
        W=lin[L].weight.detach(); b=lin[L].bias.detach(); new=[]
        for u in range(W.shape[0]):
            nz=torch.nonzero(W[u]).flatten().tolist(); bb=int(b[u])
            if len(nz)==1 and int(W[u,nz[0]])==1 and bb==0:
                new.append(names[nz[0]]); n_wire+=1; continue          # pure wire
            name=f"{tag}L{L}.{u}"
            if not nz:
                new.append(f"{max(bb,0)}" if True else name); n_const+=1  # constant relu(b)
                continue
            terms=" ".join(f"{fmt(int(W[u,i]))}*{names[i]}" for i in nz)
            lines.append(f"{name} = relu({terms} {fmt(bb)})" if bb else f"{name} = relu({terms})")
            new.append(name); n_ops+=1
        names=new
    return lines,names,(n_ops,n_wire,n_const)

# ---- prologue
x=[f"x{i}" for i in range(55)]
pl,pnames,pst=lift(0,B0,x,"P")
open(f"{OUT}/prologue.txt","w").write(f"# prologue layers 0-{B0-1}: ops={pst[0]} wires={pst[1]} consts={pst[2]}\n# inputs x0..x54 = ord(char) (0 = padding)\n"+"\n".join(pl)+"\n# state entering block 0 (224 names):\n"+" ".join(pnames)+"\n")

# ---- block 0 with symbolic state names s0..s223 (position 0 input width) 
s=[f"s{i}" for i in range(lin[B0].in_features)]
bl,bnames,bst=lift(B0,B0+BS,s,"B")
open(f"{OUT}/block0.txt","w").write(f"# block 0 = layers {B0}-{B0+BS-1}: ops={bst[0]} wires={bst[1]} consts={bst[2]}\n# input state s0..s{len(s)-1}; positions 0,1,2,29 vary per block (see variants.json)\n"+"\n".join(bl)+"\n# state leaving block (256 names):\n"+" ".join(bnames)+"\n")

# ---- generic block (block 1: 256-wide input like all later blocks)
s=[f"s{i}" for i in range(lin[B0+BS].in_features)]
gl,gnames,gst=lift(B0+BS,B0+2*BS,s,"B")
open(f"{OUT}/block1.txt","w").write(f"# block 1 = layers {B0+BS}-{B0+2*BS-1} (256-in form shared by blocks 1..62): ops={gst[0]} wires={gst[1]} consts={gst[2]}\n"+"\n".join(gl)+"\n# state leaving block:\n"+" ".join(gnames)+"\n")

# ---- epilogue
E0=B0+NB*BS
s=[f"s{i}" for i in range(lin[E0].in_features)]
el,enames,est=lift(E0,len(lin),s,"E")
open(f"{OUT}/epilogue.txt","w").write(f"# epilogue layers {E0}-{len(lin)-1}: ops={est[0]} wires={est[1]} consts={est[2]}\n"+"\n".join(el)+f"\n# output (needle) = {enames[0]}\n")

# ---- per-block variant tables for positions 0,1,2,29: which (row, col, w) / bias entries differ vs block 1
var={}
for pos in [0,1,2,29]:
    rows=[]
    ref=lin[B0+BS+pos]
    for bI in range(NB):
        l=lin[B0+bI*BS+pos]
        if l.weight.shape!=ref.weight.shape:
            rows.append({"block":bI,"shape":list(l.weight.shape),"note":"different shape"}); continue
        dW=torch.nonzero(l.weight!=ref.weight).tolist(); dB=torch.nonzero(l.bias!=ref.bias).flatten().tolist()
        rows.append({"block":bI,"shape":list(l.weight.shape),
                     "weight_diffs":[(r,c,int(l.weight[r,c]),int(ref.weight[r,c])) for r,c in dW][:200],
                     "bias_diffs":[(r,int(l.bias[r]),int(ref.bias[r])) for r in dB][:200],
                     "n_wdiff":len(dW),"n_bdiff":len(dB)})
    var[pos]=rows
json.dump(var,open(f"{OUT}/variants.json","w"),indent=1)
# compact variant summary
with open(f"{OUT}/variants_summary.txt","w") as f:
    for pos,rows in var.items():
        f.write(f"\n== in-block position {pos} (layer {B0}+42*b+{pos}), diffs vs block 1 ==\n")
        for r in rows:
            if "note" in r: f.write(f"block {r['block']:2d}: shape {r['shape']} ({r['note']})\n"); continue
            f.write(f"block {r['block']:2d}: {r['n_wdiff']:3d} weight diffs, {r['n_bdiff']:3d} bias diffs")
            if r['n_bdiff'] and r['n_bdiff']<=12: f.write("  bias: "+" ".join(f"[{i}]{a}(vs {b})" for i,a,b in r['bias_diffs']))
            if r['n_wdiff'] and r['n_wdiff']<=12: f.write("  w: "+" ".join(f"[{i},{j}]{a}(vs {b})" for i,j,a,b in r['weight_diffs']))
            f.write("\n")
print("prologue ops/wires/consts",pst,"| block0",bst,"| block1",gst,"| epilogue",est)
print("written to",OUT)


# ---------------------------------------------------------------- validation & compaction
def lift_full():
    """Lift the whole network into one flat equation list (names unique)."""
    x=[f"x{i}" for i in range(55)]
    lines,names,st=lift(0,len(lin),x,"N")
    return lines,names,st

def evaluate(lines,out_name,x_vals):
    """Evaluate emitted equations with plain Python ints. x_vals: 55 ints."""
    import re
    env={f"x{i}":int(v) for i,v in enumerate(x_vals)}
    pat=re.compile(r"([+-]\d+)\*([A-Za-z0-9_.]+)")
    for ln in lines:
        name,rhs=ln.split(" = relu(",1); rhs=rhs[:-1]
        tot=0
        for coef,var in pat.findall(rhs):
            tot+=int(coef)*(env[var] if not var.isdigit() else int(var))
        m2=re.search(r" ([+-]\d+)$",rhs)
        if m2: tot+=int(m2.group(1))
        env[name]=max(tot,0); env["__pre__"+name]=tot
    v=out_name
    return (env["__pre__"+v] if "__pre__"+v in env else int(v)), env

if __name__=="__main__" and "--validate" in sys.argv:
    from probe import load, needle, encode
    lines,names,st=lift_full()
    print("full lift: ops",st[0],"wires",st[1],"consts",st[2],"| output name:",names[0])
    mm=load(quiet=True)
    for s in ["hello","","mynbiqpmzjplsgqejeyd","a"*55,"Hello, World! 123"]:
        xv=[int(v) for v in encode(mm,s).tolist()]
        sym,env=evaluate(lines,names[0],xv); pre,_=needle(mm,s)
        # mid-network check: activations at layer 1000 output vs symbolic names
        acts={}
        lin_mm=[k for k in mm.children() if isinstance(k,torch.nn.Linear)]
        h=lin_mm[1000].register_forward_hook(lambda mod,a,o: acts.__setitem__("y",torch.relu(o.detach()).flatten().tolist()))
        with torch.no_grad(): torch.nn.Sequential.forward(mm, encode(mm,s).unsqueeze(0))
        h.remove()
        # names after layer 1000: re-lift up to 1001 to get names
        _,n1000,_=lift(0,1001,[f"x{i}" for i in range(55)],"N")
        symv=[(env[n] if n in env else int(n)) for n in n1000]
        mid_ok = symv==[int(round(v)) for v in acts["y"]]
        print(f"{s[:22]!r:26} symbolic needle {sym:+d}  torch needle {pre:+.0f}  {'OK' if sym==pre else 'MISMATCH'} | layer-1000 state {'OK' if mid_ok else 'MISMATCH'}")


# ---------------------------------------------------------------- compaction helpers
def parse(lines):
    import re
    pat=re.compile(r"([+-]\d+)\*([A-Za-z0-9_.]+)")
    eqs={}
    order=[]
    for ln in lines:
        name,rhs=ln.split(" = relu(",1); rhs=rhs[:-1]
        terms=[(int(c),v) for c,v in pat.findall(rhs)]
        m2=re.search(r" ([+-]\d+)$",rhs); b=int(m2.group(1)) if m2 else 0
        eqs[name]=(terms,b); order.append(name)
    return eqs,order

def cone(eqs,targets):
    """All equations that (transitively) feed the target names."""
    need=set(); stack=list(targets)
    while stack:
        n=stack.pop()
        if n in need or n not in eqs: continue
        need.add(n)
        for _,v in eqs[n][0]: stack.append(v)
    return need

def fold(eqs,order,keep):
    """Inline single-use, single-input units of the form relu(+1*a + b) where safe? -- no:
    only inline units used exactly once whose expression is relu(1*a) (already wires) -> nothing.
    Instead: report, for each unit, its expression with fan-in names, and mark single-use units."""
    uses=collections.Counter()
    for n in order:
        for _,v in eqs[n][0]: uses[v]+=1
    return uses

def emit_compact(lines,fname,targets=None,header=""):
    eqs,order=parse(lines)
    keep=cone(eqs,targets) if targets else set(order)
    uses=fold(eqs,order,keep)
    out=[header]
    # group by layer index for readability
    for n in order:
        if n not in keep: continue
        terms,b=eqs[n]
        expr=" ".join(f"{c:+d}*{v}" for c,v in terms)+(f" {b:+d}" if b else "")
        out.append(f"{n} = relu({expr})   # used {uses[n]}x")
    open(fname,"w").write("\n".join(out)+"\n"); return len(out)-1

if __name__=="__main__" and "--compact" in sys.argv:
    # prologue: full (it is the smallest region) with use-counts; also cones per output group
    x=[f"x{i}" for i in range(55)]
    pl,pnames,_=lift(0,B0,x,"P")
    n=emit_compact(pl,f"{OUT}/prologue_compact.txt",header=f"# prologue, all {len(pl)} real ops, with use counts. State out = {len(pnames)} names.")
    # which state outputs are non-trivial (i.e., are computed, not just x's or consts)?
    eqs,order=parse(pl)
    computed=[nm for nm in pnames if nm in eqs]; raw=[nm for nm in pnames if nm.startswith('x')]; consts=[nm for nm in pnames if nm.isdigit()]
    print(f"prologue compact: {n} lines; state out: {len(computed)} computed, {len(raw)} raw inputs passed through, {len(consts)} constants")
    # per computed output: cone size and which x's it depends on
    with open(f"{OUT}/prologue_outputs.txt","w") as f:
        f.write("# each prologue output: name, #ops in its cone, input slots it depends on\n")
        for i,nm in enumerate(pnames):
            if nm in eqs:
                c=cone(eqs,[nm]); xs=sorted({int(v[1:]) for k in c for _,v in eqs[k][0] if v.startswith('x')})
                f.write(f"state[{i}] = {nm}: cone {len(c)} ops, inputs x{xs}\n")
            elif nm.startswith('x'): f.write(f"state[{i}] = {nm} (raw input)\n")
            else: f.write(f"state[{i}] = const {nm}\n")
    # block1: cone sizes / dependencies per output state
    s=[f"s{i}" for i in range(256)]
    bl,bnames,_=lift(B0+BS,B0+2*BS,s,"B"); eqs,order=parse(bl)
    with open(f"{OUT}/block1_outputs.txt","w") as f:
        f.write("# generic block: each output state[i]: name, cone size, which input state s[j] it depends on\n")
        for i,nm in enumerate(bnames):
            if nm in eqs:
                c=cone(eqs,[nm]); ss=sorted({int(v[1:]) for k in c for _,v in eqs[k][0] if v.startswith('s')})
                f.write(f"out[{i}] = {nm}: cone {len(c)} ops, depends on s{ss}\n")
            elif nm.startswith('s'): f.write(f"out[{i}] = {nm} (wire from input state)\n")
            else: f.write(f"out[{i}] = const {nm}\n")
    # epilogue: cone of the needle back to the block state
    el,enames,_=lift(B0+NB*BS,len(lin),s,"E"); eqs,order=parse(el)
    c=cone(eqs,[enames[0]]); ss=sorted({int(v[1:]) for k in c for _,v in eqs[k][0] if v.startswith('s')})
    n=emit_compact(el,f"{OUT}/epilogue_cone.txt",targets=[enames[0]],header=f"# epilogue: only the {len(c)} ops feeding the needle; depends on final state s{ss}")
    print(f"epilogue cone: {n} ops, needle depends on {len(ss)} of 256 final-state entries")


# ---------------------------------------------------------------- annotated listings
def lift_annotated(l0,l1,in_names,tag,block_base=None):
    """Like lift(), but inserts a header line per layer with the weight-side stats."""
    lines=[]; names=list(in_names)
    for L in range(l0,l1):
        W=lin[L].weight.detach(); b=lin[L].bias.detach(); new=[]; ops=[]; nw=0; nc=0
        for u in range(W.shape[0]):
            nz=torch.nonzero(W[u]).flatten().tolist(); bb=int(b[u])
            if len(nz)==1 and int(W[u,nz[0]])==1 and bb==0: new.append(names[nz[0]]); nw+=1; continue
            if not nz: new.append(f"{max(bb,0)}"); nc+=1; continue
            name=f"{tag}L{L}.{u}"; terms=" ".join(f"{fmt(int(W[u,i]))}*{names[i]}" for i in nz)
            ops.append(f"{name} = relu({terms}{(' '+fmt(bb)) if bb else ''})"); new.append(name)
        pos=f" pos {L-block_base:2d}" if block_base is not None else ""
        wv=sorted(set(int(v) for v in torch.unique(W).tolist()) - {0})
        hdr=(f"# ---- layer {L}{pos}: {W.shape[1]}->{W.shape[0]}  ops={len(ops):4d} wires={nw:4d} consts={nc:3d}"
             f"  max|w|={int(W.abs().max())}  weight values={wv}  bias range=[{int(b.min())},{int(b.max())}]  bias nnz={int((b!=0).sum())}")
        lines.append(hdr); lines+=ops; names=new
    return lines,names

if __name__=="__main__" and "--annotate" in sys.argv:
    x=[f"x{i}" for i in range(55)]
    pl,pn=lift_annotated(0,B0,x,"P")
    open(f"{OUT}/prologue_annotated.txt","w").write("# prologue with per-layer stats\n"+"\n".join(pl)+"\n# state out:\n"+" ".join(pn)+"\n")
    s=[f"s{i}" for i in range(256)]
    bl,bn=lift_annotated(B0+BS,B0+2*BS,s,"B",block_base=B0+BS)
    open(f"{OUT}/block1_annotated.txt","w").write("# generic block (block 1) with per-layer stats; 'pos' = in-block position 0..41 (0,1,2,29 vary per block)\n"+"\n".join(bl)+"\n# state out:\n"+" ".join(bn)+"\n")
    E0=B0+NB*BS
    el,en=lift_annotated(E0,len(lin),s,"E",block_base=E0)
    open(f"{OUT}/epilogue_annotated.txt","w").write("# epilogue with per-layer stats; 'pos' = offset from epilogue start\n"+"\n".join(el)+f"\n# needle = {en[0]}\n")
    print("wrote *_annotated.txt")


# ---------------------------------------------------------------- SMT export
def emit_smt(lines,out_name,path,logic="QF_LIA",xmax=255,bits=32,goal=1):
    import re
    pat=re.compile(r"([+-]\d+)\*([A-Za-z0-9_.]+)")
    def nm(n): return "v_"+n.replace(".","_")
    bv=(logic=="QF_BV")
    def lit(c):
        if bv:
            c=int(c); return f"(_ bv{c} {bits})" if c>=0 else f"(bvneg (_ bv{-c} {bits}))"
        return str(c) if int(c)>=0 else f"(- {-int(c)})"
    def add(a,b): return f"(bvadd {a} {b})" if bv else f"(+ {a} {b})"
    def mul(a,b): return f"(bvmul {a} {b})" if bv else f"(* {a} {b})"
    def ge(a,b): return f"(bvsge {a} {b})" if bv else f"(>= {a} {b})"
    def le(a,b): return f"(bvsle {a} {b})" if bv else f"(<= {a} {b})"
    T=f"(_ BitVec {bits})" if bv else "Int"
    with open(path,"w") as f:
        f.write(f"(set-logic {logic})\n(set-option :produce-models true)\n")
        for i in range(55):
            f.write(f"(declare-const x{i} {T})\n(assert {ge(f'x{i}',lit(0))})\n(assert {le(f'x{i}',lit(xmax))})\n")
        for ln in lines:
            name,rhs=ln.split(" = relu(",1); rhs=rhs[:-1]
            terms=[(int(c),v) for c,v in pat.findall(rhs)]
            m2=re.search(r" ([+-]\d+)$",rhs); b=int(m2.group(1)) if m2 else 0
            parts=[]
            for c,v in terms:
                vv = lit(v) if v.isdigit() else (v if v.startswith("x") else nm(v))
                parts.append(vv if c==1 else mul(lit(c),vv))
            if b: parts.append(lit(b))
            expr=parts[0]
            for p in parts[1:]: expr=add(expr,p)
            y=nm(name)
            f.write(f"(declare-const {y} {T})\n")
            f.write(f"(assert (= {y} (ite {ge(expr,lit(0))} {expr} {lit(0)})))\n")
            if name==out_name:
                # goal on the pre-activation, i.e. needle >= goal
                f.write(f"(assert {ge(expr,lit(goal))})\n")
        f.write("(check-sat)\n(get-value ("+" ".join(f"x{i}" for i in range(55))+"))\n")

if __name__=="__main__" and "--smt" in sys.argv:
    lines,names,st=lift_full()
    S=os.path.join(HERE,"results","smt")
    emit_smt(lines,names[0],f"{S}/full_lia.smt2","QF_LIA")
    emit_smt(lines,names[0],f"{S}/full_bv32.smt2","QF_BV")
    print("wrote",S,"ops",st[0])
