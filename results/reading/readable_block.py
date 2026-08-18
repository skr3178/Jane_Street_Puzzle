"""Readable rewrite of ONE 42-layer block (block 1, layers 59..100).

Mechanical transforms only (NO interpretation of what it computes):
  1. rename input state slots s{j} by their zone class (bit/num/store/K/zero)
  2. collapse consecutive ops sharing a template + unit stride into indexed
     "banks" (for i in [a..b]: ...)
  3. flag payload ops (the per-block-varying pos-0 rows) with [PAYLOAD]

Run under trex:
  /home/satya/anaconda3/envs/trex/bin/python results/reading/readable_block.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "results", "static"))
from collapse import Collapser  # noqa: E402
from emu import Emu  # noqa: E402

PRO_END, PERIOD = 17, 42
BLOCK = 1                      # which block to render
LO = PRO_END + BLOCK * PERIOD  # layer 59

# ---- zone map from lane_map.txt --------------------------------------------
def load_zones():
    cls = {}
    path = os.path.join(ROOT, "results", "dynamic", "lane_map.txt")
    for ln in open(path):
        p = ln.split("\t")
        if len(p) >= 2 and p[0].isdigit():
            cls[int(p[0])] = p[1]
    pre = {"binary": "bit", "small<=8": "num", "wide": "wide",
           "const128": "K", "always0": "zero"}
    return {j: pre.get(cls.get(j, "?"), "s") for j in range(256)}


ZONE = load_zones()


def sname(j):
    return f"{ZONE[j]}[{j}]"


# ---- build the block's collapsed ops with provenance -----------------------
def build_block():
    e = Emu()
    c = Collapser(256, in_prefix="s", in_nonneg=True)
    prev = 256
    payload_rows = set()
    for p in range(PERIOD):
        k = LO + p
        r, cc, v, od, b = e.layers[k]
        Wd = np.zeros((od, prev), dtype=np.int64)
        Wd[r, cc] = v
        n_before = len(c.defs)
        c.push_layer(Wd, b, note=f"pos{p}")
        # pos-0 rows >=128 are the per-block payload (regime 1)
        if p == 0:
            for name, coeffs, bias, clamp, note in c.defs[n_before:]:
                # provenance row is not stored; mark by heuristic: pos0 ops whose
                # note is pos0 -- we flag all pos0 ops, refine below
                pass
        prev = od
    return c


def term_sig(coeffs):
    """template signature: sorted (coeff, reftype) ignoring absolute index."""
    return tuple(sorted((cf, var[0]) for var, cf in coeffs.items()))


def refs_sorted(coeffs):
    return sorted(coeffs.items(), key=lambda kv: (kv[0][0], kv[0][1]))


def bank_step(a, b):
    """True if op b is the +1-stride successor of op a (same template)."""
    (na, ca, ba, cla, _), (nb, cb, bb, clb, _) = a, b
    if cla != clb or ba != bb:
        return False
    if term_sig(ca) != term_sig(cb):
        return False
    ra, rb = refs_sorted(ca), refs_sorted(cb)
    if len(ra) != len(rb):
        return False
    for (va, fa), (vb, fb) in zip(ra, rb):
        if fa != fb or va[0] != vb[0]:
            return False
        if vb[1] != va[1] + 1:     # unit stride
            return False
    # destination ids must also be +1
    return int(nb[1:]) == int(na[1:]) + 1


def ref_str(var, off_label=None):
    idx = var[1]
    if var[0] == "in":
        base = ZONE[idx]
        return f"{base}[{idx}{off_label or ''}]"
    return f"t[{idx}{off_label or ''}]"


def op_expr(coeffs, bias, off_label=None):
    parts = []
    for var, cf in refs_sorted(coeffs):
        nm = ref_str(var, off_label)
        if cf == 1:
            parts.append(f"+ {nm}")
        elif cf == -1:
            parts.append(f"- {nm}")
        elif cf > 0:
            parts.append(f"+ {cf}*{nm}")
        else:
            parts.append(f"- {abs(cf)}*{nm}")
    s = " ".join(parts)
    if bias:
        s += f" {'+' if bias > 0 else '-'} {abs(bias)}"
    return s


def render(c):
    defs = c.defs
    out = []
    i = 0
    n = len(defs)
    while i < n:
        # extend a bank
        j = i
        while j + 1 < n and bank_step(defs[j], defs[j + 1]):
            j += 1
        name, coeffs, bias, clamp, note = defs[i]
        did = int(name[1:])
        wrap = ("relu(", ")") if clamp else ("", "")
        if j > i:
            cnt = j - i + 1
            expr = op_expr(coeffs, bias, off_label="+i")
            out.append(f"for i in [0..{cnt-1}]:  t[{did}+i] = "
                       f"{wrap[0]}{expr}{wrap[1]}   # bank x{cnt} ({note})")
        else:
            expr = op_expr(coeffs, bias)
            out.append(f"t[{did}] = {wrap[0]}{expr}{wrap[1]}   # ({note})")
        i = j + 1
    # outputs — collapse runs of identity pass-through (next X[j] = X[j])
    out.append("")
    out.append("# --- block outputs (256-wide next state) ---")

    def desc(k):
        cell = c.vec[k]
        if cell[0] == "scaled" and cell[2] == 1 and cell[1][0] == "v":
            return f"t[{cell[1][1]}]"
        if cell[0] == "const":
            return str(cell[1])
        c2 = cell[2]
        base = ref_str(cell[1])
        return f"{c2}*{base}" if c2 != 1 else base

    def is_identity(k):
        cell = c.vec[k]
        return (cell[0] == "scaled" and cell[2] == 1 and cell[1][0] == "in"
                and cell[1][1] == k)

    def kind_key(k):
        """affine descriptor of output k: ('id',), ('in', zone, offset),
        ('t', base_minus_k), ('other',)."""
        cell = c.vec[k]
        if is_identity(k):
            return ("id",)
        if cell[0] == "scaled" and cell[2] == 1 and cell[1][0] == "in":
            return ("in", ZONE[cell[1][1]], cell[1][1] - k)
        if cell[0] == "scaled" and cell[2] == 1 and cell[1][0] == "v":
            return ("t", cell[1][1] - k)
        return ("other",)

    k = 0
    while k < len(c.vec):
        kk = kind_key(k)
        if kk[0] == "other":
            out.append(f"next {sname(k)} = {desc(k)}")
            k += 1
            continue
        j = k
        while j + 1 < len(c.vec) and kind_key(j + 1) == kk \
                and ZONE[j + 1] == ZONE[k]:
            j += 1
        z = ZONE[k]
        if kk[0] == "id":
            body = "unchanged  (pass-through)"
        elif kk[0] == "in":
            d = kk[2]
            body = (f"{kk[1]}[{k+d}..{j+d}]" if j > k else f"{kk[1]}[{k+d}]")
        else:  # t
            base = k + kk[1]
            body = (f"t[{base}..{base+(j-k)}]" if j > k else f"t[{base}]")
        rng = f"{z}[{k}..{j}]" if j > k else f"{z}[{k}]"
        out.append(f"next {rng} = {body}")
        k = j + 1
    return out


def main():
    c = build_block()
    lines = render(c)
    n_banks = sum(1 for L in lines if L.startswith("for i"))
    n_single = sum(1 for L in lines if L.startswith("t["))
    hdr = [f"# Readable rewrite of BLOCK {BLOCK} (layers {LO}..{LO+PERIOD-1})",
           f"# input state = 256 slots, zone-named; t[] = block-internal temps",
           f"# {len(c.defs)} raw ops -> {n_banks} banks + {n_single} singletons",
           f"# zones: bit=binary{{0,1}}  num=small<=8  wide=large  K=const128  zero=always0",
           ""]
    path = os.path.join(HERE, f"block{BLOCK}_readable.txt")
    open(path, "w").write("\n".join(hdr + lines) + "\n")
    print("\n".join(hdr + lines[:40]))
    print("...")
    print(f"\n({len(c.defs)} ops -> {n_banks} banks + {n_single} singletons)")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
