"""Block-1 data-flow analysis: for each of the 256 OUTPUT slots (next state s2),
compute which INPUT slots (s1) it structurally depends on, classify
passthrough / permutation-copy / real-mixing, and cross-check empirically
against the int64 emulator by perturbation.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from parse_linearize import parse_file    # noqa: E402
from emu import Emu                        # noqa: E402

blk = parse_file(os.path.join(HERE, "block1_readable.txt"))

# ---- transitive input-slot support of every temp ---------------------------
support = {}   # ('t',k) -> set of ('zone',idx) input slots


def sup(ref):
    if ref[0] != "t":
        return {ref}
    if ref in support:
        return support[ref]
    support[ref] = set()          # guard (DAG, but be safe)
    s = set()
    for r in blk.by_var[ref].coeffs:
        s |= sup(r)
    support[ref] = s
    return s


# ---- parse the `next ... = ...` output mapping into per-slot sources --------
def out_map():
    m = {}   # out_slot -> ('pass', src_slot) | ('temp', temp_idx)
    for ln in blk.outputs:
        body = ln[len("next "):]
        lhs, rhs = body.split("=", 1)
        rhs = rhs.strip()
        mL = re.match(r'[a-zA-Z]+\[(\d+)(?:\.\.(\d+))?\]', lhs.strip())
        d0 = int(mL.group(1)); d1 = int(mL.group(2)) if mL.group(2) else d0
        if rhs.startswith("unchanged"):
            for off in range(d1 - d0 + 1):
                m[d0 + off] = ("pass", d0 + off)
        else:
            mT = re.match(r't\[(\d+)(?:\.\.(\d+))?\]', rhs)
            mB = re.match(r'[a-zA-Z]+\[(\d+)(?:\.\.(\d+))?\]', rhs)
            if mT:
                base = int(mT.group(1))
                for off in range(d1 - d0 + 1):
                    m[d0 + off] = ("temp", base + off)
            elif mB:
                base = int(mB.group(1))
                for off in range(d1 - d0 + 1):
                    m[d0 + off] = ("pass", base + off)
    return m


OM = out_map()

# per-output support set (as input slot indices)
out_support = {}
for o in range(256):
    kind, src = OM[o]
    if kind == "pass":
        out_support[o] = {src}
    else:
        out_support[o] = {idx for (_, idx) in sup(("t", src))}

# ---- classify ---------------------------------------------------------------
LADDER_IN = {200, 201, 202, 203}
passthrough, copy1, mixing = [], [], []
for o in range(256):
    kind, src = OM[o]
    if kind == "pass":
        passthrough.append(o)
    elif len(out_support[o]) <= 1:
        copy1.append(o)                 # computed but from a single input slot
    else:
        mixing.append(o)

print("=" * 70)
print("BLOCK-1 output classification (256 slots)")
print("=" * 70)
print(f"structural pass-through (unchanged / slot=slot copy): {len(passthrough)}")
print(f"computed-from-single-slot (permutation/relu copy)   : {len(copy1)}")
print(f"real mixing (>=2 input slots)                        : {len(mixing)}")


def rng(lst):
    if not lst:
        return "-"
    lst = sorted(lst); out = []; a = b = lst[0]
    for x in lst[1:]:
        if x == b + 1:
            b = x
        else:
            out.append(f"{a}" if a == b else f"{a}..{b}"); a = b = x
    out.append(f"{a}" if a == b else f"{a}..{b}")
    return ",".join(out)


print(f"\n  pass-through slots : {rng(passthrough)}")
print(f"  single-slot copies : {rng(copy1)}")
print(f"  mixing slots       : {rng(mixing)}")

# ---- ladder-specific queries ------------------------------------------------
print()
print("=" * 70)
print("Ladder queries")
print("=" * 70)
lad_out = [o for o in range(256) if out_support[o] & LADDER_IN]
print(f"output slots that depend on the ladder INPUTS wide[200..203]: "
      f"{rng(lad_out)}")
for o in lad_out[:4] + lad_out[-2:]:
    print(f"   out slot {o:3d}  <- inputs {sorted(out_support[o])}")

# how big is each mixing slot's input footprint?
print("\nmixing-slot input footprint (how many s1 slots each s2 slot reads):")
for label, slots in [("bit[32]", [32]), ("num[33..63]", range(33, 64)),
                     ("zero[64]", [64]), ("bit[65..95]", range(65, 96)),
                     ("bit[96..127]", range(96, 128))]:
    szs = [len(out_support[o]) for o in slots]
    print(f"   {label:14s}: footprint {min(szs)}..{max(szs)} input slots")

# =====================================================================
# empirical cross-check: perturb each input slot, see which outputs move
# =====================================================================
print()
print("=" * 70)
print("Empirical dependency cross-check (emulator, block 1, wide[200]=200)")
print("=" * 70)
e = Emu()
rngnp = np.random.default_rng(3)
x = rngnp.integers(1, 256, size=55).astype(np.int64)
_, rec = e._run(x, record_at={17 + 1 * 42, 17 + 2 * 42})
s1 = rec[17 + 1 * 42].astype(np.int64).copy()
s1[200] = 200                              # force the traced ladder input
# recompute block output from s1 via the readable block
from verify_and_census import eval_block_output    # reuse verified evaluator
o0 = eval_block_output(blk, s1)

emp_dep = {o: set() for o in range(256)}
for j in range(256):
    for dv in (1, 7, 64):
        s1p = s1.copy()
        val = int(s1p[j])
        s1p[j] = val + dv if val + dv <= 255 or j >= 192 else max(val - dv, 0)
        if s1p[j] == val:
            continue
        op = eval_block_output(blk, s1p)
        for o in np.nonzero(op != o0)[0]:
            emp_dep[o].add(j)

emp_mixing = [o for o in range(256) if len(emp_dep[o]) >= 2]
emp_const = [o for o in range(256) if len(emp_dep[o]) == 0]
print(f"outputs that MOVED under >=2 distinct input perturbations: "
      f"{len(emp_mixing)}  -> {rng(emp_mixing)}")
print(f"outputs constant under all perturbations tested          : "
      f"{len(emp_const)}  -> {rng(emp_const)}")
# structural-vs-empirical agreement on the "does it depend on wide[200..203]" query
emp_lad = [o for o in range(256) if emp_dep[o] & LADDER_IN]
print(f"empirically depend on wide[200..203]: {rng(emp_lad)}")
print(f"structural agreed?  {set(emp_lad) <= set(lad_out)}  "
      f"(structural is an over-approx; empirical ⊆ structural expected)")
