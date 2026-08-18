"""Stage 2.5 — reduced-round scaling probe.

Cryptanalysis-style small-first: for k = 1,2,4,8,16,32,63 blocks, build the
inversion instance "find input x whose state entering block k equals a KNOWN
target state" (target harvested from a real emulator trace, so guaranteed SAT
and non-answer-bearing). Time z3. The GROWTH of solve time vs k is the
deliverable — it tells us whether the full 63-round solve is plausibly
tractable or the mixer resists solvers.

Run under trex:
  /home/satya/anaconda3/envs/trex/bin/python results/solver/stage2p5_scaling.py
"""
import os
import sys
import time

import numpy as np
import z3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "results", "static"))
from collapse import Collapser  # noqa: E402
from emu import Emu  # noqa: E402

PRO_END, NBLK, PERIOD = 17, 63, 42


def build_upto(e, m):
    """Collapse layers 0..m-1; return collapser (vec = state entering layer m)."""
    c = Collapser(55, in_prefix="x", in_nonneg=True)
    prev = 55
    for k in range(m):
        r, cc, v, od, b = e.layers[k]
        Wd = np.zeros((od, prev), dtype=np.int64)
        Wd[r, cc] = v
        c.push_layer(Wd, b, note=f"L{k}")
        prev = od
    return c


def emit_inversion(c, target, path):
    n = 0
    with open(path, "w") as fh:
        fh.write("(set-logic QF_LIA)\n(set-option :produce-models true)\n")
        for i in range(55):
            fh.write(f"(declare-const x{i} Int)\n(assert (>= x{i} 0))"
                     f"(assert (<= x{i} 255))\n")
        for name, coeffs, bias, clamp, note in c.defs:
            terms = []
            for var, cf in coeffs.items():
                nm = f"x{var[1]}" if var[0] == "in" else f"v{var[1]}"
                terms.append(nm if cf == 1 else
                             f"(* {cf if cf>=0 else f'(- {-cf})'} {nm})")
            if bias != 0 or not terms:
                terms.append(str(bias) if bias >= 0 else f"(- {-bias})")
            e = terms[0] if len(terms) == 1 else "(+ " + " ".join(terms) + ")"
            fh.write(f"(declare-const {name} Int)\n")
            if clamp:
                fh.write(f"(assert (= {name} (ite (>= {e} 0) {e} 0)))\n")
            else:
                fh.write(f"(assert (= {name} {e}))\n")
            n += 1
        # target equality on each state coord
        for j, val in enumerate(target):
            cell = c.vec[j]
            if cell[0] == "const":
                # consistency (should hold by construction)
                continue
            var, cf = cell[1], cell[2]
            nm = f"x{var[1]}" if var[0] == "in" else f"v{var[1]}"
            lhs = nm if cf == 1 else f"(* {cf} {nm})"
            fh.write(f"(assert (= {lhs} {int(val)}))\n")
        fh.write("(check-sat)\n")
    return n


def main():
    e = Emu()
    xseed = e.encode("hello")
    states = e.states(xseed)   # entering blocks 0..62 + epilogue
    rows = []
    for k in [1, 2, 4, 8, 16, 32, 63]:
        m = PRO_END + k * PERIOD
        target = states[k] if k < NBLK else states[NBLK]  # state entering block k
        c = build_upto(e, m)
        path = f"{HERE}/inv_k{k}.smt2"
        nops = emit_inversion(c, target, path)
        s = z3.Solver()
        s.set("timeout", 120000)   # 120s per instance
        t = time.time()
        s.from_file(path)
        r = s.check()
        dt = time.time() - t
        # verify a returned model reproduces the target (sanity), if sat
        ok = None
        if r == z3.sat:
            mdl = s.model()
            xv = np.array([int(str(mdl[z3.Int(f"x{i}")]) or 0)
                           if mdl[z3.Int(f"x{i}")] is not None else 0
                           for i in range(55)], dtype=np.int64)
            got = e.states(xv)[k if k < NBLK else NBLK]
            ok = bool(np.array_equal(got, target))
        rows.append((k, m, nops, str(r), round(dt, 2), ok))
        print(f"k={k:2d}  layers={m:4d}  ops={nops:6d}  {r}  {dt:6.2f}s  "
              f"model_reproduces_target={ok}", flush=True)
        os.remove(path)
    import json
    json.dump([dict(zip(["k", "layers", "ops", "result", "sec", "verified"], r))
               for r in rows], open(f"{HERE}/stage2p5_scaling.json", "w"), indent=1)
    print("\nsaved stage2p5_scaling.json")


if __name__ == "__main__":
    main()
