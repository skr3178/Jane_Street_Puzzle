"""Stage 3 smoke test — does free-input search fire AT ALL, and where does it break?

Guaranteed-SAT ladder: base input "hello" (needle=-15). Free the first F bytes
(x0..x_{F-1}) in 0..255, fix the rest to hello's codes, assert (= needle -15).
hello itself is always a witness -> SAT if the solver can search that width.
Sweep F = 1,2,3,5,8,13,21,55. Report result/time/conflicts; verify model.
Input vector NOT printed. Safe: -15 is non-accepting.

  /home/satya/anaconda3/envs/trex/bin/python results/solver/stage3_smoke.py
"""
import json
import os
import sys
import time

import numpy as np
import z3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "results", "static"))
import emit_smt as E  # noqa: E402
from emu import Emu  # noqa: E402

TIMEOUT_S = 120
TARGET = -15


def emit_partial(c, nc, nb, path, base, n_free):
    """x0..x_{n_free-1} free in 0..255; rest fixed to base; needle==TARGET."""
    with open(path, "w") as fh:
        fh.write("(set-logic QF_LIA)\n(set-option :produce-models true)\n")
        for i in range(55):
            fh.write(f"(declare-const x{i} Int)\n")
            if i < n_free:
                fh.write(f"(assert (>= x{i} 0))(assert (<= x{i} 255))\n")
            else:
                fh.write(f"(assert (= x{i} {int(base[i])}))\n")
        for name, coeffs, bias, clamp, note in c.defs:
            e = E.expr_str(coeffs, bias)
            fh.write(f"(declare-const {name} Int)\n")
            fh.write(f"(assert (= {name} (ite (>= {e} 0) {e} 0)))\n" if clamp
                     else f"(assert (= {name} {e}))\n")
        ne = E.expr_str(nc, nb)
        fh.write(f"(declare-const needle Int)\n(assert (= needle {ne}))\n")
        fh.write(f"(assert (= needle {TARGET if TARGET>=0 else f'(- {-TARGET})'}))\n")
        fh.write("(check-sat)\n")


def main():
    print("building encoder...", flush=True)
    c, nc, nb = E.build_collapsed()
    e = Emu()
    base = e.encode("hello")
    assert e.needle(base) == TARGET
    rows = []
    for F in [1, 2, 3, 5, 8, 13, 21, 55]:
        path = f"{HERE}/smoke_{F}.smt2"
        emit_partial(c, nc, nb, path, base, F)
        s = z3.Solver()
        s.set("timeout", TIMEOUT_S * 1000)
        t = time.time()
        s.from_file(path)
        r = s.check()
        dt = time.time() - t
        st = s.statistics()
        conf = st.get_key_value("conflicts") if "conflicts" in st.keys() else None
        verified = None
        if r == z3.sat:
            m = s.model()
            xv = np.array([int(str(m[z3.Int(f'x{i}')])) if m[z3.Int(f'x{i}')]
                           is not None else 0 for i in range(55)], dtype=np.int64)
            verified = bool(e.needle(xv) == TARGET)
        rows.append((F, str(r), round(dt, 2), conf, verified))
        print(f"free={F:2d} bytes: {str(r):9s} {dt:7.2f}s conflicts={conf} "
              f"verified={verified}", flush=True)
        os.remove(path)
        if r != z3.sat:
            print("  -> stopped: solver failed to fire at this width")
            break
    json.dump([dict(zip(["free", "result", "sec", "conflicts", "verified"], r))
               for r in rows], open(f"{HERE}/stage3_smoke.json", "w"), indent=1)
    print("\nsaved stage3_smoke.json")


if __name__ == "__main__":
    main()
