"""Stage 3 — threshold hardness sweep (SAFE: exact non-accepting targets).

Full model, x free in 0..255. For each T in -15..0 assert (= needle T) exactly
and time z3. Pinning needle to a NON-accepting value (<=0 => output 0) means no
accepting input can ever be returned -> unfenced. This maps which needle values
are reachable and how solver effort grows as T climbs toward acceptance (1).

Records SAT/UNSAT/timeout + wall time + z3 stats. Does NOT print input vectors.

Run under trex:
  /home/satya/anaconda3/envs/trex/bin/python results/solver/stage3_threshold.py
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

TIMEOUT_S = 300


def emit_exact(c, ncoeffs, nbias, path, T):
    E.emit(c, ncoeffs, nbias, path, fix_x=None, assert_target=False)
    with open(path, "a") as fh:
        fh.write(f"(assert (= needle {T if T>=0 else f'(- {-T})'}))\n")


def main():
    print("building encoder...", flush=True)
    c, nc, nb = E.build_collapsed()
    e = Emu()
    rows = []
    for T in range(-15, 1):     # -15 .. 0  (all non-accepting)
        path = f"{HERE}/thr_{T}.smt2"
        # write base once, append target
        E.emit(c, nc, nb, path, fix_x=None, assert_target=False)
        with open(path, "a") as fh:
            fh.write(f"(assert (= needle {T if T>=0 else f'(- {-T})'}))\n")
        s = z3.Solver()
        s.set("timeout", TIMEOUT_S * 1000)
        t = time.time()
        s.from_file(path)
        r = s.check()
        dt = time.time() - t
        verified = None
        if r == z3.sat:
            m = s.model()
            xv = np.array([int(str(m[z3.Int(f'x{i}')])) if m[z3.Int(f'x{i}')]
                           is not None else 0 for i in range(55)], dtype=np.int64)
            verified = bool(e.needle(xv) == T)   # confirm, but don't print xv
        rows.append((T, str(r), round(dt, 2), verified))
        print(f"needle=={T:3d}: {str(r):9s} {dt:7.2f}s  verified={verified}",
              flush=True)
        os.remove(path)
    json.dump([dict(zip(["T", "result", "sec", "verified"], r)) for r in rows],
              open(f"{HERE}/stage3_threshold.json", "w"), indent=1)
    print("\nsaved stage3_threshold.json")


if __name__ == "__main__":
    main()
