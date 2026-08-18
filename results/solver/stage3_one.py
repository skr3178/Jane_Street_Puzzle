"""Stage 3 — SINGLE exact-value probe (safe: non-accepting target).

Full model, x free in 0..255, assert (= needle T) for ONE T (default -12,
which is known reachable). Reports result / wall time / z3 decisions+conflicts
/ model verified via emulator. Does NOT print the input vector.

  /home/satya/anaconda3/envs/trex/bin/python results/solver/stage3_one.py [T]
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
import emit_smt as E  # noqa: E402
from emu import Emu  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else -12
assert T <= 0, "safety: only non-accepting targets (T<=0) in Stage 3"
TIMEOUT_S = 300


def main():
    print(f"building encoder (one-time)...", flush=True)
    c, nc, nb = E.build_collapsed()
    path = f"{HERE}/thr_one_{T}.smt2"
    E.emit(c, nc, nb, path, fix_x=None, assert_target=False)
    with open(path, "a") as fh:
        fh.write(f"(assert (= needle {T if T>=0 else f'(- {-T})'}))\n")
    print(f"solving needle == {T} (x free in 0..255, timeout {TIMEOUT_S}s)...",
          flush=True)
    s = z3.Solver()
    s.set("timeout", TIMEOUT_S * 1000)
    t = time.time()
    s.from_file(path)
    r = s.check()
    dt = time.time() - t
    st = s.statistics()
    stat = {}
    for k in st.keys():
        if any(w in k for w in ("conflict", "decision", "restart", "memory",
                                "propagation")):
            stat[k] = st.get_key_value(k)
    verified = None
    if r == z3.sat:
        m = s.model()
        xv = np.array([int(str(m[z3.Int(f'x{i}')])) if m[z3.Int(f'x{i}')]
                       is not None else 0 for i in range(55)], dtype=np.int64)
        verified = bool(Emu().needle(xv) == T)   # confirm; xv NOT printed
    os.remove(path)
    print(f"\nT={T}: {r}  {dt:.2f}s  verified={verified}")
    print("z3 stats:", stat)


if __name__ == "__main__":
    main()
