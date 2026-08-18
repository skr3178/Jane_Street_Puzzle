"""QF_BV emitter (signed 32-bit) + smoke test with z3-BV and bitwuzla.

Width 32 is not proven sound (intervals diverged), but any SAT model is
verified against the int64 emulator, so overflow-induced false SAT is caught.
This is a tractability probe, not the exact solver.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "results", "static"))
import emit_smt as E  # noqa: E402
from emu import Emu  # noqa: E402

W = 32


def bvlit(c):
    c = int(c) % (1 << W)
    return f"(_ bv{c} {W})"


def term(cf, nm):
    if cf == 1:
        return nm
    if cf == -1:
        return f"(bvneg {nm})"
    if cf > 0:
        return f"(bvmul {bvlit(cf)} {nm})"
    return f"(bvneg (bvmul {bvlit(-cf)} {nm}))"


def expr_bv(coeffs, bias):
    parts = [term(cf, (f"x{v[1]}" if v[0] == "in" else f"v{v[1]}"))
             for v, cf in coeffs.items()]
    if bias != 0 or not parts:
        parts.append(bvlit(bias))
    if len(parts) == 1:
        return parts[0]
    e = parts[0]
    for p in parts[1:]:
        e = f"(bvadd {e} {p})"
    return e


def emit_bv(c, nc, nb, path, base=None, n_free=55, target=None, exact=True):
    with open(path, "w") as fh:
        fh.write(f"(set-logic QF_BV)\n(set-option :produce-models true)\n")
        for i in range(55):
            fh.write(f"(declare-const x{i} (_ BitVec {W}))\n")
            if base is not None and i >= n_free:
                fh.write(f"(assert (= x{i} {bvlit(int(base[i]))}))\n")
            else:
                fh.write(f"(assert (bvsle {bvlit(0)} x{i}))"
                         f"(assert (bvsle x{i} {bvlit(255)}))\n")
        for name, coeffs, bias, clamp, note in c.defs:
            e = expr_bv(coeffs, bias)
            fh.write(f"(declare-const {name} (_ BitVec {W}))\n")
            if clamp:
                fh.write(f"(assert (= {name} (ite (bvsge {e} {bvlit(0)}) {e} "
                         f"{bvlit(0)})))\n")
            else:
                fh.write(f"(assert (= {name} {e}))\n")
        ne = expr_bv(nc, nb)
        fh.write(f"(declare-const needle (_ BitVec {W}))\n"
                 f"(assert (= needle {ne}))\n")
        if target is not None:
            op = "=" if exact else "bvsge"
            fh.write(f"(assert ({op} needle {bvlit(target)}))\n")
        fh.write("(check-sat)\n")


def main():
    print("building encoder...", flush=True)
    c, nc, nb = E.build_collapsed()
    e = Emu()
    base = e.encode("hello")
    path = f"{HERE}/smoke_bv1.smt2"
    emit_bv(c, nc, nb, path, base=base, n_free=1, target=-15, exact=True)
    print(f"emitted {path} ({os.path.getsize(path)/1e6:.1f} MB); 1 free byte, "
          f"needle==-15\n")

    # z3 on BV
    import z3
    s = z3.Solver()
    s.set("timeout", 120000)
    t = time.time()
    s.from_file(path)
    r = s.check()
    dt = time.time() - t
    print(f"z3-BV   1 free byte: {r} {dt:.2f}s")

    # bitwuzla via its SMT2 parser API
    try:
        import bitwuzla as bw
        opts = bw.Options()
        opts.set(bw.Option.PRODUCE_MODELS, True)
        # parse & solve the file
        t = time.time()
        tm = bw.TermManager()
        parser = bw.Parser(tm, opts)
        parser.parse(path)
        dt = time.time() - t
        print(f"bitwuzla 1 free byte: parsed+solved in {dt:.2f}s")
    except Exception as ex:
        print("bitwuzla API path failed:", repr(ex)[:160])


if __name__ == "__main__":
    main()
