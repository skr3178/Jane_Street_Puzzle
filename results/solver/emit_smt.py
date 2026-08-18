"""Stage 1 — build the compressed, exact SMT encoder of the model.

Approach: run ONE symbolic wire-collapsing pass over all 2721 layers (reusing
the Collapser from results/static/collapse.py), with inputs asserted >= 0
(true: bytes 0..255). That makes alias/dead-slot elimination provably sound.
Emit QF_LIA (unbounded ints => no width/overflow to prove, sidesteps Stage-0-c).

output = max(needle, 0); asserting output >= 1 is equivalent to needle >= 1,
so we clamp uniformly and assert the final value >= 1.

Self-test A (no solver): evaluate the collapsed ops numerically and require
exact match to emu.out(x) on random byte-strings. This certifies the collapse
algebra before any SMT is trusted.

Run under trex:
  /home/satya/anaconda3/envs/trex/bin/python results/solver/emit_smt.py
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


def build_collapsed():
    """Symbolic collapse of layers 0..n-2 (each Linear+ReLU). The FINAL layer
    (the last Linear, whose ReLU is the output clamp) is returned separately as
    an unclamped linear form so we expose the PRE-clamp `needle`.
    Returns (collapser, needle_coeffs, needle_bias)."""
    e = Emu()
    c = Collapser(55, in_prefix="x", in_nonneg=True)   # bytes >= 0
    prev_out = 55
    for k in range(e.n - 1):
        r, cc, v, out_d, b = e.layers[k]
        Wd = np.zeros((out_d, prev_out), dtype=np.int64)
        Wd[r, cc] = v
        c.push_layer(Wd, b, note=f"L{k}")
        prev_out = out_d
    # final layer: single output, unclamped needle = b0 + sum W0j * vec[j]
    r, cc, v, out_d, b = e.layers[e.n - 1]
    Wd = np.zeros((out_d, prev_out), dtype=np.int64)
    Wd[r, cc] = v
    coeffs, bias = {}, int(b[0])
    for j in range(prev_out):
        w = int(Wd[0, j])
        if w == 0:
            continue
        val = c.vec[j]
        if val[0] == "const":
            bias += w * val[1]
        else:
            var, cf = val[1], val[2]
            coeffs[var] = coeffs.get(var, 0) + w * cf
    coeffs = {v: cc for v, cc in coeffs.items() if cc != 0}
    return c, coeffs, bias


def evaluate(collapser, needle_coeffs, needle_bias, x):
    """Numeric replay of the collapsed ops -> PRE-clamp needle for input x."""
    vals = {}
    for name, coeffs, bias, clamp, note in collapser.defs:
        vid = int(name[1:])
        s = bias
        for var, cf in coeffs.items():
            base = x[var[1]] if var[0] == "in" else vals[var[1]]
            s += cf * int(base)
        vals[vid] = max(s, 0) if clamp else s
    s = needle_bias
    for var, cf in needle_coeffs.items():
        base = x[var[1]] if var[0] == "in" else vals[var[1]]
        s += cf * int(base)
    return s


def num(c):
    c = int(c)
    return str(c) if c >= 0 else f"(- {-c})"


def ref_name(var):
    return f"x{var[1]}" if var[0] == "in" else f"v{var[1]}"


def expr_str(coeffs, bias):
    terms = []
    for var, cf in coeffs.items():
        nm = ref_name(var)
        terms.append(nm if cf == 1 else f"(* {num(cf)} {nm})")
    if bias != 0 or not terms:
        terms.append(num(bias))
    return terms[0] if len(terms) == 1 else "(+ " + " ".join(terms) + ")"


def emit(collapser, needle_coeffs, needle_bias, path, fix_x=None,
         target=1, assert_target=True):
    """Emit QF_LIA with PRE-clamp `needle`. If fix_x given, assert x==fix_x.
    If assert_target, add (>= needle target)."""
    n_ops = len(collapser.defs)
    with open(path, "w") as fh:
        fh.write("(set-logic QF_LIA)\n(set-option :produce-models true)\n\n")
        for i in range(55):
            fh.write(f"(declare-const x{i} Int)\n")
            if fix_x is None:
                fh.write(f"(assert (>= x{i} 0))\n(assert (<= x{i} 255))\n")
            else:
                fh.write(f"(assert (= x{i} {int(fix_x[i])}))\n")
        fh.write("\n")
        for name, coeffs, bias, clamp, note in collapser.defs:
            e = expr_str(coeffs, bias)
            fh.write(f"(declare-const {name} Int)\n")
            if clamp:
                fh.write(f"(assert (= {name} (ite (>= {e} 0) {e} 0)))\n")
            else:
                fh.write(f"(assert (= {name} {e}))\n")
        ne = expr_str(needle_coeffs, needle_bias)
        fh.write(f"\n(declare-const needle Int)\n(assert (= needle {ne}))\n")
        if assert_target:
            fh.write(f"(assert (>= needle {target}))\n")
        fh.write("(check-sat)\n")
        xs = " ".join(f"x{i}" for i in range(55))
        fh.write(f"(get-value ({xs} needle))\n")
    return n_ops


def main():
    print("building full-model collapse (pre-clamp needle)...", flush=True)
    c, ncoeffs, nbias = build_collapsed()
    n_ops = len(c.defs)
    n_clamp = sum(1 for d in c.defs if d[3])
    print(f"collapsed ops: {n_ops} ({n_clamp} clamp), aliases {c.n_alias}, "
          f"dead {c.n_dead}, consts {c.n_const}; needle terms {len(ncoeffs)}")

    # ---- Self-test A: numeric replay vs emulator PRE-clamp needle (no solver) ----
    e = Emu()
    rng = np.random.default_rng(3)
    ok = True
    for t in range(100):
        L = int(rng.integers(0, 56))
        x = np.zeros(55, dtype=np.int64)
        if L:
            x[:L] = rng.integers(1, 256, L)
        got = evaluate(c, ncoeffs, nbias, x)
        want = e.needle(x)      # PRE-clamp
        if got != want:
            ok = False
            print(f"  MISMATCH t={t}: collapsed={got} emu.needle={want}")
            break
    print(f"Self-test A (collapsed needle == emu.needle on 100 random "
          f"byte-strings): {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)

    # ---- emit main instance (x free in 0..255, needle>=1) ----
    p = f"{HERE}/model_lia.smt2"
    emit(c, ncoeffs, nbias, p, fix_x=None, target=1)
    sz = os.path.getsize(p) / 1e6
    print(f"emitted {p} ({sz:.1f} MB, {n_ops} ops), target needle>=1")

    # ---- emit a fixed-input faithfulness instance (no target: solver must
    #      REPORT needle == emu) ----
    xhello = e.encode("hello")
    nh = e.needle(xhello)
    emit(c, ncoeffs, nbias, f"{HERE}/selftest_hello.smt2", fix_x=xhello,
         assert_target=False)
    print(f"emitted selftest_hello.smt2 (x fixed to 'hello'); emu needle={nh}; "
          f"solver must report needle={nh}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
