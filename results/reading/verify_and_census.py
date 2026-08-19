"""Verify the readable parsers bit-exactly vs the emulator, then run the
ReLU census / linearization test on the epilogue.

  1. epilogue: parse -> eval on s63 -> compare needle to emu.needle
  2. block1  : parse -> eval on s1 -> compare next-state to s2
  3. empirical s63 bounds from random inputs
  4. propagate bounds through the epilogue; census ReLUs on the needle cone
  5. try to linearize the needle
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from emu import Emu                       # noqa: E402
from parse_linearize import parse_file, ZONES  # noqa: E402

BLK_END = 2663          # layer index of the state entering the epilogue (s63)
PRO_END, PERIOD = 17, 42
rng = np.random.default_rng(0)


def random_inputs(B):
    """Mix of valid-looking strings (prefix of nonzero bytes, 0 padding) and
    fully random byte vectors, domain 0..255 per Stage-0."""
    X = np.zeros((55, B), dtype=np.int64)
    for j in range(B):
        mode = j % 3
        if mode == 0:                      # full random 1..255, no padding
            X[:, j] = rng.integers(1, 256, size=55)
        elif mode == 1:                    # random length, nonzero prefix + pad
            L = int(rng.integers(1, 56))
            X[:L, j] = rng.integers(1, 256, size=L)
        else:                              # printable-ish prefix
            L = int(rng.integers(1, 56))
            X[:L, j] = rng.integers(32, 127, size=L)
    return X


def s63_batch(e, X):
    _, rec = e._run(X, record_at={BLK_END})
    return rec[BLK_END]                    # (256, B)


def state_at(e, X, block):
    mark = PRO_END + block * PERIOD
    _, rec = e._run(X, record_at={mark})
    return rec[mark]                       # (256, B)


# ---- block-output mapping ---------------------------------------------------
def parse_next(lines):
    """Return list of (dst_lo, dst_hi, kind, src_info)."""
    out = []
    for ln in lines:
        ln = ln[len("next "):]
        lhs, rhs = ln.split("=", 1)
        rhs = rhs.strip()
        mL = re.match(r'[a-zA-Z]+\[(\d+)(?:\.\.(\d+))?\]', lhs.strip())
        d0 = int(mL.group(1))
        d1 = int(mL.group(2)) if mL.group(2) else d0
        if rhs.startswith("unchanged"):
            out.append((d0, d1, "passthrough", d0))
        else:
            mT = re.match(r't\[(\d+)(?:\.\.(\d+))?\]', rhs)
            mB = re.match(r'[a-zA-Z]+\[(\d+)(?:\.\.(\d+))?\]', rhs)
            if mT:
                out.append((d0, d1, "temp", int(mT.group(1))))
            elif mB:
                out.append((d0, d1, "slot", int(mB.group(1))))
    return out


def eval_block_output(prog, base_vec):
    """base_vec: 256 ints. Returns 256-int next state."""
    res = prog.eval(base_vec)
    temps = res["temps"]
    outmap = parse_next(prog.outputs)
    nxt = np.array(base_vec, dtype=np.int64).copy()   # default = unchanged
    for d0, d1, kind, src in outmap:
        for off in range(d1 - d0 + 1):
            dst = d0 + off
            if kind == "passthrough":
                nxt[dst] = base_vec[src + off]
            elif kind == "temp":
                nxt[dst] = temps[("t", src + off)]
            elif kind == "slot":
                nxt[dst] = base_vec[src + off]
    return nxt


# =========================================================================
def main():
    e = Emu()
    print("=" * 70)
    print("STEP 1-2: bit-exact verification of the readable parsers")
    print("=" * 70)

    epi = parse_file(os.path.join(HERE, "epilogue_readable.txt"))
    blk = parse_file(os.path.join(HERE, "block1_readable.txt"))

    B = 64
    X = random_inputs(B)

    # ---- epilogue needle ----
    s63 = s63_batch(e, X)
    gt_needle = e.batch_needle(X)
    ok_epi = 0
    for j in range(B):
        v = epi.eval(s63[:, j])["needle"]
        ok_epi += (v == int(gt_needle[j]))
    print(f"epilogue: needle matches emu on {ok_epi}/{B} random inputs")

    # ---- block1 next-state ----
    s1 = state_at(e, X, 1)
    s2 = state_at(e, X, 2)
    ok_blk = 0
    mism = None
    for j in range(B):
        nxt = eval_block_output(blk, s1[:, j])
        if np.array_equal(nxt, s2[:, j]):
            ok_blk += 1
        elif mism is None:
            mism = np.nonzero(nxt != s2[:, j])[0]
    print(f"block1  : next-state matches s2 on {ok_blk}/{B} random inputs"
          + (f"  (first mismatch slots: {mism[:8]})" if mism is not None else ""))

    # =====================================================================
    print()
    print("=" * 70)
    print("STEP 3: empirical s63 bounds (NOT a proof) from random inputs")
    print("=" * 70)
    Bbig = 4000
    Xb = random_inputs(Bbig)
    s63b = s63_batch(e, Xb)
    emp_lo = s63b.min(axis=1)
    emp_hi = s63b.max(axis=1)
    print(f"over {Bbig} inputs: per-slot max in [{int(emp_hi.min())},"
          f" {int(emp_hi.max())}]; #slots that stay in [0,1] = "
          f"{int(((emp_lo>=0)&(emp_hi<=1)).sum())}/256; "
          f"#slots always 0 = {int((emp_hi==0).sum())}")

    # =====================================================================
    print()
    print("=" * 70)
    print("STEP 4: ReLU census on the needle's dependency cone")
    print("=" * 70)
    cone = epi.needle_cone()
    n_relu_cone = sum(1 for v in cone if epi.by_var[v].is_relu)
    print(f"needle depends on {len(cone)} temps "
          f"({n_relu_cone} of them ReLU, {len(cone)-n_relu_cone} linear)")

    def census_on_cone(lo, hi, label):
        pr = epi.propagate(lo, hi)
        st = pr["relu_state"]
        c = {"active": 0, "zero": 0, "ambiguous": 0}
        for v in cone:
            if epi.by_var[v].is_relu:
                c[st[v]] += 1
        print(f"  [{label}] cone ReLUs: active={c['active']} "
              f"zero={c['zero']} AMBIGUOUS={c['ambiguous']}   "
              f"needle in [{pr['needle_lo']}, {pr['needle_hi']}]")
        return pr, c

    # empirical bounds
    pr_emp, c_emp = census_on_cone(emp_lo, emp_hi, "empirical s63")

    # sound (diverged) bounds from stage0 npz, if present
    npz = os.path.join(ROOT, "results", "solver", "stage0_boundary_bounds.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        census_on_cone(z["lo63"], z["hi63"], "sound lo63/hi63")

    # =====================================================================
    print()
    print("=" * 70)
    print("STEP 5: linearization test")
    print("=" * 70)
    if c_emp["ambiguous"] == 0:
        coeffs, bias = epi.linearize(pr_emp["relu_state"])
        print(f"LINEARIZES (under empirical box): {len(coeffs)} nonzero slot "
              f"coeffs, bias {bias}")
        # verify the linear form reproduces needle inside the empirical box
        good = 0
        for j in range(B):
            lin = bias + sum(cf * int(s63[idx[1], j])
                             for idx, cf in coeffs.items())
            good += (lin == epi.eval(s63[:, j])["needle"])
        print(f"linear form matches full epilogue on {good}/{B} in-box inputs")
    else:
        print(f"NOT linearizable to a single affine map of s63: "
              f"{c_emp['ambiguous']} ReLUs on the needle cone stay ambiguous "
              f"even under the (tight) empirical box.")
        print("The epilogue is a genuine deep ReLU network, not an affine score.")


if __name__ == "__main__":
    main()
