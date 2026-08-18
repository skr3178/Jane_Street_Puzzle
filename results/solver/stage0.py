"""Stage 0 — faithfulness prep for the solver (SOLVER_PLAN.md).

(i)   Encoder domain sweep: true per-slot code set + string structure.
(ii)  Frozen-store identity PROOF for slots 192..255 in every block.
(iii) Interval propagation: proven |value| bounds everywhere + proven
      Boolean slots at block boundaries.
(iv)  Acceptance predicate note.

Run under trex (torch needed for (i) only):
  /home/satya/anaconda3/envs/trex/bin/python results/solver/stage0.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from emu import Emu  # noqa: E402

PRO_END, NBLK, PERIOD = 17, 63, 42
BLK_END = PRO_END + NBLK * PERIOD
Z = np.load(os.path.join(ROOT, "results", "static", "emu_layers.npz"))


def dense(k):
    r, c, v, s = Z[f"r{k}"], Z[f"c{k}"], Z[f"v{k}"], Z[f"s{k}"]
    W = np.zeros(tuple(s), dtype=np.int64)
    W[r, c] = v
    return W, Z[f"b{k}"]


# ---------------------------------------------------------------- (i) encoder
def encoder_sweep(e):
    rep = {"per_char": {}, "errors": {}}
    codes = {}
    for o in range(0, 256):
        ch = chr(o)
        try:
            v = e.encode(ch)
        except Exception as ex:
            rep["errors"][o] = repr(ex)[:80]
            continue
        nz = np.nonzero(v)[0]
        if len(nz) == 0:
            codes[o] = 0
        elif len(nz) == 1 and nz[0] == 0:
            codes[o] = int(v[0])
        else:
            codes[o] = ("MULTI", nz.tolist(), v[nz].tolist())
    rep["identity_code_chars"] = sum(
        1 for o, c in codes.items() if isinstance(c, int) and c == o)
    rep["zero_code_chars"] = sorted(
        o for o, c in codes.items() if isinstance(c, int) and c == 0)
    rep["nonidentity"] = {o: c for o, c in codes.items()
                          if not (isinstance(c, int) and c in (o, 0))}
    intcodes = [c for c in codes.values() if isinstance(c, int)]
    rep["code_range_singles"] = [int(min(intcodes)), int(max(intcodes))]
    # a couple of wider unicode probes
    uni = {}
    for ch in ["é", "ü", "€", "中", "\U0001F600"]:
        try:
            v = e.encode(ch)
            nz = np.nonzero(v)[0]
            uni[repr(ch)] = {"nnz": int(len(nz)),
                             "vals": v[nz][:4].tolist(), "pos": nz[:4].tolist()}
        except Exception as ex:
            uni[repr(ch)] = {"error": repr(ex)[:80]}
    rep["unicode_probes"] = uni
    # structure: position-independence and prefix/padding
    va = e.encode("abc")
    rep["abc_vector_head"] = va[:6].tolist()
    rep["position_independent"] = bool(
        va[0] == e.encode("a")[0] and va[1] == e.encode("b" )[0] and
        va[2] == e.encode("c")[0])
    # can a "hole" occur mid-string? test a char that maps to 0, if any
    hole_chars = rep["zero_code_chars"]
    if hole_chars:
        hc = chr(hole_chars[0])
        vh = e.encode(f"a{hc}c")
        rep["mid_string_zero_example"] = {"char_ord": hole_chars[0],
                                          "vector_head": vh[:5].tolist()}
    rep["overlong"] = {"len60_nnz": int((e.encode("a" * 60) != 0).sum()),
                       "len56_head_tail": e.encode("a" * 56)[[0, 54]].tolist()}
    with open(f"{HERE}/stage0_encoder.json", "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    return rep


# ------------------------------------------------------- (ii) frozen identity
def frozen_identity_proof():
    """Exact structural proof: within each block, each wire 192..255 must move
    through every layer via a column with exactly one nonzero (+1), landing on
    a row with exactly one nonzero and bias 0. Then (state >= 0) => identity."""
    failures = []
    for b in range(1, NBLK):   # blocks 1..62 carry the 256-wide state; block 0
        wires = {j: j for j in range(192, 256)}   # is the 224-wide init block
        ok = True
        for p in range(PERIOD):
            k = PRO_END + b * PERIOD + p
            W, bias = dense(k)
            col_nnz = (W != 0).sum(0)
            row_nnz = (W != 0).sum(1)
            new = {}
            for slot, j in wires.items():
                col = W[:, j]
                nz = np.nonzero(col)[0]
                if len(nz) != 1 or col[nz[0]] != 1:
                    failures.append((b, p, slot, "col", len(nz)))
                    ok = False
                    continue
                i = int(nz[0])
                if row_nnz[i] != 1 or bias[i] != 0:
                    failures.append((b, p, slot, "row/bias", int(row_nnz[i]),
                                     int(bias[i])))
                    ok = False
                    continue
                new[slot] = i
            wires = new
            if not ok:
                break
        if ok and any(wires.get(s) != s for s in range(192, 256)):
            failures.append((b, "final-permutation", dict(wires)))
    with open(f"{HERE}/stage0_frozen.txt", "w") as fh:
        if failures:
            fh.write(f"NOT identity: {len(failures)} failures\n")
            for f in failures[:80]:
                fh.write(f"  {f}\n")
        else:
            fh.write("PROVEN: slots 192..255 are exact identity pass-throughs "
                     "(coeff +1, bias 0, isolated wires, position-preserving) "
                     "in all 63 blocks.\n")
    return len(failures)


# ------------------------------------------------------ (iii) interval bounds
def interval_propagation(x_lo=0, x_hi=126):
    e = Emu()
    lo = np.full(55, x_lo, dtype=np.int64)
    hi = np.full(55, x_hi, dtype=np.int64)
    peak_pre = 0
    boundary_bounds = {}
    marks = {PRO_END + b * PERIOD: b for b in range(NBLK)}
    marks[BLK_END] = NBLK
    for k in range(e.n):
        if k in marks:
            boundary_bounds[marks[k]] = (lo.copy(), hi.copy())
        W, b = dense(k)
        Wp = np.maximum(W, 0)
        Wn = np.minimum(W, 0)
        nlo = b + Wp @ lo + Wn @ hi
        nhi = b + Wp @ hi + Wn @ lo
        peak_pre = max(peak_pre, int(np.abs(nlo).max()), int(np.abs(nhi).max()))
        if k < e.n - 1:
            lo = np.maximum(nlo, 0)
            hi = np.maximum(nhi, 0)
        else:
            lo, hi = nlo, nhi
    needle_bounds = (int(lo[0]), int(hi[0]))
    # proven Boolean slots: hi <= 1 at EVERY boundary from block 1 on (256-wide)
    per_slot_hi = np.zeros(256, dtype=np.int64)
    for bidx, (blo, bhi) in boundary_bounds.items():
        if len(bhi) == 256:
            per_slot_hi = np.maximum(per_slot_hi, bhi)
    proven_bool = [i for i in range(256) if per_slot_hi[i] <= 1]
    proven_zero = [i for i in range(256) if per_slot_hi[i] == 0]
    width_needed = int(np.ceil(np.log2(peak_pre + 1))) + 1
    rep = {
        "input_domain": [x_lo, x_hi],
        "peak_abs_prerelu_bound": peak_pre,
        "signed_bv_width_needed": width_needed,
        "needle_bounds": needle_bounds,
        "n_proven_bool_slots": len(proven_bool),
        "n_proven_zero_slots": len(proven_zero),
        "proven_bool_slots": proven_bool,
        "census_binary_was": 159,
        "per_slot_hi_max_over_boundaries": per_slot_hi.tolist(),
    }
    with open(f"{HERE}/stage0_bounds.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    np.savez(f"{HERE}/stage0_boundary_bounds.npz",
             **{f"lo{b}": v[0] for b, v in boundary_bounds.items()},
             **{f"hi{b}": v[1] for b, v in boundary_bounds.items()})
    return rep


def main():
    e = Emu()
    enc = encoder_sweep(e)
    print("(i) encoder:", {k: enc[k] for k in
          ("identity_code_chars", "zero_code_chars", "code_range_singles",
           "position_independent")})
    print("    nonidentity count:", len(enc["nonidentity"]),
          " unicode probes:", len(enc["unicode_probes"]))
    nf = frozen_identity_proof()
    print(f"(ii) frozen identity failures: {nf}")
    rep = interval_propagation()
    print("(iii) bounds:", {k: rep[k] for k in
          ("peak_abs_prerelu_bound", "signed_bv_width_needed",
           "needle_bounds", "n_proven_bool_slots", "n_proven_zero_slots")})
    print("(iv) acceptance: output = max(needle, 0); target must be needle>=1 "
          "(structural: model ends Linear->ReLU).")


if __name__ == "__main__":
    main()
