"""Readable rewrite of the EPILOGUE (layers 2663..2720): final state -> needle.

Same mechanical transforms as readable_block.py (zone-named slots, banks,
collapsed runs), ending with the explicit needle formula (pre-clamp) and the
acceptance condition needle >= 1. No interpretation.

Run under trex:
  /home/satya/anaconda3/envs/trex/bin/python results/reading/readable_epilogue.py
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
import readable_block as RB  # noqa: E402  (reuse ZONE, banks, rendering)

PRO_END, NBLK, PERIOD = 17, 63, 42
BLK_END = PRO_END + NBLK * PERIOD   # 2663

ZONE = RB.ZONE


def build_epilogue():
    e = Emu()
    c = Collapser(256, in_prefix="s", in_nonneg=True)
    prev = 256
    for k in range(BLK_END, e.n - 1):        # all but the final Linear
        r, cc, v, od, b = e.layers[k]
        Wd = np.zeros((od, prev), dtype=np.int64)
        Wd[r, cc] = v
        c.push_layer(Wd, b, note=f"E{k - BLK_END:02d}")
        prev = od
    # final Linear (1 x 48): needle = bias + sum w_j * vec[j], UNclamped
    r, cc, v, od, b = e.layers[e.n - 1]
    Wd = np.zeros((od, prev), dtype=np.int64)
    Wd[r, cc] = v
    coeffs, bias = {}, int(b[0])
    for j in range(prev):
        w = int(Wd[0, j])
        if w == 0:
            continue
        val = c.vec[j]
        if val[0] == "const":
            bias += w * val[1]
        else:
            var, cf = val[1], val[2]
            coeffs[var] = coeffs.get(var, 0) + w * cf
    coeffs = {v2: cf for v2, cf in coeffs.items() if cf != 0}
    return c, coeffs, bias


def main():
    c, ncoeffs, nbias = build_epilogue()
    # verify numerically against emulator epilogue before writing
    e = Emu()
    ok = True
    rng = np.random.default_rng(9)
    for t in range(30):
        L = int(rng.integers(0, 56))
        x = np.zeros(55, dtype=np.int64)
        if L:
            x[:L] = rng.integers(1, 256, L)
        s63 = e.states(x)[NBLK]
        # replay collapsed epilogue on s63
        vals = {}
        for name, coeffs, bias, clamp, note in c.defs:
            s = bias
            for var, cf in coeffs.items():
                base = s63[var[1]] if var[0] == "in" else vals[var[1]]
                s += cf * int(base)
            vals[int(name[1:])] = max(s, 0) if clamp else s
        got = nbias + sum(cf * (s63[v[1]] if v[0] == "in" else vals[v[1]])
                          for v, cf in ncoeffs.items())
        want = e.needle(x)
        if got != want:
            ok = False
            print(f"MISMATCH t={t}: {got} vs {want}")
            break
    print(f"faithfulness vs emulator on 30 random inputs: "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)

    lines = RB.render(c)
    # replace the outputs section (irrelevant here) with the needle formula
    cut = lines.index("# --- block outputs (256-wide next state) ---") - 1
    lines = lines[:cut]
    lines.append("")
    lines.append("# --- THE NEEDLE (pre-clamp; model output = max(needle, 0)) ---")
    parts = []
    for var, cf in sorted(ncoeffs.items(), key=lambda kv: str(kv[0])):
        nm = (f"{ZONE[var[1]]}[{var[1]}]" if var[0] == "in" else f"t[{var[1]}]")
        parts.append(f"{'+' if cf > 0 else '-'} {abs(cf)}*{nm}"
                     if abs(cf) != 1 else f"{'+' if cf > 0 else '-'} {nm}")
    lines.append(f"needle = {' '.join(parts)}"
                 + (f" {'+' if nbias > 0 else '-'} {abs(nbias)}" if nbias else ""))
    lines.append(f"# {len(ncoeffs)} terms, bias {nbias}")
    lines.append("")
    lines.append("# --- ACCEPTANCE: needle >= 1  (every known input <= -12) ---")

    n_banks = sum(1 for L in lines if L.startswith("for i"))
    n_single = sum(1 for L in lines if L.startswith("t["))
    hdr = [
        "# Readable rewrite of the EPILOGUE (layers 2663..2720)",
        "# input = final state s63 (256 slots, zone-named); t[] = temps",
        f"# {len(c.defs)} raw ops -> {n_banks} banks + {n_single} singletons",
        "# zones: bit=binary{0,1}  num=small<=8  wide=large  K=const128  zero=always0",
        "# (zone labels are from BLOCK-boundary statistics; inside the epilogue",
        "#  they describe the incoming s63 slots, not the temps)",
        "",
    ]
    path = os.path.join(HERE, "epilogue_readable.txt")
    open(path, "w").write("\n".join(hdr + lines) + "\n")
    print(f"{len(c.defs)} ops -> {n_banks} banks + {n_single} singletons; "
          f"needle terms {len(ncoeffs)}, bias {nbias}")
    print(f"saved {path} ({len(hdr) + len(lines)} lines)")


if __name__ == "__main__":
    main()
