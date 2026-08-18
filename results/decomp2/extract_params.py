"""Extract the per-block parameter table: everything that varies across the
63 blocks, tabulated; the invariant skeleton is discarded as boilerplate.

Varying positions (established): pos 0 (unique per block), pos 1 (3 ids),
pos 2 (2 ids), pos 29 (16 ids, 4-cycle per regime).

Uses the emu weight cache only (no torch inference).
Run:  python results/decomp2/extract_params.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(ROOT, "results", "static", "emu_layers.npz")
PRO_END, NBLK, PERIOD = 17, 63, 42


def dense(z, k):
    r, c, v, (o, i) = z[f"r{k}"], z[f"c{k}"], z[f"v{k}"], z[f"s{k}"]
    W = np.zeros((o, i), dtype=np.int64)
    W[r, c] = v
    return W, z[f"b{k}"]


def main():
    z = np.load(CACHE)
    pos_layer = lambda b, p: PRO_END + b * PERIOD + p

    # group blocks by pos-0 shape signature + pos-1/2 identity
    sig = []
    for b in range(NBLK):
        shapes = tuple(tuple(z[f"s{pos_layer(b, p)}"]) for p in (0, 1, 2))
        sig.append(shapes)
    regimes = []
    seen = {}
    for b, s in enumerate(sig):
        if s not in seen:
            seen[s] = len(seen)
        regimes.append(seen[s])

    table = {"regime_of_block": regimes, "regime_shapes": {str(v): str(k) for k, v in seen.items()}}

    # ---- pos 0: varying cells within each regime ------------------------
    pos0 = {}
    for reg in sorted(set(regimes)):
        blocks = [b for b in range(NBLK) if regimes[b] == reg]
        Ws, Bs = zip(*[dense(z, pos_layer(b, 0)) for b in blocks])
        Wst = np.stack(Ws)
        Bst = np.stack(Bs)
        wvar = np.nonzero((Wst != Wst[0]).any(0))
        bvar = np.nonzero((Bst != Bst[0]).any(0))[0]
        pos0[reg] = {
            "blocks": blocks,
            "n_varying_weight_cells": int(len(wvar[0])),
            "varying_weight_cells_rows": sorted(set(int(r) for r in wvar[0])),
            "varying_weight_cols": sorted(set(int(c) for c in wvar[1])),
            "varying_bias_idx": [int(i) for i in bvar],
            "weight_values_per_block": {str(b): Wst[i][wvar].tolist()
                                        for i, b in enumerate(blocks)},
            "bias_values_per_block": {str(b): Bst[i][bvar].tolist()
                                      for i, b in enumerate(blocks)},
            "weight_cells": [[int(r), int(c)] for r, c in zip(*wvar)],
        }
    table["pos0"] = pos0

    # ---- pos 29: variant id per block + differing cells -----------------
    var29 = {}
    hashes = {}
    ids = []
    for b in range(NBLK):
        W, bias = dense(z, pos_layer(b, 29))
        h = (W.tobytes(), bias.tobytes())
        if h not in hashes:
            hashes[h] = len(hashes)
        ids.append(hashes[h])
    var29["variant_of_block"] = ids
    # differing cells among variants (all same shape?)
    shapes = {tuple(z[f"s{pos_layer(b, 29)}"]) for b in range(NBLK)}
    var29["shapes"] = [list(s) for s in shapes]
    if len(shapes) == 1:
        Ws = np.stack([dense(z, pos_layer(b, 29))[0] for b in range(NBLK)])
        cells = np.nonzero((Ws != Ws[0]).any(0))
        var29["n_varying_cells"] = int(len(cells[0]))
        var29["varying_rows"] = sorted(set(int(r) for r in cells[0]))
        var29["varying_cols"] = sorted(set(int(c) for c in cells[1]))
        uniq = sorted(set(ids))
        var29["values_per_variant"] = {}
        for u in uniq:
            b = ids.index(u)
            var29["values_per_variant"][str(u)] = Ws[b][cells].tolist()
        var29["cells"] = [[int(r), int(c)] for r, c in zip(*cells)]
    table["pos29"] = var29

    # ---- compact per-block summary line ---------------------------------
    lines = ["block | regime | pos29_variant | pos0 payload (varying values)"]
    for b in range(NBLK):
        reg = regimes[b]
        w = pos0[reg]["weight_values_per_block"][str(b)]
        bias = pos0[reg]["bias_values_per_block"][str(b)]
        lines.append(f"{b:2d} | R{reg} | v{ids[b]:2d} | w[{len(w)}]={w} b[{len(bias)}]={bias}")
    with open(os.path.join(HERE, "param_table.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "param_table.json"), "w") as fh:
        json.dump(table, fh, indent=1, default=int)

    # ---- structure-only summary (safe to print) -------------------------
    print("regimes:", regimes)
    print("pos29 variant per block:", ids)
    for reg, d in pos0.items():
        print(f"regime {reg}: blocks {d['blocks'][0]}..{d['blocks'][-1]}, "
              f"{d['n_varying_weight_cells']} varying weight cells "
              f"in rows {d['varying_weight_cells_rows'][:6]}"
              f"{'...' if len(d['varying_weight_cells_rows'])>6 else ''} "
              f"cols {d['varying_weight_cols'][:6]}"
              f"{'...' if len(d['varying_weight_cols'])>6 else ''}, "
              f"{len(d['varying_bias_idx'])} varying bias idx")
    if "n_varying_cells" in var29:
        print(f"pos29: {var29['n_varying_cells']} varying cells, "
              f"rows {var29['varying_rows'][:8]}..., cols {var29['varying_cols'][:8]}...")
    # value-range stats only (content stays in files)
    allw = [v for reg in pos0.values() for vals in reg["weight_values_per_block"].values() for v in vals]
    print("pos0 payload value stats: n per block varies; global min/max:",
          min(allw), max(allw), " distinct:", len(set(allw)))


if __name__ == "__main__":
    main()
