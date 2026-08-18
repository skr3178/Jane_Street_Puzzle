"""Three probes on the zone structure (user's experiments 7/8/9).

7) Zone dataflow: boolean dependency closure through one block (block 1),
   aggregated to zone->zone reads; plus which slots are pass-through.
8) Slot traces: chosen slots x chosen inputs across all 63 boundaries.
9) Ladder lanes: descendants of the 4 (+128) cells at pos 1, tracked through
   the block; convergence structure, same for -128 and +-64.

Run under trex:
  /home/satya/anaconda3/envs/trex/bin/python results/dynamic/zone_probes.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from emu import Emu  # noqa: E402

PRO_END, NBLK, PERIOD = 17, 63, 42
Z = np.load(os.path.join(ROOT, "results", "static", "emu_layers.npz"))

ZONES = [("A:0-32", 0, 32), ("B:33-63", 33, 63), ("C:64", 64, 64),
         ("D:65-159", 65, 159), ("E:160", 160, 160), ("F:161-191", 161, 191),
         ("G:192-246", 192, 246), ("H:247", 247, 247), ("I:248", 248, 248),
         ("J:249-255", 249, 255)]


def adj(k):
    """|W| != 0 as boolean (out x in)."""
    r, c, s = Z[f"r{k}"], Z[f"c{k}"], Z[f"s{k}"]
    A = np.zeros(tuple(s), dtype=bool)
    A[r, c] = True
    return A


def zone_of(i):
    for name, a, b in ZONES:
        if a <= i <= b:
            return name
    return "?"


def probe7():
    # dependency closure through block 1 (layers 59..100): out slot i depends
    # on in slot j
    R = None
    for p in range(PERIOD):
        A = adj(PRO_END + PERIOD + p)
        R = A if R is None else (A.astype(np.int16) @ R.astype(np.int16) > 0)
    R = np.asarray(R, dtype=bool)          # (256 out, 256 in)
    passthrough = [i for i in range(256)
                   if R[i].sum() == 1 and R[i, i]]
    M = {}
    for on, oa, ob in ZONES:
        for iname, ia, ib in ZONES:
            cnt = int(R[oa:ob + 1, ia:ib + 1].sum())
            if cnt:
                M[(on, iname)] = cnt
    with open(f"{HERE}/zone_flow.txt", "w") as fh:
        fh.write("block-1 dependency closure, aggregated to zones\n")
        fh.write(f"pass-through slots (depend only on self): {len(passthrough)}\n")
        fh.write(f"  {passthrough}\n\n")
        fh.write("out-zone <- in-zone : #dependency pairs\n")
        for (on, iname), cnt in sorted(M.items()):
            fh.write(f"  {on:10s} <- {iname:10s} : {cnt}\n")
        # per out-zone summary: which zones feed it
        fh.write("\nsummary (out-zone <- set of in-zones):\n")
        for on, oa, ob in ZONES:
            feeds = sorted({iname for (o2, iname) in M if o2 == on})
            fh.write(f"  {on:10s} <- {feeds}\n")
    return len(passthrough)


def probe8(e):
    slots = [0, 33, 65, 100, 160, 180, 192, 220, 247, 248]
    texts = ["", "a", "b", "aa", "ab", "ba"]
    out = {}
    for t in texts:
        st = e.states(e.encode(t))
        out[t] = {s: [int(st[b][s]) for b in range(1, NBLK + 1)] for s in slots}
    with open(f"{HERE}/slot_traces.txt", "w") as fh:
        fh.write("value of slot s entering blocks 1..62 and epilogue (63 vals)\n")
        for t in texts:
            fh.write(f"\ninput {t!r}:\n")
            for s in slots:
                v = out[t][s]
                fh.write(f"  slot {s:3d} [{zone_of(s)}]: {v}\n")
    # quick diffs
    with open(f"{HERE}/slot_traces.txt", "a") as fh:
        fh.write("\npairwise: which slots differ anywhere between inputs\n")
        for a, b in [("", "a"), ("a", "b"), ("aa", "ab"), ("ab", "ba")]:
            d = [s for s in slots if out[a][s] != out[b][s]]
            fh.write(f"  {a!r} vs {b!r}: differing slots {d}\n")


def probe9():
    # lanes: rows at pos1 receiving coefficient c
    k1 = PRO_END + PERIOD + 1
    r, c, v = Z[f"r{k1}"], Z[f"c{k1}"], Z[f"v{k1}"]
    lanes = {}
    for coeff in (128, -128, 64, -64):
        kk = k1 if abs(coeff) == 128 else PRO_END + PERIOD + 3
        rr, cc, vv = Z[f"r{kk}"], Z[f"c{kk}"], Z[f"v{kk}"]
        rows = sorted(set(int(x) for x in rr[vv == coeff]))
        lanes[coeff] = (kk, rows)
    with open(f"{HERE}/ladder_lanes.txt", "w") as fh:
        for coeff, (kk, rows) in lanes.items():
            fh.write(f"coeff {coeff} at layer {kk} (pos {kk - PRO_END - PERIOD}): "
                     f"rows {rows}\n")
        # track descendants of each +128 row separately through end of block
        fh.write("\ndescendant tracking from pos-1 rows (one vector per lane):\n")
        kk, rows128 = lanes[128]
        n_out = int(Z[f"s{kk}"][0])
        vecs = []
        for row in rows128:
            u = np.zeros(n_out, dtype=bool)
            u[row] = True
            vecs.append(u)
        first_meet = {}
        for p in range(2, PERIOD):
            A = adj(PRO_END + PERIOD + p)
            vecs = [(A.astype(np.int16) @ u.astype(np.int16) > 0) for u in vecs]
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    if (i, j) not in first_meet and (vecs[i] & vecs[j]).any():
                        first_meet[(i, j)] = p
        fh.write(f"  lanes = rows {rows128} (from +128 cells)\n")
        fh.write(f"  first position where lane descendant sets intersect: "
                 f"{ {f'{rows128[i]}&{rows128[j]}': p for (i, j), p in sorted(first_meet.items())} }\n")
        for i, u in enumerate(vecs):
            hits = np.nonzero(u)[0]
            byz = {}
            for s in hits:
                byz.setdefault(zone_of(int(s)), []).append(int(s))
            fh.write(f"  lane row {rows128[i]}: descendants at block output = "
                     f"{ {z: len(ss) for z, ss in sorted(byz.items())} }\n")
        # do all four lanes end identically?
        same = all((vecs[0] == u).all() for u in vecs[1:])
        fh.write(f"  all four lanes have identical block-output footprint: {same}\n")


def main():
    e = Emu()
    n_pass = probe7()
    print(f"7) zone_flow.txt written; pass-through slots: {n_pass}")
    probe8(e)
    print("8) slot_traces.txt written")
    probe9()
    print("9) ladder_lanes.txt written")


if __name__ == "__main__":
    main()
