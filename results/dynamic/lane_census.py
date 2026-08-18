"""Test the packed-bitfield hypothesis mechanically.

A) Dynamic lane census: per state slot, observed value set across a diverse
   input corpus at all 63 block boundaries -> classify slots as const / binary
   {0,1} / small / wide. Image + table.
B) Static op-template census for block 1: normalize ops to (coeff pattern,
   lane offsets); compare first half (pos 18-28) vs second half (pos 31-41).
C) Ladder pairs: co-occurrence of c and c-1 coefficients on the same source.

Run under trex:
  /home/satya/anaconda3/envs/trex/bin/python results/dynamic/lane_census.py
"""
import json
import os
import string
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from emu import Emu  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRO_END, NBLK, PERIOD = 17, 63, 42


def corpus():
    rng = np.random.default_rng(7)
    printable = string.printable[:95]
    texts = ["", "a", "z", "azaz", "hello world", "a" * 55]
    for L in (1, 2, 3, 5, 8, 13, 21, 34, 55):
        for _ in range(12):
            texts.append("".join(rng.choice(list(printable), L)))
    for _ in range(40):
        L = int(rng.integers(1, 56))
        texts.append("".join(rng.choice(list(string.ascii_lowercase), L)))
    return texts


def lane_census(e):
    texts = corpus()
    nslots = 256
    seen = [set() for _ in range(nslots)]          # pooled over blocks 1..62
    permax = np.zeros((NBLK, nslots), dtype=np.int64)
    for t in texts:
        st = e.states(e.encode(t))
        for b in range(1, NBLK):                    # skip block-0 (width 224)
            s = st[b]
            permax[b] = np.maximum(permax[b], s)
            for i in range(len(s)):
                if len(seen[i]) < 40:
                    seen[i].add(int(s[i]))
    cls = []
    for i in range(nslots):
        v = seen[i]
        if v <= {0}:
            c = "always0"
        elif v <= {0, 1}:
            c = "binary"
        elif v == {128} or v <= {0, 128}:
            c = "const128"
        elif max(v) <= 8:
            c = "small<=8"
        else:
            c = "wide"
        cls.append(c)
    with open(f"{HERE}/lane_map.txt", "w") as fh:
        fh.write(f"corpus: {len(texts)} strings; slots pooled over blocks 1..62\n")
        fh.write("slot\tclass\tmax\tn_distinct(<=40 tracked)\tsample\n")
        for i in range(nslots):
            sv = sorted(seen[i])
            fh.write(f"{i}\t{cls[i]}\t{max(sv)}\t{len(sv)}\t{sv[:10]}\n")
    counts = Counter(cls)
    # contiguous runs of same class
    runs, start = [], 0
    for i in range(1, nslots + 1):
        if i == nslots or cls[i] != cls[start]:
            runs.append((start, i - 1, cls[start]))
            start = i
    order = {"always0": 0, "binary": 1, "small<=8": 2, "wide": 3, "const128": 4}
    img = np.array([[order[c] for c in cls]])
    plt.figure(figsize=(16, 2.2))
    plt.imshow(img, aspect="auto", interpolation="nearest", cmap="tab10", vmin=0, vmax=9)
    plt.yticks([])
    plt.xlabel("state slot")
    plt.title("slot class: 0=always0 1=binary 2=small<=8 3=wide 4=const128")
    plt.tight_layout()
    plt.savefig(f"{HERE}/lane_map.png", dpi=120)
    plt.close()
    return counts, runs


def op_templates():
    z = np.load(os.path.join(ROOT, "results", "static", "emu_layers.npz"))

    def rows(k):
        r, c, v = z[f"r{k}"], z[f"c{k}"], z[f"v{k}"]
        b = z[f"b{k}"]
        out = defaultdict(list)
        for rr, cc, vv in zip(r, c, v):
            out[int(rr)].append((int(cc), int(vv)))
        return out, b

    def layer_templates(k):
        out, b = rows(k)
        tpl = Counter()
        for i, cells in out.items():
            cells.sort()
            coeffs = tuple(sorted(v for _, v in cells))
            offs = tuple(np.diff([c for c, _ in cells])) if len(cells) > 1 else ()
            tpl[(coeffs, offs, int(b[i]))] += 1
        return tpl

    half1 = Counter()
    half2 = Counter()
    for p in range(18, 29):
        half1 += layer_templates(PRO_END + PERIOD + p)
    for p in range(31, 42):
        half2 += layer_templates(PRO_END + PERIOD + p)
    inter = sum((half1 & half2).values())
    tot1, tot2 = sum(half1.values()), sum(half2.values())
    with open(f"{HERE}/half_templates.txt", "w") as fh:
        fh.write("op template = (sorted coeffs, col offsets, bias) per row, "
                 "block 1\n")
        fh.write(f"half1 (pos18-28): {tot1} rows, {len(half1)} distinct templates\n")
        fh.write(f"half2 (pos31-41): {tot2} rows, {len(half2)} distinct templates\n")
        fh.write(f"multiset intersection: {inter} rows "
                 f"({inter/max(tot1,tot2)*100:.1f}% of larger half)\n\n")
        fh.write("top templates half1 vs count in half2:\n")
        for t, n in half1.most_common(25):
            fh.write(f"  {t}: {n} vs {half2.get(t, 0)}\n")
    return tot1, tot2, inter, len(half1), len(half2)


def ladder_pairs():
    z = np.load(os.path.join(ROOT, "results", "static", "emu_layers.npz"))
    res = []
    for p in range(0, 16):
        k = PRO_END + PERIOD + p            # block 1
        v = z[f"v{k}"]
        c = z[f"c{k}"]
        vals = Counter(int(x) for x in v)
        pows = [w for w in (2, 4, 8, 16, 32, 64, 128) if vals.get(w) or vals.get(-w)]
        pairs = []
        for w in pows:
            for s in (1, -1):
                a, b = s * w, s * w - (1 if s > 0 else -1)
                if vals.get(a) and vals.get(b):
                    cols_a = set(int(x) for x in c[v == a])
                    cols_b = set(int(x) for x in c[v == b])
                    pairs.append((a, b, vals[a], vals[b],
                                  len(cols_a & cols_b)))
        res.append((p, dict(vals), pairs))
    with open(f"{HERE}/ladder_pairs.txt", "w") as fh:
        fh.write("block-1 positions 0..15: coefficient census and (c, c-1) "
                 "co-occurrence [same source cols]\n")
        for p, vals, pairs in res:
            fh.write(f"pos {p:2d}: coeff counts {dict(sorted(vals.items()))}\n")
            for a, b, na, nb, shared in pairs:
                fh.write(f"        pair ({a},{b}): {na}x/{nb}x, "
                         f"{shared} shared source cols\n")
    return res


def main():
    e = Emu()
    counts, runs = lane_census(e)
    print("A) lane classes:", dict(counts))
    print("   contiguous runs:")
    for a, b, c in runs:
        print(f"     [{a:3d}..{b:3d}] {c}")
    t1, t2, inter, d1, d2 = op_templates()
    print(f"B) halves: {t1} vs {t2} rows; {d1}/{d2} distinct templates; "
          f"multiset overlap {inter} rows ({inter/max(t1,t2)*100:.1f}%)")
    ladder_pairs()
    print("C) ladder pair census -> ladder_pairs.txt")


if __name__ == "__main__":
    main()
