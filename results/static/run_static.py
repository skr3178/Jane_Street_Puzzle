"""Static (input-independent) analysis of the puzzle model.

Covers: value alphabet + exactness, per-position vocab, sparsity/spy plots,
structural typing, ranks, layer dedup by hash, variation at block positions
1/2/29, bias-flip bitmap, module-sequence topology, prologue/epilogue detail.

Run from repo root:  python results/static/run_static.py
Outputs land next to this file.
"""
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from probe import load  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
PRO_END, NBLK, PERIOD = 17, 63, 42
BLK_END = PRO_END + NBLK * PERIOD  # 2663


def main():
    m = load(device="cpu", quiet=True)
    mods = list(m.children())
    lins = [x for x in mods if isinstance(x, nn.Linear)]
    assert len(lins) == 2721

    # ---- topology: module sequence -------------------------------------
    seq_types = [type(x).__name__ for x in mods]
    alternates = all(
        t == ("Linear" if i % 2 == 0 else "ReLU") for i, t in enumerate(seq_types)
    )
    with open(f"{OUT}/topology.txt", "w") as fh:
        fh.write(f"container: {type(m).__name__}, {len(mods)} modules\n")
        fh.write(f"module type counts: {Counter(seq_types).most_common()}\n")
        fh.write(f"strict Linear/ReLU alternation: {alternates}\n")
        fh.write("=> pure chain: no residuals, no branches, no other op types\n")
        fh.write(f"input dim: {lins[0].weight.shape[1]}, "
                 f"final out dim: {lins[-1].weight.shape[0]}\n")

    # ---- dedup by hash --------------------------------------------------
    hash2id, layer_ids = {}, []
    for L in lins:
        h = hashlib.sha1(
            L.weight.detach().numpy().tobytes()
            + (L.bias.detach().numpy().tobytes() if L.bias is not None else b"")
        ).hexdigest()[:12]
        if h not in hash2id:
            hash2id[h] = len(hash2id)
        layer_ids.append(hash2id[h])
    n_distinct = len(hash2id)

    # RLE of the id sequence
    rle, prev, cnt = [], layer_ids[0], 1
    for i in layer_ids[1:]:
        if i == prev:
            cnt += 1
        else:
            rle.append((prev, cnt)); prev, cnt = i, 1
    rle.append((prev, cnt))
    with open(f"{OUT}/hash_sequence.txt", "w") as fh:
        fh.write(f"{n_distinct} distinct layers out of {len(lins)}\n")
        fh.write("id sequence (run-length encoded as id*count):\n")
        fh.write(" ".join(f"{i}*{c}" if c > 1 else str(i) for i, c in rle) + "\n")
        fh.write("\nfull id sequence, one per line: layer_idx\tid\n")
        for k, i in enumerate(layer_ids):
            fh.write(f"{k}\t{i}\n")

    # which in-block positions vary across the 63 blocks
    pos_variation = []
    for p in range(PERIOD):
        ids = {layer_ids[PRO_END + b * PERIOD + p] for b in range(NBLK)}
        pos_variation.append((p, len(ids)))
    varying = [p for p, c in pos_variation if c > 1]

    # ---- per-distinct-layer stats: uniques, nnz, type, rank -------------
    first_of_id = {}
    for k, i in enumerate(layer_ids):
        first_of_id.setdefault(i, k)

    def classify(W, b):
        out_d, in_d = W.shape
        nz = W != 0
        nnz = int(nz.sum())
        if nnz == 0:
            return "zero", nnz
        rows_nz = nz.sum(1)
        cols_nz = nz.sum(0)
        vals = W[nz]
        if out_d == in_d and np.array_equal(W, np.eye(out_d, dtype=W.dtype)):
            return "identity", nnz
        if (rows_nz == 1).all() and (cols_nz <= 1).all():
            return ("permutation" if (vals == 1).all() else "permutation_scaled"), nnz
        if (rows_nz <= 1).all():
            return ("selection" if (vals == 1).all() else "selection_scaled"), nnz
        if out_d == in_d:
            delta_nnz = int((W != np.eye(out_d, dtype=W.dtype)).sum())
            if delta_nnz / W.size < 0.05:
                return "identity_plus_delta", nnz
        if nnz / W.size < 0.05:
            return "sparse", nnz
        return "dense", nnz

    def dyadic_k(v):
        # smallest k<=24 with v*2^k integral, else None
        for k in range(25):
            s = v * (1 << k)
            if s == round(s):
                return k
        return None

    info = {}
    global_vals = set()
    for h, i in hash2id.items():
        L = lins[first_of_id[i]]
        W = L.weight.detach().numpy()
        b = L.bias.detach().numpy() if L.bias is not None else np.zeros(W.shape[0], W.dtype)
        wu = np.unique(W)
        bu = np.unique(b)
        global_vals.update(wu.tolist())
        global_vals.update(bu.tolist())
        typ, nnz = classify(W, b)
        rank = None
        if typ in ("sparse", "dense", "identity_plus_delta"):
            rank = int(np.linalg.matrix_rank(W.astype(np.float64)))
        info[i] = dict(
            hash=h, shape=list(W.shape), nnz=nnz, frac=nnz / W.size, type=typ,
            rank=rank, n_wvals=len(wu), n_bvals=len(bu),
            wvals=wu.tolist() if len(wu) <= 16 else None,
            bvals=bu.tolist() if len(bu) <= 16 else None,
            bias_zero=bool((b == 0).all()),
            count=layer_ids.count(i),
        )
    with open(f"{OUT}/distinct_layers.json", "w") as fh:
        json.dump(info, fh, indent=1)

    # ---- global value alphabet + exactness ------------------------------
    gv = sorted(global_vals)
    with open(f"{OUT}/global_values.txt", "w") as fh:
        fh.write(f"{len(gv)} distinct values across all weights+biases\n")
        fh.write("value\texact_fraction\tdyadic(v*2^k int, k)\n")
        for v in gv:
            fr = Fraction(v).limit_denominator(10**6)
            exact = "EXACT" if float(fr) == v else "approx"
            fh.write(f"{v!r}\t{fr}={exact}\tk={dyadic_k(v)}\n")

    # histogram (only meaningful if the set is large)
    if len(gv) > 64:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        allw = np.concatenate([lins[first_of_id[i]].weight.detach().numpy().ravel()
                               for i in info])
        plt.hist(allw, bins=201)
        plt.yscale("log")
        plt.title(f"weight values across distinct layers ({len(gv)} uniques)")
        plt.savefig(f"{OUT}/value_hist.png", dpi=110)
        plt.close()

    # ---- per-position vocabulary & typing -------------------------------
    def pos_label(k):
        if k < PRO_END:
            return f"P{k:02d}"
        if k < BLK_END:
            return f"B{(k - PRO_END) % PERIOD:02d}"
        return f"E{k - BLK_END:02d}"

    pos_vals = defaultdict(set)
    pos_types = defaultdict(Counter)
    pos_shapes = defaultdict(set)
    for k, i in enumerate(layer_ids):
        lb = pos_label(k)
        d = info[i]
        pos_types[lb][d["type"]] += 1
        pos_shapes[lb].add(tuple(d["shape"]))
        L = lins[k] if d["n_wvals"] > 16 else None
        if d["wvals"] is not None:
            pos_vals[lb].update(d["wvals"])
        else:
            pos_vals[lb].update(np.unique(lins[first_of_id[i]].weight.detach().numpy()).tolist())
        if d["bvals"] is not None:
            pos_vals[lb].update(d["bvals"])

    with open(f"{OUT}/per_position_vocab.txt", "w") as fh:
        fh.write("label\tshapes\ttypes(count)\tn_vals\tvals_if_small\n")
        for lb in sorted(pos_vals):
            vs = sorted(pos_vals[lb])
            fh.write(f"{lb}\t{sorted(pos_shapes[lb])}\t{dict(pos_types[lb])}\t"
                     f"{len(vs)}\t{vs if len(vs) <= 12 else '...'}\n")

    # ---- full listing ----------------------------------------------------
    with open(f"{OUT}/layer_types.txt", "w") as fh:
        fh.write("idx\tlabel\tid\tshape\tnnz%\ttype\trank\tbias\n")
        for k, i in enumerate(layer_ids):
            d = info[i]
            fh.write(f"{k}\t{pos_label(k)}\t{i}\t{d['shape']}\t{d['frac']*100:.2f}\t"
                     f"{d['type']}\t{d['rank'] if d['rank'] is not None else '-'}\t"
                     f"{'0' if d['bias_zero'] else 'nz'}\n")

    # one-block instruction listing
    with open(f"{OUT}/block_listing.txt", "w") as fh:
        fh.write("block-1 (layers 59..100) as 42-position listing\n")
        fh.write("pos\tidx\tid\tshape\tnnz%\ttype\trank\tn_ids_across_63_blocks\n")
        for p in range(PERIOD):
            k = PRO_END + 1 * PERIOD + p
            i = layer_ids[k]
            d = info[i]
            fh.write(f"{p}\t{k}\t{i}\t{d['shape']}\t{d['frac']*100:.2f}\t{d['type']}\t"
                     f"{d['rank'] if d['rank'] is not None else '-'}\t{pos_variation[p][1]}\n")

    # ---- spy plots -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def spy_grid(idxs, fname, title, ncol=7):
        nrow = -(-len(idxs) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.4 * nrow))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for j, k in enumerate(idxs):
            ax = axes[j // ncol][j % ncol]
            W = lins[k].weight.detach().numpy()
            ax.imshow(W != 0, aspect="auto", cmap="gray_r", interpolation="nearest")
            d = info[layer_ids[k]]
            ax.set_title(f"{pos_label(k)} {tuple(W.shape)}\n{d['type']}", fontsize=6)
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{fname}", dpi=110)
        plt.close(fig)

    spy_grid(range(PRO_END + PERIOD, PRO_END + 2 * PERIOD), "spy_block1.png",
             "block 1 (layers 59..100): nonzero pattern per layer")
    spy_grid(range(0, PRO_END), "spy_prologue.png",
             "prologue (layers 0..16): nonzero pattern per layer", ncol=6)
    spy_grid(range(BLK_END, 2721), "spy_epilogue.png",
             "epilogue (layers 2663..2720): nonzero pattern per layer", ncol=8)

    # ---- variation at positions 1, 2, 29 (and 0 for completeness) -------
    for p in varying:
        ref = lins[PRO_END + 1 * PERIOD + p]
        Wr = ref.weight.detach().numpy(); br = ref.bias.detach().numpy()
        with open(f"{OUT}/variants_pos{p}.txt", "w") as fh:
            fh.write(f"in-block position {p} (layer 17+42*b+{p}), diffs vs block 1\n")
            fh.write(f"block-1 shape {list(Wr.shape)}\n\n")
            for b in range(NBLK):
                L = lins[PRO_END + b * PERIOD + p]
                W = L.weight.detach().numpy(); bb = L.bias.detach().numpy()
                if W.shape != Wr.shape:
                    fh.write(f"block {b:2d}: shape {list(W.shape)} (different shape)\n")
                    continue
                wd = int((W != Wr).sum()); bd = int((bb != br).sum())
                where_b = np.nonzero(bb != br)[0]
                extra = ""
                if len(where_b) and len(where_b) <= 40:
                    extra = "  bias diff idx: " + ",".join(map(str, where_b.tolist()))
                fh.write(f"block {b:2d}: {wd:5d} weight diffs, {bd:3d} bias diffs{extra}\n")

    # ---- bias bitmap at position 0, indices 224..253 --------------------
    bm = np.full((NBLK, 30), -1.0)
    for b in range(NBLK):
        L = lins[PRO_END + b * PERIOD + 0]
        bias = L.bias.detach().numpy()
        if len(bias) >= 254:
            bm[b] = bias[224:254]
    np.save(f"{OUT}/bias_bitmap.npy", bm)
    plt.figure(figsize=(8, 12))
    plt.imshow(bm, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(); plt.xlabel("bias index - 224"); plt.ylabel("block")
    plt.title("position-0 bias values, indices 224..253, per block (-1 = shape mismatch)")
    plt.savefig(f"{OUT}/bias_bitmap.png", dpi=110)
    plt.close()

    # ---- summary ---------------------------------------------------------
    type_counts = Counter(info[i]["type"] for i in layer_ids)
    with open(f"{OUT}/summary.txt", "w") as fh:
        fh.write(f"distinct layers: {n_distinct} / 2721\n")
        fh.write(f"global value alphabet size: {len(gv)}\n")
        small = [v for v in gv if abs(v) <= 8]
        fh.write(f"values (|v|<=8 shown, {len(small)}/{len(gv)}): {small[:64]}\n")
        fh.write(f"layer type counts (all 2721): {dict(type_counts)}\n")
        fh.write(f"in-block positions with >1 distinct id: {varying}\n")
        fh.write("per-position distinct-id counts: "
                 f"{[(p, c) for p, c in pos_variation if c > 1]}\n")
    print(open(f"{OUT}/summary.txt").read())


if __name__ == "__main__":
    main()
