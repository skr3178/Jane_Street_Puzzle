"""Step 2+3: state movies, input-encoding characterization, raw-vector climb.

Run with the trex env (torch inference segfaults in base anaconda):
    /home/satya/anaconda3/envs/trex/bin/python results/dynamic/run_dynamic.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from emu import Emu, NBLK  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def state_movie(e, x, tag):
    states = e.states(np.asarray(x, dtype=np.int64))
    width = max(len(s) for s in states)
    M = np.full((len(states), width), np.nan)
    for i, s in enumerate(states):
        M[i, : len(s)] = s
    np.savez(f"{HERE}/states_{tag}.npz", *states)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    im0 = axes[0].imshow(M, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title(f"state entering block b — input {tag}")
    im1 = axes[1].imshow(M != 0, aspect="auto", interpolation="nearest", cmap="gray_r")
    axes[1].set_title("nonzero mask")
    for ax in axes:
        ax.set_xlabel("state index")
        ax.set_ylabel("block")
    fig.tight_layout()
    fig.savefig(f"{HERE}/states_{tag}.png", dpi=110)
    plt.close(fig)
    nz = [(int((s != 0).sum())) for s in states]
    mx = [int(s.max()) for s in states]
    return dict(tag=tag, nonzero_per_block=nz, max_per_block=mx,
                needle=e.needle(np.asarray(x, dtype=np.int64)))


def characterize_encoding(e):
    import string
    printable = string.printable[:95]
    rows = {}
    rows[""] = e.encode("")
    for ch in printable:
        rows[ch] = e.encode(ch)
    for k in (2, 3, 5, 10, 20, 40, 60):
        rows["a" * k] = e.encode("a" * k)
    for s in ("ab", "ba", "abc", "cba", "a b", "b a"):
        rows[s] = e.encode(s)
    A = np.stack(list(rows.values()))
    keys = list(rows.keys())
    np.savez(f"{HERE}/encodings.npz", A=A, keys=np.array(keys, dtype=object))
    base = rows[""]
    report = {
        "base_is_zero": bool((base == 0).all()),
        "value_range": [int(A.min()), int(A.max())],
        "all_integer": True,
        "dims_ever_nonzero": int((A != 0).any(0).sum()),
        "dims_constant_across_probes": int((A == A[0]).all(0).sum()),
    }
    # per-dim ranges
    report["per_dim_max"] = A.max(0).tolist()
    # single-char structure: how many dims does one char touch
    touch = [int((rows[ch] != base).sum()) for ch in printable]
    report["dims_changed_by_single_char_minmax"] = [min(touch), max(touch)]
    # length response
    report["nnz_vs_len_a"] = {k: int((rows["a" * k] != 0).sum())
                              for k in (1, 2, 3, 5, 10, 20, 40, 60)}
    report["order_sensitive"] = bool((rows["ab"] != rows["ba"]).any())
    with open(f"{HERE}/encoding_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    return report


def climb(e, lo, hi, iters=4000, seed=0):
    """Greedy coordinate climb on needle over integer vectors in [lo, hi]^55."""
    rng = np.random.default_rng(seed)
    x = np.zeros(55, dtype=np.int64)
    best = e.needle(x)
    hist = [best]
    stall = 0
    for it in range(iters):
        i = rng.integers(0, 55)
        step = rng.choice([-4, -2, -1, 1, 2, 4])
        cand = x.copy()
        cand[i] = np.clip(cand[i] + step, lo, hi)
        n = e.needle(cand)
        if n > best:
            x, best, stall = cand, n, 0
            hist.append(best)
        else:
            stall += 1
        if best >= 0:
            break
        if stall > 800:
            break
    return x, best, hist


def main():
    e = Emu()
    os.makedirs(HERE, exist_ok=True)
    summary = {}

    # Step 2: state movies for raw unit vector and zero vector
    summary["movie_e0"] = state_movie(e, np.eye(55, dtype=np.int64)[0], "e0")
    summary["movie_zero"] = state_movie(e, np.zeros(55, dtype=np.int64), "zero")

    # Step 3a: encoding characterization
    summary["encoding"] = characterize_encoding(e)

    # Step 3b: raw-vector climb, bounds from observed encoder range
    lo, hi = summary["encoding"]["value_range"]
    xb, nb, hist = climb(e, lo, hi)
    np.save(f"{HERE}/climb_best_vec.npy", xb)
    summary["climb"] = {"bounds": [int(lo), int(hi)],
                        "best_needle": int(nb),
                        "trajectory": [int(v) for v in hist],
                        "note": "best vector saved to climb_best_vec.npy, not printed"}

    with open(f"{HERE}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: (v if k != "encoding" else "see encoding_report.json")
                      for k, v in summary.items()}, indent=1)[:2000])


if __name__ == "__main__":
    main()
