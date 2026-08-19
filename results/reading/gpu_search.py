"""GPU-batched needle evaluator + strong search.

Loads all 2721 layers as dense float32 GPU tensors and runs batched forward,
returning the PRE-clamp needle. ~500x faster than the CPU emulator, so the
search can do millions of evaluations. Any candidate reaching needle>=1 (or the
best found) is re-checked with the EXACT int64 emulator before being trusted.
Float32 is exact for |v|<1.6e7 (observed peak ~440); used only to GUIDE search,
never as ground truth. The winning/near-miss vector is NOT printed.

  /home/satya/anaconda3/envs/trex/bin/python results/reading/gpu_search.py \
      --evals 5000000 --pop 4096
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from emu import Emu  # noqa: E402


class GPUEval:
    def __init__(self, dev="cuda"):
        e = Emu()
        self.n = e.n
        self.dev = dev
        self.W, self.b = [], []
        prev = 55
        for k in range(e.n):
            r, c, v, od, bias = e.layers[k]
            Wd = np.zeros((od, prev), dtype=np.float32)
            Wd[r, c] = v
            self.W.append(torch.tensor(Wd, device=dev))
            self.b.append(torch.tensor(bias.astype(np.float32), device=dev))
            prev = od

    @torch.no_grad()
    def needle(self, X):                      # X: (B,55) int/float -> (B,) needle
        x = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        for k in range(self.n):
            x = x @ self.W[k].T + self.b[k]
            if k < self.n - 1:
                x.clamp_(min=0)
        return x[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=3_000_000)
    ap.add_argument("--pop", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", default="byte", choices=["byte", "print"])
    args = ap.parse_args()
    g = GPUEval()
    e = Emu()
    rng = np.random.default_rng(args.seed)
    lo, hi = (32, 127) if args.domain == "print" else (0, 256)

    # population-based: keep top-K, mutate. Guided by GPU needle.
    P = args.pop
    pop = np.zeros((P, 55), dtype=np.int64)
    for i in range(P):
        L = int(rng.integers(0, 56))
        if L:
            pop[i, :L] = rng.integers(lo, hi, L)
    scores = g.needle(pop).cpu().numpy()
    best = float(scores.max()); best_x = pop[int(scores.argmax())].copy()
    t0 = time.time(); done = P
    gen = 0
    while done < args.evals and best < 1:
        gen += 1
        # keep top half, breed/mutate into bottom half
        order = np.argsort(-scores)
        keep = order[: P // 2]
        elite = pop[keep]
        children = elite[rng.integers(0, len(elite), P - len(keep))].copy()
        # mutate 1-3 bytes each
        for c in children:
            k = int(rng.integers(1, 4))
            idx = rng.integers(0, 55, k)
            c[idx] = rng.integers(lo, hi, k)
        # occasional crossover
        if gen % 3 == 0:
            partners = elite[rng.integers(0, len(elite), len(children))]
            mask = rng.random(children.shape) < 0.5
            children = np.where(mask, children, partners)
        pop = np.vstack([elite, children])
        scores = g.needle(pop).cpu().numpy()
        done += len(children)
        m = int(scores.argmax())
        if scores[m] > best:
            best = float(scores[m]); best_x = pop[m].copy()
        if gen % 20 == 0:
            print(f"gen {gen:5d}  evals {done:9d}  best(gpu) {best:.1f}  "
                  f"{done/(time.time()-t0):.0f} eval/s", flush=True)
    dt = time.time() - t0
    # verify best with EXACT emulator
    exact = int(e.needle(best_x))
    print(f"\n=== GPU search done: {done} evals in {dt:.0f}s "
          f"({done/dt:.0f} eval/s) ===")
    print(f"best needle (gpu float32): {best:.2f}")
    print(f"best needle (EXACT int64): {exact}   (acceptance needs >= 1)")
    print("SUCCESS" if exact >= 1 else "no acceptance; vector not printed")


if __name__ == "__main__":
    main()
