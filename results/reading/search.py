#!/usr/bin/env python3
"""
Optimized stochastic search for the Jane Street puzzle.

Runs simulated annealing with batch perturbations, multiple random restarts,
and early termination as soon as needle >= 1 is found.
"""

import numpy as np
import sys
import time
import argparse

sys.path.insert(0, '.')
from emu import Emu

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--restarts', type=int, default=60, help='number of restarts')
    p.add_argument('--steps', type=int, default=1500, help='steps per restart')
    p.add_argument('--batch', type=int, default=64, help='candidates per batch')
    p.add_argument('--timeout', type=int, default=540, help='max seconds')
    p.add_argument('--seed', type=int, default=None, help='random seed (None = time-based)')
    return p.parse_args()

def main():
    args = parse_args()
    e = Emu()

    # Deterministic seed for reproducibility, or use current time
    if args.seed is None:
        args.seed = int(time.time()) % 2**32
    rng = np.random.default_rng(args.seed)
    print(f"Using seed {args.seed}")

    def needle(x):
        return e.needle(np.asarray(x, dtype=np.int64))

    def batch_needle(X):
        # X: shape (B, 55) -> return array of needles
        return e.batch_needle(np.asarray(X, dtype=np.int64).T)  # emu expects (55, B)

    best_overall = -999
    best_x = None
    t0 = time.time()
    total_evals = 0

    for restart in range(args.restarts):
        if time.time() - t0 > args.timeout:
            print(f"Timeout reached after {args.timeout}s")
            break

        # ---------- Random restart ----------
        # Start with a random-length prefix of printable ASCII (32-126)
        x = np.zeros(55, dtype=np.int64)
        L = int(rng.integers(1, 56))
        x[:L] = rng.integers(32, 127, L)
        cur = needle(x)
        total_evals += 1

        # Simulated annealing schedule
        T = 4.0          # initial temperature (soft)
        cooling = 0.997  # per-step decay

        for step in range(args.steps):
            # Generate a batch of perturbed candidates
            B = args.batch
            cand = np.repeat(x[None, :], B, axis=0)

            for bi in range(B):
                # Perturb 1–3 random bytes to random 0..255 values
                k = int(rng.integers(1, 4))
                idx = rng.integers(0, 55, k)
                cand[bi, idx] = rng.integers(0, 256, k)

            vals = batch_needle(cand)
            total_evals += B

            # Best candidate in batch
            j = int(np.argmax(vals))
            best_proposal = vals[j]
            best_cand = cand[j]

            # Annealing acceptance: accept if better, or with probability exp(delta/T)
            delta = best_proposal - cur
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-6)):
                x = best_cand
                cur = int(best_proposal)

            # Track global best
            if cur > best_overall:
                best_overall = cur
                best_x = x.copy()
                if best_overall >= 1:
                    # Early exit: we found a winner
                    print(f"FOUND WINNER at restart {restart+1}, step {step+1}")
                    break

            # Cool down
            T *= cooling

        if best_overall >= 1:
            break

    # Final report
    elapsed = time.time() - t0
    print(f"\n=== Search completed ===")
    print(f"Total evaluations: {total_evals}")
    print(f"Time: {elapsed:.0f}s")
    print(f"Restarts used: {restart+1}")
    print(f"Best needle reached: {best_overall}")
    if best_overall >= 1:
        print("SUCCESS: found an input that triggers needle >= 1")
        # We do NOT print the vector itself to avoid spoilers.
        # Instead, verify with the original PyTorch model if needed.
    else:
        print("FAIL: best needle still negative; try increasing restarts/steps.")

if __name__ == "__main__":
    main()