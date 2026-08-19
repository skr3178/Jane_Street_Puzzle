#!/usr/bin/env python3
"""
Revised stochastic search adapted for the measured 10ms/input emulator speed.
Runs within a 600s timeout (~60,000 individual evals).
"""

import numpy as np
import sys
import time
import argparse

sys.path.insert(0, '.')
from emu import Emu

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--timeout', type=int, default=540, help='max seconds')
    p.add_argument('--seed', type=int, default=None, help='random seed')
    return p.parse_args()

def main():
    args = parse_args()
    e = Emu()

    if args.seed is None:
        args.seed = int(time.time()) % 2**32
    rng = np.random.default_rng(args.seed)
    print(f"Using seed {args.seed}")

    def needle(x):
        return e.needle(np.asarray(x, dtype=np.int64))

    def batch_needle(X):
        # X: shape (B, 55) -> needles
        return e.batch_needle(np.asarray(X, dtype=np.int64).T)

    best_overall = -999
    best_x = None
    t0 = time.time()
    total_evals = 0

    # Tuned hyperparameters for 10ms/input:
    RESTARTS = 8        # Fewer restarts, but longer runs per restart
    STEPS_PER = 150     # 8 * 150 = 1200 batches total
    BATCH_SIZE = 64     # 1200 * 64 = 76,800 evals (~768s) -> fits in 600s with some margin

    for restart in range(RESTARTS):
        if time.time() - t0 > args.timeout:
            print(f"Timeout after {args.timeout}s")
            break

        # Random restart: start with random prefix length (1-55)
        x = np.zeros(55, dtype=np.int64)
        L = int(rng.integers(1, 56))
        x[:L] = rng.integers(32, 127, L)
        cur = needle(x)
        total_evals += 1

        T = 5.0          # Start warm to accept bad moves (~80% acceptance at start)
        cooling = 0.99   # Cool slower to explore longer

        for step in range(STEPS_PER):
            B = BATCH_SIZE
            cand = np.repeat(x[None, :], B, axis=0)

            for bi in range(B):
                # Aggressive mutation: change 2-5 bytes per candidate
                k = int(rng.integers(2, 6))
                idx = rng.integers(0, 55, k)
                cand[bi, idx] = rng.integers(0, 256, k)

            vals = batch_needle(cand)
            total_evals += B

            j = int(np.argmax(vals))
            best_proposal = vals[j]
            best_cand = cand[j]

            delta = best_proposal - cur
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-6)):
                x = best_cand
                cur = int(best_proposal)

            if cur > best_overall:
                best_overall = cur
                best_x = x.copy()
                if best_overall >= 1:
                    print(f"FOUND WINNER at restart {restart+1}, step {step+1}")
                    break

            T *= cooling

        if best_overall >= 1:
            break

    elapsed = time.time() - t0
    print(f"\n=== Search Report ===")
    print(f"Total evals: {total_evals}")
    print(f"Time: {elapsed:.0f}s")
    print(f"Best needle: {best_overall}")
    if best_overall >= 1:
        print("SUCCESS! (vector hidden)")
    else:
        print("FAIL. Try increasing --timeout or running multiple seeds.")

if __name__ == "__main__":
    main()
