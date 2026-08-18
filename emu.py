"""Exact int64 emulator of the puzzle model.

All weights/biases are exact integers, so the whole forward pass is integer
arithmetic: x -> max(W x + b, 0), 2721 times. Sparse COO in pure numpy int64
(no scipy: its current build segfaults against numpy 2.5), bit-exact, with
free access to every intermediate value.

    from emu import Emu
    e = Emu()                      # builds/loads cache (no torch needed after first run)
    e.out(x)                       # x: int64[55] -> final scalar (post-clamp)
    e.needle(x)                    # pre-clamp value of the final layer
    e.states(x)                    # 64 state vectors entering blocks 0..62 + epilogue
    e.batch_needle(X)              # X: int64[55, B] -> needle per column
    e.encode("text")               # 55-vector via the model's own encoder (needs torch)

CLI:  python emu.py --verify      # bit-exactness check vs torch
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "results", "static", "emu_layers.npz")
PRO_END, NBLK, PERIOD = 17, 63, 42
BLK_END = PRO_END + NBLK * PERIOD


def _build_cache():
    sys.path.insert(0, HERE)
    import torch.nn as nn
    from probe import load
    m = load(device="cpu", quiet=True)
    lins = [x for x in m.children() if isinstance(x, nn.Linear)]
    blobs = {}
    for k, L in enumerate(lins):
        W = L.weight.detach().numpy()
        b = L.bias.detach().numpy()
        assert np.all(W == np.round(W)) and np.all(b == np.round(b))
        r, c = np.nonzero(W)
        blobs[f"r{k}"] = r.astype(np.int32)
        blobs[f"c{k}"] = c.astype(np.int32)
        blobs[f"v{k}"] = W[r, c].astype(np.int64)
        blobs[f"s{k}"] = np.array(W.shape)
        blobs[f"b{k}"] = b.astype(np.int64)
    np.savez_compressed(CACHE, n=len(lins), **blobs)


class Emu:
    def __init__(self):
        if not os.path.exists(CACHE):
            _build_cache()
        z = np.load(CACHE)
        self.n = int(z["n"])
        self.layers = []
        for k in range(self.n):
            self.layers.append((z[f"r{k}"], z[f"c{k}"], z[f"v{k}"],
                                int(z[f"s{k}"][0]), z[f"b{k}"]))
        self._torch_model = None

    # ---- core passes ----------------------------------------------------
    def _run(self, x, record_at=()):
        """x: int64[55] or int64[55,B]. Returns final PRE-clamp vector, recorded."""
        x = np.asarray(x, dtype=np.int64)
        rec = {}
        for k in range(self.n):
            if k in record_at:
                rec[k] = x.copy()
            r, c, v, out_d, b = self.layers[k]
            if x.ndim == 1:
                y = b.copy()
                np.add.at(y, r, v * x[c])
            else:
                y = np.repeat(b[:, None], x.shape[1], axis=1)
                np.add.at(y, r, v[:, None] * x[c])
            x = y if k == self.n - 1 else np.maximum(y, 0, out=y)
        return x, rec

    def needle(self, x):
        y, _ = self._run(x)
        return int(y[0])

    def out(self, x):
        return max(self.needle(x), 0)

    def batch_needle(self, X):
        y, _ = self._run(np.asarray(X, dtype=np.int64))
        return y[0]

    def states(self, x):
        """State vectors entering block 0..62 and the epilogue (64 vectors)."""
        marks = [PRO_END + b * PERIOD for b in range(NBLK)] + [BLK_END]
        _, rec = self._run(x, record_at=set(marks))
        return [rec[k] for k in marks]

    def max_abs_activation(self, x):
        x = np.asarray(x, dtype=np.int64)
        peak = int(np.abs(x).max())
        for k in range(self.n):
            r, c, v, out_d, b = self.layers[k]
            y = b.copy()
            np.add.at(y, r, v * x[c])
            peak = max(peak, int(np.abs(y).max()))
            x = y if k == self.n - 1 else np.maximum(y, 0, out=y)
        return peak

    # ---- string path (torch) -------------------------------------------
    def encode(self, text):
        if self._torch_model is None:
            sys.path.insert(0, HERE)
            from probe import load
            self._torch_model = load(device="cpu", quiet=True)
        v = self._torch_model._batcher.encode(text).detach().cpu().numpy()
        assert np.all(v == np.round(v)), "encoder output not integral!"
        return v.astype(np.int64)


def _verify():
    import time
    e = Emu()
    sys.path.insert(0, HERE)
    from probe import load, needle as tneedle
    m = load(device="cpu", quiet=True)
    tests = ["", "hello", "a", "Zz9 !~", "the quick brown fox", "x" * 40]
    ok = True
    peak = 0
    for s in tests:
        v = e.encode(s)
        tn, tout = tneedle(m, s)
        en = e.needle(v)
        eo = e.out(v)
        peak = max(peak, e.max_abs_activation(v))
        match = (en == int(tn)) and (eo == int(tout))
        ok &= match
        print(f"{'OK ' if match else 'MISMATCH'} torch=({tn:.10g},{tout:.10g}) "
              f"emu=({en},{eo})  {s!r}")
    print(f"peak |activation| over tests: {peak}  (int64 max ~9.2e18; "
          f"float32 exact to 1.6e7)")
    t = time.time()
    e.needle(np.zeros(55, dtype=np.int64))
    t1 = time.time() - t
    t = time.time()
    X = np.zeros((55, 256), dtype=np.int64)
    e.batch_needle(X)
    dt = time.time() - t
    print(f"single forward: {t1*1000:.1f} ms; "
          f"batched: 256 forwards in {dt*1000:.0f} ms ({dt/256*1e6:.0f} us each)")
    print("VERIFIED bit-exact" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--verify" in sys.argv:
        raise SystemExit(_verify())
    print(__doc__)
