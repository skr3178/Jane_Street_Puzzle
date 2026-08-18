"""Collapse pass-through wiring: emit each segment (prologue / one block /
epilogue) as an explicit list of integer operations.

Soundness: the model is Linear->ReLU throughout, so every inter-layer value is
>= 0. For x >= 0 and c > 0, relu(c*x) = c*x, so rows that are a single positive
scaling of a nonneg source are aliases, not ops. Rows whose affine expression
is provably <= 0 are dead (const 0); provably >= 0 rows need no clamp. The raw
55-dim input is NOT assumed nonneg (prologue rows over it always clamp).

Run:  python results/static/collapse.py
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch.nn as nn  # noqa: E402
from probe import load  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
PRO_END, NBLK, PERIOD = 17, 63, 42
BLK_END = PRO_END + NBLK * PERIOD


class Collapser:
    """Symbolic forward pass over integer Linear+ReLU layers."""

    def __init__(self, in_width, in_prefix="s", in_nonneg=True):
        # a value is ("const", c) or ("scaled", var_id, coeff>0)
        self.defs = []          # (name, expr_coeffs{var:c}, bias, clamped, note)
        self.nonneg_input = in_nonneg
        self.in_names = [f"{in_prefix}{i}" for i in range(in_width)]
        self.vec = [("scaled", ("in", i), 1) for i in range(in_width)]
        self.n_alias = self.n_dead = self.n_const = 0

    def var_name(self, v):
        return self.in_names[v[1]] if v[0] == "in" else f"v{v[1]}"

    def is_nonneg_var(self, v):
        return self.nonneg_input if v[0] == "in" else True

    def push_layer(self, W, b, note=""):
        W = W.astype(np.int64)
        b = b.astype(np.int64)
        new_vec = []
        for i in range(W.shape[0]):
            coeffs, bias = {}, int(b[i])
            for j in np.nonzero(W[i])[0]:
                w = int(W[i, j])
                val = self.vec[j]
                if val[0] == "const":
                    bias += w * val[1]
                else:
                    v, c = val[1], val[2]
                    coeffs[v] = coeffs.get(v, 0) + w * c
            coeffs = {v: c for v, c in coeffs.items() if c != 0}
            if not coeffs:                                    # pure constant
                new_vec.append(("const", max(bias, 0)))
                self.n_const += 1
                continue
            all_src_nonneg = all(self.is_nonneg_var(v) for v in coeffs)
            if all_src_nonneg and all(c < 0 for c in coeffs.values()) and bias <= 0:
                new_vec.append(("const", 0))                  # provably <= 0
                self.n_dead += 1
                continue
            if len(coeffs) == 1 and bias == 0:
                (v, c), = coeffs.items()
                if c > 0 and self.is_nonneg_var(v):           # exact alias
                    new_vec.append(("scaled", v, c))
                    self.n_alias += 1
                    continue
            provably_nonneg = (all_src_nonneg
                               and all(c > 0 for c in coeffs.values()) and bias >= 0)
            vid = len(self.defs)
            self.defs.append((f"v{vid}", coeffs, bias, not provably_nonneg,
                              f"{note} row {i}"))
            new_vec.append(("scaled", ("v", vid), 1))
        self.vec = new_vec

    def expr_str(self, coeffs, bias):
        terms = []
        for v, c in sorted(coeffs.items(), key=lambda kv: str(kv[0])):
            n = self.var_name(v)
            terms.append(f"{'+' if c > 0 else '-'}{'' if abs(c) == 1 else str(abs(c)) + '*'}{n}")
        if bias:
            terms.append(f"{'+' if bias > 0 else ''}{bias}")
        return " ".join(terms)

    def dump(self, path, title):
        with open(path, "w") as fh:
            fh.write(f"{title}\n")
            fh.write(f"inputs: {self.in_names[0]}..{self.in_names[-1]} "
                     f"({'assumed >=0' if self.nonneg_input else 'sign unknown'})\n")
            fh.write(f"ops: {len(self.defs)}   aliases: {self.n_alias}   "
                     f"dead(->0): {self.n_dead}   consts: {self.n_const}\n")
            clamped = sum(1 for d in self.defs if d[3])
            fh.write(f"of the ops, {clamped} clamp (relu matters), "
                     f"{len(self.defs) - clamped} are provably nonneg (pure affine)\n\n")
            for name, coeffs, bias, clamp, note in self.defs:
                rhs = self.expr_str(coeffs, bias)
                fh.write(f"{name} = {'relu(' + rhs + ')' if clamp else rhs:<60s} # {note}\n")
            fh.write("\noutputs:\n")
            for i, val in enumerate(self.vec):
                if val[0] == "const":
                    fh.write(f"out[{i}] = {val[1]}\n")
                else:
                    c = val[2]
                    fh.write(f"out[{i}] = {'' if c == 1 else str(c) + '*'}{self.var_name(val[1])}\n")


def collapse_range(lins, lo, hi, in_prefix, in_nonneg, label, fname):
    W0 = lins[lo].weight.detach().numpy()
    c = Collapser(W0.shape[1], in_prefix, in_nonneg)
    for k in range(lo, hi):
        L = lins[k]
        c.push_layer(L.weight.detach().numpy(), L.bias.detach().numpy(),
                     note=f"L{k}(pos{k - lo})")
        # sanity: constants stay small
    c.dump(os.path.join(OUT, fname), label)
    per_terms = Counter(len(d[1]) for d in c.defs)
    return dict(label=label, ops=len(c.defs), alias=c.n_alias, dead=c.n_dead,
                const=c.n_const, clamped=sum(1 for d in c.defs if d[3]),
                terms_hist=dict(sorted(per_terms.items())))


def main():
    m = load(device="cpu", quiet=True)
    lins = [x for x in m.children() if isinstance(x, nn.Linear)]
    stats = []
    stats.append(collapse_range(lins, 0, PRO_END, "x", False,
                                "PROLOGUE layers 0..16: 55 raw inputs -> state",
                                "prologue_collapsed.txt"))
    for b in (0, 1, 17, 33, 49):
        lo = PRO_END + b * PERIOD
        stats.append(collapse_range(
            lins, lo, lo + PERIOD, "s", True,
            f"BLOCK {b} layers {lo}..{lo + PERIOD - 1}: state -> state",
            f"block{b}_collapsed.txt"))
    stats.append(collapse_range(lins, BLK_END, 2721, "s", True,
                                "EPILOGUE layers 2663..2720: state -> scalar",
                                "epilogue_collapsed.txt"))
    with open(os.path.join(OUT, "collapse_stats.txt"), "w") as fh:
        for s in stats:
            fh.write(f"{s['label']}\n  ops={s['ops']} (clamping={s['clamped']}), "
                     f"aliases={s['alias']}, dead={s['dead']}, consts={s['const']}\n"
                     f"  terms-per-op histogram: {s['terms_hist']}\n")
    print(open(os.path.join(OUT, "collapse_stats.txt")).read())


if __name__ == "__main__":
    main()
