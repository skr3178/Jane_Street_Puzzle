"""Parse the *readable* block/epilogue rewrites and (try to) linearize the needle.

This is the Phase-1 tooling for the pasted plan, adapted to the ACTUAL file
format, which the pasted script did not handle:

  * `for i in [0..N]:  t[BASE+i] = relu(...)`   loop banks  (must be expanded)
  * index arithmetic inside brackets:  bit[32+i], t[2195+i], -2*zero[64+i]
  * zone-named base slots:  bit[] num[] wide[] K[] zero[]   (absolute slot 0..255)
  * needle line + `next ... = ...` block-output mapping

What it does
------------
  parse_file(path)            -> Program (expanded statements, needle, outputs)
  Program.eval(base_vec)      -> dict of every temp value + needle / next-state
  Program.propagate(lo, hi)   -> interval bounds on every temp + ReLU census
  Program.linearize(relu_state) -> needle as {slot: coeff}, bias   (if all fixed)

Base slots are indexed by ABSOLUTE slot number 0..255 regardless of zone name,
so bit[32] == slot 32, zero[64] == slot 64, wide[160] == slot 160.
"""
import re
import sys
from collections import defaultdict

HERE = __file__.rsplit("/", 1)[0]
ZONES = ("bit", "num", "wide", "K", "zero")

# ---------------------------------------------------------------------------
# term parsing:  " + t[5385] + 2*t[5386] - 200 "  ->  ({('t',5385):1,('t',5386):2}, -200)
# a variable ref is (kind, index) with kind in {'t','bit','num','wide','K','zero'}
_TOK = re.compile(r'([+-])\s*(\d+)\s*\*\s*([a-zA-Z]+)\[(\d+)\]'   # coeff*var
                  r'|([+-])\s*([a-zA-Z]+)\[(\d+)\]'                # +/- var
                  r'|([+-])\s*(\d+)')                              # +/- const


def parse_inner(s):
    coeffs = defaultdict(int)
    const = 0
    s = " " + s.strip()
    # normalise a leading term with no sign to '+'
    if s.lstrip()[0] not in "+-":
        s = " +" + s.lstrip()
    pos = 0
    for m in _TOK.finditer(s):
        pos = m.end()
        if m.group(3):                       # coeff*var
            sign = 1 if m.group(1) == '+' else -1
            coeffs[(m.group(3), int(m.group(4)))] += sign * int(m.group(2))
        elif m.group(6):                     # +/- var
            sign = 1 if m.group(5) == '+' else -1
            coeffs[(m.group(6), int(m.group(7)))] += sign
        else:                                # +/- const
            sign = 1 if m.group(8) == '+' else -1
            const += sign * int(m.group(9))
    return dict(coeffs), const


def subst_i(expr, i):
    """Replace the loop variable i inside bracket arithmetic: '5385+i' -> '5391'."""
    def repl(m):
        return "[" + str(int(m.group(1)) + i) + "]"
    return re.sub(r'\[(\d+)\+i\]', repl, expr)


class Stmt:
    __slots__ = ("var", "is_relu", "coeffs", "const")

    def __init__(self, var, is_relu, coeffs, const):
        self.var = var          # ('t', k)
        self.is_relu = is_relu
        self.coeffs = coeffs     # {(kind,idx): int}
        self.const = const


class Program:
    def __init__(self):
        self.stmts = []                 # in evaluation order (temp index order)
        self.by_var = {}                # ('t',k) -> Stmt
        self.needle = None              # (coeffs, const) or None
        self.outputs = []              # list of (dst(kind,idx), src) for block files
        self.n_raw_lines = 0
        self.n_loop_lines = 0

    # ---- evaluation --------------------------------------------------------
    def eval(self, base):
        """base: dict (kind,idx)->int, OR a 256-int sequence (absolute slot)."""
        if not isinstance(base, dict):
            base = {(z, i): int(base[i]) for z in ZONES for i in range(len(base))}
        val = {}

        def get(ref):
            if ref[0] == 't':
                return val[ref]
            return base.get(ref, 0)

        for st in self.stmts:
            acc = st.const + sum(c * get(r) for r, c in st.coeffs.items())
            val[st.var] = max(acc, 0) if st.is_relu else acc
        res = {"temps": val}
        if self.needle is not None:
            coeffs, const = self.needle
            res["needle"] = const + sum(c * get(r) for r, c in coeffs.items())
        return res

    # ---- interval propagation ---------------------------------------------
    def propagate(self, lo, hi):
        """lo,hi: 256-seq absolute-slot bounds. Returns per-temp (lo,hi) + census."""
        blo = {(z, i): int(lo[i]) for z in ZONES for i in range(len(lo))}
        bhi = {(z, i): int(hi[i]) for z in ZONES for i in range(len(hi))}
        tlo, thi = {}, {}
        census = {"active": 0, "zero": 0, "ambiguous": 0, "linear": 0}
        relu_state = {}   # ('t',k) -> 'active'|'zero'|'ambiguous'|None(linear)

        def L(ref):
            return tlo[ref] if ref[0] == 't' else blo[ref]

        def H(ref):
            return thi[ref] if ref[0] == 't' else bhi[ref]

        for st in self.stmts:
            a_lo = a_hi = st.const
            for r, c in st.coeffs.items():
                if c >= 0:
                    a_lo += c * L(r); a_hi += c * H(r)
                else:
                    a_lo += c * H(r); a_hi += c * L(r)
            if st.is_relu:
                if a_lo >= 0:
                    state = "active"
                elif a_hi <= 0:
                    state = "zero"
                else:
                    state = "ambiguous"
                census[state] += 1
                relu_state[st.var] = state
                tlo[st.var] = max(a_lo, 0)
                thi[st.var] = max(a_hi, 0)
            else:
                census["linear"] += 1
                relu_state[st.var] = None
                tlo[st.var] = a_lo
                thi[st.var] = a_hi
        nlo = nhi = None
        if self.needle is not None:
            coeffs, const = self.needle
            nlo = nhi = const
            for r, c in coeffs.items():
                if c >= 0:
                    nlo += c * L(r); nhi += c * H(r)
                else:
                    nlo += c * H(r); nhi += c * L(r)
        return {"tlo": tlo, "thi": thi, "census": census,
                "relu_state": relu_state, "needle_lo": nlo, "needle_hi": nhi}

    # ---- dependency cone of the needle ------------------------------------
    def needle_cone(self):
        """Set of temp vars the needle transitively depends on."""
        if self.needle is None:
            return set()
        seen = set()
        stack = [r for r in self.needle[0] if r[0] == 't']
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            st = self.by_var.get(v)
            if st:
                stack.extend(r for r in st.coeffs if r[0] == 't')
        return seen

    # ---- linearize: collapse needle to affine in base slots ----------------
    def linearize(self, relu_state):
        """Express needle as {(kind,idx): coeff}, bias, treating each ReLU per
        relu_state ('active'->identity, 'zero'->0). Raises if any needed ReLU is
        'ambiguous'. Returns (coeffs, bias, n_ambiguous_hit)."""
        memo = {}   # ('t',k) -> (dict coeffs over base slots, const)

        def resolve(ref):
            if ref[0] != 't':
                return ({ref: 1}, 0)
            if ref in memo:
                return memo[ref]
            st = self.by_var[ref]
            if st.is_relu and relu_state.get(ref) == "zero":
                memo[ref] = ({}, 0)
                return memo[ref]
            if st.is_relu and relu_state.get(ref) == "ambiguous":
                raise ValueError(f"ambiguous ReLU on cone: {ref}")
            # active relu -> identity, or plain linear
            acc = defaultdict(int)
            const = st.const
            for r, c in st.coeffs.items():
                sub_c, sub_k = resolve(r)
                for s, v in sub_c.items():
                    acc[s] += c * v
                const += c * sub_k
            memo[ref] = (dict(acc), const)
            return memo[ref]

        coeffs, const = self.needle
        acc = defaultdict(int)
        bias = const
        for r, c in coeffs.items():
            sub_c, sub_k = resolve(r)
            for s, v in sub_c.items():
                acc[s] += c * v
            bias += c * sub_k
        acc = {k: v for k, v in acc.items() if v != 0}
        return acc, bias


# ---------------------------------------------------------------------------
def parse_file(path):
    prog = Program()
    lhs_re = re.compile(r'^(?:for i in \[0\.\.(\d+)\]:\s*)?'
                        r't\[(\d+)(?:\+i)?\]\s*=\s*(.*)$')
    needle_re = re.compile(r'^needle\s*=\s*(.*)$')
    for raw in open(path):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = needle_re.match(line)
        if m:
            prog.needle = parse_inner(m.group(1))
            continue
        if line.startswith("next "):
            prog.outputs.append(line)
            continue
        m = lhs_re.match(line)
        if not m:
            continue
        prog.n_raw_lines += 1
        nloop, base_idx, rhs = m.group(1), int(m.group(2)), m.group(3)
        n = int(nloop) if nloop is not None else 0
        if nloop is not None:
            prog.n_loop_lines += 1
        for i in range(n + 1):
            rhs_i = subst_i(rhs, i)
            k = base_idx + i
            is_relu = rhs_i.strip().startswith("relu(")
            inner = rhs_i.strip()[5:-1] if is_relu else rhs_i
            coeffs, const = parse_inner(inner)
            st = Stmt(("t", k), is_relu, coeffs, const)
            prog.stmts.append(st)
            prog.by_var[("t", k)] = st
    # keep evaluation order == temp-index order (they already are, but be safe)
    prog.stmts.sort(key=lambda s: s.var[1])
    return prog


if __name__ == "__main__":
    for p in sys.argv[1:]:
        prog = parse_file(p)
        print(f"{p}: {len(prog.stmts)} temps  "
              f"({prog.n_raw_lines} source lines, {prog.n_loop_lines} loop banks)  "
              f"needle={'yes' if prog.needle else 'no'}  outputs={len(prog.outputs)}")
