"""Symbolically trace ONE ladder lane through block-1 and tabulate its exact
input->output map. Lane i depends only on the scalar wide[200+i], so we can
(a) print the raw per-level definitions and (b) sweep the whole input range.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_linearize import parse_file    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
blk = parse_file(os.path.join(HERE, "block1_readable.txt"))


def fmt(st):
    parts = []
    for (kind, idx), c in sorted(st.coeffs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        nm = f"{kind}[{idx}]"
        parts.append(f"{'+' if c >= 0 else '-'}{'' if abs(c)==1 else str(abs(c))+'*'}{nm}")
    if st.const:
        parts.append(f"{'+' if st.const >= 0 else '-'}{abs(st.const)}")
    body = " ".join(parts)
    return f"relu({body})" if st.is_relu else body


# ---- the ladder chain for lane i=0, grouped by peel level ------------------
LANE = 0
levels = [
    (128, 96, 100, 104, 144, 140),
    (64, 225, 229, 233, 369, 365),
    (32, 467, 471, 475, 576, 572),
    (16, 702, 706, 710, 839, 835),
    (8, 959, 963, 967, 1089, 1085),
    (4, 1197, 1201, 1205, 1313, 1309),
    (2, 1397, 1401, 1405, 1485, 1481),
]

print("=" * 72)
print(f"RAW per-level definitions, lane i={LANE} (input scalar = wide[{200+LANE}])")
print("=" * 72)
for L, a, b, cc, comb, bit in levels:
    print(f"\n--- peel level {L:3d} ---")
    for t in (a, b, cc, comb, bit):
        v = ("t", t + LANE)
        print(f"  t[{t+LANE}] = {fmt(blk.by_var[v])}")

# ---- exact sweep: everything else 0, vary wide[200] ------------------------
bit_temps = [b + LANE for (_, _, _, _, _, b) in levels]     # 128,64,...,2 bits
remainder_temp = 1485 + LANE
print()
print("=" * 72)
print("EXACT input->output sweep (all other slots = 0)")
print("=" * 72)
print(" v   | bits @128 64 32 16  8  4  2 | rem | reconstruct  match?")
print("-" * 72)


def run(v):
    base = [0] * 256
    base[200 + LANE] = v
    val = blk.eval(base)["temps"]
    bits = [int(val[("t", t)]) for t in bit_temps]
    rem = int(val[("t", remainder_temp)])
    return bits, rem


recon_ok = True
weights = [128, 64, 32, 16, 8, 4, 2]
shown = list(range(0, 16)) + [63, 64, 65, 126, 127, 128, 129, 191, 192,
                              200, 254, 255, 256, 257, 300, 440, 510, 511]
for v in shown:
    bits, rem = run(v)
    recon = sum(w * b for w, b in zip(weights, bits)) + rem
    ok = (recon == v)
    print(f"{v:4d} |        {bits[0]}  {bits[1]}  {bits[2]}  {bits[3]}  "
          f"{bits[4]}  {bits[5]}  {bits[6]} | {rem:3d} | {recon:4d}"
          f"        {'ok' if ok else 'NO'}")

# full-range integrity + where each bit really equals the binary digit
print("-" * 72)
lo_ok = hi_break = None
for v in range(0, 1024):
    bits, rem = run(v)
    recon = sum(w * b for w, b in zip(weights, bits)) + rem
    per_bit_binary = all(bits[k] == ((v >> (7 - k)) & 1) for k in range(7)) and rem == (v & 1)
    if recon == v and per_bit_binary:
        if lo_ok is None:
            lo_ok = v
    else:
        if lo_ok is not None and hi_break is None:
            hi_break = v
            break
print(f"exact binary decomposition (bit_k == digit_k AND rem == v&1 AND "
      f"reconstruct==v) holds for v in [{lo_ok}, {hi_break-1 if hi_break else 1023}]")
