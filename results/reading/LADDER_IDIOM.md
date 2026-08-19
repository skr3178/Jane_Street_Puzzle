# Reading-route finding — the bit-extraction ladder (2026-08-18)

Traced one "ladder lane" through BLOCK 1 symbolically (per the ±128/±64/±32/…/±2
alternating-coefficient lead) and pinned down the idiom exactly.

## Setup / why it's tractable
Lane `i` (i = 0..3) is a **pure function of the single scalar `wide[200+i]`** —
every temp in the chain traces back to just that one input slot. So the lane can
be both traced symbolically *and* swept over its entire input range for an exact,
non-approximate input→output map (no ReLU-state guessing needed).

Tooling: [parse_linearize.py](parse_linearize.py) (parser, verified **bit-exact**
vs the int64 emulator, 64/64) + [trace_ladder.py](trace_ladder.py) (this trace).

## The idiom: 7-level restoring bit-extraction
Each peel level `L ∈ {128, 64, 32, 16, 8, 4, 2}` is the **same three-comparator
gadget** on a running value `x` (starts at `wide[200+i]`):

```
lt   = relu(L - x)                 # > 0  iff  x < L
ge-1 = relu(x - (L-1))             # the two "x ≥ L ?" comparators
ge   = relu(x - L)
next = relu(x - L*ge-1 + L*ge)     # = x      if x < L
                                   # = x - L  if x ≥ L      (clears the L-bit)
bit  = relu(1 - lt)                # = [x ≥ L]              (emits that bit)
```

Seven levels in series emit bits 7…1; the final leftover (`t[1485+i]`) is bit 0.

Raw per-level temp indices for lane i=0 (add `+i` for the other lanes):

| level L | comparators (lt, ge-1, ge) | combine→next | bit out |
|--------:|----------------------------|-------------:|--------:|
| 128 | t[96],  t[100], t[104] | t[144]  | t[140]  |
|  64 | t[225], t[229], t[233] | t[369]  | t[365]  |
|  32 | t[467], t[471], t[475] | t[576]  | t[572]  |
|  16 | t[702], t[706], t[710] | t[839]  | t[835]  |
|   8 | t[959], t[963], t[967] | t[1089] | t[1085] |
|   4 | t[1197],t[1201],t[1205]| t[1313] | t[1309] |
|   2 | t[1397],t[1401],t[1405]| t[1485] | t[1481] |

## Verified behavior (exact sweep, all other slots = 0)
- For input in **[0, 255]**: exact 8-bit binary decomposition — `bit_k == digit_k`,
  `rem == v & 1`, and the weighted sum reconstructs `v` perfectly.
- For input **> 255**: saturating — the 7 top bits pin to 1 and the remainder
  absorbs the overflow (`rem = v - 254`).
- Holds for **all four lanes**: `wide[200]→bit[167..161]+wide[160]`, and
  identically for `wide[201..203]` (same ops via the `for i in [0..3]` bank).

Sample rows (lane 0):
```
 v=128 -> 1 0 0 0 0 0 0 | rem 0     (10000000b)
 v=200 -> 1 1 0 0 1 0 0 | rem 0     (11001000b)
 v=255 -> 1 1 1 1 1 1 1 | rem 1
 v=256 -> 1 1 1 1 1 1 1 | rem 2     (saturates)
```

## Scope (mechanical only)
This documents what **BLOCK 1** does to four `wide` slots: it spreads each stored
integer into 7 bit-slots + a remainder — the inverse direction of the `1,2,4,…,128`
weighted sums that rebuild bytes in the epilogue's E56 stage. It is a data-flow
idiom of the compiled network; it is **not** a claim about what the whole 63-round
loop "checks" or about the answer input. Next unfoldings (num-lanes; where
`bit[161..191]` get consumed in round 2) can reuse `trace_ladder.py`.
