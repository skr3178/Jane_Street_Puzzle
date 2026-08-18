# Block anatomy — lane census, half-equivalence, ladder census (2026-08-18)

ASCII summary of the three mechanical tests of the packed-bitfield hypothesis
(user's hypothesis; raw findings only, interpretation deliberately withheld).
Data files: `dynamic/lane_map.{txt,png}`, `dynamic/half_templates.txt`,
`dynamic/ladder_pairs.txt`, script `dynamic/lane_census.py`.

## A) The state vector (the "memory" between blocks) — 256 slots, zoned

Measured over 154 inputs x 63 block boundaries; a slot is "0/1" only if it
never held anything else anywhere in the corpus.

```
slot: 0        33       64 65        160 161      192        247 249..255
     ┌─────────┬────────┬─┬───────────┬─┬──────────┬──────────┬─┬────────┐
     │ 0/1 bits│ small  │0│  0/1 bits │w│ 0/1 bits │   wide   │128│ dead │
     │  (33)   │ ints≤8 │ │   (95)    │ │   (31)   │  values  │const│(7) │
     └─────────┴────────┴─┴───────────┴─┴──────────┴──────────┴─┴────────┘
      flag-like  counter-   flag-like    flag-like  data-like   pinned
      lanes      like       lanes        lanes      lanes
```

Class counts: 159 binary slots, 31 small(<=8), 57 wide, 8 always-0, 1 const-128.

## B) One 42-layer block — same instruction runs twice on different operands

This is ONE unit; it repeats 63x. Only pos 0 (payload) and pos 29 (selector)
change between copies; the other 40 positions are byte-identical in all 63.

```
pos:  0   1..16          17  18 ─────── 28  29  30  31 ─────── 41
    ┌───┬──────────────┬───┬───────────────┬───┬───┬───────────────┐
    │pay│   ladder     │   │   HALF 1      │sel│   │   HALF 2      │
    │load│  section    │   │ 3265 ops      │ect│   │ 3265 ops      │
    │(76 │ (see C)     │   │ 26 templates  │or │   │ 26 templates  │
    │ops)│             │   │      │        │   │   │      │        │
    └───┴──────────────┴───┴──────┼────────┴───┴───┴──────┼────────┘
         varies                   │ same 26 templates,    │
         per block                │ same counts: 100%     │
         (payload)                └──── match ────────────┘
                                  but different slots wired in
                                  => same operation, two pieces of state
```

- Op template := (sorted coefficients, source-column offsets, bias) per row,
  i.e. arithmetic shape with slot identities erased.
- Halves match 100% as template multisets (3265 rows, 26 templates each) yet
  share almost no actual cells (see static/repetition_scan.txt) — operation-
  level equivalence, not copied wiring.
- Varying per block: pos 0 payload (values in {-1,0,1,2});
  pos 29 selector: variant = 4*floor(b/16) + (b mod 4).

## C) The ladder (block positions 1..13) — powers of two, 4 lanes at a time

```
pos  1:   ±128 ±128 ±128 ±128      <- 4 cells +, 4 cells -
pos  2:    (only ±1 coeffs)
pos  3:   ±64  ±64  ±64  ±64
pos  4:    (only ±1 coeffs)
pos  5:   ±32  ±32  ±32  ±32
  ...          halving every 2 layers
pos 13:   ±2   ±2   ±2   ±2
```

Comparator test: a layer containing both (x - 128) and (x - 127) on the same
source would be a threshold gadget (difference = 0/1 step). Result in the
skeleton: NO such (c, c-1) pairs anywhere in positions 0..15 of block 1. The
-127/-63/-31 neighbor values seen in the position vocabulary live in OTHER
blocks' PAYLOAD cells, not in the shared skeleton.

## The whole model in these terms

```
input string (<=55 chars)
   │
   ▼
[ prologue: 17 layers ]  — builds the initial 256-slot state
   │
   ▼            ┌────────────────────────────────────────┐
 state (A) ───► │ block b  =  diagram B (with payload    │
   ▲            │ row b and selector variant plugged in) │──► state (A)
   │            └────────────────────────────────────────┘      │
   └───────────────────── repeat for b = 0 .. 62 ◄──────────────┘
   │
   ▼
[ epilogue: 58 layers ]  — collapses the final state to 1 number
```

Zone characters persist across all 63 boundaries (bit-lanes stay bit-lanes,
wide slots stay wide) — that is what the census pooled over.
