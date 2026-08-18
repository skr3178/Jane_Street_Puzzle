# Stage 0 results — faithfulness prep (2026-08-18)

Purpose was NOT to rediscover the zones (lane census already did, empirically)
but to test which empirical facts are PROVABLE, so the solver can assert them
without risking a false UNSAT. Files: stage0.py, stage0_encoder.json,
stage0_frozen.txt, stage0_bounds_*.json, stage0_bounds_empirical.json.

## (i) Input domain — RESOLVED, corrects a trap
- Encoder maps every byte 0..255 IDENTICALLY: chr(o) -> code o at position 0,
  position-independent; only NUL(0) -> 0 (padding). 256/256 identity chars.
- => True sound domain is x_i in {0..255}, NOT the corpus-observed 0..126.
  Freezing 0..126 or 32..126 as ground truth could have excluded the answer.
- Structure: value 0 = padding; a string is a prefix of nonzero codes.
- Unicode > 255: see stage0_encoder.json (multi/att. — irrelevant, domain is bytes).

## (ii) Frozen store 192..255 — value-constancy PROVEN operationally
- Direct check: slots 192..255 are identical across all block boundaries 1..62
  for every tested input (values 0..440). Safe to model as ONE shared variable
  per input across all 63 rounds.
- Caveat: they are NOT structurally isolated — the pos-0 payload READS them
  (multi-consumer columns) and a few slots fan out. Value is preserved; that is
  what the solver needs. (My first proof demanded isolation and "failed" 62x at
  pos-0 payload slots; that was the wrong property.)

## (iii) Bounds / Boolean typing — NEGATIVE result, changes the plan
- Naive interval propagation DIVERGES: bounds overflow int64 (~9.2e18) over
  2721 layers because the abstraction cannot track cancellation.
- It proves only 35 Boolean slots vs 159 empirical. The gap is unbridgeable by
  intervals.
- Empirical peak |activation| = 440 over 400 random full-byte strings
  (was 320 on the smaller printable corpus). 224/256 slots stay in {0,1}
  empirically (incl. always-0).

### Consequence for the encoder (supersedes SOLVER_PLAN Stage 0/1 sub-steps)
- DROP "interval-proven widths + proven Boolean set" — not achievable cheaply.
- Sound replacement = empirical-bound + GUARDED widths: pick a signed width
  with large headroom over 440 (e.g. 16-bit +-32767) and ADD overflow-guard
  assertions per intermediate; if the solver ever needs a value outside range,
  it is flagged, not silently wrapped -> soundness without a divergent proof.
- Boolean typing: assert only under the same guard, or run the exact version
  without Boolean typing and let the solver discover ranges. If a tighter
  PROOF is wanted later, needs a relational/zonotope domain, not intervals.

## (iv) Acceptance predicate
- output = max(needle, 0); model ends Linear->ReLU. Target is needle >= 1.
