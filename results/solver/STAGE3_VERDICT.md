# Stage 3 verdict — monolithic SMT/SAT inversion FAILS (2026-08-18)

## Experiments (full model, exact-value non-accepting targets, SAFE)

| task | solver / logic | width | result |
|---|---|---|---|
| x fully fixed (hello), needle==-15 | z3 QF_LIA | int | SAT ~4s (evaluation) |
| Stage 2.5 full target state fixed, 63 rounds | z3 QF_LIA | int | SAT ~5s (propagation) |
| needle==-12, x FREE (0..255) | z3 QF_LIA | int | TIMEOUT 300s |
| 1 free byte, needle==-15 (witness exists) | z3 QF_LIA | int | TIMEOUT 145s |
| 1 free byte, needle==-15 | z3 QF_BV | 32 | TIMEOUT >400s |
| 1 free byte, needle==-15 | bitwuzla QF_BV | 32 | TIMEOUT >150s |
| 1 free byte, needle==-15 | bitwuzla QF_BV | 16 | SAT but ~196s |

## Diagnosis
- Determined input -> instant (both solvers just evaluate).
- Free even ONE byte -> all solvers time out or take ~196s, while the emulator
  brute-forces 256 values in microseconds.
- Cause: one free input byte avalanches (dynamic finding) into ~185,000
  undetermined ReLU sign-decisions; general SMT/SAT must case-split on them.
  Symbolic forward evaluation through 2721 layers is ~256x more expensive than
  one numeric pass, so even bounded enumeration is catastrophic for the solver.

## Conclusion
- Monolithic SMT/SAT inversion is the WRONG methodology for this network
  (tested across z3-LIA, z3-BV, bitwuzla-BV @ 16/32-bit). Not a tuning gap.
- Combined with earlier local-search plateaus (hill-climb -12, coord-descent
  -13), the forward-easy / inverse-hard signature is exactly an avalanche
  mixer. The intended route is almost certainly READING the compiled algorithm
  (collapsed transition function + 63-row parameter table), not inversion.

## Untried solver cards (low expected value, noted for completeness)
- CryptoMiniSat (XOR-aware) — but net is ReLU, not XOR-structured.
- Custom bit-blast + kissat with 10-bit widths — marginal vs bitwuzla-16.
- Structure-aware decomposition / meet-in-the-middle — blocked by 256-wide
  state and avalanche.
