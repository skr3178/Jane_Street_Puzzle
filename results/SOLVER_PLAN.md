# Solver plan — staged SMT/SAT attack on the compressed recurrence (2026-08-18)

Status: PLANNING ONLY. Nothing has been solved. A background design workflow
(4 encoding designs -> 4 adversarial soundness audits -> 1 synthesis) is
producing the build-ready spec; its output will be appended/merged here.

## The formulation

```
variables:   x[0..54]                     (the 55 input slots)
constraints: s0    = Prologue(x)          (SYMBOLIC in x — not a constant)
             s{k+1}= Block(s{k}; params[k])   k = 0..62
             needle= Epilogue(s63)
target:      needle >= 1                  (NOT >= 0: output = max(needle,0),
                                           so needle 0 still prints 0.0)
ask:         SAT?  ->  x is answer-bearing (fenced)
```

Logic: QF_BV (bit-vectors, proven widths) for the exact version; possibly a
Boolean/CNF (kissat/CaDiCaL) fast version after slot domains are PROVEN.

## Corrections locked in from review (traps that would make a run worthless)

1. Acceptance target is needle >= 1, never >= 0.
2. Prologue consumes x: the initial state is symbolic, never precomputed.
3. Zone/Boolean domains from the lane census are EMPIRICAL (154-string
   corpus). Asserting them unproven = possible false UNSAT. They must be
   proven by interval/abstract propagation first, or the run is heuristic-only.
4. Bit-vector widths need PROVEN bounds (measured peak |activation| 320 is an
   observation; weights reach +-256). Interval-propagate to get sound widths.
5. Input domain: observed codes 0..126 over printable probes only. 32..126 is
   a flagged heuristic (fast version); the exact version uses the domain the
   Stage-0 encoder sweep establishes. Tightening can exclude the answer.
6. Real strings = prefix of codes then zero padding. 55 free vars is a
   superset -> any SAT vector needs a realizability check (decode ->
   re-encode -> compare) before it counts.
7. It is QF_LIA/QF_BV, NOT QF_NIA — all multiplies are by constants.

## Sound compressions to build in

- Collapsed listings (~78% of rows are pass-through wires) — already on disk.
- Skeleton instantiated once, stamped 63x with the operand table
  (results/decomp2/param_table.json); only pos 0/1/2/29 vary.
- Frozen store (slots 192-255): single shared variables across all 63 rounds
  — AFTER verifying literal identity pass-through (coeff +1, bias 0).
- Per-block distillation: E/F (slots 160-191) depend only on the frozen store
  within a block (function varies per block via pos-29 selector) -> precompute
  63 small E_b(x), F_b(x); the recurrence core then lives on slots 0-159.
- Drop always-0 slots and constants (64, 249-255, 247).
- Boolean-type the PROVEN {0,1} slots (turn census observation into invariant
  first).

## The stages

| Stage | What | Safety | Go/no-go signal |
|---|---|---|---|
| Design (running) | 4 designs -> audits -> synthesized spec | safe | spec lands |
| 0 Faithfulness prep | encoder domain sweep (wide charset, prefix structure); frozen-store identity check; interval propagation -> proven bounds + proven Boolean set; acceptance-predicate note | safe, pure measurement | all checks pass |
| 1 Encoder build | emit QF_BV instance from collapsed listings + table; SELF-TEST: fix a known input, solver needle must equal emulator needle exactly | safe | self-test exact match |
| 2 Epilogue backward probe | solve needle>=1 over final 256-state alone (no 63-block core) | low risk (state vector, not input) | sat/unsat + which terminal configs can fire |
| 3 Hardness sweep | full instance, incremental thresholds -12, -11, -10, ... ; portfolio z3 / bitwuzla / yices (fix its :produce-models-before-set-logic bug), ~1h timebox per step | near-miss vectors only | do thresholds fall? which runtime bucket |
| 4 Full exact solve | needle >= 1, portfolio, escalating timeouts | FENCED: a SAT assignment IS the answer. Stop, flag, reveal only on explicit user authorization | explicit user go required |

Validation chain for any SAT: decode x -> re-encode via the REAL encoder ->
bit-exact emulator (emu.py) -> torch (trex env), all agreeing.

## Runtime expectations (honest)

Instance size (~190k ReLU case-splits, few-million-clause CNF) is routine;
hardness is not predictable. Buckets: minutes (structure helps) / hours-days
(with portfolio + backward seeding) / effectively never (solver-resistant
mixer — the measured avalanche is the warning sign). Stage 3 exists to learn
the bucket cheaply. CPU-bound, single-core dominated, RAM-sensitive; GPU
irrelevant.

## Tooling on hand

z3, bitwuzla, yices (results/smt/ has earlier validated sanity encodings —
check_hello.smt2 returned sat under z3/bitwuzla, proving encoding
faithfulness is achievable). kissat/CaDiCaL only if the pure-SAT path is
chosen. Emulator (emu.py) is the ground truth for every cross-check.
