# Solver plan — staged SMT/SAT attack on the compressed recurrence (2026-08-18, rev 3)

Status: Stage 0 DONE (see STAGE0_RESULTS.md). Nothing answer-bearing solved.
Rev 2 incorporated the external review (Stage 1.5 checkpoint equivalence,
epilogue target analysis, BV-width invariant, encoder/realizability split,
non-monotonic thresholds, formulation x solver matrix).
Rev 3 folds in Stage 0 FINDINGS — three changes below marked [S0]:
  [S0-a] True input domain is bytes 0..255 (identity encoder), NOT 0..126.
  [S0-b] Frozen store 192..255 value-constant across all boundaries -> shared
         variable per input across rounds is sound.
  [S0-c] Interval propagation DIVERGES (int64 overflow; proves only 35/159
         Boolean slots). "Interval-proven widths + proven Boolean set" is
         DROPPED; replaced by empirical bound (peak |act| = 440) + GUARDED
         widths (overflow-guard assertions per intermediate).

## The formulation

```
variables:   x[0..54]                     (the 55 input slots)
constraints: s0    = Prologue(x)          (SYMBOLIC in x — not a constant)
             s{k+1}= Block(s{k}; params[k])   k = 0..62
             needle= Epilogue(s63)
target:      needle >= 1                  (HARD INVARIANT at every stage:
                                           output = max(needle,0), so
                                           needle 0 still prints 0.0)
ask:         SAT?  ->  x is answer-bearing (fenced)
```

Logic: linear integer arithmetic + ReLU case splits (NOT QF_NIA — all
multiplies are by constants). Emitted as QF_BV with PROVEN widths and/or
QF_LIA; possibly pure CNF (kissat/CaDiCaL) once slot domains are proven.

## Locked-in trap corrections (a run violating any of these is worthless)

1. Acceptance target is needle >= 1, never >= 0. Hard invariant everywhere.
2. Prologue consumes x: S0 is symbolic; only the transformation STRUCTURE is
   precomputable, never the initial state.
3. Zone/Boolean domains from the lane census are EMPIRICAL (154-string
   corpus). Asserting them unproven = possible false UNSAT. Prove first
   (interval/abstract propagation) or mark the run heuristic-only.
4. [S0-c REVISED] BV widths: interval propagation DIVERGES here (overflows
   int64 over 2721 layers; cancellation-blind), so proven widths are NOT
   cheaply obtainable. Sound replacement: pick a signed width with large
   headroom over the empirical peak (|act| = 440 over 400 random full-byte
   strings) — e.g. 16-bit +-32767 — and ADD an overflow-guard assertion per
   intermediate (assert value fits the word). If the solver ever needs a value
   outside range it is FLAGGED, not silently wrapped -> soundness without a
   divergent proof. A true proof would need a relational domain (zonotope),
   deferred unless needed.
5. ENCODING RANGE vs ALLOWED INPUT RANGE are separate questions:
     (i)  string -> x   (what vectors can the encoder actually produce:
          per-slot code set AND cross-slot structure, e.g. prefix-of-codes
          then zero padding)
     (ii) x in D        (what the solver is told)
   Solving with 55 independent vars is a SUPERSET of (i): mathematically
   valid x may correspond to no string. Every SAT needs the realizability
   check (decode -> re-encode -> compare). This is the biggest remaining
   faithfulness question and Stage 0's core job.
6. [S0-a] Domain versions, formalized:
     A (ground truth): x_i in {0..255} (PROVEN: encoder is byte-identity,
                       0 = padding, position-independent).
     B (heuristic):    x_i in [32,126] printable.
   Interpretation: B SAT -> candidate; B UNSAT -> no printable solution ONLY;
   A SAT -> possibly non-printable solution; A UNSAT -> strong result.
   The heuristic domain must never silently become the puzzle definition.

## Sound compressions to build in

- Collapsed listings (~78% of rows are pass-through wires) — on disk.
- Skeleton instantiated once, stamped 63x with the operand table
  (results/decomp2/param_table.json); only pos 0/1/2/29 vary.
- [S0-b] Frozen store (slots 192-255): single shared variables across all 63
  rounds — VERIFIED value-constant across all boundaries (payload reads them
  but does not modify them; a few fan out). Value preservation confirmed.
- Per-block distillation: E/F (slots 160-191) depend only on the frozen store
  within a block (function varies per block via pos-29 selector) -> precompute
  63 small E_b(x), F_b(x); the recurrence core then lives on slots 0-159.
- Drop always-0 slots and constants (64, 249-255, 247) once proven.
- [S0-c] Boolean-type slots only under overflow-guard (not via interval proof;
  only 35/159 are interval-provable). Empirically 224/256 stay in {0,1}.

## The stages

| Stage | What | Safety | Go/no-go signal |
|---|---|---|---|
| Design (running) | 4 designs -> audits -> synthesized spec | safe | spec lands |
| 0 Faithfulness prep [DONE] | (i) domain = bytes 0..255 identity, 0=pad [DONE]; (ii) frozen store value-constant [DONE]; (iii) interval prop DIVERGES -> use empirical peak 440 + guarded widths [DONE]; (iv) target needle>=1 [DONE]. See STAGE0_RESULTS.md | safe, pure measurement | PASSED |
| 1 Encoder build | emit compressed instance; SELF-TEST: fix a known input, solver needle == emulator needle exactly | safe | exact match |
| 1.5 Checkpoint equivalence | dozens-hundreds of random valid x: BV/LIA model == int64 emulator at checkpoints S0, S1, S2, S16, S32, S48, S63, needle — not just the final scalar. Deliberately include ReLU-boundary cases, extreme propagated values, and inputs straddling regime transitions 16/32/48 | safe | zero mismatches; catches "right needle, wrong intermediate" encodings |
| 2 Epilogue target analysis | more than sat-probing: extract the needle's dependency cone through the epilogue (start: results/decomp/epilogue_cone.txt) — which s63 coordinates carry nonzero influence, signs, bounds; then solve needle>=1 over s63 alone to characterize (minimally) which terminal combinations can fire; ideally reduce the target to a small explicit condition c1*s_i1 + ... + ck*s_ik >= 1 | low risk (state-level, not input-level) | a compact target condition for the 63-round core |
| 2.5 Reduced-round scaling probe | small-first, cryptanalysis-style: generate targets FROM KNOWN RUNS (emulator traces of e.g. "hello"), then solve inversion instances — 1-block (given s1 find s0), k-block for k=1,2,4,8,16,32 (given s_k recover an input), partial-input (fix 50 of 55 slots). All guaranteed-SAT (target is reachable by construction: any UNSAT = encoding bug, so this doubles as a faithfulness test) and non-answer-bearing (targets from known inputs). Record solve time vs k -> scaling curve. CAVEAT: reduced-round success does NOT extrapolate for a strong mixer (difficulty can grow exponentially in rounds); the curve's GROWTH RATE is the deliverable, not a promise | safe, unfenced by construction | shallow growth -> full solve plausible; explosive growth -> reading route is primary |
| 3 Threshold/hardness sweep | full instance, thresholds -12, -11, -10, ...; RECORD per run: threshold, SAT/UNSAT/timeout, wall time, conflicts/decisions, memory. Do NOT read runtimes as a smooth difficulty curve — predicates are nested but solver effort is NOT monotonic | near-miss vectors only | which runtime bucket; do thresholds fall at all |
| 4 Formulation optimization | build the matrix, not one instance: Formulation A = exact LIA+ReLU; B = proven-width BV; C = BV with aggressive state elimination (distillation precompute, Boolean typing). Portfolio: z3 x {A,B,C}, bitwuzla x {B,C}, yices x {A,B} (fix its :produce-models-before-set-logic bug). Representation often matters more than solver | safe (no needle>=1 asked yet beyond Stage 3 thresholds) | best formulation/solver pair chosen |
| 5 Exact solve | needle >= 1 on domain A (and B in parallel, labeled), best pairs from Stage 4, escalating timeouts | FENCED: a SAT assignment IS the answer. Stop, flag, reveal only on explicit user authorization | explicit user go required |
| 6 Independent verification | decode x -> re-encode via the REAL encoder (realizability) -> bit-exact emulator (emu.py) -> torch (trex env): all four agree | fenced output handling continues | all agree -> flagged candidate |

## Runtime expectations (honest)

Instance size (~190k ReLU case-splits, few-million-clause CNF) is routine;
hardness is not predictable. Buckets: minutes (structure helps) / hours-days
(portfolio + backward seeding) / effectively never (solver-resistant mixer —
the measured avalanche is the warning sign). Stage 3 exists to learn the
bucket cheaply. The biggest danger is NOT solver performance but (a) semantic
mismatch between abstract x and the real string encoder, (b) silent BV
overflow — both addressed in Stage 0 by construction. CPU-bound, single-core
dominated, RAM-sensitive; GPU irrelevant.

## Tooling on hand

z3, bitwuzla, yices (earlier sanity encodings validated: check_hello.smt2
sat under z3/bitwuzla). kissat/CaDiCaL if the pure-SAT path is chosen.
emu.py is ground truth for every cross-check; torch runs in the trex env.
