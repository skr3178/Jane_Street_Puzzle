# Reading-route finding — block-1 input→output data flow (2026-08-18)

Full 256×256 structural dependency of one block (parser-exact, cross-checked
empirically against the int64 emulator by perturbation). Tool:
[trace_dataflow.py](trace_dataflow.py).

## The 256 output slots split cleanly into 4 roles

| role | count | slots | what it is |
|------|------:|-------|-----------|
| **pass-through** | 128 | 0..31, 128..159, 192..255 | frozen store 192..255 (64, Stage-0 constant) + two bit-lane permutations (0..31←in 128..159; 128..159←in 96..127) |
| **ladder outputs** | 32 | 160..191 | bit-extraction; each reads exactly ONE `wide` lane |
| **shift/copy** | 32 | 96..127 | `next bit[96+i] = relu(bit[32+i] − 2·zero[64+i]) = bit[32+i]` (zero≡0) — a plain copy of bit[32..63] |
| **avalanche mix** | 64 | 32..95 | bit[32], num[33..63], zero[64], bit[65..95]; each reads **126–192 of 256** input slots through the ReLU stack |

## Ladder wiring (confirms 4 independent lanes)
- `wide[200..203]` feed **only** output slots 160..191 (nothing in the mixing
  core reads them). Each ladder output depends on a single lane:
  `wide[160]+bit[161..167] ← wide[200]`, `bit[168..175] ← wide[201]`, etc.

## Consumption / the loop-carried recurrence
- The ladder's *output* bits (`bit[161..191]`, `wide[160]`) are **terminal** in
  this block — no other block-1 output reads them; they are consumed **next round**.
- The *previous* round's ladder state (input `bit[161..191]`, `wide[160]`) is read
  by **all 64** mixing-core outputs (64/64). Example fan-in of `num[40]`:
  bit[0..31]:29, bit[32..63]:29, bit[65..95]:28, bit[96..159]:58, ladder[160..191]:29.

## Mixing operation — NOT a per-slot XOR / additive accumulator
- Permutations & copies (0..31, 96..159): direct relabel/copy.
- Ladder (160..191): comparator-cascade bit extraction (not XOR/add).
- Core (32..95): each output is a deep ReLU function of **~half the state**
  (126–192 slots), not `state XOR bit` or `state + bit·const`.

**Consequence for the "linear/CRC recurrence solvable by linear algebra" idea:**
it does not hold. The per-block map is high-fan-in and nonlinear (matches the
already-proven non-linearizability and the Stage-3 forward-easy/inverse-hard
avalanche verdict). The 63-round state is not a linear combination of input bytes.

## The useful narrowing (positive takeaway)
Per block, the hard nonlinearity is confined to **64 of 256 slots** (the 32..95
mixing core). The other 192 are frozen (64), trivially per-lane-invertible
ladders (32), or permutation/copy (96). This is mechanical structure of the
compiled data flow — not a claim about what the network "checks" or the answer.
