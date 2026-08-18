# Findings log — static + dynamic reverse-engineering campaign (2026-08-18)

Spoiler-safe summary of what was measured and where the artifacts live.
Interpretation of the block semantics and the parameter table is deliberately
left to the user (see CLAUDE.md ground rules).

## 0. Environment

- The documented `lingbot` env and `.shim/` are gone (`/media/skr/storage` not
  mounted). Static weight reading works in base anaconda (py3.12, torch 2.6).
- **torch inference segfaults in base anaconda** — the crash is inside the
  model's cloudpickled encoder lambda (numpy 2.5 ABI). Use
  `/home/satya/anaconda3/envs/trex/bin/python` (py3.11, torch 2.6, cloudpickle)
  for anything that runs the model. scipy in base anaconda also segfaults
  against numpy 2.5 (why `emu.py` is pure numpy).

## 1. Provenance — settled

The model is machine-generated, not trained:
- All 118 distinct parameter values are **exact integers** (range ±256).
- Layers are ~99.5% sparse, banded; 2,721 layers dedup to **197 distinct**.
- Pure `nn.Sequential`, strict Linear→ReLU alternation ×2721 — no branches.
Files: `static/global_values.txt`, `static/topology.txt`, `static/summary.txt`.

## 2. Program structure

- 17-layer prologue → **63 blocks × 42 layers** → 58-layer epilogue.
  Annotated width plot: `../width_profile_annotated.png`, `arch/blocks.json`.
- Only 4 in-block positions vary across blocks: pos 0 (63 ids), pos 1 (3),
  pos 2 (2), pos 29 (16). Regimes: block 0 | 1–31 | 32–47 | 48–62.
- pos-29 selector is exact: variant = 4*floor(b/16) + (b mod 4).
Files: `static/hash_sequence.txt`, `static/layer_types.txt`,
`static/block_listing.txt`, `static/per_position_vocab.txt`,
`static/distinct_layers.json`, spy plots `static/spy_{prologue,block1,epilogue}.png`.

## 3. Collapsed transition functions (exact, ReLU-aware)

Wire-collapsing decompiler: `static/collapse.py`. Because inter-layer values
are >=0, positive copy-wires pass ReLU exactly; ~78% of rows collapse away.
- One block ≈ **3,000 irreducible integer ops**, almost all 1–3 terms.
- Prologue: 530 ops incl. 169 wide (~110-term) sums. Epilogue: 5,624 ops incl.
  the only other wide ops (8/16/48-term) producing the scalar.
Files: `static/{prologue,block0,block1,block17,block33,block49,epilogue}_collapsed.txt`,
`static/collapse_stats.txt`.

## 4. Exact int64 emulator

`../emu.py` — pure-numpy sparse int64 forward pass, **verified bit-exact vs
torch** on needle+output for 6 strings. ~14 ms/forward, full intermediate
state access. Peak |activation| observed: 320 (no overflow anywhere).
Cache: `static/emu_layers.npz` (all weights as integer COO).

## 5. Dynamic instrumentation (`dynamic/`)

- **State movies**: state entering each of the 63 blocks + epilogue for inputs
  e0 and zero: `dynamic/states_{e0,zero}.{png,npz}`. ~70–100 of ~256 slots
  nonzero per block; max value is a constant 128 (one stripe at idx ~192);
  active region idx ~0–160.
- **Input encoding characterized** (`dynamic/encoding_report.json`,
  `encodings.npz`): encode("") = 0; each character sets exactly one dim;
  dim i = integer code of char at position i; observed codes 0–126; strings
  capped at **55 chars**; order-sensitive.
- **Raw-vector climbs**: greedy stalls at needle −14; exhaustive per-coordinate
  descent converges at **−13** (nnz 2) — a true single-coordinate local
  optimum. Vectors saved (`climb_best_vec.npy`, `coord_descent_best.npy`),
  not printed (spoiler protocol). Earlier string-space beam search: −12.
Script: `dynamic/run_dynamic.py` (run under trex).

## 6. Parameter table — the candidate decompiled program (`decomp2/`)

Everything that varies across the 63 blocks, tabulated:
- R1 blocks: 448 weight cells + 32 bias cells; R2/R3: 240 + 32. Values only
  {−1, 0, 1, 2}. Payload renders as diagonal *staircases* (values marching one
  step per block) + an irregular bias-bit section: `decomp2/payload_bitmap.png`.
- Payload lands in exactly **76 ops** of the ~3,000 in a block:
  `decomp2/payload_ops_block1.txt` (family: v = relu(+s32k −2·s64k +s96k −1)).
- pos-29 variants differ in 3,072 cells (rows 32–127 × cols 96–159),
  values {−2, 0, 1}.
Files: `decomp2/param_table.{txt,json}`, `decomp2/extract_params.py`.
Also the older 63×30 bias-bit matrix: `static/bias_bitmap.{png,npy}`.

## 7. Repetition fully inventoried (`static/repetition_scan.txt`)

- Exact: skeleton as in §2.
- Near: exactly two pairs — blocks 19~31 and 40~47 share **identical pos-0
  weights**, differing only in bias bits → staircase and bias bits vary
  independently.
- Shape-echo only: block halves (pos 18–28 vs 31–41) and the epilogue share
  width layout but NOT contents.

## 8. Lane census, half-equivalence, ladder census (see BLOCK_ANATOMY.md)

Mechanical tests of the user's packed-bitfield hypothesis:
- **Lane census** (154 inputs × 63 boundaries): state slots form contiguous
  zones — 159 slots only ever hold {0,1}, 31 hold small ints ≤8, 57 wide,
  1 pinned at 128, 8 always zero. `dynamic/lane_map.{txt,png}`.
- **Half-equivalence**: block positions 18–28 vs 31–41 are a **100% match as
  op-template multisets** (3,265 rows, 26 templates each) while sharing almost
  no cells — same operation set on different operands. `dynamic/half_templates.txt`.
- **Ladder census**: ±2^k coefficients (128→2, halving every 2 layers) appear
  in exactly 4+4 cells per step; **no (c, c−1) comparator pairs** anywhere in
  the skeleton — the −127/−63/… values are payload-only. `dynamic/ladder_pairs.txt`.
ASCII diagrams of all three + the whole-model loop: `BLOCK_ANATOMY.md`.

## Open threads

- Read the block semantics from the collapsed listings (user's task).
- Read the 63-row parameter table (user's task).
- Optional tooling: symbolic payload listing (p0..pN aligned with the table),
  state-movie diffs between chosen inputs, pair-coordinate search, SMT re-run
  with the 63×-instantiated compressed encoding (yices options bug is fixable).
