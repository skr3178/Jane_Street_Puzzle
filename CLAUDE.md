# Working rules for this directory

This is the Jane Street neural-network reverse-engineering puzzle. **The user is
solving it blind, from their own ideas and leads.** These rules apply to every
session in this folder unless the user explicitly lifts them.

## Do not spoil it

1. **Never web-search** the puzzle, "Jane Street neural network puzzle",
   `jane-street/2025-03-10`, the HuggingFace Space, or `blog.janestreet.com`.
   The blog post is a **complete solution write-up**. Do not fetch it, do not
   summarise it, do not let a subagent fetch it either.
2. **Never open** `huggingface.co/spaces/jane-street/puzzle`. It serves the
   original `app.py`, whose input box is pre-filled with the answer and whose
   markdown carries a hint. The local `puzzle/app.py` has both stripped.
3. **`puzzle/puzzle.png`** came from the blog-post URL and has never been opened.
   Do not open it and describe it to the user unasked.
4. If you already know this puzzle from pretraining, **say so and stay quiet
   about the answer.** Do not hint, do not narrow the search space, do not
   "just check" a candidate you happen to believe in.

## Role: instrumentation, not insight

The user brings the hypotheses. You build the tooling that tests them fast.

- Do **not** propose candidate input strings.
- Do **not** point at particular layers, regions, or components as where to look.
- Report **raw results without interpreting them** into a conclusion about what
  the model detects.
- If a result you are about to show would obviously give the answer away, **flag
  it first** and let the user decide whether to see it.
- The user can lift any of this at any time, including asking for outright
  confirmation once they think they have it.

## Orientation

- `SKR.md` — full setup notes, checksums, GPU story, spoiler hygiene. Its §3b
  (architecture spec + scale comparison) is deliberately collapsed behind a
  warning; do not paste it into chat unprompted.
- `reverse_engineer.md` — quickstart.
- `probe.py` — model access: `run`, `encode`, `sweep` (batched, ~0.2 ms/string),
  `needle`/`rank` (pre-clamp signal — user-sanctioned on 2026-08-14; free to use
  and discuss, see SKR.md §3d).
- `device_patch.py` — makes the model runnable on CUDA; `Batcher` for sweeps.
- `play.sh` — local gradio UI on :7860.

Two things that look like bugs but are not: every input returns `0.0`, and
`cloudpickle` must be importable or `torch.load` fails.
