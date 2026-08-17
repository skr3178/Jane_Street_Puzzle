Jane Street neural-network reverse-engineering puzzle.

Files are in `puzzle/` — load `model.pt` (Python 3.10) or `model_3_11.pt`
(Python 3.11+) and figure out what the network does. `app.py` shows how the
model is invoked: it takes a text string and outputs a number.

(The original source link was removed from this file because it leads to a full
solution write-up. Don't search for the blog post.)

---

## Quickstart

```
                     ┌──────────────┐
   type a string ──► │  :7860 UI    │ ──► a float
                     └──────┬───────┘
                            │  ./play.sh   → http://100.98.123.127:7860
                            ▼
              lingbot env + .shim/cloudpickle
                            │
                            ▼
              model_3_11.pt on CUDA (device_patch)
```

Interactive playground:

```bash
./play.sh                      # then browse to 100.98.123.127:7860
PUZZLE_DEVICE=cpu ./play.sh    # bit-exact CPU fallback
pkill -f play.py               # stop
```

Direct, no UI:

```bash
PYTHONPATH=.shim /media/skr/storage/conda_envs/lingbot/bin/python
# import torch, sys; sys.path.insert(0, '.')
# m = torch.load('puzzle/model_3_11.pt', weights_only=False, map_location='cpu').eval()
# from device_patch import to_device; to_device(m, 'cuda', calibrate_with='x')
# float(m("your string here"))
```

Two things that look like bugs but aren't: **every input returns `0.0`** (the
network genuinely computes — activations are input-dependent throughout — so the
constant output is by design), and **`cloudpickle` must be importable** or
`torch.load` fails, because these files are pickled objects rather than state
dicts.

Full setup notes, checksums, the GPU story, and spoiler hygiene: **`SKR.md`**.
