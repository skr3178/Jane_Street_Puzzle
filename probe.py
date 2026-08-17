"""Instrumentation for poking at the puzzle model. No analysis, just access.

    PYTHONPATH=.shim /media/skr/storage/conda_envs/lingbot/bin/python probe.py "some string"
    PYTHONPATH=.shim ... probe.py --encode "some string"      # the 55-vector
    PYTHONPATH=.shim ... probe.py --file candidates.txt       # batched sweep
    PYTHONPATH=.shim ... probe.py --nonzero candidates.txt    # only non-zero outputs
    PYTHONPATH=.shim ... probe.py --needle "s1" "s2" ...      # pre-clamp signal
    PYTHONPATH=.shim ... probe.py --rank candidates.txt       # sort by needle

Or import it:

    from probe import load, encode, run, sweep, needle, rank
    m = load()
    run(m, "text") ; encode(m, "text") ; sweep(m, ["a", "b", ...])
    needle(m, "text")          # (pre-clamp value, output)
    rank(m, texts)             # [(needle, output, text)] closest-to-firing first
"""
import sys
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from device_patch import to_device, Batcher, NeedleTap  # noqa: E402

_MODEL = "model_3_11.pt" if sys.version_info >= (3, 11) else "model.pt"


def load(device=None, quiet=False):
    path = os.path.join(HERE, "puzzle", _MODEL)
    m = torch.load(path, weights_only=False, map_location="cpu").eval()
    device = device or os.environ.get(
        "PUZZLE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device != "cpu":
        try:
            to_device(m, device, calibrate_with="cal")
        except Exception as e:
            if not quiet:
                print(f"[probe] {device} unavailable ({e}); CPU", file=sys.stderr)
            device = "cpu"
    m._batcher = Batcher(m)
    m._needle = NeedleTap(m)
    m._device = device
    if not quiet:
        print(f"[probe] {_MODEL} on {device}", file=sys.stderr)
    return m


def run(model, text):
    with torch.no_grad():
        return float(model(text))


def encode(model, text):
    """The vector entering the first parameterised module."""
    return model._batcher.encode(text)


def sweep(model, texts):
    """Batched. ~0.2 ms/string on GPU vs ~47 ms one at a time."""
    return model._batcher(list(texts))


def needle(model, text):
    """(pre-clamp value, output). The needle moves where the output is flat."""
    out = run(model, text)
    return model._needle.last()[0], out


def rank(model, texts):
    """Batched needle sweep -> [(needle, output, text)], closest to firing first."""
    texts = list(texts)
    b, tap = model._batcher, model._needle
    rows = []
    with torch.no_grad():
        for i in range(0, len(texts), b.chunk):
            part = texts[i : i + b.chunk]
            x = torch.stack([b.encode(t) for t in part])
            y = torch.nn.Sequential.forward(model, x)
            rows += list(zip(tap.last(), y.flatten().tolist(), part))
    rows.sort(key=lambda r: -r[0])
    return rows


def _main(argv):
    if not argv:
        print(__doc__)
        return 1
    mode, rest = "run", argv
    if argv[0].startswith("--"):
        mode, rest = argv[0][2:], argv[1:]

    m = load()
    if mode == "needle":
        for s in rest:
            pre, out = needle(m, s)
            print(f"{pre:.10g}\t{out:.10g}\t{s}")
    elif mode == "rank":
        with open(rest[0]) as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        import time

        t = time.time()
        rows = rank(m, lines)
        print(
            f"# {len(rows)} ranked in {(time.time()-t)*1000:.0f} ms — "
            "needle, output, input; closest to firing first",
            file=sys.stderr,
        )
        for pre, out, s in rows:
            print(f"{pre:.10g}\t{out:.10g}\t{s}")
    elif mode == "encode":
        v = encode(m, rest[0])
        torch.set_printoptions(precision=6, sci_mode=False, threshold=10_000)
        print(f"shape {tuple(v.shape)} {v.dtype}")
        print(v.cpu())
    elif mode in ("file", "nonzero"):
        with open(rest[0]) as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        import time

        t = time.time()
        vals = sweep(m, lines)
        dt = time.time() - t
        hits = [(v, s) for v, s in zip(vals, lines) if v != 0.0]
        print(
            f"# {len(lines)} inputs in {dt*1000:.0f} ms "
            f"({dt/max(len(lines),1)*1000:.3f} ms each) on {m._device}; "
            f"{len(hits)} non-zero",
            file=sys.stderr,
        )
        for v, s in (hits if mode == "nonzero" else zip(vals, lines)):
            print(f"{v:.10g}\t{s}")
    else:
        for s in rest:
            print(f"{run(m, s):.10g}\t{s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
