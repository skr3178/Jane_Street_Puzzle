"""Local playground for the Jane Street neural-net puzzle.

Same interaction as the HuggingFace Space -- type a string, get a number -- but
served from local, checksum-verified weights, with the authors' hints absent.

Launch via ./play.sh (it sets the cloudpickle shim on PYTHONPATH).
"""
import sys
import os
import time
import torch
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
PUZZLE = os.path.join(HERE, "puzzle")

# The two .pt files are whole pickled objects, so the file has to match the
# interpreter's minor version; cloudpickle bakes in version-specific bytecode.
MODEL_FILE = "model_3_11.pt" if sys.version_info >= (3, 11) else "model.pt"
MODEL_PATH = os.environ.get("PUZZLE_MODEL", os.path.join(PUZZLE, MODEL_FILE))

print(f"[play] python {'%d.%d.%d' % sys.version_info[:3]}  torch {torch.__version__}")
print(f"[play] loading {MODEL_PATH}")
model = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
model.eval()
n_params = sum(p.numel() for p in model.parameters())

# Default to CUDA when present. The model needs device_patch to run there at all:
# its string-encoding stage builds a CPU tensor, so a bare .cuda() mismatches.
want = os.environ.get("PUZZLE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
if want != "cpu":
    try:
        from device_patch import to_device
        to_device(model, want, calibrate_with="calibration")
        DEVICE = want
        print(f"[play] on {want} via device_patch ({model._device_hooks} hook)")
    except Exception as e:
        print(f"[play] {want} unavailable ({type(e).__name__}: {e}); staying on CPU")

try:
    from device_patch import Batcher
    BATCHER = Batcher(model)
    BATCHER(["warmup"])
    print("[play] batched path enabled for the Batch tab")
except Exception as e:
    BATCHER = None
    print(f"[play] batched path unavailable ({type(e).__name__}: {e}); using loop")

try:
    from device_patch import NeedleTap
    NEEDLE = NeedleTap(model)
    print("[play] needle tap installed (Tune tab)")
except Exception as e:
    NEEDLE = None
    print(f"[play] needle unavailable ({type(e).__name__}: {e})")

print(f"[play] ready: {type(model).__name__}, {n_params:,} parameters, {DEVICE}")


def predict(text):
    with torch.no_grad():
        return float(model(text))


def predict_one(text):
    try:
        return f"{predict(text):.10g}"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def predict_many(blob):
    """One input per line -> aligned output/input table. Useful for sweeps.

    Uses the batched path when available: a forward pass is latency-bound, so
    hundreds of strings cost barely more than one.
    """
    lines = [ln for ln in blob.split("\n") if ln.strip() != ""]
    if not lines:
        return "(nothing to run)"
    t0 = time.time()
    if BATCHER is not None:
        try:
            vals = BATCHER(lines)
        except Exception as e:
            return f"batched run failed ({type(e).__name__}: {e})"
    else:
        vals = [predict(ln) for ln in lines]
    dt = time.time() - t0
    rows = [f"{v:>16.10g}  |  {ln}" for v, ln in zip(vals, lines)]
    head = f"{len(lines)} inputs in {dt*1000:.0f} ms ({dt/len(lines)*1000:.2f} ms each) on {DEVICE}"
    return "\n".join([head, f"{'output':>16}  |  input", "-" * 60] + rows)


def tune_one(text):
    """Live: output plus the needle — the value entering the final clamp."""
    if NEEDLE is None:
        return "needle unavailable", ""
    try:
        out = predict(text)
        pre = NEEDLE.last()[0]
        return f"{out:.10g}", f"{pre:.10g}"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}", ""


def rank_batch(blob):
    """Rank one-per-line candidates by needle, closest to firing first."""
    if NEEDLE is None:
        return "needle unavailable"
    lines = [ln for ln in blob.split("\n") if ln.strip() != ""]
    if not lines:
        return "(nothing to rank)"
    t0 = time.time()
    rows = []
    try:
        if BATCHER is not None:
            with torch.no_grad():
                for i in range(0, len(lines), BATCHER.chunk):
                    part = lines[i : i + BATCHER.chunk]
                    x = torch.stack([BATCHER.encode(t) for t in part])
                    y = torch.nn.Sequential.forward(model, x)
                    rows += list(zip(NEEDLE.last(), y.flatten().tolist(), part))
        else:
            for ln in lines:
                out = predict(ln)
                rows.append((NEEDLE.last()[0], out, ln))
    except Exception as e:
        return f"rank failed ({type(e).__name__}: {e})"
    rows.sort(key=lambda r: -r[0])
    dt = time.time() - t0
    head = f"{len(rows)} ranked in {dt*1000:.0f} ms on {DEVICE} — closest to firing first"
    body = [f"{pre:>16.10g}  {out:>10.10g}  |  {s}" for pre, out, s in rows]
    return "\n".join([head, f"{'needle':>16}  {'output':>10}  |  input", "-" * 64] + body)


with gr.Blocks(title="Puzzle (local)") as demo:
    gr.Markdown(
        f"### Jane Street puzzle — local playground\n"
        f"`{os.path.basename(MODEL_PATH)}` · {type(model).__name__} · "
        f"{n_params:,} parameters · {DEVICE}\n\n"
        f"Type a string and press **Enter**. No hints here."
    )

    with gr.Tab("Single"):
        inp = gr.Textbox(label="Model Input", value="", autofocus=True)
        out = gr.Textbox(label="Model Output", interactive=False)
        inp.submit(fn=predict_one, inputs=inp, outputs=out)
        gr.Button("Run").click(fn=predict_one, inputs=inp, outputs=out)

    with gr.Tab("Batch"):
        binp = gr.Textbox(
            label="Inputs (one per line)", lines=10,
            placeholder="alpha\nbeta\ngamma",
        )
        bout = gr.Textbox(label="Results", lines=16, interactive=False)
        gr.Button("Run batch").click(fn=predict_many, inputs=binp, outputs=bout)

    with gr.Tab("Tune"):
        gr.Markdown(
            "The **needle** is the value entering the model's final clamp — the "
            "continuous signal behind the flat `0.0` output. Less negative = "
            "closer to firing. Updates live as you type."
        )
        t_in = gr.Textbox(label="Input (live)", value="")
        with gr.Row():
            t_out = gr.Textbox(label="Model Output", interactive=False)
            t_pre = gr.Textbox(label="Needle (pre-clamp)", interactive=False)
        t_in.input(fn=tune_one, inputs=t_in, outputs=[t_out, t_pre])
        t_in.submit(fn=tune_one, inputs=t_in, outputs=[t_out, t_pre])
        r_in = gr.Textbox(
            label="Rank candidates (one per line)", lines=8,
            placeholder="paste variants here — e.g. the same phrase with one thing changed per line",
        )
        r_out = gr.Textbox(label="Ranked", lines=14, interactive=False)
        gr.Button("Rank by needle").click(fn=rank_batch, inputs=r_in, outputs=r_out)

if __name__ == "__main__":
    demo.queue(max_size=64)
    demo.launch(
        server_name=os.environ.get("PUZZLE_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PUZZLE_PORT", "7860")),
        share=False,
        inbrowser=False,
    )
