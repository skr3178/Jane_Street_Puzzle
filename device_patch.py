"""Run the puzzle model on a non-CPU device without knowing its architecture.

The model's leading stage builds a tensor from the input string on the CPU, so a
plain `model.cuda()` moves the parameters but not that tensor, and the first
matmul dies with a device mismatch.

Rather than hard-coding where the hop belongs, `to_device` attaches a
forward-pre-hook to every parameterised submodule that moves incoming tensors to
that module's own parameter device. Device only -- never dtype, since integer
index tensors must stay integer.
"""
import torch


class _Stop(Exception):
    pass


class NeedleTap:
    """Capture the value entering the model's final child module.

    The stack ends in a clamp that maps every negative number to 0, so the
    observable output is flat almost everywhere. The tensor feeding that last
    module is the continuous signal behind it -- the "needle". Less negative
    means closer to a non-zero output. Installed once; read after any forward
    (single or batched).
    """

    def __init__(self, model):
        self.box = {}
        last = list(model.children())[-1]

        def cap(_m, args, _out):
            self.box["pre"] = args[0].detach()

        self.handle = last.register_forward_hook(cap)

    def last(self):
        """Needle value(s) from the most recent forward, as a list."""
        t = self.box.get("pre")
        return None if t is None else t.flatten().tolist()


class Batcher:
    """Evaluate many strings in one pass instead of one call each.

    A single forward is ~2721 dependent modules over a 55-element vector, so it
    is pure latency -- the GPU's parallelism sits unused. Adding a batch
    dimension costs almost nothing: 256 strings take about as long as one.

    The model's `forward` is a lambda that encodes the string and then calls
    `Sequential.forward`. We intercept the tensor entering the first
    parameterised module (so the encoder is used, never inspected), stack those
    tensors, and drive the Sequential once on the batch.
    """

    def __init__(self, model, chunk=256):
        self.model = model
        self.chunk = chunk
        self.first = next(
            m for m in model.modules() if any(True for _ in m.parameters(recurse=False))
        )

    def encode(self, text):
        box = {}

        def grab(_m, args):
            box["x"] = args[0]
            raise _Stop

        h = self.first.register_forward_pre_hook(grab)
        try:
            self.model(text)
        except _Stop:
            pass
        finally:
            h.remove()
        return box["x"]

    def __call__(self, texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.chunk):
                part = texts[i : i + self.chunk]
                x = torch.stack([self.encode(t) for t in part])
                y = torch.nn.Sequential.forward(self.model, x)
                out.extend(y.flatten().tolist())
        return out


def _align_inputs(module, args):
    try:
        dev = next(module.parameters(recurse=False)).device
    except StopIteration:
        return None
    moved, out = False, []
    for a in args:
        if torch.is_tensor(a) and a.device != dev:
            a, moved = a.to(dev), True
        out.append(a)
    return tuple(out) if moved else None


def to_device(model, device="cuda", strict_fp32=True, calibrate_with=None):
    """Move `model` to `device` and install the alignment hooks. Returns model.

    strict_fp32 disables TF32 so GPU matmuls stay closer to the CPU result --
    worth keeping on while reverse-engineering, where comparing activations
    across runs matters more than throughput.

    calibrate_with: a sample input. The model is a fixed Sequential, so only the
    first module to receive a CPU tensor ever actually moves anything. Running
    one calibration pass lets us drop the hooks that never fire, which removes
    thousands of no-op Python calls from every subsequent forward.
    """
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        if strict_fp32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    model.to(device)
    handles = {}
    for m in model.modules():
        if any(True for _ in m.parameters(recurse=False)):
            handles[m] = m.register_forward_pre_hook(_align_inputs)
    model._device_hooks = len(handles)

    if calibrate_with is not None:
        fired = set()

        def watch(module, args):
            r = _align_inputs(module, args)
            if r is not None:
                fired.add(module)
            return r

        for m, h in handles.items():
            h.remove()
            handles[m] = m.register_forward_pre_hook(watch)
        with torch.no_grad():
            model(calibrate_with)
        for m, h in handles.items():
            h.remove()
            if m in fired:
                m.register_forward_pre_hook(_align_inputs)
        model._device_hooks = len(fired)

    return model
