"""GPU setup. Call once per process, before any model is built.

On an A100 the reader is many small convolutions, so the defaults matter more
than usual: TF32 roughly doubles conv throughput at precision that is far
beyond what a flow correlation needs, and cudnn autotuning pays for itself
because our shapes are fixed for a whole run.
"""
from __future__ import annotations

import torch


def setup(device: str = "cuda", tf32: bool = True, benchmark: bool = True,
          threads: int | None = None, verbose: bool = False) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if benchmark:
            # Shapes are constant within a run, so autotuning is a pure win.
            torch.backends.cudnn.benchmark = True
        if verbose:
            p = torch.cuda.get_device_properties(0)
            print(f"[perf] {p.name}  {p.total_memory / 1e9:.0f} GB  "
                  f"tf32={tf32} benchmark={benchmark}")
        return device
    if threads:
        torch.set_num_threads(int(threads))
    if verbose:
        print(f"[perf] CPU, {torch.get_num_threads()} threads "
              "(expect this to be very slow: the jitter ensemble is "
              "J*(K-1) reader calls per optimisation step)")
    return "cpu"


def estimate_runtime(image_size: int, steps: int, n_dir: int, k: int,
                     guide_every: int = 1, device: str = "cuda") -> str:
    """Rough wall-clock estimate, to catch a run that cannot finish.

    Calibrated from measured CPU throughput; the GPU factor is indicative only.
    """
    calls = n_dir * max(k - 1, 1) * (steps / max(guide_every, 1))
    px = (image_size / 192.0) ** 2
    per_call = 0.30 * px if device == "cpu" else 0.004 * px
    sec = calls * per_call
    return (f"~{sec / 60:.1f} min/image  ({int(calls)} batched reader calls, "
            f"{image_size}px, {device})")
