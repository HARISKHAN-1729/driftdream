"""Residual illusory flow: the entire signal this project optimises.

The flow read from a jittered pair is dominated by the known global shift.
The illusory component is what is LEFT OVER after removing it.

Two ways to remove it, and the choice matters:

  mode="nominal"  f_res = f - (d_i - d_{i-1})
      Correct only if the reader's gain is exactly 1.  Real motion models
      systematically under-read speed (the built-in reader reads ~0.4x), so
      this leaves a large spurious residual pointing along the jitter -- which
      would swamp the illusory signal and produce confident nonsense.

  mode="global"   f_res = f - spatial_median(f)          [DEFAULT]
      Removes whatever uniform component the reader actually reported,
      whatever its gain.  Also the more faithful model: a uniform retinal slip
      is indistinguishable from eye movement, so only the SPATIALLY STRUCTURED
      departure from it is perceived as object motion.  The cost is that a
      genuinely uniform illusory drift over the whole frame is unmeasurable --
      acceptable, since every target here is spatially structured.

If this subtraction is wrong, every downstream number is meaningless and will
still look like success.  tests/test_fixation.py guards it.
"""
from __future__ import annotations

import torch


def _remove_global(f: torch.Tensor) -> torch.Tensor:
    """Subtract the spatial median of each flow component (robust to outliers)."""
    med = f.flatten(1).median(dim=1).values
    return f - med.view(-1, 1, 1)


def _pair_residual(reader, a, b, true_shift, mode):
    f = reader.flow(torch.stack([a, b], 0))
    if mode == "nominal":
        return f - true_shift.to(f.device, f.dtype).view(2, 1, 1)
    if mode == "global":
        return _remove_global(f)
    raise ValueError(f"unknown residual mode {mode!r}")


def residual_flow(reader, movie: torch.Tensor, deltas: torch.Tensor,
                  mode: str = "global") -> torch.Tensor:
    """Ensemble-averaged residual flow. (K,C,H,W) + (K,2) -> (2,H,W).

    Readers declaring n_frames == 2 are run on every consecutive pair;
    multi-frame readers (e.g. the 15-frame Dual-Channel model) consume the
    whole clip at once.  Averaging over a trajectory with varied directions is
    what stops the optimiser exploiting structure tuned to one shift direction.
    """
    k = movie.shape[0]
    if k < 2:
        # Degenerate by construction (the K=1 ablation): zero flow, zero grad.
        return torch.zeros(2, *movie.shape[-2:], device=movie.device, dtype=movie.dtype)

    nf = getattr(reader, "n_frames", 2)
    if nf == 2:
        acc = None
        for i in range(1, k):
            res = _pair_residual(reader, movie[i - 1], movie[i],
                                 deltas[i] - deltas[i - 1], mode)
            acc = res if acc is None else acc + res
        return acc / (k - 1)

    clip = movie
    if nf > k:
        clip = torch.cat([movie, movie[-1:].expand(nf - k, *movie.shape[1:])], 0)
        step = (deltas[-1] - deltas[0]) / max(k - 1, 1)
    else:
        clip = movie[:nf]
        step = (deltas[nf - 1] - deltas[0]) / max(nf - 1, 1)
    f = reader.flow(clip)
    if mode == "nominal":
        return f - step.to(f.device, f.dtype).view(2, 1, 1)
    return _remove_global(f)


def flow_stats(f: torch.Tensor) -> dict:
    mag = f.pow(2).sum(0).sqrt()
    return {"flow_mean_mag": float(mag.mean()), "flow_max_mag": float(mag.max())}
