"""Simulated fixational eye movements.

Static-image illusory motion does not exist in the pixels; it is a transient
artefact of microsaccades and drift.  This module builds the short "movie"
that a frozen motion reader is shown, by rigidly jittering one image.

Convention: warp(x, d) produces an image whose CONTENT has moved by +d pixels
(d = (dx, dy), dy positive downwards).  Hence the true optical flow from
warp(x, d0) to warp(x, d1) is the constant field (d1 - d0).  flow.py relies on
this exactly; see tests/test_fixation.py.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def make_trajectory(k: int, delta_px: float, n_turns: int = 0,
                    seed: int = 0, angle_deg: Optional[float] = None,
                    device=None) -> torch.Tensor:
    """Cumulative offsets (k, 2) for a k-frame fixation clip.

    A straight drift of step `delta_px`, with at most `n_turns` direction
    changes (Rosario et al. allow up to 3 location changes per <1 s clip).
    Averaging the residual over a trajectory with varied directions is what
    stops the optimiser from exploiting structure tuned to one shift direction.
    """
    g = torch.Generator().manual_seed(int(seed))
    if angle_deg is None:
        ang = float(torch.rand(1, generator=g).item()) * 2 * math.pi
    else:
        ang = math.radians(angle_deg)
    turn_at = sorted(torch.randperm(max(k - 1, 1), generator=g)[:n_turns].tolist()) if n_turns else []
    offs = [torch.zeros(2)]
    cur = torch.tensor([math.cos(ang), math.sin(ang)]) * delta_px
    for i in range(1, k):
        if (i - 1) in turn_at:
            ang += (float(torch.rand(1, generator=g).item()) - 0.5) * math.pi
            cur = torch.tensor([math.cos(ang), math.sin(ang)]) * delta_px
        offs.append(offs[-1] + cur)
    out = torch.stack(offs, 0)
    return out.to(device) if device is not None else out


def warp(x: torch.Tensor, d: torch.Tensor, padding_mode: str = "reflection") -> torch.Tensor:
    """Rigidly translate x by d = (dx, dy) pixels. Differentiable.

    x: (C, H, W) or (B, C, H, W).  Reflection padding avoids black borders,
    which would otherwise inject spurious flow at the frame edge.
    """
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(0)
    b, c, h, w = x.shape
    d = d.to(x.device, x.dtype)
    ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
    xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    # output(p) = x(p - d)  =>  sample location is p - d
    gx = gx - 2.0 * d[0] / max(w - 1, 1)
    gy = gy - 2.0 * d[1] / max(h - 1, 1)
    grid = torch.stack([gx, gy], -1).unsqueeze(0).expand(b, h, w, 2)
    out = F.grid_sample(x, grid, mode="bilinear",
                        padding_mode=padding_mode, align_corners=True)
    return out.squeeze(0) if squeeze else out


def build_movie(x: torch.Tensor, deltas: torch.Tensor,
                padding_mode: str = "reflection") -> torch.Tensor:
    """(C, H, W) + (K, 2) -> (K, C, H, W) jittered clip. Differentiable in x."""
    return torch.stack([warp(x, deltas[i], padding_mode) for i in range(deltas.shape[0])], 0)


class Fixation:
    """The jitter protocol: n_dir trajectories of k frames each.

    n_dir is not a detail.  With a single trajectory direction the residual is
    dominated by direction-specific reader artefacts: measured rho swings
    between +0.13 and -0.09 across jitter seeds on the SAME image.  Spanning
    directions evenly and averaging collapses that variance and is what makes
    the objective a measurement rather than a coin flip.  It is also what stops
    the optimiser from exploiting structure tuned to one shift direction.

    Set n_dir=1, delta_px=0 or k=1 to obtain the no-jitter ablation (H2).
    """

    def __init__(self, k: int = 4, delta_px: float = 6.0, n_dir: int = 8,
                 n_turns: int = 0, seed: int = 0,
                 angle_deg: Optional[float] = None,
                 padding_mode: str = "reflection"):
        self.k, self.delta_px = int(k), float(delta_px)
        self.n_dir, self.n_turns = int(n_dir), int(n_turns)
        self.seed, self.angle_deg, self.padding_mode = seed, angle_deg, padding_mode

    def trajectories(self, device=None, seed: Optional[int] = None):
        """J directions, evenly spaced, with a RANDOM GLOBAL ROTATION.

        The rotation offset matters. Without it the fan is the fixed set of
        angles {360j/J}, identical at every optimisation step, and the image
        can be tuned to those specific angles: measured rho reaches 0.99 on
        them while rho on any other fan sits near zero. That is an adversarial
        example for one ensemble, not an image with illusory structure. A fresh
        offset each step forces the image to satisfy the distribution of eye
        movement directions instead of J memorised samples.
        """
        base = self.seed if seed is None else seed
        n = max(self.n_dir, 1)
        if self.angle_deg is None:
            g = torch.Generator().manual_seed(int(base) & 0x7FFFFFFF)
            phase = float(torch.rand(1, generator=g).item()) * 360.0 / n
        else:
            phase = 0.0
        out = []
        for j in range(n):
            ang = (self.angle_deg if self.angle_deg is not None
                   else phase + 360.0 * j / n)
            out.append(make_trajectory(self.k, self.delta_px, self.n_turns,
                                       int(base) * 1000 + j, ang, device))
        return out

    def movies(self, x: torch.Tensor, seed: Optional[int] = None):
        """Yield (movie, deltas) for every trajectory. Differentiable in x."""
        for d in self.trajectories(x.device, seed):
            yield build_movie(x, d, self.padding_mode), d

    def movie(self, x: torch.Tensor, seed: Optional[int] = None):
        d = self.trajectories(x.device, seed)[0]
        return build_movie(x, d, self.padding_mode), d

    @property
    def n_reader_calls(self) -> int:
        return max(self.n_dir, 1) * max(self.k - 1, 0)

    def __repr__(self):
        return (f"Fixation(k={self.k}, delta_px={self.delta_px}, "
                f"n_dir={self.n_dir}, n_turns={self.n_turns})")
