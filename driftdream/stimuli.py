"""Procedural peripheral-drift stimuli for calibration and initialisation.

We generate our own Rotating-Snakes-style patterns rather than shipping
Kitaoka's copyrighted images.  The mechanism reproduced here is the asymmetric
luminance sequence: repeating steps such as black -> dark -> white -> light
around an annulus.  Differential response latency across luminance levels
makes a motion detector read a systematic drift.

Used for:
  * scripts/day1_gate.py  -- does the reader see the illusion at all?
  * the E7 structure-prior ablation (optional SDEdit-style initialisation);
  * unit tests that must not depend on any download.
"""
from __future__ import annotations

import math

import torch

# Classic asymmetric 4-step sequence (black, dark, white, light).
SNAKE_SEQ = (0.0, 0.28, 1.0, 0.72)


def peripheral_drift(h: int = 256, w: int = 256, n_repeats: int | None = None,
                     n_rings: int = 4, sequence=SNAKE_SEQ, reverse: bool = False,
                     r_in: float = 0.16, r_out: float = 0.47,
                     ring_gap: float = 0.18, bg: float = 0.5,
                     device=None) -> torch.Tensor:
    """(3, H, W) in [0, 1].  reverse=True flips the luminance order, which
    reverses the perceived direction -- the cleanest possible sanity check.

    n_repeats defaults to a resolution-dependent value: a fixed repeat count
    aliases badly at small sizes (24 repeats at 96px puts each luminance step
    below a pixel), which silently kills the illusion.
    """
    seq = list(sequence)[::-1] if reverse else list(sequence)
    s = min(h, w)
    if n_repeats is None:
        n_repeats = max(6, int(round(s / 11.0)))
    cx, cy = w / 2.0, h / 2.0
    ys = torch.arange(h, dtype=torch.float32, device=device).view(-1, 1)
    xs = torch.arange(w, dtype=torch.float32, device=device).view(1, -1)
    dx, dy = xs - cx, ys - cy
    r = torch.sqrt(dx ** 2 + dy ** 2)
    ang = torch.atan2(dy, dx)

    img = torch.full((h, w), bg, device=device)
    ri, ro = r_in * s, r_out * s
    band = (ro - ri) / n_rings
    for k in range(n_rings):
        lo, hi = ri + k * band, ri + (k + 1) * band * (1 - ring_gap)
        ring = (r >= lo) & (r < hi)
        if not bool(ring.any()):
            continue
        # Alternate ring orientation, as in the original stimulus.
        phase = (ang / (2 * math.pi) * n_repeats + (0.5 if k % 2 else 0.0)) % 1.0
        idx = torch.clamp((phase * len(seq)).long(), 0, len(seq) - 1)
        lum = torch.tensor(seq, device=device)[idx]
        img = torch.where(ring, lum, img)
    return img.unsqueeze(0).repeat(3, 1, 1)


def concentric_noise(h: int = 256, w: int = 256, seed: int = 0,
                     device=None) -> torch.Tensor:
    """Control stimulus: same spatial statistics, no asymmetric sequence."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.rand(1, 1, h // 8, w // 8, generator=g).to(device or "cpu")
    up = torch.nn.functional.interpolate(base, size=(h, w), mode="bicubic",
                                         align_corners=False)[0, 0].clamp(0, 1)
    return up.unsqueeze(0).repeat(3, 1, 1)
