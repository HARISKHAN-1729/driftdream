"""Negative controls: the falsifiability machinery (experiment E4).

The strongest single piece of evidence that the method found the real
mechanism rather than adversarial texture is that transforms which abolish the
HUMAN percept also collapse the measured rho.  If rho survives luminance
permutation, you have optimised an artefact -- report that honestly.
"""
from __future__ import annotations

import torch

from .readers.base import to_gray


def _relevel(x: torch.Tensor, out_gray: torch.Tensor) -> torch.Tensor:
    gray = to_gray(x.unsqueeze(0))[0]
    ratio = (out_gray / (gray + 1e-4)).clamp(0, 4).unsqueeze(0)
    return (x * ratio).clamp(0, 1)


def luminance_permute(x: torch.Tensor, n_levels: int = 8, seed: int = 0) -> torch.Tensor:
    """ONE global permutation of the quantised luminance levels.

    WARNING -- this is NOT an abolishing control, and measuring that was
    informative: a global permutation preserves a coherent (merely different)
    asymmetric sequence, so it produces a DIFFERENT illusion rather than none.
    Empirically it can raise |rho| (0.10 -> 0.39 on our procedural stimulus).

    Keep it in the suite as a POSITIVE control -- structure is preserved, so
    rho should remain large -- and use luminance_scramble for the negative one.
    """
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_levels, generator=g).to(x.device)
    gray = to_gray(x.unsqueeze(0))[0]
    idx = torch.clamp((gray * n_levels).long(), 0, n_levels - 1)
    levels = (torch.arange(n_levels, device=x.device).float() + 0.5) / n_levels
    return _relevel(x, levels[perm][idx])


def luminance_scramble(x: torch.Tensor, n_levels: int = 8, seed: int = 0,
                       block: int = 24) -> torch.Tensor:
    """THE negative control: an INDEPENDENT luminance permutation per block.

    Preserves the local luminance histogram, the spatial layout and the
    contrast everywhere, but destroys the globally coherent luminance ORDER
    that peripheral drift depends on.  Humans lose the directional percept;
    rho must collapse toward zero.  If it does not, an artefact was optimised
    rather than the mechanism -- report that.
    """
    g = torch.Generator().manual_seed(int(seed))
    gray = to_gray(x.unsqueeze(0))[0]
    h, w = gray.shape
    idx = torch.clamp((gray * n_levels).long(), 0, n_levels - 1)
    levels = (torch.arange(n_levels, device=x.device).float() + 0.5) / n_levels
    out = torch.empty_like(gray)
    for i in range(0, h, block):
        for j in range(0, w, block):
            perm = torch.randperm(n_levels, generator=g).to(x.device)
            sub = idx[i:i + block, j:j + block]
            out[i:i + block, j:j + block] = levels[perm][sub]
    return _relevel(x, out)


def grayscale(x: torch.Tensor) -> torch.Tensor:
    return to_gray(x.unsqueeze(0))[0].unsqueeze(0).repeat(3, 1, 1)


def phase_scramble(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Randomise Fourier phase, preserve the amplitude spectrum.

    Same spatial-frequency content, no coherent structure. rho must collapse.
    """
    g = torch.Generator().manual_seed(int(seed))
    out = []
    for c in range(x.shape[0]):
        F_ = torch.fft.fft2(x[c])
        ph = torch.rand(F_.shape, generator=g).to(x.device) * 2 * torch.pi
        out.append(torch.fft.ifft2(F_.abs() * torch.exp(1j * ph)).real)
    y = torch.stack(out, 0)
    y = (y - y.amin()) / (y.amax() - y.amin() + 1e-8)
    return y


def rotate180(x: torch.Tensor) -> torch.Tensor:
    """Rotating the image should rotate the flow, not abolish it. Positive control."""
    return torch.flip(x, dims=(-2, -1))


#: Negative controls must collapse rho; positive controls must preserve it.
NEGATIVE_CONTROLS = ("luminance_scramble", "phase_scramble")
POSITIVE_CONTROLS = ("luminance_permute", "rotate180")

CONTROLS = {"luminance_permute": luminance_permute,
            "luminance_scramble": luminance_scramble, "grayscale": grayscale,
            "phase_scramble": phase_scramble, "rotate180": rotate180}


def apply_control(name: str, x: torch.Tensor, **kw) -> torch.Tensor:
    if name not in CONTROLS:
        raise KeyError(f"unknown control {name!r}; have {list(CONTROLS)}")
    return CONTROLS[name](x, **kw)
