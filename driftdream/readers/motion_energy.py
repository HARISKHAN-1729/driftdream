"""A weight-free, fully differentiable motion reader.

Why this exists.  The proposal's in-loop reader is the Sun & Nishida
Dual-Channel model, whose repo is a research artefact of unknown gradient
health.  Depending on a gated download for the project to run at all is a bad
bet, so this module implements a self-contained biologically-motivated reader:

  1. optional CONTRAST-DEPENDENT TEMPORAL LATENCY -- darker regions respond
     more slowly.  This is the standard account of peripheral drift (Faubert &
     Herbert 1999; Conway et al. 2005): asymmetric luminance sequences produce
     differential latencies, which a motion detector integrates into a
     systematic directional bias.  Without this stage the reader cannot
     exhibit illusory motion at all, and the whole objective is vacuous.
  2. a quadrature Gabor bank at N orientations x M spatial frequencies;
  3. PHASE-BASED component velocities (Fleet & Jepson 1990):
     for I(p) = cos(2*pi*f*(p.n)), a content shift of d gives a true phase
     change of -2*pi*f*(d.n).  NOTE torch.conv2d is cross-correlation, which
     negates the measured phase, so the recovered component velocity is
     v_comp = +dphi_measured / (2*pi*f).  (Getting this sign wrong flips every
     result in the paper while still looking plausible -- tests guard it.);
  4. amplitude-weighted least-squares combination across orientations
     (intersection of constraints) with Tikhonov damping.

Honest limitation: this is a weaker model of human motion perception than
Dual-Channel, and it has no peripheral/eccentricity mechanism.  It is a valid
IN-LOOP reader and a valid weight-free baseline, but held-out evaluation
should use readers from a different family (SEA-RAFT, PredNet).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import Reader, to_gray

EPS = 1e-6


def gabor_pair(sigma: float, freq: float, theta: float, ksize: int, device=None):
    """Zero-DC even/odd Gabor kernels, each (1, 1, k, k)."""
    r = ksize // 2
    ax = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    xr = xx * math.cos(theta) + yy * math.sin(theta)
    env = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    even = env * torch.cos(2 * math.pi * freq * xr)
    odd = env * torch.sin(2 * math.pi * freq * xr)
    even = even - even.mean()          # kill DC so flat regions give no response
    odd = odd - odd.mean()
    n = env.sum() + EPS
    return (even / n).view(1, 1, ksize, ksize), (odd / n).view(1, 1, ksize, ksize)


def _wrap(d: torch.Tensor) -> torch.Tensor:
    """Wrap phase difference to (-pi, pi] differentiably."""
    return torch.atan2(torch.sin(d), torch.cos(d))


class MotionEnergyReader(Reader):
    n_frames = 2
    differentiable = True

    def __init__(self, name: str = "motion_energy", n_orient: int = 8,
                 freqs=(0.12, 0.20), sigma_scale: float = 1.6,
                 ksize: int = 15, latency_gain: float = 0.75,
                 damping: float = 1e-2, device: str = "cpu"):
        self.name = name
        self.n_orient, self.freqs = int(n_orient), tuple(freqs)
        self.sigma_scale, self.ksize = float(sigma_scale), int(ksize) | 1
        self.latency_gain, self.damping = float(latency_gain), float(damping)
        self.device = device
        self._kernels = None

    def to(self, device):
        self.device, self._kernels = device, None
        return self

    def _bank(self, dtype):
        if self._kernels is None:
            ks = []
            for f in self.freqs:
                sigma = self.sigma_scale / f / (2 * math.pi) * 2.0
                sigma = max(1.5, min(sigma, self.ksize / 3.0))
                for i in range(self.n_orient):
                    th = math.pi * i / self.n_orient
                    e, o = gabor_pair(sigma, f, th, self.ksize, self.device)
                    ks.append((th, f, e.to(dtype), o.to(dtype)))
            self._kernels = ks
        return self._kernels

    def _latency(self, gray: torch.Tensor) -> torch.Tensor:
        """Contrast/luminance-dependent temporal lag.

        I_eff[t] = (1-a)*I[t] + a*I[t-1],  a = gain * (1 - L)
        Dark regions lag; bright regions are prompt.  On an asymmetric
        luminance sequence this differential latency is what a motion detector
        reads out as systematic drift.
        """
        if self.latency_gain <= 0:
            return gray
        a = self.latency_gain * (1.0 - gray.clamp(0, 1))
        prev = torch.cat([gray[:1], gray[:-1]], 0)
        return (1 - a) * gray + a * prev

    def flow(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[0] < 2:
            return torch.zeros(2, *frames.shape[-2:], device=frames.device,
                               dtype=frames.dtype)
        gray = to_gray(frames.to(self.device))
        gray = self._latency(gray)
        i0, i1 = gray[0].unsqueeze(0).unsqueeze(0), gray[-1].unsqueeze(0).unsqueeze(0)
        pad = self.ksize // 2
        h, w = gray.shape[-2:]
        dt = float(max(frames.shape[0] - 1, 1))

        A = torch.zeros(2, 2, h, w, device=i0.device, dtype=i0.dtype)
        b = torch.zeros(2, h, w, device=i0.device, dtype=i0.dtype)
        for th, f, ke, ko in self._bank(i0.dtype):
            e0 = F.conv2d(i0, ke, padding=pad)[0, 0]
            o0 = F.conv2d(i0, ko, padding=pad)[0, 0]
            e1 = F.conv2d(i1, ke, padding=pad)[0, 0]
            o1 = F.conv2d(i1, ko, padding=pad)[0, 0]
            a0 = torch.sqrt(e0 ** 2 + o0 ** 2 + EPS)
            a1 = torch.sqrt(e1 ** 2 + o1 ** 2 + EPS)
            dphi = _wrap(torch.atan2(o1, e1) - torch.atan2(o0, e0))
            v = dphi / (2 * math.pi * f) / dt           # see sign note in docstring
            wgt = a0 * a1
            nx, ny = math.cos(th), math.sin(th)
            A[0, 0] += wgt * nx * nx
            A[0, 1] += wgt * nx * ny
            A[1, 0] += wgt * nx * ny
            A[1, 1] += wgt * ny * ny
            b[0] += wgt * v * nx
            b[1] += wgt * v * ny

        lam = self.damping * (A[0, 0] + A[1, 1] + EPS)
        a11, a12, a22 = A[0, 0] + lam, A[0, 1], A[1, 1] + lam
        det = a11 * a22 - a12 * a12 + EPS
        vx = (a22 * b[0] - a12 * b[1]) / det
        vy = (-a12 * b[0] + a11 * b[1]) / det
        return torch.stack([vx, vy], 0)
