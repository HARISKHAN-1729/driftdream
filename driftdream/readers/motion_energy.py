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
        self._stacked = None

    def to(self, device):
        self.device, self._kernels, self._stacked = device, None, None
        return self

    def _stacked_bank(self, dtype):
        """Concatenated filter bank, built once. Rebuilding it per call added
        real overhead inside the optimisation loop."""
        if self._stacked is None or self._stacked[0].dtype != dtype:
            bank = self._bank(dtype)
            self._stacked = (
                torch.cat([k[2] for k in bank], 0),
                torch.cat([k[3] for k in bank], 0),
                torch.tensor([k[1] for k in bank], device=self.device,
                             dtype=dtype).view(1, -1, 1, 1),
                torch.tensor([k[0] for k in bank], device=self.device,
                             dtype=dtype).view(1, -1, 1, 1),
            )
        return self._stacked

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

    def _latency_batch(self, gray: torch.Tensor) -> torch.Tensor:
        """(B,T,H,W) version of _latency; lag is applied along T."""
        if self.latency_gain <= 0:
            return gray
        a = self.latency_gain * (1.0 - gray.clamp(0, 1))
        prev = torch.cat([gray[:, :1], gray[:, :-1]], 1)
        return (1 - a) * gray + a * prev

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

    def _solve(self, i0: torch.Tensor, i1: torch.Tensor, dt: float) -> torch.Tensor:
        """(B,1,H,W) x2 -> (B,2,H,W). All orientations share one conv batch."""
        pad = self.ksize // 2
        ke, ko, freqs, th = self._stacked_bank(i0.dtype)
        e0 = F.conv2d(i0, ke, padding=pad)               # (B,F,H,W)
        o0 = F.conv2d(i0, ko, padding=pad)
        e1 = F.conv2d(i1, ke, padding=pad)
        o1 = F.conv2d(i1, ko, padding=pad)

        a0 = torch.sqrt(e0 ** 2 + o0 ** 2 + EPS)
        a1 = torch.sqrt(e1 ** 2 + o1 ** 2 + EPS)
        dphi = _wrap(torch.atan2(o1, e1) - torch.atan2(o0, e0))
        v = dphi / (2 * math.pi * freqs) / dt            # see sign note above
        wgt = a0 * a1
        nx, ny = torch.cos(th), torch.sin(th)
        a11 = (wgt * nx * nx).sum(1)
        a12 = (wgt * nx * ny).sum(1)
        a22 = (wgt * ny * ny).sum(1)
        b0 = (wgt * v * nx).sum(1)
        b1 = (wgt * v * ny).sum(1)

        lam = self.damping * (a11 + a22 + EPS)
        a11d, a22d = a11 + lam, a22 + lam
        det = a11d * a22d - a12 * a12 + EPS
        vx = (a22d * b0 - a12 * b1) / det
        vy = (-a12 * b0 + a11d * b1) / det
        return torch.stack([vx, vy], 1)                  # (B,2,H,W)

    def flow(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[0] < 2:
            return torch.zeros(2, *frames.shape[-2:], device=frames.device,
                               dtype=frames.dtype)
        return self.flow_batch(frames.unsqueeze(0))[0]

    def flow_batch(self, clips: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) -> (B, 2, H, W). One conv batch for the whole set."""
        if clips.dim() == 4:
            clips = clips.unsqueeze(0)
        x = clips.to(self.device)
        bsz, t = x.shape[0], x.shape[1]
        if t < 2:
            return torch.zeros(bsz, 2, *x.shape[-2:], device=x.device, dtype=x.dtype)
        gray = to_gray(x.reshape(bsz * t, *x.shape[2:])).reshape(bsz, t, *x.shape[-2:])
        gray = self._latency_batch(gray)
        i0 = gray[:, 0].unsqueeze(1)
        i1 = gray[:, -1].unsqueeze(1)
        return self._solve(i0, i1, float(max(t - 1, 1)))
