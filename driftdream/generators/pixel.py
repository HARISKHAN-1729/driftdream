"""Direct pixel optimisation -- no diffusion prior, no downloads.

Two jobs:
  1. The Day-0 shortcut.  If rho does not climb on a FREE image tensor, it will
     never climb under a diffusion prior, and you have learned that in minutes
     rather than days.
  2. A paper ablation: unconstrained optimisation is the upper bound on
     achievable rho.  The gap to DriftDream quantifies what the prior costs in
     illusion strength and buys in content.

Parameterised through a sigmoid so pixels stay in [0, 1] without projection.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import Generator, GuidanceHook


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return ((x[..., 1:, :] - x[..., :-1, :]).abs().mean()
            + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())


class PixelGenerator(Generator):
    name = "pixel"

    def __init__(self, size: int = 256, steps: int = 300, lr: float = 0.05,
                 tv_weight: float = 0.02, device: str = "cpu",
                 init: str = "noise", init_scale: float = 0.15,
                 log_every: int = 50, band_limit: int = 0):
        self.size, self.steps, self.lr = int(size), int(steps), float(lr)
        self.tv_weight, self.device = float(tv_weight), device
        self.init, self.init_scale, self.log_every = init, float(init_scale), int(log_every)
        # band_limit: optimise a coarser grid and upsample. Nothing else in the
        # objective constrains spatial scale, so the optimiser is free to use
        # the finest structure the reader responds to -- which produces
        # high-frequency texture rather than the coarse discrete elements that
        # human peripheral drift requires. Set band_limit to the grid size
        # (e.g. 64) to force elements a human could resolve peripherally.
        self.band_limit = int(band_limit)

    def _latent_size(self) -> int:
        return self.band_limit if self.band_limit else self.size

    def _to_image(self, z: torch.Tensor) -> torch.Tensor:
        if self.band_limit and self.band_limit != self.size:
            z = torch.nn.functional.interpolate(
                z.unsqueeze(0), size=(self.size, self.size), mode="bicubic",
                align_corners=False)[0]
        return torch.sigmoid(z)

    def _init_latent(self, seed: int) -> torch.Tensor:
        n = self._latent_size()
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        z = torch.randn(3, n, n, generator=g) * self.init_scale
        if self.init == "snakes":       # E7 structure-prior variant
            from ..stimuli import peripheral_drift
            base = peripheral_drift(n, n).clamp(1e-3, 1 - 1e-3)
            z = torch.log(base / (1 - base)) + 0.1 * z
        return z.to(self.device).requires_grad_(True)

    def sample(self, prompt: str = "", seed: int = 0,
               hook: Optional[GuidanceHook] = None, **kw) -> Tuple[torch.Tensor, dict]:
        steps = int(kw.get("steps", self.steps))
        z = self._init_latent(seed)
        opt = torch.optim.Adam([z], lr=self.lr)
        trace = {"loss": [], "rho": [], "step": []}
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            x = self._to_image(z)
            loss = torch.zeros((), device=x.device)
            info = {}
            if hook is not None:
                loss, info = hook(x, i)
            if self.tv_weight > 0:
                loss = loss + self.tv_weight * total_variation(x)
            loss.backward()
            opt.step()
            if i % self.log_every == 0 or i == steps - 1:
                trace["step"].append(i)
                trace["loss"].append(float(loss.detach()))
                trace["rho"].append(info.get("rho", float("nan")))
        with torch.no_grad():
            x = self._to_image(z)
        return x.detach(), trace
