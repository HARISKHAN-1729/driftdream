"""Gradient-free fallbacks -- the insurance policy.

Write these before you need them.  If the in-loop reader turns out to have a
broken gradient path (a hidden .detach(), a hard argmax in the motion-energy
front end), the contribution survives intact: the jitter-ensemble objective and
the held-out evaluation are unchanged, only the optimiser is swapped.  Both
remain strictly training-free.

  best_of_n     Ma et al., "Inference-Time Scaling for Diffusion Models",
                CVPR 2025 -- search over initial noises with a verifier.
  fk_steering   Singhal et al., ICML 2025 -- Feynman-Kac interacting-particle
                SMC; particles are resampled in proportion to exp(beta * reward).
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def best_of_n(generator, objective, prompt: str, n: int = 16, base_seed: int = 0,
              **kw):
    """Sample n seeds unguided, keep the best by rho. Compute-matched baseline."""
    best, best_r, scores = None, -2.0, []
    for i in range(n):
        img, _ = generator.sample(prompt, seed=base_seed * 10_000 + i, hook=None, **kw)
        r = objective.measure(img)["rho"]
        scores.append(r)
        if r > best_r:
            best, best_r = img, r
    return best, {"rho": best_r, "n": n, "all_rho": scores,
                  "mean_rho": sum(scores) / max(len(scores), 1)}


def fk_steering(generator, objective, prompt: str, n_particles: int = 8,
                beta: float = 8.0, n_rounds: int = 4, base_seed: int = 0,
                sigma: float = 0.08, **kw):
    """Simplified FK/SMC over image space: propose, score, resample, perturb.

    Generic: works with any generator, because it only needs samples and a
    scalar reward.  Not a substitute for guidance when gradients work, but a
    genuine training-free optimiser when they do not.
    """
    parts = []
    for i in range(n_particles):
        img, _ = generator.sample(prompt, seed=base_seed * 10_000 + i, hook=None, **kw)
        parts.append(img)
    hist = []
    for rnd in range(n_rounds):
        rew = torch.tensor([objective.measure(p)["rho"] for p in parts])
        hist.append({"round": rnd, "mean": float(rew.mean()), "max": float(rew.max())})
        w = torch.softmax(beta * rew, 0)
        idx = torch.multinomial(w, n_particles, replacement=True)
        g = torch.Generator().manual_seed(base_seed * 977 + rnd)
        s = sigma * (1.0 - rnd / max(n_rounds, 1))
        parts = [(parts[int(j)] + s * torch.randn(parts[int(j)].shape, generator=g)
                  ).clamp(0, 1) for j in idx]
    rew = [objective.measure(p)["rho"] for p in parts]
    b = int(max(range(len(rew)), key=lambda i: rew[i]))
    return parts[b], {"rho": rew[b], "history": hist, "n_particles": n_particles}
