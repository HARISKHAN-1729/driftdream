"""The DriftDream objective: jitter-ensemble motion guidance.

    L = lam_m * (1 - rho(f_bar, F*)) + lam_c * (1 - CLIP(x, prompt))
        + lam_q * quiescence(f_bar, Omega)

where f_bar is the residual flow ENSEMBLE-AVERAGED over simulated
microsaccades spanning several directions.  That averaging is the contributed
component: on a static frame the flow is ~0 and the objective is vacuous, so
the ensemble is what makes it a signal at all.

The object is callable as a guidance hook: obj(x, step) -> (loss, info).
"""
from __future__ import annotations

from typing import Optional

import torch

from .fixation import Fixation
from .flow import residual_flow
from .metrics import ContentScorer, quiescence, rho
from .targets import Target


class MotionObjective:
    def __init__(self, reader, target: Target, fixation: Fixation,
                 lam_motion: float = 1.0, lam_content: float = 0.0,
                 lam_quiet: float = 0.0, prompt: str = "",
                 content_scorer: Optional[ContentScorer] = None,
                 residual_mode: str = "global", ensemble_seed: int = 0):
        self.reader, self.target, self.fix = reader, target, fixation
        self.lam_motion, self.lam_content, self.lam_quiet = (
            float(lam_motion), float(lam_content), float(lam_quiet))
        self.prompt = prompt
        self.content = content_scorer
        self.residual_mode, self.ensemble_seed = residual_mode, int(ensemble_seed)

    # ---- core -----------------------------------------------------------
    def ensemble_flow(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """Residual flow averaged over the direction-spanning jitter ensemble.

        All J*(K-1) clips go to the reader in ONE batched call when the reader
        supports it. Looping instead leaves a GPU mostly idle -- each clip is a
        handful of small convolutions, so launch overhead dominates.
        """
        seed = self.ensemble_seed if seed is None else seed
        trajs = self.fix.trajectories(x.device, seed)
        if not trajs or self.fix.k < 2:
            return torch.zeros(2, *x.shape[-2:], device=x.device, dtype=x.dtype)

        if getattr(self.reader, "n_frames", 2) != 2:
            acc = None
            for movie, deltas in self.fix.movies(x, seed):
                f = residual_flow(self.reader, movie, deltas,
                                  mode=self.residual_mode)
                acc = f if acc is None else acc + f
            return acc / len(trajs)

        from .fixation import build_movie
        clips, trues = [], []
        for d in trajs:
            mov = build_movie(x, d, self.fix.padding_mode)
            for i in range(1, self.fix.k):
                clips.append(mov[i - 1:i + 1])
                trues.append(d[i] - d[i - 1])
        flows = self.reader.flow_batch(torch.stack(clips, 0))    # (N,2,H,W)
        if self.residual_mode == "nominal":
            t = torch.stack(trues, 0).to(flows.device, flows.dtype)
            res = flows - t.view(-1, 2, 1, 1)
        else:
            med = flows.flatten(2).median(dim=2).values
            res = flows - med.view(-1, 2, 1, 1)
        return res.mean(0)

    def measure(self, x: torch.Tensor, seed: Optional[int] = None) -> dict:
        """No-grad evaluation. Returns rho and flow statistics."""
        with torch.no_grad():
            f = self.ensemble_flow(x, seed)
            r = float(rho(f, self.target.field, self.target.mask))
            q = float(quiescence(f, self.target.mask))
            mag = f.pow(2).sum(0).sqrt()
        return {"rho": r, "quiescence": q,
                "flow_mean_mag": float(mag.mean()), "flow_max_mag": float(mag.max())}

    def __call__(self, x: torch.Tensor, step: int = 0):
        f = self.ensemble_flow(x)
        r = rho(f, self.target.field, self.target.mask)
        loss = self.lam_motion * (1.0 - r)
        info = {"rho": float(r.detach())}
        if self.lam_quiet > 0:
            q = quiescence(f, self.target.mask)
            loss = loss + self.lam_quiet * q
            info["quiescence"] = float(q.detach())
        if self.lam_content > 0 and self.content is not None and self.prompt:
            c = self.content.score(x, self.prompt)
            loss = loss + self.lam_content * (1.0 - c)
            info["clip"] = float(c.detach())
        info["loss"] = float(loss.detach())
        return loss, info
