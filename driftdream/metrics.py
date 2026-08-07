"""Objective metrics.  rho is the headline; everything else is support."""
from __future__ import annotations

from typing import Optional

import torch

EPS = 1e-8


def rho(f: torch.Tensor, target_field: torch.Tensor,
        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Masked normalised flow correlation in [-1, 1] (Rosario et al.).

    Scale-invariant: measures agreement of DIRECTION and relative magnitude,
    not absolute speed.  rho(F, F) = 1, rho(F, -F) = -1, rho(F, F_perp) = 0.
    """
    if mask is None:
        mask = torch.ones(f.shape[-2:], device=f.device, dtype=f.dtype)
    m = mask.unsqueeze(0).to(f.device, f.dtype)
    a, b = f * m, target_field.to(f.device, f.dtype) * m
    return (a * b).sum() / (a.flatten().norm() * b.flatten().norm() + EPS)


def sign_agreement(rho_value: float, requested_sign: float = 1.0) -> float:
    """1.0 if the measured motion runs the way it was ASKED to, else 0.0.

    rho is already computed against the SIGNED target field, so a clockwise
    request and an anticlockwise request both score positive rho when correct.
    requested_sign is accepted for call-site clarity but must not be applied
    twice -- doing so scored every anticlockwise run as a failure.
    """
    return float(rho_value > 0)


def quiescence(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared residual flow OUTSIDE the support -- 'only the word moves'."""
    outside = (1.0 - mask).unsqueeze(0).to(f.device, f.dtype)
    denom = outside.sum() * f.shape[0] + EPS
    return (f * outside).pow(2).sum() / denom


def endpoint_error(f: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (f - gt.to(f.device, f.dtype)).pow(2).sum(0).sqrt().mean()


class ContentScorer:
    """CLIP / DINO prompt- and identity-alignment. Lazy, optional, cached."""

    def __init__(self, clip_name: str = "openai/clip-vit-large-patch14",
                 device: str = "cpu", enabled: bool = True):
        self.device, self.enabled, self.clip_name = device, enabled, clip_name
        self._clip = self._proc = None

    def _load(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor
            self._clip = CLIPModel.from_pretrained(self.clip_name).to(self.device).eval()
            for p in self._clip.parameters():
                p.requires_grad_(False)
            self._proc = CLIPProcessor.from_pretrained(self.clip_name)
        return self._clip, self._proc

    def score(self, x: torch.Tensor, prompt: str) -> torch.Tensor:
        """Differentiable CLIP cosine similarity. x: (C,H,W) in [0,1]."""
        if not self.enabled or not prompt:
            return torch.zeros((), device=x.device)
        import torch.nn.functional as F
        clip, proc = self._load()
        mean = torch.tensor([0.4815, 0.4578, 0.4082], device=x.device).view(3, 1, 1)
        std = torch.tensor([0.2686, 0.2613, 0.2758], device=x.device).view(3, 1, 1)
        img = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear",
                            align_corners=False)
        img = (img - mean) / std
        tok = proc(text=[prompt], return_tensors="pt", padding=True).to(self.device)
        ie = clip.get_image_features(pixel_values=img)
        te = clip.get_text_features(**tok)
        return F.cosine_similarity(ie, te).squeeze()
