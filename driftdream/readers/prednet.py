"""Adapter for PredNet (held-out reader B2).

Watanabe, Kitaoka, Sakamoto, Yasugi & Tanaka (Front. Psychol. 2018) showed a
PredNet trained only for next-frame prediction reproduces rotating-snakes
illusory flow.  Kirubeswaran & Storrs (2023) found this behaviour inconsistent
across colour and greyscale variants -- disclose that when citing it.

PredNet predicts frames, not flow.  We read out flow as the displacement
between the last observed frame and PredNet's prediction of it, estimated with
the weight-free phase-based estimator so the read-out itself adds no new
learned parameters.

Set DRIFTDREAM_PREDNET_ROOT and DRIFTDREAM_PREDNET_CKPT.
"""
from __future__ import annotations

import os

import torch

from .base import Reader

_INSTALL = """
PredNet not found. To install:

    git clone https://github.com/eijwat/prednet_in_pytorch.git
    export DRIFTDREAM_PREDNET_ROOT=$PWD/prednet_in_pytorch
    export DRIFTDREAM_PREDNET_CKPT=/path/to/rotating_snakes_weights.pth

Disclose the weight provenance in the paper.
"""


class PredNetReader(Reader):
    name = "prednet"
    n_frames = 4
    differentiable = True

    def __init__(self, device: str = "cpu", root: str | None = None,
                 ckpt: str | None = None, n_frames: int | None = None):
        self.device = device
        self.root = root or os.environ.get("DRIFTDREAM_PREDNET_ROOT")
        self.ckpt = ckpt or os.environ.get("DRIFTDREAM_PREDNET_CKPT")
        if n_frames:
            self.n_frames = int(n_frames)
        self._model = self._est = None

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.root or not os.path.isdir(self.root):
            raise FileNotFoundError(_INSTALL)
        import sys
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        try:
            from prednet import PredNet
        except Exception as e:                           # pragma: no cover
            raise ImportError(f"{_INSTALL}\noriginal error: {e}") from e
        model = PredNet(R_channels=(3, 48, 96, 192), A_channels=(3, 48, 96, 192))
        if self.ckpt and os.path.isfile(self.ckpt):
            sd = torch.load(self.ckpt, map_location="cpu")
            model.load_state_dict(sd.get("state_dict", sd), strict=False)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        return model

    def _estimator(self):
        if self._est is None:
            from .motion_energy import MotionEnergyReader
            self._est = MotionEnergyReader(latency_gain=0.0, device=self.device)
        return self._est

    def flow(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.to(self.device)
        if x.shape[0] < self.n_frames:
            x = torch.cat([x, x[-1:].expand(self.n_frames - x.shape[0], *x.shape[1:])], 0)
        pred = self._load()(x[:self.n_frames].unsqueeze(0))
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = pred[0] if pred.dim() == 4 else pred
        if pred.dim() == 4:
            pred = pred[-1]
        return self._estimator().flow(torch.stack([x[self.n_frames - 1], pred], 0))

    def to(self, device):
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self
