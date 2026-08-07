"""Adapter for the Sun & Nishida Dual-Channel multi-order motion model (reader A).

Sun, Chen, Yang, Li & Nishida, "Machine learning modelling for multi-order
human visual motion processing", Nature Machine Intelligence, 2025.
Repo: https://github.com/anoymized/multi-order-motion-model  (Apache-2.0)

This is the ONE component whose gradient health cannot be verified without the
download, which is why scripts/day1_gate.py exercises it first and why the
search-based fallback exists.  Set DRIFTDREAM_DUAL_ROOT to the cloned repo.

Adapter contract (edit `_forward` if the upstream API differs):
  input  (T, 3, H, W) in [0, 1], T == n_frames
  output (2, H, W) flow in pixels, x right / y down, autograd preserved
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .base import Reader

_INSTALL = """
Dual-Channel model not found. To install:

    git clone https://github.com/anoymized/multi-order-motion-model.git
    export DRIFTDREAM_DUAL_ROOT=$PWD/multi-order-motion-model
    # download the checkpoint (see the repo README) to $DRIFTDREAM_DUAL_ROOT

Then re-run.  If the upstream forward signature differs from the one assumed
in DualChannelReader._forward, edit that method -- it is the only place the
external API is touched.
"""


class DualChannelReader(Reader):
    name = "dual_channel"
    n_frames = 15
    differentiable = True

    def __init__(self, device: str = "cpu", root: str | None = None,
                 ckpt: str | None = None, n_frames: int | None = None):
        self.device = device
        self.root = root or os.environ.get("DRIFTDREAM_DUAL_ROOT")
        self.ckpt = ckpt or os.environ.get("DRIFTDREAM_DUAL_CKPT")
        if n_frames:
            self.n_frames = int(n_frames)
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.root or not os.path.isdir(self.root):
            raise FileNotFoundError(_INSTALL)
        import sys
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        try:
            from model.motion_model import MotionModel  # noqa: F401
        except Exception as e:                           # pragma: no cover
            raise ImportError(f"{_INSTALL}\noriginal error: {e}") from e
        model = MotionModel()
        if self.ckpt and os.path.isfile(self.ckpt):
            sd = torch.load(self.ckpt, map_location="cpu")
            model.load_state_dict(sd.get("state_dict", sd), strict=False)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)     # frozen: no training, ever
        self._model = model
        return model

    def _forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: (1, T, 3, H, W) -> (1, 2, H, W).  EDIT HERE if the API differs."""
        out = self._load()(clip)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("flow", next(iter(out.values())))
        if out.dim() == 5:
            out = out[:, -1]
        return out

    def flow(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.to(self.device)
        if x.shape[0] < self.n_frames:
            x = torch.cat([x, x[-1:].expand(self.n_frames - x.shape[0], *x.shape[1:])], 0)
        x = x[:self.n_frames]
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        f = self._forward(x.unsqueeze(0))[0]
        return f[:, :h, :w]

    def to(self, device):
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self
