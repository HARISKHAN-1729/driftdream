"""Adapter for SEA-RAFT (held-out reader B1).

Wang, Lipson & Deng, "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical
Flow", ECCV 2024 (Oral).  BSD-3.  Repo: https://github.com/princeton-vl/SEA-RAFT

Held out: never used inside any gradient. Two-frame.
Set DRIFTDREAM_SEARAFT_ROOT and DRIFTDREAM_SEARAFT_CKPT.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .base import Reader

_INSTALL = """
SEA-RAFT not found. To install:

    git clone https://github.com/princeton-vl/SEA-RAFT.git
    export DRIFTDREAM_SEARAFT_ROOT=$PWD/SEA-RAFT
    # download a checkpoint, e.g. Tartan-C-T-TSKH-spring540x960-M
    export DRIFTDREAM_SEARAFT_CKPT=/path/to/checkpoint.pth
"""


class SEARAFTReader(Reader):
    name = "searaft"
    n_frames = 2
    differentiable = True

    def __init__(self, device: str = "cpu", root: str | None = None,
                 ckpt: str | None = None, iters: int = 12):
        self.device = device
        self.root = root or os.environ.get("DRIFTDREAM_SEARAFT_ROOT")
        self.ckpt = ckpt or os.environ.get("DRIFTDREAM_SEARAFT_CKPT")
        self.iters, self._model = int(iters), None

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.root or not os.path.isdir(self.root):
            raise FileNotFoundError(_INSTALL)
        import sys
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        try:
            from core.raft import RAFT
        except Exception as e:                           # pragma: no cover
            raise ImportError(f"{_INSTALL}\noriginal error: {e}") from e
        import argparse
        import json
        cfg_path = os.path.join(self.root, "config", "eval", "spring-M.json")
        args = argparse.Namespace(**(json.load(open(cfg_path))
                                     if os.path.isfile(cfg_path) else {}))
        model = RAFT(args)
        if self.ckpt and os.path.isfile(self.ckpt):
            sd = torch.load(self.ckpt, map_location="cpu")
            model.load_state_dict({k.replace("module.", ""): v
                                   for k, v in sd.get("state_dict", sd).items()},
                                  strict=False)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        return model

    def flow(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.to(self.device)[:2] * 255.0
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        out = self._load()(x[0:1], x[1:2], iters=self.iters, test_mode=True)
        f = out[-1] if isinstance(out, (list, tuple)) else out
        if isinstance(f, dict):
            f = f.get("flow", next(iter(f.values())))
        if f.dim() == 5:
            f = f[:, -1]
        return f[0][:, :h, :w]

    def to(self, device):
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self
