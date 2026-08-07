"""Generator interface.  The guidance loop must not know which model it steers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch

GuidanceHook = Callable[[torch.Tensor, int], Tuple[torch.Tensor, dict]]


class Generator(ABC):
    name: str = "generator"
    #: which stage the hook sees; resolves the resolution question empirically
    stages = ("base",)

    @abstractmethod
    def sample(self, prompt: str, seed: int, hook: Optional[GuidanceHook] = None,
               **kw) -> Tuple[torch.Tensor, dict]:
        """Return ((3, H, W) image in [0, 1], trace dict)."""

    def to(self, device):
        return self
