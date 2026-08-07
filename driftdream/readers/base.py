"""One interface for every motion reader, so swapping A for B is a config string.

This matters beyond tidiness: the paper's central defence against reward
hacking is that the evaluation readers never appear in any gradient.  That
claim is only credible if swapping readers is trivial and enforced in code
(see registry.get_eval_readers).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Reader(ABC):
    name: str = "reader"
    n_frames: int = 2          # frames consumed per flow estimate
    differentiable: bool = True

    @abstractmethod
    def flow(self, frames: torch.Tensor) -> torch.Tensor:
        """(T, C, H, W) in [0, 1] -> (2, H, W) flow in pixels.

        Component 0 is x (right positive), component 1 is y (down positive).
        Must preserve autograd if `differentiable` is True.
        """

    def to(self, device):
        return self

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name}, n_frames={self.n_frames})"


def to_gray(frames: torch.Tensor) -> torch.Tensor:
    """(T, C, H, W) -> (T, H, W) luminance."""
    if frames.shape[1] == 1:
        return frames[:, 0]
    w = torch.tensor([0.299, 0.587, 0.114], device=frames.device,
                     dtype=frames.dtype).view(1, 3, 1, 1)
    return (frames[:, :3] * w).sum(1)
