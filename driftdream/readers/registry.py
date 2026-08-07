"""Reader registry, and the code that enforces held-out evaluation.

The paper's defence against reward hacking is that evaluation readers never
appear in any gradient.  get_eval_readers() turns that from a promise in the
prose into an assertion in the code.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from .base import Reader

_REGISTRY: Dict[str, Callable[..., Reader]] = {}
#: Readers that belong to the same family as the built-in one.  Using these
#: for held-out evaluation is better than nothing but is NOT independent
#: evidence; the paper must report a cross-family reader.
SAME_FAMILY = {"motion_energy", "motion_energy_alt"}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def available() -> List[str]:
    return sorted(_REGISTRY)


def get_reader(name: str, device: str = "cpu", **kw) -> Reader:
    if name not in _REGISTRY:
        raise KeyError(f"unknown reader {name!r}; available: {available()}")
    return _REGISTRY[name](device=device, **kw)


def get_eval_readers(loop_reader: str, names: List[str], device: str = "cpu",
                     strict: bool = True) -> List[Reader]:
    """Instantiate held-out readers, refusing to include the in-loop one."""
    out = []
    for n in names:
        if n == loop_reader:
            raise ValueError(
                f"Evaluation reader {n!r} is the in-loop reader. The held-out "
                "claim would be false. Choose a different reader.")
        if strict and n in SAME_FAMILY and loop_reader in SAME_FAMILY:
            import warnings
            warnings.warn(
                f"Eval reader {n!r} shares a family with in-loop {loop_reader!r}. "
                "Fine for smoke tests; NOT independent evidence for the paper.")
        out.append(get_reader(n, device=device))
    return out


@register("motion_energy")
def _me(device="cpu", **kw):
    from .motion_energy import MotionEnergyReader
    return MotionEnergyReader(name="motion_energy", device=device, **kw)


@register("motion_energy_alt")
def _me_alt(device="cpu", **kw):
    """Same family, different parameters. Smoke-test stand-in only."""
    from .motion_energy import MotionEnergyReader
    p = dict(n_orient=6, freqs=(0.09, 0.17), ksize=19, sigma_scale=2.0,
             latency_gain=0.65)
    p.update(kw)
    return MotionEnergyReader(name="motion_energy_alt", device=device, **p)


@register("searaft")
def _searaft(device="cpu", **kw):
    from .searaft import SEARAFTReader
    return SEARAFTReader(device=device, **kw)


@register("prednet")
def _prednet(device="cpu", **kw):
    from .prednet import PredNetReader
    return PredNetReader(device=device, **kw)


@register("dual_channel")
def _dual(device="cpu", **kw):
    from .dual_channel import DualChannelReader
    return DualChannelReader(device=device, **kw)
