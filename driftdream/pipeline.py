"""Config -> objects.  One place where configs become runnable components,
so every script (gate, generate, sweep, evaluate) builds them identically.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from .config import load_config
from .fixation import Fixation
from .generators import get_generator
from .metrics import ContentScorer
from .objective import MotionObjective
from .readers import get_eval_readers, get_reader
from .targets import Target, build_target


_PERF_DONE = False


def resolve_device(name: str = "cuda", verbose: bool = False) -> str:
    """Resolve the device and apply GPU settings once per process."""
    global _PERF_DONE
    from .perf import setup
    if name.startswith("cuda") and not torch.cuda.is_available():
        name = "cpu"
    if not _PERF_DONE:
        setup(name, verbose=verbose)
        _PERF_DONE = True
    return name


def build_fixation(cfg: Dict[str, Any], image_size: int, seed: int = 0) -> Fixation:
    """Jitter step is specified at 256px and scaled, so a delta means the same
    thing at every resolution."""
    f = dict(cfg.get("fixation", {}))
    f["delta_px"] = float(f.get("delta_px", 6.0)) * image_size / 256.0
    f["seed"] = seed
    return Fixation(**f)


def build_target_from_cfg(cfg: Dict[str, Any], image_size: int, device) -> Target:
    return build_target(dict(cfg["target"]), image_size, image_size, device=device)


def build_objective(cfg: Dict[str, Any], device: Optional[str] = None,
                    seed: int = 0, reader=None, target: Optional[Target] = None,
                    for_eval_reader: Optional[str] = None) -> MotionObjective:
    device = device or resolve_device(cfg.get("device", "cuda"))
    size = int(cfg.get("image_size", 256))
    tgt = target if target is not None else build_target_from_cfg(cfg, size, device)
    if reader is None:
        name = for_eval_reader or cfg.get("loop_reader", "motion_energy")
        reader = get_reader(name, device=device, **cfg.get("loop_reader_kwargs", {}))
    scorer = None
    lam_c = float(cfg.get("lam_content", 0.0))
    if lam_c > 0:
        scorer = ContentScorer(cfg.get("content_model"), device=device)
    return MotionObjective(
        reader=reader, target=tgt, fixation=build_fixation(cfg, size, seed),
        lam_motion=float(cfg.get("lam_motion", 1.0)), lam_content=lam_c,
        lam_quiet=float(cfg.get("lam_quiet", 0.0)),
        prompt=cfg.get("prompt", ""), content_scorer=scorer,
        residual_mode=cfg.get("residual_mode", "global"), ensemble_seed=seed,
        resample=bool(cfg.get("resample_ensemble", True)))


def build_eval_objectives(cfg: Dict[str, Any], device=None, seed: int = 0,
                          target: Optional[Target] = None):
    """One objective per held-out reader.  get_eval_readers refuses to return
    the in-loop reader, so the held-out claim is enforced, not promised."""
    device = device or resolve_device(cfg.get("device", "cuda"))
    size = int(cfg.get("image_size", 256))
    tgt = target if target is not None else build_target_from_cfg(cfg, size, device)
    names = cfg.get("eval_readers", []) or []
    readers = get_eval_readers(cfg.get("loop_reader", "motion_energy"), names,
                              device=device)
    out = {}
    for r in readers:
        o = build_objective(cfg, device=device, seed=seed, reader=r, target=tgt)
        o.lam_content = 0.0
        out[r.name] = o
    return out


def build_generator(cfg: Dict[str, Any], device: Optional[str] = None):
    """Build the generator, dropping kwargs it does not accept.

    generator_kwargs is shared across a sweep, but generators take different
    arguments: `lr` and `tv_weight` are meaningful for the pixel generator and
    nonsense for SDXL.  Without this filter, e6_generator raises TypeError on
    every sdxl/deepfloyd cell.  Dropped keys are reported once so a genuine
    typo is not silently swallowed.
    """
    import inspect
    import warnings

    device = device or resolve_device(cfg.get("device", "cuda"))
    kw = dict(cfg.get("generator_kwargs", {}))
    name = cfg.get("generator", "pixel")
    if name == "pixel":
        kw.setdefault("size", int(cfg.get("image_size", 256)))

    gen_cls = {"pixel": "driftdream.generators.pixel.PixelGenerator",
               "sdxl": "driftdream.generators.sdxl.SDXLGenerator",
               "deepfloyd": "driftdream.generators.deepfloyd.DeepFloydGenerator"}
    if name in gen_cls:
        import importlib
        mod, cls = gen_cls[name].rsplit(".", 1)
        sig = inspect.signature(getattr(importlib.import_module(mod), cls).__init__)
        accepted = set(sig.parameters) - {"self"}
        dropped = sorted(set(kw) - accepted)
        if dropped:
            warnings.warn(f"generator {name!r} does not accept {dropped}; dropping. "
                          f"Accepted: {sorted(accepted)}")
        kw = {k: v for k, v in kw.items() if k in accepted}
    return get_generator(name, device=device, **kw)


def load_method_config(method: str, target: str, overrides=None) -> Dict[str, Any]:
    from .config import apply_overrides, deep_merge
    cfg = load_config(os.path.join("methods", method))
    cfg = deep_merge(cfg, load_config(os.path.join("targets", target)))
    cfg["method"], cfg["target_name"] = method, target
    return apply_overrides(cfg, overrides or [])
