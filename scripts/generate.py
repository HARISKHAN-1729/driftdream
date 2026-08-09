#!/usr/bin/env python
"""Generate one image. For debugging and for figure one-offs."""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")
from driftdream.io import save_image                                    # noqa: E402
from driftdream.pipeline import (build_eval_objectives, build_generator,  # noqa: E402
                                 build_objective, load_method_config,
                                 resolve_device)
from driftdream.search import best_of_n, fk_steering                    # noqa: E402


def run_one(cfg, prompt, seed, device=None):
    """Returns (image, metrics, trace). The single entry point every script uses."""
    device = device or resolve_device(cfg.get("device", "cuda"))
    cfg = dict(cfg); cfg["prompt"] = prompt
    obj = build_objective(cfg, device=device, seed=seed)
    gen = build_generator(cfg, device=device)
    search = cfg.get("search")
    trace = {}
    if search:
        kind = search.get("kind", "best_of_n")
        fn = best_of_n if kind == "best_of_n" else fk_steering
        kw = {k: v for k, v in search.items() if k != "kind"}
        img, trace = fn(gen, obj, prompt, base_seed=seed, **kw)
    elif cfg.get("guided", True):
        img, trace = gen.sample(prompt, seed=seed, hook=obj)
    else:
        img, trace = gen.sample(prompt, seed=seed, hook=None)
    img = img.to(device)
    if not torch.isfinite(img).all():
        # A NaN image yields NaN metrics and a corrupt PNG. Better to record
        # the failure than to fill the results tree with silent garbage.
        raise RuntimeError("generator returned a non-finite image; refusing "
                           "to record NaN metrics")
    m = {f"{k}_loop": v for k, v in obj.measure(img).items()}

    # Held-out PROBE ensemble: fixed, large, different seed, never optimised
    # against. rho_probe is the honest generalisation number; rho_loop is
    # measured on the same trajectories the optimiser saw.
    from driftdream.fixation import Fixation
    size = int(cfg.get("image_size", 256))
    pf = dict(cfg.get("probe_fixation", {}))
    # MAGNITUDE-MATCHED to training. A probe at a different jitter magnitude
    # measures the reader's magnitude tuning rather than the image: a genuine
    # drift pattern itself scores -0.01 at delta=2 and +0.15 at delta=6. Only
    # the directions and the seed are held out.
    train_delta = float(cfg.get("fixation", {}).get("delta_px", 6.0)) * size / 256.0
    probe = Fixation(k=int(pf.get("k", 2)),
                     delta_px=float(pf.get("delta_px", train_delta)),
                     n_dir=int(pf.get("n_dir", 16)), n_turns=0,
                     seed=int(pf.get("seed", 99991)))
    m["rho_probe"] = obj.measure(img, seed=int(pf.get("seed", 99991)),
                                 fixation=probe)["rho"]
    for name, eo in build_eval_objectives(cfg, device=device, seed=seed).items():
        for k, v in eo.measure(img).items():
            m[f"{k}_{name}"] = v
    sign = float(cfg.get("target", {}).get("sign", 1.0))
    from driftdream.metrics import sign_agreement
    m["sign_agreement_loop"] = sign_agreement(m["rho_loop"], sign)
    return img.cpu(), m, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="driftdream")
    ap.add_argument("--target", default="rotation_cw")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/figures/single.png")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_method_config(a.method, a.target, a.override)
    img, m, _ = run_one(cfg, a.prompt, a.seed)
    save_image(img, a.out)
    print(f"saved {a.out}")
    for k, v in sorted(m.items()):
        print(f"  {k:26s} {v:+.4f}")


if __name__ == "__main__":
    main()
