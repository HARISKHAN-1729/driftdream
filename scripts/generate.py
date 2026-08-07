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
    m = {f"{k}_loop": v for k, v in obj.measure(img).items()}
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
