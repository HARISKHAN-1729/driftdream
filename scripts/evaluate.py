#!/usr/bin/env python
"""Re-run held-out readers over an existing results tree.

Deliberately decoupled from generation: if a reader adapter breaks on day 4,
you re-evaluate in minutes instead of re-generating hundreds of images.
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")
from driftdream.io import RunDir, load_image, walk_runs                 # noqa: E402
from driftdream.pipeline import build_eval_objectives, resolve_device   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--readers", nargs="*", default=None,
                    help="override eval_readers stored in each run's config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="also (re)compute rho_probe on a held-out ensemble")
    ap.add_argument("--probe-ndir", type=int, default=16)
    ap.add_argument("--probe-seed", type=int, default=99991)
    a = ap.parse_args()

    device = resolve_device(a.device)
    n = 0
    for meta in walk_runs(a.results_root, a.exp):
        cfg = meta.get("config")
        if not cfg or "error" in meta:
            continue
        if a.readers:
            cfg = dict(cfg); cfg["eval_readers"] = a.readers
        names = cfg.get("eval_readers", []) or []
        need_probe = a.probe and (a.force or "rho_probe" not in meta.get("metrics", {}))
        if not names and not need_probe:
            continue
        if (not a.force and not need_probe
                and all(f"rho_{r}" in meta.get("metrics", {}) for r in names)):
            continue
        path = meta["_dir"] + "/image.png"
        try:
            img = load_image(path, device)
        except Exception as e:
            print(f"  !! {path}: {e}")
            continue
        new = {}
        for rn, eo in build_eval_objectives(cfg, device=device,
                                            seed=int(meta.get("seed", 0))).items():
            for k, v in eo.measure(img).items():
                new[f"{k}_{rn}"] = v
        if need_probe:
            # Score with an ensemble the optimiser never saw. rho_loop is
            # measured on the trajectories that were optimised, so it cannot
            # show whether spanning directions bought anything.
            from driftdream.fixation import Fixation
            from driftdream.pipeline import build_objective
            size = int(cfg.get("image_size", 256))
            probe = Fixation(k=2, delta_px=6.0 * size / 256.0,
                             n_dir=a.probe_ndir, n_turns=0, seed=a.probe_seed)
            obj = build_objective(cfg, device=device,
                                  seed=int(meta.get("seed", 0)))
            new["rho_probe"] = obj.measure(img, seed=a.probe_seed,
                                           fixation=probe)["rho"]
            new["rho_gap"] = meta.get("metrics", {}).get("rho_loop",
                                                         float("nan")) - new["rho_probe"]
        rd = RunDir.__new__(RunDir)
        rd.path, rd.meta_path = meta["_dir"], meta["_dir"] + "/meta.json"
        rd.update_metrics(new)
        n += 1
        print(f"  {meta['_dir']}  " + "  ".join(f"{k}={v:+.4f}" for k, v in new.items()
                                                if k.startswith("rho")))
    print(f"evaluated {n} runs")


if __name__ == "__main__":
    main()
