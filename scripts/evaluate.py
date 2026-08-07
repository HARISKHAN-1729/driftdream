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
        if not names:
            continue
        if not a.force and all(f"rho_{r}" in meta.get("metrics", {}) for r in names):
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
        rd = RunDir.__new__(RunDir)
        rd.path, rd.meta_path = meta["_dir"], meta["_dir"] + "/meta.json"
        rd.update_metrics(new)
        n += 1
        print(f"  {meta['_dir']}  " + "  ".join(f"{k}={v:+.4f}" for k, v in new.items()
                                                if k.startswith("rho")))
    print(f"evaluated {n} runs")


if __name__ == "__main__":
    main()
