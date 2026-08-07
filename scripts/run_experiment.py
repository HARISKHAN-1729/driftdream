#!/usr/bin/env python
"""Config-driven experiment sweeps, with resume.

Every cell writes image.png + meta.json and is skipped if already complete, so
an interrupted sweep continues rather than restarting. Tables and figures are
rebuilt from those files, never from memory.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time

import torch

sys.path.insert(0, ".")
from driftdream.config import apply_overrides, deep_merge, load_config   # noqa: E402
from driftdream.controls import apply_control                           # noqa: E402
from driftdream.io import RunDir, load_prompts, save_image              # noqa: E402
from driftdream.pipeline import (build_eval_objectives, build_objective,  # noqa: E402
                                 load_method_config, resolve_device)
from scripts.generate import run_one                                    # noqa: E402


def sweep_grid(sweep: dict):
    """{'a': [1,2], 'b': [3]} -> [{'a':1,'b':3}, {'a':2,'b':3}]"""
    if not sweep:
        return [{}]
    keys = list(sweep)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(sweep[k] for k in keys))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="config name under configs/experiments")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0, help="cap cells (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    exp = apply_overrides(load_config(f"experiments/{a.exp}"), a.override)
    name = exp.get("exp", a.exp)
    device = resolve_device(exp.get("device", "cuda"))
    prompts = load_prompts(exp.get("prompts_file", "assets/prompts.txt"),
                           int(exp.get("n_prompts", 4)))
    grid = sweep_grid(exp.get("sweep", {}))
    cells = list(itertools.product(exp["methods"], exp["targets"], enumerate(prompts),
                                   exp.get("seeds", [0]), grid))
    if a.limit:
        cells = cells[:a.limit]
    from driftdream.perf import estimate_runtime
    gk = exp.get("generator_kwargs", {})
    fx = exp.get("fixation", {})
    est = estimate_runtime(int(exp.get("image_size", 256)),
                           int(gk.get("steps", 300)), int(fx.get("n_dir", 8)),
                           int(fx.get("k", 4)), int(gk.get("guide_every", 1)),
                           device)
    print(f"[{name}] {len(cells)} cells | device={device} | "
          f"methods={exp['methods']} targets={exp['targets']}")
    print(f"[{name}] {est}  -> whole sweep on the order of "
          f"{len(cells)} x that")
    if device == "cpu" and len(cells) > 12:
        print(f"[{name}] WARNING: {len(cells)} cells on CPU. The jitter "
              "ensemble is many reader calls per step; this will not finish. "
              "Use a GPU, or --limit for a smoke test.")
    if a.dry_run:
        for c in cells[:10]:
            print("   ", c[0], c[1], f"p{c[2][0]}", f"seed{c[3]}", c[4])
        return

    done = skipped = 0
    t_start = time.time()
    for method, target, (pid, prompt), seed, sw in cells:
        cfg = load_method_config(method, target, a.override)
        cfg = deep_merge(cfg, {k: v for k, v in exp.items()
                               if k in ("image_size", "eval_readers", "loop_reader",
                                        "device", "results_root", "generator",
                                        "generator_kwargs", "lam_content")})
        if sw:
            cfg = apply_overrides(cfg, [f"{k}={v!r}" for k, v in sw.items()])
        tag = method + ("_" + "_".join(f"{k.split('.')[-1]}{v}" for k, v in sw.items())
                        if sw else "")
        rd = RunDir(cfg.get("results_root", "results"), name, tag, target, pid, seed)
        if rd.is_complete():
            skipped += 1
            continue
        try:
            img, metrics, trace = run_one(cfg, prompt, seed, device=device)
        except Exception as e:                                   # keep the sweep alive
            print(f"  !! {tag}/{target}/p{pid}/s{seed}: {type(e).__name__}: {e}")
            rd.make(); rd.write({"error": f"{type(e).__name__}: {e}", "metrics": {}})
            continue
        rd.make()
        save_image(img, rd.image_path())
        for cname in exp.get("controls", []) or []:
            ctl = apply_control(cname, img.to(device))
            save_image(ctl, rd.path + f"/control_{cname}.png")
            obj = build_objective(cfg, device=device, seed=seed)
            metrics[f"rho_loop__{cname}"] = obj.measure(ctl)["rho"]
            for rn, eo in build_eval_objectives(cfg, device=device, seed=seed).items():
                metrics[f"rho_{rn}__{cname}"] = eo.measure(ctl)["rho"]
        rd.write({"exp": name, "method": method, "tag": tag, "target": target,
                  "prompt_id": pid, "prompt": prompt, "seed": seed, "sweep": sw,
                  "config": cfg, "metrics": metrics, "trace": trace})
        done += 1
        el = time.time() - t_start
        print(f"  [{done + skipped}/{len(cells)}] {tag}/{target}/p{pid}/s{seed}  "
              f"rho_loop={metrics.get('rho_loop', float('nan')):+.4f}  "
              f"({el:.0f}s elapsed)")
    print(f"[{name}] done={done} skipped={skipped} total={len(cells)} "
          f"in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
