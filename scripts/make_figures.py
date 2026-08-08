#!/usr/bin/env python
"""Build the paper figures from results on disk.

  fig2_gallery      images with residual-flow arrows, CW/CCW pairs adjacent
  fig3_jitter       rho vs delta_px and vs n_dir  (E2 -- the money figure)
  fig4_controls     rho collapse under percept-abolishing transforms (E4)
  fig5_methods      held-out rho by method, with the zero line (E1)
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
from collections import defaultdict

import torch

sys.path.insert(0, ".")
from driftdream.io import load_image, walk_runs                          # noqa: E402
from driftdream.pipeline import (build_objective, build_target_from_cfg,  # noqa: E402
                                 resolve_device)
from driftdream.viz import bars, curve, gallery                          # noqa: E402


def _runs(root, exp):
    return [m for m in walk_runs(root, exp) if "error" not in m and m.get("metrics")]


def _ceiling(root):
    """rho of a genuine drift pattern, for the reference line. Without it the
    plots imply 1.0 is the target, which nothing reachable ever attains."""
    import json
    p = os.path.join(root, "reference_ceiling.json")
    if os.path.isfile(p):
        try:
            return json.load(open(p)).get("ceiling")
        except Exception:
            return None
    return None


def fig_gallery(root, exp, out, device, n=8):
    rs = [m for m in _runs(root, exp) if m.get("method") == "driftdream"][:n]
    if not rs:
        return None
    imgs, flows, titles, mask = [], [], [], None
    for m in rs:
        cfg = m["config"]
        img = load_image(m["_dir"] + "/image.png", device)
        obj = build_objective(cfg, device=device, seed=int(m.get("seed", 0)))
        with torch.no_grad():
            f = obj.ensemble_flow(img)
        mask = obj.target.mask
        imgs.append(img.cpu()); flows.append(f.cpu())
        titles.append(f"{m['target']}  rho={m['metrics'].get('rho_loop', 0):+.3f}")
    return gallery(imgs, flows, titles, out, mask=mask.cpu() if mask is not None else None,
                   scale=float(max(1e-3, max(f.abs().max() for f in flows))) * 2)


def fig_jitter(root, exp, out_dir):  # noqa: C901
    rs = _runs(root, exp)
    made = []
    for key, xlabel, fname in [("fixation.delta_px", "jitter magnitude (px)",
                                "fig3a_delta.png"),
                               ("fixation.n_dir", "ensemble directions n_dir",
                                "fig3b_ndir.png")]:
        # Hold the OTHER swept variable fixed. Averaging over it silently mixes
        # in the delta_px=0 cells, which are zero by construction: every n_dir
        # then lands on the same value ((0 + 5*0.99)/6 = 0.83) and the panel
        # looks flat for a reason that has nothing to do with n_dir.
        others = sorted({k for m in rs for k in (m.get("sweep") or {}) if k != key})
        fixed = {}
        for o in others:
            vals = sorted({(m.get("sweep") or {}).get(o) for m in rs
                           if (m.get("sweep") or {}).get(o) is not None})
            nz = [v for v in vals if not (isinstance(v, (int, float)) and v == 0)]
            if nz:
                fixed[o] = nz[len(nz) // 2]        # a representative non-zero level
        # Split by METHOD. Averaging driftdream and fixedseed together turns
        # a 0.99-vs-0.5 contrast into a meaningless middle line.
        buck = defaultdict(lambda: defaultdict(list))
        for m in rs:
            sw = m.get("sweep") or {}
            if key not in sw:
                continue
            if any(sw.get(o) != v for o, v in fixed.items()):
                continue
            meth = m.get("method") or "run"
            for mk, mv in m["metrics"].items():
                if mk in ("rho_loop", "rho_probe"):
                    buck[f"{meth}:{mk.replace('rho_', '')}"][sw[key]].append(mv)
        if not buck:
            continue
        xs = sorted({x for d in buck.values() for x in d})
        series = {mk: [st.mean(d[x]) if d.get(x) else float("nan") for x in xs]
                  for mk, d in buck.items()}
        sub = ", ".join(f"{o.split('.')[-1]}={v}" for o, v in fixed.items())
        made.append(curve(xs, series, xlabel, "rho", os.path.join(out_dir, fname),
                          title="H2: jitter is required"
                                + (f"  ({sub} held fixed)" if sub else ""),
                          hline=_ceiling(root)))
    return made


def fig_controls(root, exp, out):
    rs = _runs(root, exp)
    if not rs:
        return None
    buck = defaultdict(list)
    for m in rs:
        for k, v in m["metrics"].items():
            if k.startswith("rho_loop"):
                buck["none" if "__" not in k else k.split("__", 1)[1]].append(v)
    labels = ["none"] + [k for k in buck if k != "none"]
    labels = [l for l in labels if l in buck]
    vals = [st.mean(buck[l]) for l in labels]
    errs = [st.pstdev(buck[l]) if len(buck[l]) > 1 else 0.0 for l in labels]
    return bars(labels, vals, errs, "rho", out, hline=0.0,
                title="E4: percept-abolishing transforms must collapse rho")


def fig_methods(root, exp, out):
    rs = _runs(root, exp)
    if not rs:
        return None
    buck = defaultdict(list)
    for m in rs:
        held = [v for k, v in m["metrics"].items()
                if k.startswith("rho_") and not k.startswith("rho_loop") and "__" not in k]
        if held:
            buck[m.get("tag") or m.get("method")].append(st.mean(held))
    if not buck:
        for m in rs:
            buck[m.get("tag") or m.get("method")].append(m["metrics"].get("rho_loop", 0.0))
    labels = sorted(buck)
    vals = [st.mean(buck[l]) for l in labels]
    errs = [st.pstdev(buck[l]) if len(buck[l]) > 1 else 0.0 for l in labels]
    return bars(labels, vals, errs, "held-out rho", out, hline=0.0,
                title="E1: held-out rho by method")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--which", nargs="*",
                    default=["gallery", "jitter", "controls", "methods"])
    a = ap.parse_args()
    dev = resolve_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    made = []
    if "gallery" in a.which:
        made.append(fig_gallery(a.results_root, a.exp, f"{a.out}/fig2_gallery.png", dev))
    if "jitter" in a.which:
        made += fig_jitter(a.results_root, a.exp, a.out) or []
    if "controls" in a.which:
        made.append(fig_controls(a.results_root, a.exp, f"{a.out}/fig4_controls.png"))
    if "methods" in a.which:
        made.append(fig_methods(a.results_root, a.exp, f"{a.out}/fig5_methods.png"))
    for p in [m for m in made if m]:
        print("wrote", p)


if __name__ == "__main__":
    main()
