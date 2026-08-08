#!/usr/bin/env python
"""Measure the REFERENCE CEILING: what a genuine illusion scores on the probe.

Without this number rho is uninterpretable. A correlation of 1.0 is not the
target and is not reachable: a real peripheral-drift pattern scores about
+0.15 under this measurement, because the reader responds to a small
directional bias riding on a much larger unstructured residual. Judging
generated images against 1.0 makes a good result look like a failure and an
adversarial result look like a triumph.

It also shows that a REAL illusion does not generalise across jitter
magnitude under this reader (-0.01 at delta=2, +0.15 at delta=6). So a probe
at a different magnitude than training measures the reader's magnitude
tuning, not the image. Probes must be magnitude-matched.

Writes results/reference_ceiling.json for the figure scripts to draw as a line.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from driftdream.fixation import Fixation                       # noqa: E402
from driftdream.objective import MotionObjective               # noqa: E402
from driftdream.pipeline import resolve_device                 # noqa: E402
from driftdream.readers import get_reader                      # noqa: E402
from driftdream.stimuli import concentric_noise, peripheral_drift  # noqa: E402
from driftdream.targets import build_target                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="motion_energy")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--deltas", type=float, nargs="*",
                    default=[2.0, 4.0, 6.0, 10.0, 16.0])
    ap.add_argument("--n-dir", type=int, default=16)
    ap.add_argument("--seed", type=int, default=99991)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/reference_ceiling.json")
    a = ap.parse_args()

    dev = resolve_device(a.device)
    reader = get_reader(a.reader, device=dev)
    tgt = build_target({"kind": "rotation", "sign": 1.0}, a.size, a.size, device=dev)
    stim = {"drift_cw": peripheral_drift(a.size, a.size, device=dev),
            "drift_ccw": peripheral_drift(a.size, a.size, reverse=True, device=dev),
            "noise": concentric_noise(a.size, a.size, device=dev)}

    out = {"reader": a.reader, "size": a.size, "n_dir": a.n_dir, "by_delta": {}}
    print(f"reference ceiling | reader={a.reader} size={a.size} n_dir={a.n_dir}\n")
    print(f"{'delta':>7} " + " ".join(f"{k:>11}" for k in stim))
    for d in a.deltas:
        fx = Fixation(k=2, delta_px=d, n_dir=a.n_dir, seed=a.seed)
        obj = MotionObjective(reader, tgt, fx)
        row = {k: obj.measure(v, seed=a.seed, fixation=fx)["rho"]
               for k, v in stim.items()}
        out["by_delta"][str(d)] = row
        print(f"{d:>7} " + " ".join(f"{row[k]:>+11.4f}" for k in stim))

    ceil = max(r["drift_cw"] for r in out["by_delta"].values())
    out["ceiling"] = ceil
    out["ceiling_delta"] = max(out["by_delta"],
                               key=lambda d: out["by_delta"][d]["drift_cw"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nceiling = {ceil:+.4f} at delta={out['ceiling_delta']}")
    print("Report generated rho RELATIVE to this. Anything near 1.0 is not a")
    print("better illusion; it is a solution tuned to the exact probe.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
