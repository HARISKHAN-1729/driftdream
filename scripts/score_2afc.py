#!/usr/bin/env python
"""Score 2AFC responses against key.csv.

Reports accuracy over directed responses (chance = 50%), a binomial p-value,
and the control condition separately -- if participants report motion in
luminance-permuted images at the same rate, the percept claim fails.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


def binom_p(k, n, p=0.5):
    """Two-sided exact-ish p via the normal approximation with continuity."""
    if n == 0:
        return float("nan")
    z = (abs(k - n * p) - 0.5) / math.sqrt(n * p * (1 - p))
    return math.erfc(z / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="results/study")
    ap.add_argument("--responses", nargs="+", required=True,
                    help="one JSON file per participant (or a glob)")
    a = ap.parse_args()
    key = {r["id"]: r for r in csv.DictReader(open(f"{a.study}/key.csv"))}
    files = [f for pat in a.responses for f in glob.glob(pat)]
    per_p, agg = {}, defaultdict(lambda: [0, 0])
    for fp in files:
        resp = json.load(open(fp))
        c = t = 0
        for r in resp:
            k = key.get(r["id"])
            if not k:
                continue
            kind = k["kind"]
            if kind == "test" and r["answer"] in ("cw", "ccw"):
                t += 1; agg["test"][1] += 1
                if r["answer"] == k["truth"]:
                    c += 1; agg["test"][0] += 1
            elif kind == "control":
                agg["control"][1] += 1
                if r["answer"] != "none":
                    agg["control"][0] += 1
        per_p[os.path.basename(fp)] = (c, t, c / t if t else float("nan"))
    print("per participant (correct/directed, accuracy):")
    for k, (c, t, acc) in per_p.items():
        print(f"  {k:28s} {c:3d}/{t:3d}  {acc:.3f}")
    c, t = agg["test"]
    print(f"\npooled direction accuracy: {c}/{t} = {c/max(t,1):.3f} "
          f"(chance 0.5, p={binom_p(c, t):.4g})")
    cc, ct = agg["control"]
    if ct:
        print(f"control 'motion reported' rate: {cc}/{ct} = {cc/ct:.3f} "
              "(should be low; if not, the percept claim fails)")


if __name__ == "__main__":
    main()
