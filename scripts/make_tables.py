#!/usr/bin/env python
"""Aggregate results/runs/**/meta.json into CSV + Markdown tables.

Rebuilt from disk every time, so any number in the paper can be regenerated
after a crash, and "what is the variance across seeds?" is a query.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from driftdream.io import walk_runs                                     # noqa: E402


def collect(root, exp):
    rows = []
    for m in walk_runs(root, exp):
        if "error" in m or not m.get("metrics"):
            continue
        base = {"exp": m.get("exp"), "method": m.get("method"), "tag": m.get("tag"),
                "target": m.get("target"), "prompt_id": m.get("prompt_id"),
                "seed": m.get("seed")}
        base.update({k: v for k, v in (m.get("sweep") or {}).items()})
        base.update(m["metrics"])
        rows.append(base)
    return rows


def agg(rows, group_keys, metrics):
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r.get(k) for k in group_keys)].append(r)
    out = []
    for key, rs in sorted(buckets.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(group_keys, key))
        row["n"] = len(rs)
        for mname in metrics:
            vals = [r[mname] for r in rs if isinstance(r.get(mname), (int, float))]
            if not vals:
                continue
            row[f"{mname}_mean"] = st.mean(vals)
            row[f"{mname}_sd"] = st.pstdev(vals) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def to_markdown(rows, cols=None, floatfmt="{:+.4f}"):
    if not rows:
        return "_(no rows)_"
    cols = cols or list(rows[0])
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--group", nargs="*", default=["exp", "tag", "target"])
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()

    rows = collect(a.results_root, a.exp)
    if not rows:
        print("no completed runs found"); return
    metrics = a.metrics or sorted({k for r in rows for k, v in r.items()
                                   if isinstance(v, float) and
                                   (k.startswith("rho") or k.startswith("sign")
                                    or k.startswith("quiesc") or k.startswith("clip"))})
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.out, f"{a.exp or 'all'}_raw.csv")
    cols = sorted({k for r in rows for k in r})
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    summ = agg(rows, a.group, metrics)
    scols = a.group + ["n"] + [c for c in (f"{m}_{s}" for m in metrics
                                            for s in ("mean", "sd"))
                               if any(c in r for r in summ)]
    md = to_markdown(summ, scols)
    mdp = os.path.join(a.out, f"{a.exp or 'all'}_summary.md")
    with open(mdp, "w") as f:
        f.write(f"# {a.exp or 'all'}\n\nn runs: {len(rows)}\n\n{md}\n")
    with open(os.path.join(a.out, f"{a.exp or 'all'}_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scols, extrasaction="ignore")
        w.writeheader(); w.writerows(summ)
    print(md)
    print(f"\nwrote {raw}\n      {mdp}")


if __name__ == "__main__":
    main()
