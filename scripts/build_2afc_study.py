#!/usr/bin/env python
"""Emit a self-contained 2AFC direction-discrimination study + answer key.

Direction discrimination only -- no aesthetic judgement, no "is this pretty".
Participants see an image and choose clockwise / anticlockwise / no motion.
Because the ground truth is the requested sign, accuracy is objective and
chance is 50% over the two directed responses.

Ships as one HTML file (no server) plus key.csv for scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys

sys.path.insert(0, ".")
from driftdream.io import walk_runs                                     # noqa: E402

HTML = """<!doctype html><meta charset=utf-8><title>Motion direction study</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}
 img{width:100%;max-width:520px;display:block;margin:1rem auto;border:1px solid #ddd}
 button{font-size:1rem;padding:.6rem 1.1rem;margin:.3rem}
 #done{display:none} .bar{height:6px;background:#eee}.bar>div{height:6px;background:#4c6ef5}
</style>
<h2>Which way does the pattern appear to move?</h2>
<p>Look at the centre of the image and let your eyes relax. Blink normally.
Answer quickly with your first impression. There are <b id=n></b> images.</p>
<div class=bar><div id=prog style=width:0></div></div>
<div id=task><img id=img><div style=text-align:center>
<button onclick=ans('cw')>Clockwise</button>
<button onclick=ans('ccw')>Anticlockwise</button>
<button onclick=ans('none')>No motion</button></div></div>
<div id=done><h3>Finished - thank you.</h3>
<p>Copy the text below and send it to the experimenter.</p>
<textarea id=out rows=12 style=width:100%></textarea></div>
<script>
const T=__TRIALS__; let i=0, res=[], t0=Date.now();
document.getElementById('n').textContent=T.length;
function show(){ if(i>=T.length){fin();return;}
 document.getElementById('img').src=T[i].file; t0=Date.now();
 document.getElementById('prog').style.width=(100*i/T.length)+'%';}
function ans(a){res.push({id:T[i].id,answer:a,ms:Date.now()-t0});i++;show();}
function fin(){document.getElementById('task').style.display='none';
 document.getElementById('done').style.display='block';
 document.getElementById('out').value=JSON.stringify(res);}
show();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e3_heldout")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default="results/study")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-controls", action="store_true",
                    help="add luminance-permuted versions as catch trials")
    a = ap.parse_args()

    os.makedirs(f"{a.out}/img", exist_ok=True)
    trials, key = [], []
    runs = [m for m in walk_runs(a.results_root, a.exp)
            if "error" not in m and m.get("metrics")]
    rng = random.Random(a.seed)
    rng.shuffle(runs)
    for m in runs:
        src = m["_dir"] + "/image.png"
        if not os.path.isfile(src):
            continue
        sign = float(m.get("config", {}).get("target", {}).get("sign", 1.0))
        if m.get("target", "").startswith("rotation"):
            truth = "cw" if sign > 0 else "ccw"
        else:
            continue                      # 2AFC only well-defined for rotation
        tid = f"t{len(trials):03d}"
        shutil.copy(src, f"{a.out}/img/{tid}.png")
        trials.append({"id": tid, "file": f"img/{tid}.png"})
        key.append({"id": tid, "truth": truth, "kind": "test",
                    "method": m.get("method"), "dir": m["_dir"]})
        if a.include_controls:
            cpath = m["_dir"] + "/control_luminance_permute.png"
            if os.path.isfile(cpath):
                cid = f"c{len(trials):03d}"
                shutil.copy(cpath, f"{a.out}/img/{cid}.png")
                trials.append({"id": cid, "file": f"img/{cid}.png"})
                key.append({"id": cid, "truth": "none", "kind": "control",
                            "method": m.get("method"), "dir": m["_dir"]})
        if len(trials) >= a.n:
            break
    rng.shuffle(trials)
    with open(f"{a.out}/index.html", "w") as f:
        f.write(HTML.replace("__TRIALS__", json.dumps(trials)))
    with open(f"{a.out}/key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "truth", "kind", "method", "dir"])
        w.writeheader(); w.writerows(key)
    print(f"wrote {a.out}/index.html  ({len(trials)} trials) and key.csv")
    print("Open the HTML locally; participants paste the JSON back. Score with:")
    print("  python scripts/score_2afc.py --study", a.out, "--responses resp.json")


if __name__ == "__main__":
    main()
