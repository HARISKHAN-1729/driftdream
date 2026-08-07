"""Run bookkeeping.

Two properties everything else depends on:

  1. Generation and evaluation are SEPARATE.  Every run writes image.png plus
     meta.json; evaluate.py walks the tree afterwards.  If a reader adapter
     breaks on day 4 you re-run evaluation in minutes instead of re-running
     hundreds of generations.
  2. Tables and figures regenerate FROM DISK, never from memory.  After a crash
     you can rebuild every number in the paper, and "what is the variance
     across seeds?" is a query rather than a re-run.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Dict, Iterator, Optional

import torch


def git_sha(default: str = "nogit") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def save_image(x: torch.Tensor, path: str) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = (x.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255)
    Image.fromarray(arr.astype("uint8")).save(path)


def load_image(path: str, device="cpu") -> torch.Tensor:
    from PIL import Image
    import numpy as np
    a = np.array(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).to(device)


class RunDir:
    """results/runs/<exp>/<method>/<target>/<prompt_id>/<seed>/"""

    def __init__(self, root: str, exp: str, method: str, target: str,
                 prompt_id: str, seed: int):
        self.path = os.path.join(root, "runs", exp, method, target,
                                 str(prompt_id), f"seed{seed}")
        self.meta_path = os.path.join(self.path, "meta.json")

    def exists(self) -> bool:
        return os.path.isfile(self.meta_path)

    def is_complete(self, require=("rho_loop",)) -> bool:
        if not self.exists():
            return False
        m = self.read()
        return all(k in m.get("metrics", {}) for k in require)

    def make(self):
        os.makedirs(self.path, exist_ok=True)
        return self

    def read(self) -> Dict[str, Any]:
        with open(self.meta_path) as f:
            return json.load(f)

    def write(self, meta: Dict[str, Any]) -> None:
        self.make()
        meta = dict(meta)
        meta.setdefault("git_sha", git_sha())
        meta.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        os.replace(tmp, self.meta_path)          # atomic: no half-written meta

    def update_metrics(self, new: Dict[str, Any]) -> None:
        m = self.read() if self.exists() else {}
        m.setdefault("metrics", {}).update(new)
        self.write(m)

    def image_path(self) -> str:
        return os.path.join(self.path, "image.png")

    def flow_path(self, reader: str) -> str:
        return os.path.join(self.path, f"flow_{reader}.pt")


def walk_runs(root: str, exp: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Yield every meta.json under results/runs (optionally one experiment)."""
    base = os.path.join(root, "runs")
    if exp:
        base = os.path.join(base, exp)
    for dirpath, _, files in os.walk(base):
        if "meta.json" in files:
            try:
                with open(os.path.join(dirpath, "meta.json")) as f:
                    m = json.load(f)
                m["_dir"] = dirpath
                yield m
            except Exception:
                continue


def load_prompts(path: str, n: Optional[int] = None):
    with open(path) as f:
        ps = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return ps[:n] if n else ps
