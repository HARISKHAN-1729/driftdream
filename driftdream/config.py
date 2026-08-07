"""Minimal YAML config: `_base_` inheritance, deep merge, dotted CLI overrides.

Deliberately tiny -- no hydra, no omegaconf, no dependency you have to debug at
2am four days before a deadline.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from typing import Any, Dict, List

import yaml

CONFIG_ROOT = os.environ.get("DRIFTDREAM_CONFIGS", "configs")


def deep_merge(a: Dict, b: Dict) -> Dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if (k in out and isinstance(out[k], dict)
                                           and isinstance(v, dict)) else copy.deepcopy(v)
    return out


def load_config(path: str, _seen=None) -> Dict[str, Any]:
    """Load YAML, resolving `_base_` (a path or list of paths) recursively."""
    _seen = _seen or set()
    path = path if os.path.isabs(path) else os.path.join(CONFIG_ROOT, path)
    if not path.endswith((".yaml", ".yml")):
        path += ".yaml"
    real = os.path.realpath(path)
    if real in _seen:
        raise ValueError(f"circular _base_ at {path}")
    _seen.add(real)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        merged = deep_merge(merged, load_config(b, _seen))
    return deep_merge(merged, cfg)


def _coerce(v: str):
    try:
        return ast.literal_eval(v)
    except Exception:
        return v


def apply_overrides(cfg: Dict, overrides: List[str]) -> Dict:
    """Apply `a.b.c=value` strings from the command line."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(v)
    return cfg


def config_hash(cfg: Dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str)
                        .encode()).hexdigest()[:10]
