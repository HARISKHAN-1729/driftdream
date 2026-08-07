"""Figures.  Flow-arrow overlays need care to be legible at print size."""
from __future__ import annotations

import os
from typing import List, Optional

import torch


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def flow_overlay(ax, img: torch.Tensor, flow: torch.Tensor,
                 mask: Optional[torch.Tensor] = None, step: int = 16,
                 scale: float = 1.0, color: str = "#ff2d55"):
    """Quiver of the residual flow over the image."""
    ax.imshow(img.detach().cpu().permute(1, 2, 0).numpy())
    f = flow.detach().cpu()
    h, w = f.shape[-2:]
    ys = torch.arange(step // 2, h, step)
    xs = torch.arange(step // 2, w, step)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    u = f[0][gy, gx]
    v = f[1][gy, gx]
    if mask is not None:
        m = mask.detach().cpu()[gy, gx] > 0.5
        u, v = u * m, v * m
    ax.quiver(gx, gy, u, -v, color=color, angles="xy",
              scale_units="xy", scale=1.0 / max(scale, 1e-6),
              width=0.004, headwidth=3.5)
    ax.set_xticks([]); ax.set_yticks([])


def gallery(images: List[torch.Tensor], flows: List[torch.Tensor],
            titles: List[str], out: str, mask=None, ncol: int = 4,
            scale: float = 1.0):
    plt = _mpl()
    n = len(images)
    ncol = min(ncol, max(n, 1))
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow),
                             squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off"); continue
        flow_overlay(ax, images[i], flows[i], mask, scale=scale)
        ax.set_title(titles[i], fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def curve(xs, series: dict, xlabel: str, ylabel: str, out: str,
          title: str = "", hline: Optional[float] = 0.0):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", label=name, linewidth=1.8, markersize=4)
    if hline is not None:
        ax.axhline(hline, color="#888", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def bars(labels, values, errs, ylabel: str, out: str, title: str = "",
         hline: Optional[float] = None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(range(len(labels)), values, yerr=errs, capsize=4, color="#4c6ef5")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    if hline is not None:
        ax.axhline(hline, color="#c92a2a", linestyle="--", linewidth=1)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out
