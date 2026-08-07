"""Target motion fields F* and their support masks Omega.

A Target is the user's specification: "clockwise rotation in this annulus",
"radial expansion", "this word drifts left".  Fields are normalised to unit
Frobenius norm over the mask, because rho is scale-invariant -- only the
direction and relative magnitude of F* carry meaning.

Convention: field[0] = x-component (right positive), field[1] = y-component
(DOWN positive, matching image row order).  A "clockwise" rotation in screen
coordinates therefore has sign = +1 under this convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

EPS = 1e-8


@dataclass
class Target:
    field: torch.Tensor   # (2, H, W)
    mask: torch.Tensor    # (H, W), float in [0, 1]
    name: str

    def to(self, device) -> "Target":
        return Target(self.field.to(device), self.mask.to(device), self.name)

    @property
    def shape(self):
        return tuple(self.field.shape[-2:])

    def flipped(self) -> "Target":
        """Sign-reversed target; used for the H6 sign-control experiment."""
        return Target(-self.field, self.mask.clone(), self.name + "_flipped")


def _coords(h: int, w: int, device=None):
    ys = torch.arange(h, dtype=torch.float32, device=device)
    xs = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def _normalise(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(0)
    n = (field * m).flatten().norm()
    return field / (n + EPS)


def _soft_annulus(h, w, cx, cy, r_in, r_out, feather, device=None):
    xx, yy = _coords(h, w, device)
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = torch.sigmoid((r - r_in) / max(feather, 1e-3))
    outer = torch.sigmoid((r_out - r) / max(feather, 1e-3))
    return inner * outer


def rotation(h: int, w: int, sign: float = 1.0, r_in: float = 0.18,
             r_out: float = 0.46, gamma: float = 1.0, feather: float = 4.0,
             center: Optional[tuple] = None, device=None) -> Target:
    """Rotating-Snakes analogue.  sign=+1 clockwise on screen, -1 anticlockwise.

    Radii are given as fractions of min(h, w).
    """
    s = min(h, w)
    cx, cy = (w / 2.0, h / 2.0) if center is None else center
    ri, ro = r_in * s, r_out * s
    xx, yy = _coords(h, w, device)
    dx, dy = xx - cx, yy - cy
    r = torch.sqrt(dx ** 2 + dy ** 2) + EPS
    # Clockwise tangent in (x, y-down) coordinates is (-dy, dx)... check:
    # at the top of the circle (dy<0) clockwise motion points +x, and
    # (-dy, dx) = (+|dy|, 0). Correct.
    tx, ty = -dy / r, dx / r
    profile = torch.clamp(r / ro, 0.0, 1.5) ** gamma
    field = sign * torch.stack([tx * profile, ty * profile], 0)
    mask = _soft_annulus(h, w, cx, cy, ri, ro, feather, device)
    return Target(_normalise(field, mask), mask,
                  f"rotation_{'cw' if sign > 0 else 'ccw'}")


def radial(h: int, w: int, sign: float = 1.0, r_in: float = 0.10,
           r_out: float = 0.46, gamma: float = 1.0, feather: float = 4.0,
           center: Optional[tuple] = None, device=None) -> Target:
    """Radial expansion (sign=+1) or contraction (sign=-1)."""
    s = min(h, w)
    cx, cy = (w / 2.0, h / 2.0) if center is None else center
    ri, ro = r_in * s, r_out * s
    xx, yy = _coords(h, w, device)
    dx, dy = xx - cx, yy - cy
    r = torch.sqrt(dx ** 2 + dy ** 2) + EPS
    profile = torch.clamp(r / ro, 0.0, 1.5) ** gamma
    field = sign * torch.stack([dx / r * profile, dy / r * profile], 0)
    mask = _soft_annulus(h, w, cx, cy, ri, ro, feather, device)
    return Target(_normalise(field, mask), mask,
                  f"radial_{'out' if sign > 0 else 'in'}")


def laminar(h: int, w: int, angle_deg: float = 180.0,
            stencil: Optional[torch.Tensor] = None, device=None) -> Target:
    """Uniform drift inside a stencil mask; zero outside.

    stencil: (H, W) float in [0, 1].  If None, a centred horizontal band.
    """
    th = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32, device=device))
    ux, uy = torch.cos(th), torch.sin(th)
    if stencil is None:
        yy = torch.arange(h, dtype=torch.float32, device=device).view(-1, 1)
        band = ((yy > 0.35 * h) & (yy < 0.65 * h)).float().expand(h, w).clone()
        stencil = band
    stencil = stencil.to(device=device, dtype=torch.float32)
    field = torch.stack([ux.expand(h, w), uy.expand(h, w)], 0) * stencil.unsqueeze(0)
    return Target(_normalise(field, stencil), stencil, f"laminar_{int(angle_deg)}")


def stencil_from_text(text: str, h: int, w: int, device=None) -> torch.Tensor:
    """Rasterise text to a (H, W) mask in [0, 1].  Falls back to a band."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        size = max(12, int(h * 0.28))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except Exception:
            font = ImageFont.load_default()
        bbox = d.textbbox((0, 0), text, font=font)
        d.text(((w - (bbox[2] - bbox[0])) / 2 - bbox[0],
                (h - (bbox[3] - bbox[1])) / 2 - bbox[1]), text, fill=255, font=font)
        import numpy as np
        return torch.from_numpy(np.array(img).astype("float32") / 255.0).to(device)
    except Exception:
        return torch.zeros(h, w, device=device)


BUILDERS = {"rotation": rotation, "radial": radial, "laminar": laminar}


def build_target(cfg: dict, h: int, w: int, device=None) -> Target:
    """Build a Target from a config dict, e.g.
    {kind: rotation, sign: 1, r_in: 0.18, r_out: 0.46, gamma: 1.0}
    {kind: laminar, angle_deg: 180, text: DRIFT}
    """
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    if kind not in BUILDERS:
        raise KeyError(f"unknown target kind {kind!r}; have {list(BUILDERS)}")
    if kind == "laminar" and "text" in cfg:
        cfg["stencil"] = stencil_from_text(cfg.pop("text"), h, w, device)
    cfg.pop("text", None)
    return BUILDERS[kind](h, w, device=device, **cfg)
