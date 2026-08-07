import torch

from driftdream.targets import build_target, rotation


def test_unit_norm_on_mask():
    t = rotation(64, 64)
    assert abs(float((t.field * t.mask).flatten().norm()) - 1.0) < 1e-4


def test_clockwise_convention():
    """At the TOP of the circle, clockwise motion points +x."""
    t = rotation(64, 64, sign=1.0)
    v = t.field[:, 16, 32]
    assert float(v[0]) > 0 and abs(float(v[1])) < 1e-5


def test_sign_flip_negates_field():
    a, b = rotation(32, 32, sign=1.0), rotation(32, 32, sign=-1.0)
    assert torch.allclose(a.field, -b.field, atol=1e-6)


def test_build_dispatch():
    for cfg in [{"kind": "rotation", "sign": 1.0}, {"kind": "radial"},
                {"kind": "laminar", "angle_deg": 90.0}]:
        t = build_target(cfg, 48, 48)
        assert t.field.shape == (2, 48, 48) and t.mask.shape == (48, 48)
