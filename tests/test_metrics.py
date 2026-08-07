import torch

from driftdream.metrics import quiescence, rho, sign_agreement
from driftdream.targets import rotation


def test_rho_identity_and_negation():
    t = rotation(48, 48)
    assert abs(float(rho(t.field, t.field, t.mask)) - 1.0) < 1e-3
    assert abs(float(rho(t.field, -t.field, t.mask)) + 1.0) < 1e-3


def test_rho_orthogonal_is_zero():
    t = rotation(48, 48)
    perp = torch.stack([-t.field[1], t.field[0]], 0)
    assert abs(float(rho(t.field, perp, t.mask))) < 1e-3


def test_rho_is_scale_invariant():
    t = rotation(48, 48)
    assert abs(float(rho(t.field * 7.3, t.field, t.mask))
               - float(rho(t.field, t.field, t.mask))) < 1e-4


def test_sign_agreement():
    assert sign_agreement(0.4, 1.0) == 1.0 and sign_agreement(-0.4, 1.0) == 0.0


def test_sign_agreement_not_applied_twice():
    """A correct anticlockwise result has POSITIVE rho against its own signed
    target and must score 1.0, not 0.0."""
    assert sign_agreement(0.9, -1.0) == 1.0


def test_quiescence_penalises_outside_only():
    t = rotation(48, 48)
    inside = t.field * t.mask.unsqueeze(0)
    outside = t.field * (1 - t.mask).unsqueeze(0)
    assert float(quiescence(inside, t.mask)) < float(quiescence(outside, t.mask))
