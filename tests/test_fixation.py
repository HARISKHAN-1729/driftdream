"""LOAD-BEARING. If the residual of a pure shift is not ~0, every downstream
number is meaningless and will still look like success."""
import torch
import torch.nn.functional as F

from driftdream.fixation import Fixation, build_movie, make_trajectory, warp
from driftdream.flow import residual_flow
from driftdream.readers import get_reader


def _texture(n=96, seed=0):
    torch.manual_seed(seed)
    return F.avg_pool2d(torch.rand(3, n, n).unsqueeze(0), 3, 1, 1)[0]


def test_warp_moves_content_by_plus_delta():
    x = torch.zeros(1, 9, 9)
    x[0, 4, 4] = 1.0
    y = warp(x, torch.tensor([2.0, 0.0]), padding_mode="zeros")
    assert float(y[0, 4, 6]) > 0.9, "content must move by +dx"


def test_warp_is_differentiable():
    x = torch.rand(3, 16, 16, requires_grad=True)
    warp(x, torch.tensor([1.5, -0.5])).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_trajectory_shape_and_start():
    d = make_trajectory(5, 3.0, n_turns=1, seed=0)
    assert d.shape == (5, 2) and torch.allclose(d[0], torch.zeros(2))


def test_pure_shift_residual_collapses():
    """THE test. Global mode must remove the rigid component despite the
    reader's gain error (it under-reads speed by roughly 2.5x)."""
    x, reader = _texture(), get_reader("motion_energy", latency_gain=0.0)
    d = make_trajectory(4, 5.0, angle_deg=25.0)
    mov = build_movie(x, d)
    nominal = residual_flow(reader, mov, d, mode="nominal")
    glob = residual_flow(reader, mov, d, mode="global")
    m_nom = float(nominal.pow(2).sum(0).sqrt().mean())
    m_glo = float(glob.pow(2).sum(0).sqrt().mean())
    assert m_glo < 0.35 * m_nom, f"global {m_glo:.3f} not << nominal {m_nom:.3f}"
    assert m_glo < 1.0


def test_ensemble_cancels_gain_error():
    """Evenly spaced directions cancel the nominal offset -- a further reason
    the ensemble matters, and why gate B must use a single direction."""
    x, reader = _texture(), get_reader("motion_energy", latency_gain=0.0)
    acc = None
    for d in Fixation(k=3, delta_px=5.0, n_dir=4).trajectories():
        r = residual_flow(reader, build_movie(x, d), d, mode="nominal")
        acc = r if acc is None else acc + r
    assert float((acc / 4).pow(2).sum(0).sqrt().mean()) < 1.5


def test_k1_is_degenerate_by_construction():
    x, reader = _texture(48), get_reader("motion_energy")
    d = make_trajectory(1, 0.0)
    f = residual_flow(reader, build_movie(x, d), d)
    assert float(f.abs().max()) == 0.0
