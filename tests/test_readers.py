import pytest
import torch
import torch.nn.functional as F

from driftdream.fixation import warp
from driftdream.readers import available, get_eval_readers, get_reader


def _texture(n=80, seed=1):
    torch.manual_seed(seed)
    return F.avg_pool2d(torch.rand(3, n, n).unsqueeze(0), 3, 1, 1)[0]


def test_registry_lists_expected():
    for n in ["motion_energy", "motion_energy_alt", "searaft", "prednet",
              "dual_channel"]:
        assert n in available()


def test_heldout_guard_refuses_loop_reader():
    with pytest.raises(ValueError):
        get_eval_readers("motion_energy", ["motion_energy"])


def test_shape_and_finiteness():
    r = get_reader("motion_energy")
    f = r.flow(torch.rand(2, 3, 64, 64))
    assert f.shape == (2, 64, 64) and torch.isfinite(f).all()


@pytest.mark.parametrize("d", [(2.0, 0.0), (0.0, -2.0), (1.5, 1.5)])
def test_recovers_shift_direction(d):
    """Sign convention. torch conv2d is cross-correlation, which negates the
    measured phase; if this test fails, every result flips sign."""
    x = _texture()
    r = get_reader("motion_energy", latency_gain=0.0)
    f = r.flow(torch.stack([x, warp(x, torch.tensor(d))], 0))
    med = f.flatten(1).median(dim=1).values
    for i in range(2):
        if abs(d[i]) > 1e-6:
            assert (float(med[i]) > 0) == (d[i] > 0), f"axis {i}: {med.tolist()} vs {d}"


def test_gradients_flow_to_image():
    x = torch.rand(3, 48, 48, requires_grad=True)
    r = get_reader("motion_energy")
    f = r.flow(torch.stack([x, warp(x, torch.tensor([1.0, 0.0]))], 0))
    g, = torch.autograd.grad(f.abs().mean(), x)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0
