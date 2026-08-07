import torch

from driftdream.config import apply_overrides, deep_merge, load_config
from driftdream.controls import CONTROLS, apply_control
from driftdream.fixation import Fixation
from driftdream.objective import MotionObjective
from driftdream.pipeline import (build_fixation, build_objective,
                                 load_method_config, resolve_device)
from driftdream.readers import get_reader
from driftdream.targets import rotation


def test_configs_load_and_inherit():
    cfg = load_method_config("driftdream", "rotation_cw")
    assert cfg["method"] == "driftdream" and cfg["target"]["kind"] == "rotation"
    assert cfg["fixation"]["n_dir"] >= 1


def test_nojitter_config_is_degenerate():
    cfg = load_method_config("nojitter", "rotation_cw")
    f = build_fixation(cfg, 128)
    assert f.delta_px == 0.0 and f.n_dir == 1


def test_overrides_and_merge():
    c = apply_overrides({"a": {"b": 1}}, ["a.b=5", "c=2.5"])
    assert c["a"]["b"] == 5 and c["c"] == 2.5
    assert deep_merge({"x": {"y": 1}}, {"x": {"z": 2}})["x"] == {"y": 1, "z": 2}


def test_objective_returns_loss_and_info():
    obj = MotionObjective(get_reader("motion_energy"), rotation(64, 64),
                          Fixation(k=2, delta_px=3.0, n_dir=2))
    x = torch.rand(3, 64, 64, requires_grad=True)
    loss, info = obj(x, 0)
    assert torch.isfinite(loss) and "rho" in info
    g, = torch.autograd.grad(loss, x)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_objective_measure_is_nograd():
    obj = MotionObjective(get_reader("motion_energy"), rotation(64, 64),
                          Fixation(k=2, delta_px=3.0, n_dir=2))
    m = obj.measure(torch.rand(3, 64, 64))
    assert set(["rho", "quiescence"]).issubset(m)


def test_all_controls_preserve_shape():
    x = torch.rand(3, 48, 48)
    for name in CONTROLS:
        assert apply_control(name, x).shape == x.shape


def test_resolve_device_falls_back():
    assert resolve_device("cuda") in ("cuda", "cpu")


def test_generator_kwargs_are_filtered_per_generator():
    """generator_kwargs is shared across a sweep, but generators take different
    arguments. Without filtering, e6_generator raises TypeError on every
    sdxl/deepfloyd cell."""
    import warnings

    from driftdream.pipeline import build_generator
    cfg = {"generator": "pixel", "device": "cpu", "image_size": 64,
           "generator_kwargs": {"steps": 5, "lr": 0.1, "tv_weight": 0.01}}
    assert type(build_generator(cfg)).__name__ == "PixelGenerator"
    cfg["generator"] = "sdxl"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_generator(cfg)              # must not raise
        assert any("does not accept" in str(x.message) for x in w)
