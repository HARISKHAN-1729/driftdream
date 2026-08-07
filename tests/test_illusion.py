"""The functional test: does the reader see illusory motion, does it reverse
with the luminance sequence, and does it vanish on a control?"""
import torch

from driftdream.controls import apply_control
from driftdream.fixation import build_movie, make_trajectory
from driftdream.flow import residual_flow
from driftdream.metrics import rho
from driftdream.readers import get_reader
from driftdream.stimuli import concentric_noise, peripheral_drift
from driftdream.targets import rotation

SIZE = 192


def _rho(img, reader, tgt, n_dir=8, k=4, delta=4.5):
    acc = None
    for j in range(n_dir):
        d = make_trajectory(k, delta, 0, j, 360.0 * j / n_dir)
        f = residual_flow(reader, build_movie(img, d), d, mode="global")
        acc = f if acc is None else acc + f
    return float(rho(acc / n_dir, tgt.field, tgt.mask))


def test_illusion_detected_and_reverses():
    r, t = get_reader("motion_energy"), rotation(SIZE, SIZE, sign=1.0)
    fwd = _rho(peripheral_drift(SIZE, SIZE), r, t)
    rev = _rho(peripheral_drift(SIZE, SIZE, reverse=True), r, t)
    assert fwd > 0.03, f"no illusion detected (rho={fwd:+.4f})"
    assert rev < 0 < fwd, f"reversal failed (fwd={fwd:+.4f}, rev={rev:+.4f})"


def test_latency_stage_is_necessary():
    """With no contrast-dependent latency there is no illusion at all."""
    t = rotation(SIZE, SIZE)
    off = _rho(peripheral_drift(SIZE, SIZE), get_reader("motion_energy",
                                                        latency_gain=0.0), t)
    on = _rho(peripheral_drift(SIZE, SIZE), get_reader("motion_energy"), t)
    assert abs(off) < 0.02 and on > abs(off)


def test_control_stimulus_near_zero():
    r, t = get_reader("motion_energy"), rotation(SIZE, SIZE)
    assert abs(_rho(concentric_noise(SIZE, SIZE), r, t)) < 0.03


def test_luminance_scramble_collapses_rho():
    """E4 in miniature: the negative control must collapse rho.

    Block-wise independent permutation preserves local histogram and contrast
    but destroys the globally coherent luminance ORDER.
    """
    r, t = get_reader("motion_energy"), rotation(SIZE, SIZE)
    img = peripheral_drift(SIZE, SIZE)
    base = _rho(img, r, t)
    scr = _rho(apply_control("luminance_scramble", img, seed=3), r, t)
    assert abs(scr) < 0.5 * abs(base), f"base {base:+.4f} vs scrambled {scr:+.4f}"


def test_global_permutation_does_not_abolish():
    """Documents a real finding: ONE global luminance permutation preserves a
    coherent (different) asymmetric sequence, so it yields a different illusion
    rather than none. It is a positive control, not a negative one."""
    r, t = get_reader("motion_energy"), rotation(SIZE, SIZE)
    img = peripheral_drift(SIZE, SIZE)
    perm = _rho(apply_control("luminance_permute", img, seed=3), r, t)
    assert abs(perm) > 0.03


def test_phase_scramble_collapses_rho():
    r, t = get_reader("motion_energy"), rotation(SIZE, SIZE)
    img = peripheral_drift(SIZE, SIZE)
    base = _rho(img, r, t)
    scr = _rho(apply_control("phase_scramble", img, seed=1), r, t)
    assert abs(scr) < 0.5 * abs(base)
