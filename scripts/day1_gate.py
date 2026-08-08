#!/usr/bin/env python
"""THE GO/NO-GO. Run this before writing or trusting anything else.

Four gates, in the order that fails cheapest first:

  A  READER HEALTH   loads, runs, and produces finite gradients.
  B  SIGN CONVENTION a known rigid shift is recovered with the right sign, and
                     the residual of a pure shift collapses.  If this fails,
                     every downstream number is meaningless AND will still look
                     like success.
  C  ILLUSION        does the reader see illusory rotation in a peripheral-drift
                     pattern, does it REVERSE when the luminance sequence is
                     reversed, and is it ~0 on a control?
  C2 RESOLUTION      does the illusion survive downsampling to 64px?  This is
                     what decides guidance_stage: if it dies, DeepFloyd stage-I
                     (64x64) cannot work and you must guide at sr1 or use SDXL.
  D  OPTIMISABILITY  does rho climb on a FREE pixel tensor in a few dozen steps?
                     If not on a free tensor, never under a diffusion prior.

Exit code 0 = proceed with the gradient path; 1 = switch to scripts search
(FK / best-of-N), today, not on day 3.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from driftdream.fixation import Fixation, build_movie, make_trajectory  # noqa: E402
from driftdream.flow import residual_flow                                # noqa: E402
from driftdream.generators import get_generator                          # noqa: E402
from driftdream.metrics import rho                                       # noqa: E402
from driftdream.objective import MotionObjective                         # noqa: E402
from driftdream.pipeline import resolve_device                           # noqa: E402
from driftdream.readers import get_reader                                # noqa: E402
from driftdream.stimuli import concentric_noise, peripheral_drift        # noqa: E402
from driftdream.targets import build_target                              # noqa: E402

OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def line(name, ok, detail=""):
    print(f"  [{OK if ok else BAD}] {name:34s} {detail}")
    return ok


def gate_a(reader, device, size):
    print("\nGATE A  reader health")
    x = torch.rand(3, size, size, device=device, requires_grad=True)
    mov = torch.stack([x, torch.roll(x, 3, dims=-1)], 0)
    try:
        f = reader.flow(mov)
    except Exception as e:
        return line("forward runs", False, f"{type(e).__name__}: {e}")
    ok = line("forward runs", tuple(f.shape) == (2, size, size), f"shape {tuple(f.shape)}")
    ok &= line("output finite", bool(torch.isfinite(f).all()))
    if not getattr(reader, "differentiable", True):
        line("gradients", False, "reader declares differentiable=False")
        return False
    try:
        g, = torch.autograd.grad(f.abs().mean(), x)
        gok = bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
        ok &= line("gradients flow", gok, f"|grad| = {float(g.abs().sum()):.3e}")
    except Exception as e:
        ok &= line("gradients flow", False, f"{type(e).__name__}: {e} "
                                            "-> USE THE SEARCH FALLBACK")
    return ok


def gate_b(reader, device, size):
    print("\nGATE B  sign convention and residual")
    torch.manual_seed(0)
    x = F.avg_pool2d(torch.rand(3, size, size, device=device).unsqueeze(0), 3, 1, 1)[0]
    ok = True
    for d in [(2.0, 0.0), (0.0, -1.5)]:
        dv = torch.tensor(d, device=device)
        from driftdream.fixation import warp
        f = reader.flow(torch.stack([x, warp(x, dv)], 0))
        med = f.flatten(1).median(dim=1).values
        agree = all((float(med[i]) > 0) == (d[i] > 0) for i in range(2) if abs(d[i]) > 1e-6)
        ok &= line(f"recovers shift {d}", agree,
                   f"read ({float(med[0]):+.2f}, {float(med[1]):+.2f})")
    # ONE direction on purpose. With an evenly spaced ensemble the nominal
    # shifts cancel and the gain error becomes invisible -- which is a further
    # argument for the ensemble, but makes this a useless test.
    fx = Fixation(k=4, delta_px=6.0, n_dir=1, angle_deg=25.0)
    mags = {}
    for mode in ("nominal", "global"):
        mov, dd = fx.movie(x)
        r = residual_flow(reader, mov, dd, mode=mode)
        mags[mode] = float(r.pow(2).sum(0).sqrt().mean())
    ok &= line("global-mode residual collapses", mags["global"] < 0.5 * mags["nominal"],
               f"nominal {mags['nominal']:.3f} vs global {mags['global']:.3f}")
    return ok


def _ensemble_rho(img, reader, tgt, n_dir=8, k=4, delta=6.0, mode="global"):
    acc = None
    for j in range(n_dir):
        d = make_trajectory(k, delta, 0, j, 360.0 * j / n_dir, img.device)
        f = residual_flow(reader, build_movie(img, d), d, mode=mode)
        acc = f if acc is None else acc + f
    return float(rho(acc / n_dir, tgt.field, tgt.mask))


def gate_c(reader, device, size, thresh):
    # Peripheral drift is under-resolved below ~128px: at 96px the measured
    # rho is +0.003, at 128px +0.051, at 192px +0.100. Testing "can the reader
    # see the illusion" on an under-resolved stimulus tests the stimulus, not
    # the reader, so this gate floors the size.
    size = max(size, 160)
    print(f"\nGATE C  does the reader see the illusion?  (stimulus at {size}px)")
    tgt = build_target({"kind": "rotation", "sign": 1.0}, size, size, device=device)
    d = max(size / 256.0 * 6.0, 2.0)
    a = _ensemble_rho(peripheral_drift(size, size, device=device), reader, tgt, delta=d)
    b = _ensemble_rho(peripheral_drift(size, size, reverse=True, device=device),
                      reader, tgt, delta=d)
    c = _ensemble_rho(concentric_noise(size, size, device=device), reader, tgt, delta=d)
    ok = line("illusion detected", a > thresh, f"rho = {a:+.4f} (need > {thresh})")
    ok &= line("reverses with luminance order", b < 0 < a,
               f"reversed rho = {b:+.4f}")
    ok &= line("control near zero", abs(c) < max(abs(a), 1e-6) * 0.6,
               f"noise rho = {c:+.4f}")
    return ok


def gate_c2(reader, device, size):
    size = max(size, 160)
    print("\nGATE C2 resolution -- decides guidance_stage")
    tgt_lo = build_target({"kind": "rotation", "sign": 1.0}, 64, 64, device=device)
    hi = peripheral_drift(size, size, device=device)
    lo = F.interpolate(hi.unsqueeze(0), size=(64, 64), mode="area")[0]
    r_lo = _ensemble_rho(lo, reader, tgt_lo, delta=2.0)
    tgt_hi = build_target({"kind": "rotation", "sign": 1.0}, size, size, device=device)
    r_hi = _ensemble_rho(hi, reader, tgt_hi, delta=max(size / 256.0 * 6.0, 2.0))
    survives = r_lo > 0.25 * max(r_hi, 1e-6)
    line("illusion survives 64px", survives, f"64px {r_lo:+.4f} vs {size}px {r_hi:+.4f}")
    print("       -> " + ("DeepFloyd stage-I (64px) guidance is viable."
                          if survives else
                          "64px is TOO COARSE. Use generator=sdxl, or "
                          "deepfloyd with stage=sr1. Set this now."))
    return True                     # informative, never blocking


def _optimise(reader, device, size, steps, resample, n_dir=4, lr=0.15):
    tgt = build_target({"kind": "rotation", "sign": 1.0}, size, size, device=device)
    obj = MotionObjective(reader, tgt,
                          Fixation(k=2, delta_px=max(size / 256.0 * 6.0, 2.0),
                                   n_dir=n_dir),
                          lam_motion=1.0, resample=resample)
    gen = get_generator("pixel", device=device, size=size, steps=steps,
                        lr=lr, tv_weight=0.005, log_every=max(steps // 4, 1))
    x, tr = gen.sample("", seed=0, hook=obj)
    return obj, x, tr


def gate_d(reader, device, size, steps, thresh):
    """Gradient health only: can this objective be descended at all?

    Deliberately runs with a FROZEN jitter fan. Whether a solution generalises
    to unseen fan orientations is a separate question (gate D2) and mixing the
    two would make a hard research problem look like a broken gradient.
    """
    print("\nGATE D  optimisability (the Day-0 shortcut, frozen fan)")
    t0 = time.time()
    obj, x, tr = _optimise(reader, device, size, steps, resample=False)
    r = obj.measure(x)["rho"]
    print(f"       rho trace: {[round(v, 3) for v in tr['rho']]}  ({time.time()-t0:.1f}s)")
    return line("rho climbs on a free tensor", r > thresh,
                f"final rho = {r:+.4f} (need > {thresh})")


def gate_d2(reader, device, size, steps):
    """Informative, never blocking. How much does a frozen fan flatter you?

    A frozen fan lets the optimiser tune the image to J specific jitter
    directions; the result scores well on those and can score near zero on any
    other. Resampling the fan each step removes that escape route, and the
    gap between the two is the size of the problem.
    """
    print("\nGATE D2 generalisation across jitter directions")
    probe = Fixation(k=2, delta_px=max(size / 256.0 * 6.0, 2.0), n_dir=16,
                     seed=99991)
    out = {}
    for tag, rs in (("frozen fan", False), ("resampled", True)):
        obj, x, _ = _optimise(reader, device, size, steps, resample=rs)
        out[tag] = (obj.measure(x)["rho"],
                    obj.measure(x, seed=99991, fixation=probe)["rho"])
        print(f"       {tag:11s} rho_loop {out[tag][0]:+.4f}   "
              f"rho_probe {out[tag][1]:+.4f}   gap {out[tag][0]-out[tag][1]:+.4f}")
    fz, rsm = out["frozen fan"], out["resampled"]
    if fz[0] - fz[1] > 0.2:
        print("       -> A frozen fan overfits its own directions here. Keep")
        print("          resample_ensemble=true and report rho_probe, not rho_loop.")
    if rsm[1] < 0.3:
        print("       -> Resampling does not converge in this budget. Expect")
        print("          LOWER honest rho, and raise steps / lower lr. This is")
        print("          the difficulty of the problem, not a broken gradient.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="motion_energy")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--rho-thresh", type=float, default=0.03)
    ap.add_argument("--opt-thresh", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-d2", action="store_true")
    a = ap.parse_args()

    dev = resolve_device(a.device)
    print(f"DriftDream day-1 gate | reader={a.reader} device={dev} size={a.size}")
    reader = get_reader(a.reader, device=dev)

    results = {"A reader health": gate_a(reader, dev, a.size)}
    results["B sign convention"] = gate_b(reader, dev, a.size)
    results["C illusion"] = gate_c(reader, dev, a.size, a.rho_thresh)
    gate_c2(reader, dev, a.size)
    results["D optimisability"] = gate_d(reader, dev, a.size, a.steps, a.opt_thresh)
    if not a.skip_d2:
        gate_d2(reader, dev, a.size, a.steps)

    print("\n" + "=" * 62)
    for k, v in results.items():
        print(f"  {OK if v else BAD}  {k}")
    if all(results.values()):
        print("\n  GO. Gradient path is viable; proceed to day 2.")
        return 0
    print("\n  NO-GO on the gradient path. Switch TODAY to the search fallback:")
    print("      python scripts/run_experiment.py --exp e1_main "
          "--override method=fk")
    print("  The contribution survives: the jitter-ensemble objective and the")
    print("  held-out protocol are unchanged; only the optimiser differs.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:                     # environment, not science
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 62)
        print("  The gate CRASHED before producing a result.")
        print(f"  {type(exc).__name__}: {exc}")
        print("\n  This says nothing about whether the method works. Fix the")
        print("  install, then re-run:")
        print("      python -c 'import driftdream, driftdream.perf'")
        print("      python -m pytest tests/ -q")
        raise SystemExit(2)
