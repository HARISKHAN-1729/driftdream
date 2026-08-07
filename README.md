# DriftDream

**Training-free generation of static images with a prescribed illusory-motion field.**

Give it a target flow field — *clockwise rotation in this annulus*, *this word
drifts left* — and it steers a frozen generator until the still image is read as
moving that way by a frozen motion model, under simulated microsaccades.

Zero gradient updates to any network weights, ever. The only things optimised
are pixels (or diffusion latents).

---

## The one idea

Static-image illusory motion is not in the pixels. It is a transient artefact of
fixational eye movements — which is why a naive objective ("read this image's
optical flow, match it to the target") is **vacuous**: a static frame has zero
flow and therefore zero gradient.

The contributed component is **jitter-ensemble motion guidance**. The loss is
defined on the residual flow *ensemble-averaged over simulated microsaccades
spanning several directions*:

```
f_res(i) = M(frame[i-1], frame[i]) − (global component)
f_bar    = mean over frames and over trajectory DIRECTIONS
L        = λ_m (1 − ρ(f_bar, F*)) + λ_c (1 − CLIP) + λ_q ‖f_bar‖²_outside
```

Three consequences: the objective becomes non-vacuous, it becomes *signed* and
*spatially localised* (so direction and layout are controllable), and averaging
across directions suppresses solutions tuned to one particular shift.

---

## Quickstart (no downloads, ~3 minutes)

```bash
pip install -r requirements.txt
python -m pytest tests/ -q              # 37 tests, ~50 s
python scripts/day1_gate.py --size 192 --steps 60
```

The gate is the go/no-go. Expected output ends with `GO.` — meaning the reader
is healthy, the sign convention is right, the illusion is detectable and
reverses, and ρ climbs under optimisation.

Then generate one image:

```bash
python scripts/generate.py --method driftdream --target rotation_cw \
    --override device=cpu image_size=128 generator_kwargs.steps=120 \
    --out results/figures/demo.png
```

Everything above runs offline on CPU. The heavy models are optional plug-ins.

---

## What runs without any download

| Component | Offline default | Paper version |
|---|---|---|
| Motion reader (in-loop) | `motion_energy` — phase-based Gabor bank + contrast-dependent latency, weight-free, differentiable | `dual_channel` (Sun & Nishida, Nature MI 2025) |
| Held-out reader | `motion_energy_alt` — **same family, not independent evidence** | `searaft`, `prednet` |
| Generator | `pixel` — direct optimisation, no prior | `sdxl`, `deepfloyd` |

The offline path is a complete, runnable pipeline, not a mock. It exists so that
a broken gated download cannot stop the project, and so the Day-0 shortcut
(below) costs minutes rather than days.

**The `motion_energy_alt` warning is not noise.** Held-out evaluation with a
same-family reader is a smoke test. The paper needs a cross-family reader
(`searaft`, `prednet`), or the anti-reward-hacking claim is unsupported.

---

## Plugging in the real models

```bash
# Reader A (in the loop)
git clone https://github.com/anoymized/multi-order-motion-model.git
export DRIFTDREAM_DUAL_ROOT=$PWD/multi-order-motion-model
export DRIFTDREAM_DUAL_CKPT=/path/to/checkpoint.pth

# Held-out reader B1
git clone https://github.com/princeton-vl/SEA-RAFT.git
export DRIFTDREAM_SEARAFT_ROOT=$PWD/SEA-RAFT
export DRIFTDREAM_SEARAFT_CKPT=/path/to/Tartan-C-T-TSKH-spring540x960-M.pth

# Held-out reader B2
git clone https://github.com/eijwat/prednet_in_pytorch.git
export DRIFTDREAM_PREDNET_ROOT=$PWD/prednet_in_pytorch
export DRIFTDREAM_PREDNET_CKPT=/path/to/rotating_snakes.pth

# Diffusion generators + CLIP content term
pip install "diffusers>=0.27" "transformers>=4.40" accelerate safetensors
```

Verify each one through the gate before trusting it:

```bash
python scripts/day1_gate.py --reader dual_channel --size 256 --steps 60
```

If Gate A reports that gradients do not flow, **switch to the search fallback
that same day** — the contribution survives intact, only the optimiser changes:

```bash
python scripts/run_experiment.py --exp e1_main --override method=fk
```

Each adapter isolates the external API in a single method (`_forward` for
Dual-Channel, `flow` for the others). If upstream signatures differ, that is the
only place to edit.

---

## Running the experiments

**The CPU path is a smoke test only.** The jitter ensemble is `n_dir x (k-1)`
reader calls per optimisation step, so a full sweep on CPU takes months. Every
sweep now prints a runtime estimate before it starts and warns if the numbers
do not add up. Check first:

```bash
make cost                          # per-image estimate under the GPU profile
python scripts/run_experiment.py --exp e1_main --dry-run
```

Then:

```bash
./run_all.sh --smoke     # ~10 min, CPU, proves the plumbing
make e2 e1 e3 e4 e5 e7   # GPU profile (see configs/gpu.yaml)
```

### Two settings the day-1 gate decides for you

**Guidance stage.** Gate C2 reports whether the illusion survives at 64px. It
does not: measured rho is `-0.006` at 64px against `+0.100` at 192px. A
repeating element needs enough pixels to hold four distinguishable luminance
steps, and at 64px it does not have them. DeepFloyd IF stage I generates at
64x64, so guiding there gives the reader nothing to read and the gradient is
noise. Use `generator=sdxl`, or `deepfloyd` with `stage=sr1`. Keep
`image_size >= 192` for the objective; 128px is already half strength.

**Ensemble size.** `n_dir=8, k=4` is 24 clips per loss call. `n_dir=4, k=2` is
4, and was enough for a clean sign-reversing signal in the gate. Run
`e2_jitter` first and let it decide, rather than paying 6x for every other
experiment on an assumption.

Or individually:

```bash
python scripts/run_experiment.py --exp e2_jitter
python scripts/make_tables.py    --exp e2_jitter
python scripts/make_figures.py   --exp e2_jitter --which jitter
```

| Experiment | Tests | What it produces |
|---|---|---|
| `e1_main` | H1, H4, H6 | Main table: driftdream vs unguided / best-of-N / no-jitter |
| `e2_jitter` | **H2** | ρ vs `delta_px` and vs `n_dir` — **the money figure** |
| `e3_heldout` | H3 | ρ under readers that never entered a gradient |
| `e4_controls` | H5 | ρ collapse under percept-abolishing transforms |
| `e5_terms` | — | Objective-term and residual-mode ablations |
| `e6_generator` | — | pixel vs SDXL vs DeepFloyd (needs downloads) |
| `e7_prior_cost` | — | Unconstrained optimisation as the upper bound on ρ |

Sweeps **resume**: a completed cell (one with `meta.json`) is skipped, so an
interrupted run continues rather than restarting.

### Human study

```bash
python scripts/build_2afc_study.py --exp e3_heldout --n 40 --include-controls
# participants open results/study/index.html, paste JSON back
python scripts/score_2afc.py --study results/study --responses 'resp_*.json'
```

Direction discrimination only — no aesthetic judgement. Chance is 50%.

---

## Repository layout

```
driftdream/
├── targets.py          F* fields and support masks
├── fixation.py         jitter trajectories, differentiable warp
├── flow.py             residual extraction  ← read this before trusting a number
├── metrics.py          rho, sign agreement, quiescence, CLIP/DINO
├── objective.py        jitter-ensemble motion guidance (the contribution)
├── stimuli.py          procedural peripheral-drift patterns
├── controls.py         negative and positive controls
├── search.py           best-of-N and FK-steering fallbacks
├── pipeline.py         config → objects (one construction path for all scripts)
├── config.py  io.py  viz.py
├── readers/            base · registry · motion_energy · dual_channel · searaft · prednet
└── generators/         base · pixel · sdxl · deepfloyd
scripts/                day1_gate · generate · run_experiment · evaluate
                        make_tables · make_figures · build_2afc_study · score_2afc
configs/                base · targets/ · methods/ · experiments/
tests/                  37 tests
assets/                 prompts.txt · stencils/ · calibration/ (gitignored)
LICENSE  Makefile  requirements.txt  run_all.sh  .gitignore
```

70 files, ~3,200 lines. `make help`-free by design: see the Makefile targets
(`install test gate smoke e1..e7 tables figures clean all`).

**Generation and evaluation are decoupled.** Every run writes `image.png` plus
`meta.json`; `evaluate.py` walks the tree afterwards. If a reader adapter breaks
on day 4, re-evaluate in minutes instead of re-generating hundreds of images.
Tables and figures rebuild **from disk**, so every number in the paper survives a
crash and "what is the variance across seeds?" is a query, not a re-run.

---

## Five things this codebase learned the hard way

These are all encoded as tests. Each one produced results that looked fine.

**1. The phase sign is inverted.** `torch.conv2d` is cross-correlation, which
negates the measured phase. Getting this wrong flips every result in the paper
while remaining perfectly plausible. → `test_readers.py::test_recovers_shift_direction`

**2. Subtracting the nominal shift is wrong.** Real motion models under-read
speed — the built-in reader by about 2.5× — so `f − d` leaves a residual pointing
along the jitter that swamps the illusory signal. Removing the flow field's own
global component is robust to gain error *and* more faithful: uniform retinal
slip is indistinguishable from eye movement. Measured on a pure shift: nominal
residual **6.08**, global residual **0.16**. → `test_fixation.py::test_pure_shift_residual_collapses`

**3. A direction-spanning ensemble cancels gain error by itself.** With evenly
spaced directions the nominal offsets sum to zero. This is another argument for
the ensemble — and the reason Gate B must use a *single* direction, or it tests
nothing. → `test_fixation.py::test_ensemble_cancels_gain_error`

**4. The obvious negative control is not a negative control.** One global
permutation of luminance levels preserves a coherent (merely different)
asymmetric sequence, so it yields a *different* illusion, not none — |ρ| rose
from 0.10 to **0.39**. The real negative control is `luminance_scramble`:
independent permutations per block, destroying global order while preserving
local histogram and contrast. → `test_illusion.py::test_global_permutation_does_not_abolish`

**5. Peripheral drift is under-resolved below ~128px.** Measured ρ: 96px
**+0.003**, 128px **+0.051**, 192px **+0.100**. A fixed angular repeat count
aliases at small sizes and silently kills the illusion. This is why Gate C2
exists and why `guidance_stage` is a config dimension: DeepFloyd stage I
generates at 64×64, where the structure does not yet exist.

---

## Validation status

Verified on CPU in this repository:

- 37/37 tests pass.
- Day-1 gate passes all four gates.
- The built-in reader detects illusory rotation in a procedural drift pattern
  (ρ = **+0.078**), **reverses** with the luminance sequence (ρ = **−0.080**),
  and stays near zero on a noise control (ρ = **+0.017**).
- The latency stage is **necessary**: with `latency_gain=0` there is no illusion
  at all (ρ ≈ 0.000).
- Day-0 shortcut: ρ climbs from **−0.002 → +0.53** in 40 steps on a free pixel
  tensor; to **+0.90** in 20 steps at 112px.
- End-to-end sweep: driftdream ρ ≈ **+0.96** vs unguided ρ ≈ **±0.01** (H1 in
  miniature), with tables and figures generated from disk.

Not yet verified — needs the gated downloads and a GPU: Dual-Channel gradient
health, SEA-RAFT and PredNet adapters, SDXL/DeepFloyd guidance, and whether the
illusion transfers across reader families. **That last one is the real risk**,
and `e3_heldout` exists to measure it. If ρ is high on the in-loop reader and
zero on held-out readers, do not claim a human illusion — report it as
"generating stimuli that selectively drive biologically-inspired motion models",
which is still novel and still falsifiable.

---

## Licence notes

- `motion_energy`, `stimuli` and all pipeline code here: original, no external
  weights.
- Dual-Channel model: Apache-2.0. SEA-RAFT: BSD-3. SDXL: Stability AI Community
  Licence. **DeepFloyd IF: non-commercial research only** — report the SDXL
  variant as the licence-clean headline.
- Peripheral-drift stimuli are generated procedurally. Use Kitaoka's copyrighted
  images only locally for calibration (`assets/calibration/`, gitignored); ship
  only generated stimuli.
