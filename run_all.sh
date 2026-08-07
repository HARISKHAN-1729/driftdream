#!/usr/bin/env bash
# Reproduce every experiment. Safe to re-run: completed cells are skipped.
#
#   ./run_all.sh              full run (needs a GPU for anything but a smoke test)
#   ./run_all.sh --smoke      tiny CPU version, ~10 min, proves the plumbing
set -euo pipefail
cd "$(dirname "$0")"

SMOKE=""
[[ "${1:-}" == "--smoke" ]] && SMOKE="--override device=cpu image_size=96 n_prompts=1 seeds=[0] generator_kwargs.steps=25 fixation.n_dir=2 fixation.k=2 --limit 6"

echo "=== day-1 gate (go/no-go) ==="
if ! python scripts/day1_gate.py --size 192 --steps 60; then
  echo
  echo "!! GATE FAILED. Do not proceed on the gradient path."
  echo "!! Switch to the search fallback TODAY:  --override method=fk"
  exit 1
fi

for exp in e1_main e2_jitter e3_heldout e4_controls e5_terms e7_prior_cost; do
  echo; echo "=== $exp ==="
  python scripts/run_experiment.py --exp "$exp" $SMOKE
done

echo; echo "=== held-out evaluation ==="
python scripts/evaluate.py || true

echo; echo "=== tables ==="
for exp in e1_main e2_jitter e3_heldout e4_controls e5_terms e7_prior_cost; do
  python scripts/make_tables.py --exp "$exp" || true
done

echo; echo "=== figures ==="
python scripts/make_figures.py --exp e1_main    --which gallery methods || true
python scripts/make_figures.py --exp e2_jitter  --which jitter          || true
python scripts/make_figures.py --exp e4_controls --which controls       || true

echo; echo "done. tables in results/tables, figures in results/figures"
