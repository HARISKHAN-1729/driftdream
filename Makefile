.PHONY: install test gate smoke e1 e2 e3 e4 e5 e7 tables figures clean all

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -q

gate:
	python scripts/day1_gate.py --size 192 --steps 60

smoke:            ## 5-minute end-to-end check on CPU
	python scripts/run_experiment.py --exp e1_main --limit 4 \
	  --override device=cpu image_size=96 n_prompts=1 seeds='[0]' \
	             generator_kwargs.steps=25 fixation.n_dir=2 fixation.k=2
	python scripts/make_tables.py --exp e1_main

# GPU profile applied to every experiment. Edit configs/gpu.yaml, not these.
GPU = --override device=cuda image_size=256 \
      fixation.k=2 fixation.n_dir=4 \
      generator=sdxl generator_kwargs.steps=50 generator_kwargs.guide_every=2 \
      loop_reader=dual_channel "eval_readers=['searaft','prednet']"

e1: ; python scripts/run_experiment.py --exp e1_main        $(GPU)
e2: ; python scripts/run_experiment.py --exp e2_jitter      $(GPU)
e3: ; python scripts/run_experiment.py --exp e3_heldout     $(GPU)
e4: ; python scripts/run_experiment.py --exp e4_controls    $(GPU)
e5: ; python scripts/run_experiment.py --exp e5_terms       $(GPU)
e7: ; python scripts/run_experiment.py --exp e7_prior_cost  $(GPU)

cost:             ## print a runtime estimate before committing to a sweep
	python -c "from driftdream.perf import estimate_runtime as e; \
	  print('per image:', e(256, 50, 4, 2, 2, 'cuda'))"

tables:
	for e in e1_main e2_jitter e3_heldout e4_controls e5_terms e7_prior_cost; do \
	  python scripts/make_tables.py --exp $$e || true; done

figures:
	python scripts/make_figures.py --exp e1_main --which gallery methods
	python scripts/make_figures.py --exp e2_jitter --which jitter
	python scripts/make_figures.py --exp e4_controls --which controls

all: gate e1 e2 e3 e4 e5 e7 tables figures

clean:
	rm -rf results __pycache__ .pytest_cache
