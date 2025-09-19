Grader Guide — Ant Rescue

This short guide explains how to run and grade the Ant Rescue project without needing prior coding knowledge.

Quick checks (recommended order)

1) Install dependencies (PowerShell):


python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt


2) Smoke-run (very short, saves a run config and metrics):


python scripts\smoke_run.py


- Output: `runs/run_config_<TS>.json` and `runs/metrics_<TS>.json`.
- Check the JSON files to see the exact run settings and the number of casualties rescued.

3) Run experiments (several seeds) and aggregate results:


python scripts\run_experiments.py --model ppo_fix_continuous_action.cleanrl_model --episodes 8 --base-seed 0
python scripts\aggregate_runs.py
python scripts\stats_analysis.py


- Outputs: `runs/experiments_<TS>.csv`, `runs/metrics_aggregated.csv`, `runs/metrics_stats.csv`, and PNG summary plots in `runs/`.

What to check while grading

- `runs/` contains timestamped `run_config_*.json` and `metrics_*.json` files. Open a `run_config` to confirm the experiment parameters. Each metric JSON includes `rescued`, `ttr_median`, `steps_done`, `total_xy_disp`, and `run_seed`.

- Inspect `ant_rescue/controller.py` if you want to understand the main loop. The top of that file contains a plain-English explanation of the overall control flow.

- Look `README.md` for more context.

Notes and assumptions

- If you don't have MuJoCo or a GPU available, run the scripts headless; the repo supports `render_mode=rgb_array` and headless runs.

- Reproducibility: every run saves the `RunConfig` and the terrain `run_seed`. To reproduce a run exactly, pass the saved `simulation.seed` back to the CLI.

Contact

- If you want an analysis notebook with the exact figure-generation steps used for a paper, run `scripts/stats_analysis.py` or ask me to add a notebook that loads `runs/metrics_aggregated.csv` and produces publication-ready plots.
