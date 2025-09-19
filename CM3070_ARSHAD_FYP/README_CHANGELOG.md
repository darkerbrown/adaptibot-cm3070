Changelog and README notes

This project contains the Ant rescue controller demo. Below are recent noteworthy changes and how to reproduce runs and aggregate metrics.

Reproducibility
- Each run now saves a snapshot of the `RunConfig` used to launch the run in the `runs/` directory with a timestamped filename `run_config_<TS>.json`.
- Each run also saves `metrics_<TS>.json` in the `runs/` directory containing a JSON object with keys like `rescued`, `ttr_median`, `steps_done`, `n_casualties`, `total_xy_disp`, `interrupted`, `run_seed`, and `run_config`.
- The `RunConfig` saved includes the `simulation.seed` (if provided) and other deterministic flags so experiments can be reproduced.

Quick smoke-run (PowerShell-friendly)
- A small helper script is included to run a deterministic headless rollout and save metrics into `runs/`:

  python scripts\smoke_run.py

- This helper constructs a short deterministic `RunConfig`, runs a headless rollout (no GUI), and saves the run config and metrics to the `runs/` directory.

Aggregating metrics
- Use `scripts/aggregate_runs.py` to collect all `metrics_*.json` files under `runs/` into a single CSV and produce a simple plot (PNG).

  python scripts\aggregate_runs.py

- The script will write `runs/metrics_aggregated.csv` and `runs/metrics_summary.png` (if matplotlib is available). If plotting dependencies are not present, the CSV will still be written.

Notes
- Some MuJoCo-related runtime objects are dynamic C-extension objects. During a recent type-checking cleanup, the code was updated to avoid spurious type-checker errors by using small, local `Any` casts for `model` and `data` access. This preserves runtime behavior and maintains stronger typing elsewhere in the codebase.
- If you plan to run many experiments, consider mounting a separate storage location for `runs/` and/or adding a small job script that runs `scripts/smoke_run.py` with different seeds and aggregates results afterwards.
