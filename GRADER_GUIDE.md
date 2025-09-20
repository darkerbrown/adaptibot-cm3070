# Grader Guide — Ant Rescue (Adaptibot CM3070)

This guide is examiner-focused. It explains exactly how to run and grade the project, what outputs to expect, and where to look in the code. No prior coding knowledge is assumed.

> Repo audited against https://github.com/darkerbrown/adaptibot-cm3070 on 20 Sep 2025 (SGT).

---

## Repository Layout

- `adaptibot/` — core package
  - `controller.py` — target selection, capture logic, action smoothing
  - `policy.py`, `config.py`, `runtime.py`, `terrain.py`, `app.py`
- `adaptibot_main.py` — launcher (imports `adaptibot.app:main`)
- `scripts/` — helpers
  - `smoke_run.py` — fast sanity check (writes to `runs/`)
  - `aggregate.py` — collate JSON metrics
  - `plot_results.py` — produce PNG charts
- `tests/` — minimal integration tests
  - `test_step_50.py`
- `runs/` — auto-generated configs, metrics, CSVs, plots
- `ppo_fix_continuous_action.cleanrl_model` — pretrained PPO checkpoint (at repo root)
- `requirements.txt`, `requirements-locked.txt` — dependencies
- `README.md`, `LICENSE`, `GRADER_GUIDE.md`

> If you later move the checkpoint into `models/`, update the `--model` path accordingly.

---

## Quick Checks (Recommended Order)

1. **Install dependencies**  
   ```powershell
    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
   ```

2. **Smoke run (very short, verifies setup)**  
   ```powershell
   python scripts\smoke_run.py
   ```
   Expected: completes without error and writes a run under `./runs/`:
   - `./runs/<timestamp>/run_config.json`
   - `./runs/<timestamp>/metrics.json`

3. **Deterministic headless run (single episode)**  
   ```powershell
   python adaptibot_main.py --render_mode rgb_array --seed 123 --steps 500 --model .\ppo_fix_continuous_action.cleanrl_model
   ```
   This produces an additional `./runs/<timestamp>/metrics.json` and optional plots if you then run:
   ```powershell
   python scripts\aggregate.py
   python scripts\plot_results.py
   ```

---

## What to Check While Grading

- **Configuration evidence**: each run folder has `run_config.json` with seed, steps, and flags.
- **Metrics evidence**: `metrics.json` includes casualties rescued, median time‑to‑rescue (TTR), steps, displacement, and seed.
- **Code transparency**: open `adaptibot/controller.py` for a plain-English description at the top and the main control loop.
- **Documentation**: `README.md` plus this guide provide reproducibility instructions.
- **Visual outputs**: `runs/plots/*.png` after `plot_results.py`.

---

## Notes and Assumptions

- No MuJoCo or GPU required. Runs headless with PyBullet.
- Use `--render_mode rgb_array` for headless evaluation.
- Reproducibility: pass a saved `--seed` to repeat behavior on the same machine.
- Typical runtimes on CPU:
  - Smoke run: < 1 minute
  - Single deterministic run (500 steps): ~1–3 minutes

---

## Typical Grading Workflow

1. Run `scripts/smoke_run.py`. Confirm `runs/<ts>/metrics.json` exists and `rescued > 0` or reasonable depending on seed.
2. Run a deterministic headless run as above. Confirm new `metrics.json` and check fields.
3. Aggregate and plot to generate `runs/plots/*.png`.
4. Skim `adaptibot/controller.py` to verify the loop and capture logic.

Done.

