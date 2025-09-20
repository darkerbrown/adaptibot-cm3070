# Adaptibot: CM3070 Final Year Project

## Abstract
Adaptibot is a search-and-rescue (SAR) simulator that augments a pre-trained locomotion policy (PPO on MuJoCo Ant) with a light goal-seeking controller to locate and “rescue” casualties. The system targets a clean, reproducible demo: deterministic seeds, headless runs, and a minimal dependency surface. The novelty is not new gait learning but the integration of a robust, pre-trained gait with a task-level controller that achieves SAR behavior reliably in an examiner-friendly setup.

## Contributions
- **Task integration:** Robust walking from PPO + minimal controller achieving SAR way-finding.  
- **Deterministic demo path:** Seeds, fixed camera, repeatable headless runs.  
- **Inspector ergonomics:** Simple CLI, smoke test, metrics aggregation, and plots.  
- **Optional perception path:** CNN stubs gated behind flags to keep the core demo lean.  

## Repository Structure
```
adaptibot/
  app.py            # CLI entry and run orchestrator
  controller.py     # Target selection, capture logic, action smoothing
  config.py         # Dataclasses/enums for run configuration
  policy.py         # PPO loader + policy wrapper
  runtime.py        # Device/seed helpers, optional bindings
  terrain.py        # Flat terrain and casualty placement
adaptibot_main.py   # Launcher (imports app.main)
scripts/
  smoke_run.py      # 5–10s sanity check
  aggregate.py      # Collate JSON metrics
  plot_results.py   # Produce simple PNG charts
ppo_fix_continuous_action.cleanrl_model  # Pre-trained PPO checkpoint
tests/
  test_step_50.py   # Minimal integration test (rgb_array, 50 steps)
```

## Quick Start

### 1) Environment
```bash
# Python 3.10. Windows 11 tested.

# Create and activate a virtual environment (example: venv on Windows)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install runtime dependencies
pip install -r requirements.txt

# CPU-only works out of the box. GPU optional if enabling perception.
```

### 2) Smoke Test (headless, fast)
```bash
python scripts/smoke_run.py

# Expected: completes without error, writes a small run under ./runs/
```

### 3) Interactive Demo (GUI)
```bash
python adaptibot_main.py --gui --model ./ppo_fix_continuous_action.cleanrl_model

# Use ESC to quit the viewer.
# Casualties are red spheres. The ant rescues them sequentially.
```

## How It Works
- **Control stack:** PPO policy generates base locomotion actions. A small controller biases actions toward the nearest casualty while applying smoothing and clipping to reduce flips.  
- **Rescue model:** If the ant enters a capture radius, that casualty is removed and the next nearest target is selected.  
- **Rendering:** `human` for GUI demos; `rgb_array` for tests and headless evaluation.  
- **Terrain:** Flat floor with neutralized texture. Obstacles disabled for stability in grading runs.  

## Reproducibility
```bash
# Deterministic headless evaluation
python adaptibot_main.py --render_mode rgb_array --seed 123   --model ./ppo_fix_continuous_action.cleanrl_model --steps 500

# Aggregate metrics from ./runs/
python scripts/aggregate.py

# Plot simple figures to ./runs/plots/
python scripts/plot_results.py
```

## Configuration
Key flags (see `RunConfig` in `adaptibot/run_config.py`):
- `--render_mode` = `human` | `rgb_array`  
- `--seed` integer seed  
- `--steps` rollout length  
- `--model` path to PPO checkpoint  
- `--gui` boolean convenience flag for human mode  

## Expected Outputs
- **Metrics JSON** under `./runs/<timestamp>/metrics.json` with keys: `steps`, `episodes`, `rescued`, `flips`, timing info.  
- **Plots** under `./runs/plots/` (PNG) when `plot_results.py` is run.  

## Evaluation Guidance (for graders)
- Primary success signal: number of casualties rescued before termination.  
- Sanity checks: no crashes, consistent target reselection, ant remains upright longer with smoothing.  
- Deterministic seeds should yield near-identical metrics on the same machine.  

## Design Choices
- **Keep the demo simple:** Prioritize a reliable, short demo over fragile complexity.  
- **Controller, not retraining:** Use a pre-trained gait to focus on SAR control logic.  
- **Optional perception:** Vision is disabled by default to avoid runtime drag on modest hardware.  

## Limitations
- No obstacles in the grading path; rubble and mud are excluded for stability.  
- Physics nondeterminism can produce minor variance across machines.  
- The PPO checkpoint is treated as a fixed gait provider; no fine-tuning here.  

## Ethical and Safety Notes
This is a simulated SAR scenario for research/education. It does not validate performance in real rescue contexts.  

## Troubleshooting
```bash
# 1) Stuck at import or viewer
pip show mujoco gymnasium >NUL 2>&1 || pip install mujoco gymnasium

# 2) Model not found
# Ensure the checkpoint path is correct:
python adaptibot_main.py --gui --model ./ppo_fix_continuous_action.cleanrl_model

# 3) Headless CI failing due to render
python adaptibot_main.py --render_mode rgb_array --steps 50
```

## Tech Stack
- **RL:** PPO (CleanRL-style checkpoint), MLP policy 2×256.  
- **Sim:** Gymnasium Ant-v4 (MuJoCo).  
- **Lang/Tooling:** Python 3.10, pytest, minimal CI hooks.  

## Citations
- Raffin et al., Stable-Baselines3.  
- OpenAI Gym / Gymnasium, MuJoCo.  
- **Pretrained PPO gait**: sdpkjc, *Ant-v4-ppo_fix_continuous_action-seed3*, Hugging Face Hub.  
  Source: https://huggingface.co/sdpkjc/Ant-v4-ppo_fix_continuous_action-seed3 
- **ResNet-18** (baseline classifier): torchvision.models.resnet18, PyTorch Vision.  
  Source: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
- **EfficientNet-B0** (baseline classifier): google, *efficientnet-b0*, Hugging Face Hub.  
  Source: https://huggingface.co/google/efficientnet-b0 

## License
MIT
