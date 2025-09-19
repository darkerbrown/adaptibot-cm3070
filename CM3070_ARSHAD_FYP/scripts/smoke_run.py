"""Small headless smoke-run helper for the Ant rescue controller.

Usage (PowerShell friendly):
  python scripts\smoke_run.py

This creates a short deterministic RunConfig, prepares the environment, loads the policy
artifacts, runs a limited number of steps, and prints summary metrics.
"""
from __future__ import annotations

import sys

import pathlib

# Ensure the repository root is on sys.path so local packages (adaptibot) can be
# imported when this script is invoked directly (e.g. `python scripts\smoke_run.py`).
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from adaptibot.run_config import _assemble_run_config
from adaptibot.controller import _prepare_environment, _init_policy_artifacts, _run_controller


def main() -> int:
    try:
        cfg = _assemble_run_config(
            "ppo_fix_continuous_action.cleanrl_model",
            1,  # n_casualties
            0.6,  # radius
            0.2,  # leg_radius
            100,  # steps
            False,  # gui
            0.0,  # desired_angle_deg
            0.0,  # heading_bias
            1.0,  # drive_scale
            False,  # print_actuators
            "",  # hip_left_override
            "",  # hip_right_override
            False,  # gait_enable
            2.0,  # gait_freq
            0.0,  # gait_hip_amp
            0.0,  # gait_ankle_amp
            0.0,  # gait_turn_gain
            1.0,  # ppo_weight_go
            1.0,  # ppo_weight_turn
            0.0,  # turn_enter
            0.0,  # turn_exit
            0.0,  # turn_amp
            0,  # pause_steps
            False,  # perception
            None,  # terrain_weights
            None,  # difficulty_weights
            "rgb_array",  # render_mode
            False,  # rubble
            "cluster",  # rubble_style
            30,  # rubble_max_parts
            40,  # rubble_draw_max
            20,  # perception_every_k
            224,  # perception_img
            0,  # max_fps
            False,  # show_rgb
            False,  # show_models
            True,  # deterministic
            0,  # seed
            False,  # use_terrain
            False,  # use_difficulty
            "mixed",  # terrain_theme
        )

        print("[smoke] built run config")
        setup = _prepare_environment(cfg, cfg.viewer.render_mode, cfg.terrain.theme, cfg.simulation.rescue_radius, cfg.terrain.draw_overlay)
        print("[smoke] environment prepared")
        # Use the public controller entrypoint which will prepare artifacts
        # and run the control loop; this avoids calling internal helpers
        # with the wrong signature.
        metrics = _run_controller(cfg)
        print("[smoke] run completed; metrics:\n", metrics)
        return 0
    except Exception as e:
        print("[smoke] run failed with exception:", repr(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
