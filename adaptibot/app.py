"""CLI entry point for the Ant rescue demo."""

from __future__ import annotations

import argparse
from typing import List, Optional

from .config import LOGGER
from .controller import build_run_config_from_args, do_ablation, run


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser used by the rescue demo."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_fix_continuous_action.cleanrl_model", help="path to .cleanrl_model")
    ap.add_argument("--n-casualties", type=int, default=3)
    ap.add_argument("--radius", type=float, default=0.6, help="rescue radius in meters (torso)")
    ap.add_argument("--leg-radius", type=float, default=0.45, help="additional rescue radius for leg contacts")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--det", action="store_true", help="use deterministic (mean) actions")
    ap.add_argument("--seed", type=int, default=None, help="seed for env and spawns")
    ap.add_argument("--desired-angle-deg", type=float, default=0.0, help="map target to this heading in agent frame (deg)")
    ap.add_argument("--heading-bias", type=float, default=0.0, help="left/right hip bias from yaw error (0 to disable)")
    ap.add_argument("--drive-scale", type=float, default=1.2, help="scale PPO actions to maintain pace")
    ap.add_argument("--print-actuators", action="store_true", help="print actuator names and mapped hip indices")
    ap.add_argument("--hip-left", type=str, default="", help="comma-separated indices to bias as left hips, e.g. 0,2")
    ap.add_argument("--hip-right", type=str, default="", help="comma-separated indices to bias as right hips, e.g. 1,3")
    ap.add_argument("--gait-enable", type=lambda x: x.lower() in ["1", "true", "yes", "y"], default=False)
    ap.add_argument("--gait-freq", type=float, default=1.4)
    ap.add_argument("--gait-hip-amp", type=float, default=0.6)
    ap.add_argument("--gait-ankle-amp", type=float, default=0.4)
    ap.add_argument("--gait-turn-gain", type=float, default=0.6)
    ap.add_argument("--ppo-weight-go", type=float, default=1.0)
    ap.add_argument("--ppo-weight-turn", type=float, default=1.0)
    ap.add_argument("--turn-enter", type=float, default=0.6, help="enter TURN when |yaw error| > this (rad)")
    ap.add_argument("--turn-exit", type=float, default=0.25, help="exit TURN when |yaw error| < this (rad)")
    ap.add_argument("--turn-amp", type=float, default=0.30, help="hip torque bias during TURN (0-1)")
    ap.add_argument("--pause-steps", type=int, default=20, help="pause frames after a rescue to recalibrate")
    ap.add_argument("--no-terrain-gating", dest="use_terrain", action="store_false",
        help="Disable terrain-aware speed reduction when perception is enabled")
    ap.add_argument("--no-difficulty-gating", dest="use_difficulty", action="store_false",
        help="Disable difficulty-aware speed reduction when perception is enabled")
    ap.add_argument("--perception", action="store_true", help="enable EfficientNet-B0 + ResNet-18 perception")
    ap.add_argument("--terrain-weights", type=str, default="", help="path to EfficientNet-B0 terrain weights (.pth)")
    ap.add_argument("--difficulty-weights", type=str, default="", help="path to ResNet-18 difficulty weights (.pth)")
    ap.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"], help="env render mode")
    ap.add_argument("--rubble", action="store_true", help="draw rubble overlay on the floor")
    ap.add_argument("--rubble-style", type=str, default="building", choices=["building", "random"], help="rubble generator style")
    ap.add_argument("--rubble-max-parts", type=int, default=400, help="max parts generated for rubble scene")
    ap.add_argument("--rubble-draw-max", type=int, default=300, help="max markers drawn per frame for rubble")
    ap.add_argument("--show-models", action="store_true", help="print perception model summaries on startup")
    ap.add_argument("--perception-every-k", type=int, default=5, help="run perception every K frames (>=1)")
    ap.add_argument("--perception-img", type=int, default=160, help="perception input image size (e.g., 160 or 224)")
    ap.add_argument("--max-fps", type=int, default=30, help="cap GUI at this FPS (0 to disable)")
    ap.add_argument("--rgb-view", action="store_true", help="open a separate RGB overlay window for perception")
    ap.add_argument("--waypoint-every-k", type=int, default=6, help="recompute waypoint every K frames")
    ap.add_argument("--waypoint-max-parts", type=int, default=80, help="max rubble parts to consider for waypointing")
    ap.add_argument("--terrain-theme", type=str, default="showcase", choices=["flat", "mud", "rubble", "mixed", "showcase"], help="visual terrain theme for floor and debris")
    ap.add_argument("--ablate", action="store_true", help="run 4-way ablation and print a table")
    ap.add_argument("--episodes", type=int, default=5, help="episodes per configuration")

    ap.set_defaults(
        use_terrain=True,
        use_difficulty=True,
        gui=True,
        render_mode="human",
        rubble=True,
        rubble_style="building",
        rubble_max_parts=700,
        rubble_draw_max=600,
        perception=True,
        perception_every_k=8,
        perception_img=144,
        max_fps=24,
        show_models=False,
        rgb_view=False,
    )
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the command-line demo and ablation runner."""
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    config = build_run_config_from_args(args)

    if args.ablate:
        do_ablation(args, config)
        return

    try:
        run(config)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user", exc_info=False)
