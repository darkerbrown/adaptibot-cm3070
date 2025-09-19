"""Simple experiment runner for adaptibot.

Runs multiple seeds headless (no GUI), aggregates per-episode metrics saved by
`adaptibot.controller` into a single CSV under `runs/` for later analysis.

Usage example:
    python -m scripts.run_experiments --model ppo_fix_continuous_action.cleanrl_model --episodes 8 --base-seed 100 --steps 1500

The runner constructs a RunConfig for each episode, sets viewer off, disables
heavy perception by default, and calls the library run() entrypoint. After each
run it looks up the newly created metrics JSON in `runs/` and appends a row to
a CSV file.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from datetime import datetime
from dataclasses import replace

# Optional TensorBoard SummaryWriter (best-effort)
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        from tensorboardX import SummaryWriter  # type: ignore
    except Exception:
        SummaryWriter = None  # type: ignore

# Import library constructors
from adaptibot.controller import _assemble_run_config, _run_controller


def list_metrics_files(runs_dir: str):
    pattern = os.path.join(runs_dir, "metrics_*.json")
    return sorted(glob.glob(pattern))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to policy checkpoint (cleanrl style)")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--n-casualties", type=int, default=3)
    p.add_argument("--rubble", action="store_true", help="Enable rubble overlay in terrain (visual only)")
    p.add_argument("--runs-dir", default=os.path.abspath(os.path.join(os.getcwd(), "runs")))
    p.add_argument("--tensorboard", action="store_true", help="Write simple scalars to TensorBoard logs")
    p.add_argument("--tb-dir", default=None, help="Optional TensorBoard root directory (defaults to runs/tb-<ts>)")
    args = p.parse_args()

    os.makedirs(args.runs_dir, exist_ok=True)
    csv_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_path = os.path.join(args.runs_dir, f"experiments_{csv_ts}.csv")

    # baseline list of pre-existing metrics to detect new files

    # Build a minimal RunConfig using the same defaults used by the library.
    base_cfg = _assemble_run_config(
        model_path=args.model,
        n_casualties=args.n_casualties,
        radius=0.6,
        leg_radius=0.45,
        steps=int(args.steps),
        gui=False,
        desired_angle_deg=0.0,
        heading_bias=0.0,
        drive_scale=1.0,
        print_actuators=False,
        hip_left_override='',
        hip_right_override='',
        gait_enable=False,
        gait_freq=1.4,
        gait_hip_amp=0.6,
        gait_ankle_amp=0.4,
        gait_turn_gain=0.6,
        ppo_weight_go=1.0,
        ppo_weight_turn=1.0,
        turn_enter=0.6,
        turn_exit=0.25,
        turn_amp=0.3,
        pause_steps=20,
        perception=False,
        terrain_weights=None,
        difficulty_weights=None,
        render_mode='rgb_array',
        rubble=args.rubble,
        rubble_style='building',
        rubble_max_parts=400,
        rubble_draw_max=300,
        perception_every_k=10,
        perception_img=160,
        max_fps=0,
        show_rgb=False,
        show_models=False,
        deterministic=True,
        seed=None,
        use_terrain=False,
        use_difficulty=False,
        terrain_theme='mixed',
        waypoint_every_k=6,
        waypoint_max_parts=80,
    )

    rows = []

    tb_writer = None
    tb_root = None
    if args.tensorboard and SummaryWriter is not None:
        tb_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if args.tb_dir:
            tb_root = os.path.abspath(args.tb_dir)
        else:
            tb_root = os.path.join(args.runs_dir, f"tb-{tb_ts}")
        try:
            os.makedirs(tb_root, exist_ok=True)
            tb_writer = SummaryWriter(tb_root)
            print(f"[exp] tensorboard logs -> {tb_root}")
        except Exception as e:
            print(f"[exp] could not create tensorboard writer: {e}")
            tb_writer = None
    elif args.tensorboard:
        print("[exp] tensorboard requested but SummaryWriter not available; skipping TB logs")
    for i in range(args.episodes):
        seed = int(args.base_seed) + i
        print(f"[exp] running episode {i+1}/{args.episodes} seed={seed}")
        sim_cfg = replace(base_cfg.simulation, seed=int(seed))
        viewer_cfg = replace(base_cfg.viewer, gui=False, render_mode='rgb_array', max_fps=0, show_rgb=False)
        perception_cfg = replace(base_cfg.perception, enabled=False)
        run_cfg = replace(base_cfg, simulation=sim_cfg, viewer=viewer_cfg, perception=perception_cfg)

        # Snapshot list of metrics before
        before_set = set(list_metrics_files(args.runs_dir))

        # Run (this executes in-process)
        try:
            _run_controller(run_cfg)
        except Exception as e:
            print(f"[exp] run error: {e}")

        # Allow filesystem to flush
        time.sleep(0.2)

        after = list_metrics_files(args.runs_dir)
        new = [p for p in after if p not in before_set]
        if not new:
            print("[exp] warning: no metrics file created for this run; skipping row")
            continue
        # pick latest created
        new_path = sorted(new, key=os.path.getmtime)[-1]
        try:
            with open(new_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception:
            print(f"[exp] could not read metrics {new_path}")
            continue
        row = {
            'seed': seed,
            'metrics_file': os.path.basename(new_path),
            'rescued': data.get('rescued'),
            'ttr_median': data.get('ttr_median'),
            'steps_done': data.get('steps_done'),
            'total_xy_disp': data.get('total_xy_disp'),
            'interrupted': data.get('interrupted'),
            'run_seed': data.get('run_seed'),
            'run_config': data.get('run_config'),
        }
        rows.append(row)
        # write scalars to TensorBoard if available
        try:
            if tb_writer is not None:
                step = int(seed)
                if data.get('rescued') is not None:
                    tb_writer.add_scalar('rescued', float(data.get('rescued')), global_step=step)
                if data.get('ttr_median') is not None:
                    tb_writer.add_scalar('ttr_median', float(data.get('ttr_median')), global_step=step)
                if data.get('steps_done') is not None:
                    tb_writer.add_scalar('steps_done', float(data.get('steps_done')), global_step=step)
                if data.get('total_xy_disp') is not None:
                    tb_writer.add_scalar('total_xy_disp', float(data.get('total_xy_disp')), global_step=step)
                if data.get('run_seed') is not None:
                    try:
                        tb_writer.add_scalar('run_seed', float(data.get('run_seed')), global_step=step)
                    except Exception:
                        pass
        except Exception:
            pass

    # Write aggregated CSV
    if rows:
        fieldnames = ['seed', 'metrics_file', 'rescued', 'ttr_median', 'steps_done', 'total_xy_disp', 'interrupted', 'run_seed', 'run_config']
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"[exp] wrote aggregate CSV to {csv_path}")
        except Exception as e:
            print(f"[exp] could not write CSV: {e}")
    else:
        print("[exp] no results to write")

    if tb_writer is not None:
        try:
            tb_writer.flush()
            tb_writer.close()
            print(f"[exp] closed tensorboard writer at {tb_root}")
        except Exception:
            pass


if __name__ == '__main__':
    main()
