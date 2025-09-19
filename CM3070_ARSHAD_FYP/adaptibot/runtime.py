"""Small helpers used by the controller.

- These functions take the internal simulator state (positions and
    velocities) and convert them into the numeric list the learned
    controller expects. Think of it as measuring where the robot is and
    packaging those measurements neatly before giving them to the brain
    (the neural network).

Short descriptions:
- `build_ant_obs`: collect the robot's positions and velocities into a
    single array.
- `normalize_obs`: shift and scale the numbers so they match what the
    controller saw during training (this keeps the controller stable).
- `warp_obs_axis_toward_target`: rotate the velocity measurements so
    the controller always sees the casualty as if it were in the same
    forward direction; this helps the controller generalise.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

    # In minimal builds mujoco_py may be missing; swallow errors gracefully.
try:
    import mujoco as mj  # type: ignore
except Exception:  # pragma: no cover
    mj = None  # type: ignore


__all__ = [
    "normalize_obs",
    "get_ant_xy",
    "build_ant_obs",
    "rotate_xy",
    "warp_obs_axis_toward_target",
    "_yaw_quat",
    "_set_base_yaw",
    "_tune_viewer_camera",
]


def normalize_obs(obs: np.ndarray, obs_rms: Mapping[str, np.ndarray]) -> np.ndarray:
    """Standardize observations using stored running-mean statistics."""
        # Match CleanRL normalisation with clip to avoid exploding values.
    norm = (obs - obs_rms["mean"]) / np.sqrt(obs_rms["var"] + 1e-8)
    return np.clip(norm, -10.0, 10.0)


def get_ant_xy(env) -> np.ndarray:
    """Return the planar (x, y) position of the Ant base."""
        # Access MuJoCo qpos: first two entries are the planar translation.
    return env.unwrapped.data.qpos[0:2].copy()


def build_ant_obs(env) -> np.ndarray:
    """Match CleanRL's observation layout by concatenating qpos[2:], qvel."""
    qpos = env.unwrapped.data.qpos.ravel().copy()
    qvel = env.unwrapped.data.qvel.ravel().copy()
        # Rebuild observation vector exactly as the policy expects.
    return np.concatenate([qpos[2:], qvel]).astype(np.float32)


def rotate_xy(vec2: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
        # Basic 2D rotation used by warp/waypoint logic.
    return np.array([c * vec2[0] - s * vec2[1], s * vec2[0] + c * vec2[1]], dtype=np.float32)


def warp_obs_axis_toward_target(
    raw_obs: np.ndarray,
    env,
    target_xy: np.ndarray,
    desired_angle_rad: float = 0.0,
) -> np.ndarray:
    """Rotate the root velocity so the target aligns with a desired heading."""
    qpos = env.unwrapped.data.qpos.ravel().copy()
    qvel = env.unwrapped.data.qvel.ravel().copy()

        # Determine target direction in world frame before remapping velocity.
    ant_xy = qpos[0:2]
    delta = target_xy - ant_xy
    theta = math.atan2(delta[1], delta[0])

    rot = desired_angle_rad - theta
    v_xy_rot = rotate_xy(qvel[0:2], rot)
    qvel_mod = qvel.copy()
    qvel_mod[0:2] = v_xy_rot

    return np.concatenate([qpos[2:], qvel_mod]).astype(np.float32)


def _yaw_quat(angle: float) -> np.ndarray:
    """Construct a quaternion representing a pure yaw rotation."""
        # Convert yaw angle to quaternion for direct state manipulation.
    half = 0.5 * angle
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _set_base_yaw(env, yaw: float) -> None:
    """Best-effort reset of the Ant base orientation to a given yaw."""
    try:
        data = env.unwrapped.data
        qpos = data.qpos.ravel().copy()
        qvel = data.qvel.ravel().copy()
        qpos[3:7] = _yaw_quat(yaw)
        if hasattr(env.unwrapped, "set_state"):
            env.unwrapped.set_state(qpos, qvel)
        else:
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            if mj is not None:
                try:
                    mj.mj_forward(env.unwrapped.model, data)
                except Exception:
                    pass
    except Exception:
        pass


def _tune_viewer_camera(
    env,
    distance: float = 10.0,
    elevation: float = -25.0,
    azimuth: float = 35.0,
) -> None:
    """Move the MuJoCo viewer camera to showcase rubble near the origin."""
    try:
        viewer = None
                # Try to respect new Gymnasium renderer API where possible.
        if hasattr(env.unwrapped, "mujoco_renderer"):
            mr = env.unwrapped.mujoco_renderer
            viewer = getattr(mr, "viewer", None)
            if viewer is None and hasattr(mr, "_get_viewer"):
                try:
                    viewer = mr._get_viewer(render_mode="human")
                except Exception:
                    viewer = None
        if viewer is not None and hasattr(viewer, "cam"):
            cam = viewer.cam
            cam.distance = distance
            cam.elevation = elevation
            cam.azimuth = azimuth
    except Exception:
        pass
