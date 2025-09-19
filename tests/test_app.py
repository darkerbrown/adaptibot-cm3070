import numpy as np
import argparse
from typing import List, Dict

import pytest

from adaptibot.controller import (
    ViewerConfig,
    RunConfig,
    _parse_index_list,
    _resolve_render_mode,
    _assemble_run_config,
    build_run_config_from_args,
    _prepare_environment,
    EnvironmentSetup,
)
from adaptibot.config import TerrainTheme, TerrainThemeSpec


def test_parse_index_list_handles_duplicates_and_whitespace():
    assert _parse_index_list('0, 1,1, 2 , ,3') == (0, 1, 2, 3)
    assert _parse_index_list('') == ()


def test_resolve_render_mode_prefers_human_when_gui_enabled():
    viewer = ViewerConfig(gui=True, render_mode='rgb_array', max_fps=30, show_rgb=False)
    assert _resolve_render_mode(viewer, 'rgb_array', True) == 'human'


def test_resolve_render_mode_forces_rgb_array_when_perception():
    viewer = ViewerConfig(gui=False, render_mode='human', max_fps=30, show_rgb=False)
    assert _resolve_render_mode(viewer, 'human', True) == 'rgb_array'


def test_assemble_run_config_round_trip_defaults():
    cfg = _assemble_run_config(
        model_path='model',
        n_casualties=3,
        radius=0.6,
        leg_radius=0.45,
        steps=100,
        gui=True,
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
        render_mode='human',
        rubble=False,
        rubble_style='building',
        rubble_max_parts=400,
        rubble_draw_max=300,
        perception_every_k=5,
        perception_img=160,
        max_fps=24,
        show_rgb=False,
        show_models=False,
        deterministic=False,
        seed=None,
        use_terrain=True,
        use_difficulty=True,
        terrain_theme='mixed',
        waypoint_every_k=6,
        waypoint_max_parts=80,
    )
    assert isinstance(cfg, RunConfig)
    assert cfg.viewer.gui is True
    assert cfg.simulation.n_casualties == 3
    assert cfg.terrain.theme == TerrainTheme.MIXED
    assert cfg.perception.enabled is False


def _make_args(**overrides):
    defaults = dict(
        model='model',
        n_casualties=3,
        radius=0.6,
        leg_radius=0.45,
        steps=100,
        gui=True,
        desired_angle_deg=0.0,
        heading_bias=0.0,
        drive_scale=1.0,
        print_actuators=False,
        hip_left='',
        hip_right='',
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
        render_mode='human',
        rubble=False,
        rubble_style='building',
        rubble_max_parts=400,
        rubble_draw_max=300,
        perception_every_k=5,
        perception_img=160,
        max_fps=24,
        rgb_view=False,
        show_models=False,
        det=False,
        seed=None,
        use_terrain=True,
        use_difficulty=True,
        terrain_theme='mixed',
        waypoint_every_k=6,
        waypoint_max_parts=80,
        ablate=False,
        episodes=5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_run_config_from_args_uses_namespace_values():
    args = _make_args(gui=False, perception=True, use_terrain=False)
    cfg = build_run_config_from_args(args)
    assert cfg.viewer.gui is False
    assert cfg.perception.enabled is True
    assert cfg.perception.use_terrain is False


class DummyAssets:
    def __init__(self):
        self.casualties: List[np.ndarray] = []
        self.rocks: List[Dict[str, object]] = []
        self.mud_patches: List[Dict[str, object]] = []
        self.flat_tiles: List[Dict[str, object]] = []
        self.floor_rgba = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.floor_friction = (1.0, 0.0, 0.0)
        self.solid_rubble = False


class DummyModel:
    ngeom = 0
    geom_rgba = np.zeros((1, 4), dtype=np.float32)


class DummyEnv:
    def __init__(self, render_mode='human'):
        self.render_mode = render_mode
        self.unwrapped = self
        self.model = DummyModel()

    def close(self):
        pass


@pytest.mark.parametrize('gui,expected_mode', [(True, 'human'), (False, 'rgb_array')])
def test_prepare_environment_returns_setup(monkeypatch, gui, expected_mode):
    monkeypatch.setattr('ant_rescue.controller.generate_terrain_assets', lambda *args, **kwargs: DummyAssets())
    monkeypatch.setattr('ant_rescue.controller.build_theme_spec', lambda theme: TerrainThemeSpec(False, False, np.zeros(4, dtype=np.float32), (1.0, 0.0, 0.0)))
    monkeypatch.setattr('ant_rescue.controller.gym.make', lambda *args, **kwargs: DummyEnv(render_mode=kwargs.get('render_mode', expected_mode)))

    config = _assemble_run_config(
        model_path='model',
        n_casualties=0,
        radius=0.6,
        leg_radius=0.45,
        steps=10,
        gui=gui,
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
        render_mode='human',
        rubble=False,
        rubble_style='building',
        rubble_max_parts=400,
        rubble_draw_max=300,
        perception_every_k=5,
        perception_img=160,
        max_fps=24,
        show_rgb=False,
        show_models=False,
        deterministic=False,
        seed=None,
        use_terrain=True,
        use_difficulty=True,
        terrain_theme='mixed',
        waypoint_every_k=6,
        waypoint_max_parts=80,
    )

    setup = _prepare_environment(
        config,
        expected_mode,
        config.terrain.theme,
        config.simulation.rescue_radius,
        config.terrain.draw_overlay,
    )

    assert isinstance(setup, EnvironmentSetup)
    assert setup.env.render_mode == expected_mode
    assert setup.casualties == []
    assert setup.rubble_overlay == config.terrain.draw_overlay
    assert setup.theme_spec.enable_mud is False
