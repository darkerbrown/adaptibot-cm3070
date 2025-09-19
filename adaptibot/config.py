from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger("adaptibot")
LOGGER.addHandler(logging.NullHandler())

CasualtyArray: TypeAlias = NDArray[np.float32]


class TerrainTheme(str, Enum):
    FLAT = "flat"
    MUD = "mud"
    RUBBLE = "rubble"
    MIXED = "mixed"
    SHOWCASE = "showcase"


@dataclass(frozen=True)
class TerrainShowcaseLayout:
    casualties: List[CasualtyArray]
    mud_patches: List[Dict[str, object]]
    extra_rubble: List[Dict[str, object]]
    flat_tiles: List[Dict[str, object]]


@dataclass(frozen=True)
class TerrainThemeSpec:
    enable_rubble: bool
    enable_mud: bool
    floor_rgba: np.ndarray
    floor_friction: Tuple[float, float, float]
    showcase: Optional[TerrainShowcaseLayout] = None


@dataclass(frozen=True)
class TerrainAssets:
    casualties: List[CasualtyArray]
    rocks: List[Dict[str, object]]
    mud_patches: List[Dict[str, object]]
    flat_tiles: List[Dict[str, object]]
    floor_rgba: np.ndarray
    floor_friction: Tuple[float, float, float]
    solid_rubble: bool
    run_seed: Optional[int] = None


@dataclass(frozen=True)
class RescueMetrics:
    rescued: int
    ttr_median: float

    def as_dict(self) -> Dict[str, float]:
        return {"rescued": float(self.rescued), "ttr_median": float(self.ttr_median)}


def _floor_rgba_for_theme(theme: TerrainTheme) -> np.ndarray:
    mapping: Dict[TerrainTheme, Tuple[float, float, float, float]] = {
        TerrainTheme.FLAT: (0.93, 0.93, 0.93, 1.0),
        TerrainTheme.MUD: (0.38, 0.26, 0.18, 1.0),
        TerrainTheme.RUBBLE: (0.72, 0.72, 0.72, 1.0),
        TerrainTheme.MIXED: (0.82, 0.82, 0.82, 1.0),
        TerrainTheme.SHOWCASE: (0.88, 0.88, 0.92, 1.0),
    }
    rgba = mapping.get(theme, mapping[TerrainTheme.MIXED])
    return np.array(rgba, dtype=np.float32)


def _floor_friction_for_theme(theme: TerrainTheme) -> Tuple[float, float, float]:
    mapping: Dict[TerrainTheme, Tuple[float, float, float]] = {
        TerrainTheme.FLAT: (1.0, 0.005, 0.0001),
        TerrainTheme.MUD: (0.35, 0.004, 0.00005),
        TerrainTheme.RUBBLE: (1.2, 0.01, 0.0003),
        TerrainTheme.MIXED: (0.9, 0.006, 0.0001),
        TerrainTheme.SHOWCASE: (0.85, 0.006, 0.00012),
    }
    return mapping.get(theme, mapping[TerrainTheme.MIXED])


def coerce_theme(theme: str) -> TerrainTheme:
    try:
        return TerrainTheme(theme.lower())
    except Exception:
        LOGGER.warning("[terrain] unknown theme %r -> using mixed", theme)
        return TerrainTheme.MIXED


def build_theme_spec(theme: TerrainTheme) -> TerrainThemeSpec:
    theme = TerrainTheme(theme)
    base_spec = TerrainThemeSpec(
        enable_rubble=theme in (TerrainTheme.RUBBLE, TerrainTheme.MIXED, TerrainTheme.SHOWCASE),
        enable_mud=theme in (TerrainTheme.MUD, TerrainTheme.MIXED, TerrainTheme.SHOWCASE),
        floor_rgba=_floor_rgba_for_theme(theme),
        floor_friction=_floor_friction_for_theme(theme),
    )

    if theme != TerrainTheme.SHOWCASE:
        return base_spec

    showcase_layout = TerrainShowcaseLayout(
        casualties=[
            np.array([-3.4, 3.6], dtype=np.float32),
            np.array([3.5, 3.1], dtype=np.float32),
            np.array([0.0, -3.8], dtype=np.float32),
        ],
        mud_patches=[
            {'pos': (3.5, 3.1, 0.02), 'size': (2.0, 1.5, 0.03), 'rgba': [0.34, 0.22, 0.14, 0.95]},
            {'pos': (3.4, 2.5, 0.02), 'size': (1.4, 0.9, 0.025), 'rgba': [0.37, 0.24, 0.16, 0.9]},
        ],
        extra_rubble=[
            {'type': 1, 'pos': np.array([-3.2, 3.6, 0.28], dtype=np.float32), 'size': [0.9, 0.6, 0.5], 'rgba': [0.42, 0.39, 0.36, 1.0]},
            {'type': 1, 'pos': np.array([-3.8, 3.3, 0.24], dtype=np.float32), 'size': [0.7, 0.5, 0.45], 'rgba': [0.5, 0.48, 0.46, 1.0]},
            {'type': 2, 'pos': np.array([-3.5, 3.9, 0.35], dtype=np.float32), 'size': [0.45, 0.45, 0.45], 'rgba': [0.6, 0.58, 0.55, 1.0]},
        ],
        flat_tiles=[
            {'pos': (0.0, 0.0, 0.02), 'size': (2.6, 0.65, 0.02), 'rgba': [0.97, 0.97, 0.99, 1.0]},
            {'pos': (0.0, 0.0, 0.02), 'size': (0.65, 2.6, 0.02), 'rgba': [0.97, 0.97, 0.99, 1.0]},
            {'pos': (0.1, -3.8, 0.02), 'size': (1.8, 1.1, 0.02), 'rgba': [0.98, 0.98, 1.0, 1.0]},
        ],
    )

    return TerrainThemeSpec(
        enable_rubble=base_spec.enable_rubble,
        enable_mud=base_spec.enable_mud,
        floor_rgba=base_spec.floor_rgba,
        floor_friction=base_spec.floor_friction,
        showcase=showcase_layout,
    )
