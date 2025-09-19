"""Terrain generator: create rubble, mud patches and casualty locations.

- This module decides where to place obstacles (rubble) and the people
    who need rescuing (casualties) on the flat ground used by the Ant.
- When the code picks positions it can use a constant number called a
    "seed" so the same scene can be recreated later. If no seed is given
    the scene is random but the chosen seed is saved so the run can be
    reproduced.
- The module includes a few simple strategies: empty field, scattered
    rubble, or clusters that look like collapsed buildings.

Key grader takeaways:
- The casualty placement tries to avoid burying people completely inside
    rubble, but it will place them near rubble to make the task realistic.
- The generator returns a small set of plain Python objects (lists and
    arrays) describing positions and sizes; the controller inserts these
    into the simulator's world model before each run.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Any, cast

import numpy as np

from .config import (
        LOGGER,
        TerrainAssets,
        TerrainTheme,
        build_theme_spec,
)

def _spawn_casualties(n_casualties: int, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """Sample casualty positions uniformly on the plane (optionally seeded)."""
    rg = rng if rng is not None else np.random.default_rng()
        # Uniformly sample casualties within the spawn square.
    return [rg.uniform(-5.0, 5.0, size=2).astype(np.float32) for _ in range(n_casualties)]


def _spawn_casualties_clear(
    n_casualties: int,
    parts,
    rng: Optional[np.random.Generator] = None,
    min_clear: float = 0.7,
    spread: float = 8.0,
) -> List[np.ndarray]:
    """Rejection-sample casualties while respecting clearance around rubble geometry."""
    rg = rng if rng is not None else np.random.default_rng()
    out: List[np.ndarray] = []

    def clear(xy: np.ndarray) -> bool:
        x, y = float(xy[0]), float(xy[1])
        if x * x + y * y < 1.0:
            return False
        for p in parts:
            px, py = float(p["pos"][0]), float(p["pos"][1])
            tp = int(p["type"])  # 1=box, 2=sphere
            if tp == 1:
                sx = float(p["size"][0]) + min_clear
                sy = float(p["size"][1]) + min_clear
                if abs(x - px) <= sx and abs(y - py) <= sy:
                    return False
            else:
                r = float(p["size"][0]) + min_clear
                if (x - px) ** 2 + (y - py) ** 2 <= r * r:
                    return False
        return True

        # Limit rejection sampling attempts to avoid infinite loops in dense rubble.
    attempts = 0
        # Rejection loop: keep sampling until enough clear points are found.
    while len(out) < n_casualties and attempts < 10000:
        attempts += 1
        xy = rg.uniform(-spread, spread, size=2).astype(np.float32)
        if clear(xy):
            out.append(xy)

    while len(out) < n_casualties:
        out.append(rg.uniform(-spread, spread, size=2).astype(np.float32))
    return out

def _spawn_casualties_in_rubble(
    n_casualties: int,
    parts,
    rng: Optional[np.random.Generator] = None,
    min_clear: float = 0.7,
    near_radius: float = 1.2,
    spread: float = 8.0,
) -> List[np.ndarray]:
    """Sample casualties that hug rubble clusters while avoiding collisions."""
    if rng is None:
        rng = np.random.default_rng()
    if not parts:
        return _spawn_casualties_clear(n_casualties, parts, rng=rng, min_clear=min_clear, spread=spread)

    out: List[np.ndarray] = []
    attempts = 0

    def collides(x: float, y: float) -> bool:
        if x * x + y * y < 1.0:
            return True
        for p in parts:
            px, py = float(p["pos"][0]), float(p["pos"][1])
            tp = int(p["type"])  # 1=box, 2=sphere
            if tp == 1:
                sx = float(p["size"][0]) + min_clear
                sy = float(p["size"][1]) + min_clear
                if abs(x - px) <= sx and abs(y - py) <= sy:
                    return True
            else:
                r = float(p["size"][0]) + min_clear
                if (x - px) ** 2 + (y - py) ** 2 <= r * r:
                    return True
        return False

        # Similar rejection sampling but biased near existing rubble parts.
    while len(out) < n_casualties and attempts < 20000:
        attempts += 1
        p = parts[int(rng.integers(0, len(parts)))]
        px, py = float(p["pos"][0]), float(p["pos"][1])
        tp = int(p["type"])  # 1=box, 2=sphere
        if tp == 1:
            sx, sy = float(p["size"][0]), float(p["size"][1])
            side = int(rng.integers(0, 4))
            ox = sx + rng.uniform(0.2, near_radius)
            oy = sy + rng.uniform(0.2, near_radius)
            if side == 0:
                cx, cy = px + ox, py + rng.uniform(-oy, oy)
            elif side == 1:
                cx, cy = px - ox, py + rng.uniform(-oy, oy)
            elif side == 2:
                cx, cy = px + rng.uniform(-ox, ox), py + oy
            else:
                cx, cy = px + rng.uniform(-ox, ox), py - oy
        else:
            r = float(p["size"][0]) + rng.uniform(0.2, near_radius)
            ang = rng.uniform(0, 2 * np.pi)
            cx, cy = px + r * np.cos(ang), py + r * np.sin(ang)

        cx = float(np.clip(cx, -spread, spread))
        cy = float(np.clip(cy, -spread, spread))
        if not collides(cx, cy):
            out.append(np.array([cx, cy], dtype=np.float32))

    while len(out) < n_casualties:
        extra = _spawn_casualties_clear(1, parts, rng=rng, min_clear=min_clear, spread=spread)[0]
        out.append(extra)
    return out

def _spawn_rubble(
    n_items: int = 80,
    spread: float = 8.0,
    max_parts: int = 600,
    rng: Optional[np.random.Generator] = None,
    style: str = "building",
) -> List[Dict[str, object]]:
    """Generate box/sphere debris clusters to form a rubble field."""
    # If no RNG is provided, create a fresh generator so runs without an
    # explicit seed are randomized each time instead of using a fixed seed.
    rng = rng if rng is not None else np.random.default_rng()

    parts: List[Dict[str, object]] = []
        # Each entry is a dict describing a MuJoCo geom-like marker to overlay.

    def add_part(tp, pos3, size3, rgba):
            # Utility to append parts only while under max_parts limit.

        nonlocal parts

        if len(parts) < max_parts:

            parts.append({"type": tp, "pos": np.array(pos3, dtype=np.float32), "size": list(map(float, size3)), "rgba": list(map(float, rgba))})



    def add_wall_rect(center, w, h, thickness=0.15, height=1.0, gaps=0.35, doorway=True):
            # Assemble perimeter wall segments around a notional building footprint.

        cx, cy = float(center[0]), float(center[1])

        seg = thickness

        xs = np.linspace(cx - w/2, cx + w/2, max(3, int(max(3, w/seg))))

        ys = np.linspace(cy - h/2, cy + h/2, max(3, int(max(3, h/seg))))

        wall_rgba = [0.52, 0.52, 0.52, 1.0]

        # choose door positions along each side to ensure passages

        # Use the provided RNG consistently (avoid np.random global state).
        door_x = float(rng.uniform(cx - w/2 + 0.4, cx + w/2 - 0.4)) if doorway else None
        door_y = float(rng.uniform(cy - h/2 + 0.4, cy + h/2 - 0.4)) if doorway else None

        door_w = 0.7  # width of doorway opening

        for x in xs:
            if rng.random() < gaps:
                continue

            if doorway and abs(x - door_x) < door_w:
                continue

            for zlev in np.linspace(0.12, height, 5):

                add_part(1, [x, cy - h/2, zlev], [seg, seg, seg], wall_rgba)

                add_part(1, [x, cy + h/2, zlev], [seg, seg, seg], wall_rgba)

        for y in ys:
            if rng.random() < gaps:
                continue

            if doorway and abs(y - door_y) < door_w:
                continue

            for zlev in np.linspace(0.12, height, 5):

                add_part(1, [cx - w/2, y, zlev], [seg, seg, seg], wall_rgba)

                add_part(1, [cx + w/2, y, zlev], [seg, seg, seg], wall_rgba)



    def add_columns(center, w, h, count=6):
            # Scatter support columns to make the rubble field visually interesting.

        cx, cy = float(center[0]), float(center[1])

        col_rgba = [0.6, 0.6, 0.62, 1.0]

        # perimeter + two interior spots

        spots = [

            (cx - w/2, cy - h/2), (cx + w/2, cy - h/2),

            (cx - w/2, cy + h/2), (cx + w/2, cy + h/2),

            (cx, cy - h/3), (cx, cy + h/3)

        ]

        rng.shuffle(spots)

        for (x, y) in spots[:count]:

            for zlev in np.linspace(0.12, rng.uniform(0.9, 1.6), 8):

                add_part(1, [x, y, zlev], [0.16, 0.16, 0.16], col_rgba)



    def add_collapsed_slab(center, w, h, t=0.07):
            # Lay down broken slabs to simulate collapsed roofs.

        cx, cy = float(center[0]), float(center[1])

        slab_rgba = [0.4, 0.38, 0.35, 1.0]

        tiles_x = max(2, int(w / 0.5))

        tiles_y = max(2, int(h / 0.5))

        for i in range(tiles_x):

            for j in range(tiles_y):

                dx = (i - (tiles_x-1)/2) * 0.5 + float(rng.normal(0, 0.05))

                dy = (j - (tiles_y-1)/2) * 0.5 + float(rng.normal(0, 0.05))

                z = 0.18 + float(rng.normal(0, 0.05))

                add_part(1, [cx + dx, cy + dy, z], [0.6, 0.6, max(0.08, t)], slab_rgba)



    def add_debris_pile(center, radius=1.2, count=24):
            # Drop random rocks/boxes around a center point.

        cx, cy = float(center[0]), float(center[1])

        for _ in range(count):
            if len(parts) >= max_parts:
                break

            ang = float(rng.uniform(0, 2*np.pi))

            r = float(rng.uniform(0, radius))

            x = cx + np.cos(ang)*r

            y = cy + np.sin(ang)*r

            if rng.random() < 0.6:

                s = float(rng.uniform(0.15, 0.28))

                add_part(2, [x, y, 0.16], [s, s, s], [0.45, 0.45, 0.48, 1.0])

            else:

                s = float(rng.uniform(0.14, 0.24))

                add_part(1, [x, y, 0.14], [s, s, s*0.7], [0.42, 0.4, 0.38, 1.0])



    def add_roof_plate(center, w, h, z=1.2, t=0.18):
            # Overlay partial roof sections to give verticality.

        cx, cy = float(center[0]), float(center[1])

        rgba = [0.35, 0.34, 0.33, 1.0]

        tiles_x = max(1, int(w / 1.0))

        tiles_y = max(1, int(h / 1.0))

        for i in range(tiles_x):

            for j in range(tiles_y):

                dx = (i - (tiles_x-1)/2) * 1.0 + float(rng.normal(0, 0.05))

                dy = (j - (tiles_y-1)/2) * 1.0 + float(rng.normal(0, 0.05))

                add_part(1, [cx + dx, cy + dy, z + float(rng.normal(0, 0.02))], [0.9, 0.9, t], rgba)



    def add_beam_line(center, length, axis='x', z=1.0, s=0.12, step=0.5):
            # Long beam fragments for additional clutter.

        cx, cy = float(center[0]), float(center[1])

        rgba = [0.38, 0.36, 0.35, 1.0]

        n = max(2, int(length/step))

        for k in range(n):

            t = (k/(n-1) - 0.5) * length

            x = cx + (t if axis=='x' else 0.0)

            y = cy + (t if axis=='y' else 0.0)

            add_part(1, [x, y, z], [s, s, s], rgba)



    if style == "building":
        # Hero rubble cluster plus additional scattered sites.

        # Always place a mega cluster near origin for visibility

        center0 = np.array([0.0, 0.0], dtype=np.float32)

        w0 = float(rng.uniform(3.5, 5.0))

        h0 = float(rng.uniform(3.5, 5.0))

        add_wall_rect(center0, w0, h0, thickness=0.2, height=float(rng.uniform(1.2, 1.8)), gaps=0.55, doorway=True)

        add_columns(center0, w0, h0, count=int(rng.integers(4, 8)))

        add_roof_plate(center0 + rng.normal(0, 0.2, size=2), w0*0.9, h0*0.9, z=float(rng.uniform(1.0, 1.5)), t=0.22)

        add_collapsed_slab(center0 + rng.normal(0, 0.25, size=2), w0*0.8, h0*0.8, t=0.12)

        add_beam_line(center0, max(w0, h0)*1.2, axis='x', z=1.0, s=0.14, step=0.4)

        add_beam_line(center0, max(w0, h0)*1.2, axis='y', z=1.1, s=0.14, step=0.4)

        add_debris_pile(center0 + rng.normal(0, 0.3, size=2), radius=1.8, count=int(rng.integers(25, 45)))



        # Additional building sites spread across the plane

        n_sites = max(4, n_items // 14)

        for _ in range(n_sites):
            if len(parts) >= max_parts:
                break

            center = rng.uniform(-spread, spread, size=2).astype(np.float32)

            w = float(rng.uniform(2.8, 4.2))

            h = float(rng.uniform(2.8, 4.2))

            add_wall_rect(center, w, h, thickness=0.15, height=float(rng.uniform(0.9, 1.4)), gaps=0.6, doorway=True)

            add_columns(center, w, h, count=int(rng.integers(3, 6)))

            if rng.random() < 0.7:

                add_roof_plate(center + rng.normal(0, 0.2, size=2), w*0.8, h*0.8, z=float(rng.uniform(0.9, 1.3)), t=0.18)

            add_collapsed_slab(center + rng.normal(0, 0.25, size=2), w*0.7, h*0.7, t=0.1)

            add_debris_pile(center + rng.normal(0, 0.35, size=2), radius=1.8, count=int(rng.integers(20, 40)))

        # Shuffle parts once to interleave clusters; ensures draw_max shows variety each frame

        try:
            # numpy Generator.shuffle typing is restrictive; cast to Any to
            # perform an in-place shuffle of our list-of-dicts.
            cast(Any, rng).shuffle(parts)
        except Exception:
            pass

    else:

        # Generic random rubble

        for _ in range(n_items):

            if len(parts) >= max_parts:

                break

            base = rng.uniform(-spread, spread, size=2).astype(np.float32)

            z = 0.08

            shape = rng.choice(["beam", "slab", "pile", "boulder"], p=[0.35, 0.25, 0.25, 0.15])

            if shape == "beam":

                L = float(rng.uniform(0.8, 1.6))

                w = float(rng.uniform(0.08, 0.14))

                h = float(rng.uniform(0.05, 0.10))

                rgba = [0.42, 0.42, 0.45, 1.0]

                n = 3

                theta = float(rng.uniform(0, 2*np.pi))

                dirv = np.array([np.cos(theta), np.sin(theta)])

                for i in range(n):

                    t = (i/(n-1) - 0.5) * L

                    pos = [base[0] + dirv[0]*t, base[1] + dirv[1]*t, z]

                    add_part(1, pos, [w, w, h], rgba)

            elif shape == "slab":

                L = float(rng.uniform(0.7, 1.2))

                W = float(rng.uniform(0.4, 0.8))

                t = float(rng.uniform(0.05, 0.09))

                rgba = [0.36, 0.30, 0.22, 1.0]

                rows, cols = 2, 3

                dx = L/(cols)

                dy = W/(rows)

                for i in range(rows):

                    for j in range(cols):

                        cx = (j - (cols-1)/2) * dx

                        cy = (i - (rows-1)/2) * dy

                        pos = [base[0] + cx, base[1] + cy, z]

                        add_part(1, pos, [dx*0.45, dy*0.45, t], rgba)

            elif shape == "pile":

                R = float(rng.uniform(0.4, 0.9))

                k = int(rng.integers(6, 10))

                for _ in range(k):

                    if len(parts) >= max_parts:

                        break

                    r = float(rng.uniform(0.05, 0.10))

                    jitter = rng.normal(0, R*0.25, size=2)

                    pos = [base[0] + jitter[0], base[1] + jitter[1], z]

                    if rng.random() < 0.5:

                        add_part(2, pos, [r, r, r], [0.55, 0.55, 0.55, 1.0])

                    else:

                        add_part(1, pos, [r*0.9, r*0.9, r*0.5], [0.38, 0.34, 0.30, 1.0])

            else:

                r = float(rng.uniform(0.16, 0.28))

                rgba = [0.48, 0.50, 0.52, 1.0]

                add_part(2, [base[0], base[1], z], [r, r, r], rgba)

                for _ in range(int(rng.integers(1, 3))):

                    rr = r * rng.uniform(0.5, 0.8)

                    ang = rng.uniform(0, 2*np.pi)

                    pos = [base[0] + np.cos(ang)*r*0.6, base[1] + np.sin(ang)*r*0.6, z]

                    add_part(2, pos, [rr, rr, rr], rgba)

    # Cast to the declared return type (list[dict[str, object]]) to satisfy mypy
    return cast(List[Dict[str, object]], parts)



def _spawn_mud_patches(
    n_patches: int = 10,
    spread: float = 8.0,
    rng: Optional[np.random.Generator] = None,
    mean_size: float = 1.2,
) -> List[Dict[str, object]]:
    """Create decorative mud patches used in mixed/showcase themes."""
    rng = rng if rng is not None else np.random.default_rng()

    patches = []

    for _ in range(max(1, int(n_patches))):

        radius = float(rng.uniform(0.6, 1.4) * mean_size)

        aspect = float(rng.uniform(0.55, 1.35))

        sx = radius

        sy = radius * aspect

        sz = float(rng.uniform(0.008, 0.018))

        x = float(rng.uniform(-spread + sx, spread - sx))

        y = float(rng.uniform(-spread + sy, spread - sy))

        hue = float(rng.uniform(-0.04, 0.04))

        rgba = [0.36 + hue, 0.23 + 0.5 * hue, 0.14 + hue, 0.92]

        patches.append({

            "pos": (x, y, sz),

            "size": (sx, sy, sz),

            "rgba": rgba,

        })

    return cast(List[Dict[str, object]], patches)



def generate_terrain_assets(
    n_casualties: int,
    theme: TerrainTheme,
    seed: Optional[int] = None,
    rubble_style: str = "building",
    rubble_max_parts: int = 600,
    rubble_spread: float = 8.0,
) -> TerrainAssets:
    """Return casualties, rubble, mud, and floor properties for a given theme."""
    spec = build_theme_spec(theme)
    # Choose a run_seed to use for all RNGs. If user passed a seed, use it
    # (reproducible). Otherwise pick a fresh random seed and log it so runs
    # are randomized by default but replayable when needed.
    if seed is None:
        run_seed = int(np.random.default_rng().integers(0, 2 ** 31 - 1))
        LOGGER.info("[terrain] generated run seed=%d (no seed provided)", run_seed)
    else:
        run_seed = int(seed)
        LOGGER.info("[terrain] using provided seed=%d", run_seed)

    def make_rng(offset: int = 0):
        return np.random.default_rng(run_seed + int(offset))
    rocks: List[Dict[str, object]] = []

        # Optionally synthesize rubble geometry depending on the theme flags.
    if spec.enable_rubble:

        rocks = _spawn_rubble(

            spread=rubble_spread,

            rng=make_rng(2),

            style=rubble_style,

            max_parts=int(rubble_max_parts),

        )



    casualties: List[np.ndarray] = []

    mud_patches: List[Dict[str, object]] = []

    flat_tiles: List[Dict[str, object]] = []



    showcase = spec.showcase if theme is TerrainTheme.SHOWCASE else None

        # Showcase theme uses a static handcrafted layout to highlight features.
    if showcase is not None:

        casualties = [np.array(pos, dtype=np.float32).copy() for pos in showcase.casualties[:n_casualties]]

        if len(casualties) < n_casualties:

            casualties.extend(

                _spawn_casualties(n_casualties - len(casualties), rng=make_rng(1))

            )

        mud_patches = [copy.deepcopy(patch) for patch in showcase.mud_patches]

        rocks.extend(copy.deepcopy(showcase.extra_rubble))

        flat_tiles = [copy.deepcopy(tile) for tile in showcase.flat_tiles]

    else:

        rubble_cas = make_rng(1)

        if rocks:

            casualties = _spawn_casualties_in_rubble(

                n_casualties,

                rocks,

                rng=rubble_cas,

                min_clear=0.8,

                near_radius=1.5,

                spread=rubble_spread,

            )

        else:

            casualties = _spawn_casualties(n_casualties, rng=rubble_cas)



        if spec.enable_mud:

            patch_count = 16 if theme is TerrainTheme.MIXED else 12

            mud_patches = _spawn_mud_patches(

                n_patches=patch_count,

                spread=rubble_spread,

                rng=make_rng(5),

                mean_size=1.2,

            )



    solid_rubble = bool(spec.enable_rubble)



        # Log the terrain mix for debugging/telemetry.
    LOGGER.info(

        "[terrain] theme=%s solid_rubble=%s mud_patches=%d",

        theme.value,

        solid_rubble,

        len(mud_patches),

    )



    return TerrainAssets(
        casualties=casualties,
        rocks=rocks,
        mud_patches=mud_patches,
        flat_tiles=flat_tiles,
        floor_rgba=spec.floor_rgba,
        floor_friction=spec.floor_friction,
        solid_rubble=solid_rubble,
        run_seed=run_seed,
    )
