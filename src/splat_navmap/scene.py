"""Seeded procedural flats with exact geometry.

A scene is a rectangular flat split into rooms by binary space partitioning. Every split adds
one interior wall with one door, so the room adjacency graph is a tree and all rooms are
connected. Furniture is chosen to exercise the height band of the map extraction:

==========  =====================================  ===================================
piece       geometry                               why it is there
==========  =====================================  ===================================
table       top at 0.72 m on four 5 cm legs        top is above the robot, legs are not
crate       box 10-16 cm high                      lower than a mid-height slice
console     wall cabinet from 0.25 m to 0.65 m     floats above a low slice
sofa/shelf  solid boxes from the floor             plain obstacles
plant       pot, thin trunk, foliage above 0.6 m   overhang above the robot
stool       cylinder                               curved footprint
rug         1 cm slab on the floor                 below the robot's ground clearance
==========  =====================================  ===================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from splat_navmap.geometry import Box, Cylinder, Solid, rasterize
from splat_navmap.grid import GridSpec, inflate, reachable_components

WALL_HEIGHT = 2.5
DOOR_HEIGHT = 2.05
OUTER_WALL = 0.16  # thickness, metres
INNER_WALL = 0.10
MIN_ROOM_SIDE = 2.4
DOOR_WIDTH = (0.85, 1.0)
DOOR_END_MARGIN = 0.35  # distance kept between a door and the ends of its wall
DOOR_KEEPOUT = 1.0  # depth in front of a door kept free of furniture
FURNITURE_GAP = 0.70  # minimum gap between two pieces, and between a free-standing piece and a wall
AREA_PER_PIECE = 4.0  # one piece of furniture per this many square metres of room
MIN_CONNECTED_SHARE = 0.97  # share of the free space that must be mutually reachable
SHOWCASE_SEED = 21

WALL_COLOR = (0.93, 0.91, 0.86)
FLOOR_COLOR = (0.66, 0.50, 0.34)
_PALETTE = {
    "wood": (0.60, 0.43, 0.28), "walnut": (0.34, 0.25, 0.19), "teal": (0.16, 0.42, 0.46),
    "mustard": (0.80, 0.62, 0.20), "slate": (0.27, 0.30, 0.36), "cardboard": (0.74, 0.58, 0.38),
    "terracotta": (0.72, 0.38, 0.25), "leaf": (0.22, 0.46, 0.24), "trunk": (0.36, 0.27, 0.18),
    "white": (0.90, 0.90, 0.88), "rust": (0.66, 0.30, 0.22), "steel": (0.55, 0.57, 0.60),
}
Rect = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


@dataclass(frozen=True)
class Robot:
    """Cylindrical robot: its body spans heights ``(ground_clearance, height)``."""

    radius: float = 0.20
    height: float = 0.40
    ground_clearance: float = 0.02


@dataclass(frozen=True)
class Scene:
    """Analytic scene: building footprint, floor slab, and every other solid."""

    bounds: Rect
    floor: Box
    solids: tuple[Solid, ...]
    rooms: tuple[Rect, ...]

    def grid(self, resolution: float = 0.05) -> GridSpec:
        """Grid frame covering the building."""
        return GridSpec.covering(self.bounds, resolution)

    def truth(self, spec: GridSpec, robot: Robot) -> np.ndarray:
        """Exact occupancy of the height range swept by the robot's body."""
        return rasterize(self.solids, spec, robot.ground_clearance, robot.height)


@dataclass(frozen=True)
class _Door:
    vertical: bool  # True when the wall runs along y (at x = coord)
    coord: float
    lo: float
    hi: float

    def keepout(self) -> Rect:
        side = 0.15
        if self.vertical:
            return (self.coord - DOOR_KEEPOUT, self.lo - side, self.coord + DOOR_KEEPOUT,
                    self.hi + side)
        return (self.lo - side, self.coord - DOOR_KEEPOUT, self.hi + side,
                self.coord + DOOR_KEEPOUT)


def generate_scene(seed: int, robot: Robot | None = None, max_attempts: int = 20) -> Scene:
    """Return the first layout drawn from ``seed`` whose free space is (almost) all connected."""
    robot = robot or Robot()
    for attempt in range(max_attempts):
        scene = _draw_scene(np.random.default_rng([seed, attempt]))
        spec = scene.grid()
        blocked = inflate(scene.truth(spec, robot), robot.radius, spec.resolution)
        labels = reachable_components(blocked)
        sizes = np.bincount(labels.ravel())[1:]
        if sizes.size and sizes.max() >= MIN_CONNECTED_SHARE * sizes.sum():
            return scene
    raise RuntimeError(f"no connected layout found for seed {seed}")


def showcase_scene() -> Scene:
    """The scene used for the timings."""
    return generate_scene(SHOWCASE_SEED)


def _draw_scene(rng: np.random.Generator) -> Scene:
    width, depth = rng.uniform(9.0, 12.0), rng.uniform(7.0, 9.0)
    rooms, doors, solids = _partition(rng, width, depth)
    t = OUTER_WALL
    for cx, cy, hx, hy in ((width / 2, -t / 2, width / 2 + t, t / 2),
                           (width / 2, depth + t / 2, width / 2 + t, t / 2),
                           (-t / 2, depth / 2, t / 2, depth / 2),
                           (width + t / 2, depth / 2, t / 2, depth / 2)):
        solids.append(Box(cx, cy, hx, hy, 0.0, WALL_HEIGHT, color=WALL_COLOR, label="wall"))
    keepouts = [door.keepout() for door in doors]
    for room in rooms:
        solids.extend(_furnish(rng, room, keepouts, width, depth))
    bounds = (-t, -t, width + t, depth + t)
    floor = Box(width / 2, depth / 2, width / 2 + t, depth / 2 + t, -0.05, 0.0,
                color=FLOOR_COLOR, label="floor")
    return Scene(bounds, floor, tuple(solids), tuple(rooms))


def _partition(rng: np.random.Generator, width: float, depth: float,
               ) -> tuple[list[Rect], list[_Door], list[Solid]]:
    """Split the flat into rooms; return rooms, doors and the interior wall boxes."""
    rooms: list[Rect] = [(0.0, 0.0, width, depth)]
    doors: list[_Door] = []
    walls: list[Solid] = []
    target = int(rng.integers(4, 7))
    for _ in range(4 * target):
        if len(rooms) >= target:
            break
        index = max(range(len(rooms)), key=lambda i: _area(rooms[i]))
        x0, y0, x1, y1 = rooms[index]
        vertical = (x1 - x0) >= (y1 - y0)  # cut across the longer side
        lo, hi = (x0, x1) if vertical else (y0, y1)
        span_lo, span_hi = (y0, y1) if vertical else (x0, x1)
        if hi - lo < 2 * MIN_ROOM_SIDE + INNER_WALL:
            break
        coord = lo + rng.uniform(0.4, 0.6) * (hi - lo)
        # A new wall must not end inside a door of the two walls it runs into.
        clash = any(d.vertical != vertical and d.coord in (span_lo, span_hi)
                    and d.lo - 0.3 < coord < d.hi + 0.3 for d in doors)
        if clash:
            continue
        door_width = rng.uniform(*DOOR_WIDTH)
        door_lo = rng.uniform(span_lo + DOOR_END_MARGIN, span_hi - DOOR_END_MARGIN - door_width)
        door = _Door(vertical, coord, door_lo, door_lo + door_width)
        doors.append(door)
        pieces = ((span_lo, door.lo, 0.0, WALL_HEIGHT), (door.hi, span_hi, 0.0, WALL_HEIGHT),
                  (door.lo, door.hi, DOOR_HEIGHT, WALL_HEIGHT))
        for a, b, z0, z1 in pieces:
            mid, half = (a + b) / 2, (b - a) / 2
            walls.append(Box(coord, mid, INNER_WALL / 2, half, z0, z1, color=WALL_COLOR,
                             label="wall") if vertical else
                         Box(mid, coord, half, INNER_WALL / 2, z0, z1, color=WALL_COLOR,
                             label="wall"))
        rooms[index : index + 1] = ([(x0, y0, coord, y1), (coord, y0, x1, y1)] if vertical else
                                    [(x0, y0, x1, coord), (x0, coord, x1, y1)])
    return rooms, doors, walls


def _area(rect: Rect) -> float:
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def _overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _grow(rect: Rect, margin: float) -> Rect:
    return (rect[0] - margin, rect[1] - margin, rect[2] + margin, rect[3] + margin)


def _furnish(rng: np.random.Generator, room: Rect, keepouts: list[Rect], width: float,
             depth: float) -> list[Solid]:
    """Place furniture in one room by rejection sampling on axis-aligned bounding boxes."""
    # Interior rectangle: interior walls take half their thickness on each side they border.
    inner = tuple(v if v in (0.0, limit) else v + sign * INNER_WALL / 2
                  for v, limit, sign in zip(room, (width, depth, width, depth), (1, 1, -1, -1),
                                            strict=True))
    placed: list[Rect] = []
    solids: list[Solid] = []
    kinds = ["sofa", "shelf", "console", "table", "crate", "plant", "stool", "crate", "table"]
    count = int(np.clip(round(_area(room) / AREA_PER_PIECE), 2, 7))
    for kind in rng.choice(kinds, size=count):
        for _ in range(30):
            parts, size_x, size_y, against_wall = _piece(rng, str(kind))
            if against_wall:
                side = int(rng.integers(4))  # 0: south wall, 1: east, 2: north, 3: west
                yaw = side * math.pi / 2
                along = (inner[2] - inner[0]) if side % 2 == 0 else (inner[3] - inner[1])
                if along < size_x + 0.2:
                    continue
                offset = rng.uniform(size_x / 2 + 0.05, along - size_x / 2 - 0.05)
                cx, cy = ((inner[0] + offset, inner[1] + size_y / 2),
                          (inner[2] - size_y / 2, inner[1] + offset),
                          (inner[2] - offset, inner[3] - size_y / 2),
                          (inner[0] + size_y / 2, inner[3] - offset))[side]
            else:
                yaw = rng.uniform(0.0, math.pi)
                reach = math.hypot(size_x, size_y) / 2 + FURNITURE_GAP
                if inner[2] - inner[0] < 2 * reach or inner[3] - inner[1] < 2 * reach:
                    continue
                cx = rng.uniform(inner[0] + reach, inner[2] - reach)
                cy = rng.uniform(inner[1] + reach, inner[3] - reach)
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            half_x, half_y = (size_x * c + size_y * s) / 2, (size_x * s + size_y * c) / 2
            box = (cx - half_x, cy - half_y, cx + half_x, cy + half_y)
            guarded = _grow(box, FURNITURE_GAP)
            if any(_overlap(guarded, other) for other in placed) or any(
                    _overlap(box, k) for k in keepouts):
                continue
            placed.append(box)
            solids.extend(_to_world(part, cx, cy, yaw) for part in parts)
            break
    rug = _grow(inner, -rng.uniform(0.5, 0.9))
    if rug[2] - rug[0] > 1.0 and rug[3] - rug[1] > 1.0:
        color = _PALETTE[str(rng.choice(["rust", "slate", "teal", "mustard"]))]
        solids.append(Box((rug[0] + rug[2]) / 2, (rug[1] + rug[3]) / 2, (rug[2] - rug[0]) / 2,
                          (rug[3] - rug[1]) / 2, 0.0, 0.01, color=color, label="rug"))
    return solids


def _to_world(part: Solid, cx: float, cy: float, yaw: float) -> Solid:
    """Move a part defined in the piece frame (origin at the footprint centre) to the world."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = cx + c * part.cx - s * part.cy, cy + s * part.cx + c * part.cy
    if isinstance(part, Cylinder):
        return Cylinder(x, y, part.radius, part.z0, part.z1, part.color, part.label)
    return Box(x, y, part.hx, part.hy, part.z0, part.z1, yaw + part.yaw, part.color, part.label)


def _piece(rng: np.random.Generator, kind: str) -> tuple[list[Solid], float, float, bool]:
    """Build one piece in its own frame: parts, footprint size (x, y), and whether it hugs a wall.

    For wall pieces the local -y side is the back, placed against the wall.
    """
    u = rng.uniform
    if kind == "table":
        lx, ly, top, leg = u(1.2, 1.8), u(0.7, 0.9), 0.72, 0.025
        color = _PALETTE[str(rng.choice(["wood", "walnut", "white"]))]
        parts: list[Solid] = [Box(0, 0, lx / 2, ly / 2, top, top + 0.04, color=color, label=kind)]
        parts += [Box(sx * (lx / 2 - 0.08), sy * (ly / 2 - 0.08), leg, leg, 0.0, top,
                      color=_PALETTE["steel"], label=kind) for sx in (-1, 1) for sy in (-1, 1)]
        return parts, lx, ly, False
    if kind == "crate":
        side, height = u(0.35, 0.6), u(0.10, 0.16)
        return ([Box(0, 0, side / 2, side / 2, 0.0, height, color=_PALETTE["cardboard"],
                     label=kind)], side, side, False)
    if kind == "stool":
        radius = u(0.16, 0.20)
        return ([Cylinder(0, 0, radius, 0.0, 0.45, _PALETTE["mustard"], kind)],
                2 * radius, 2 * radius, False)
    if kind == "plant":
        pot, crown = u(0.14, 0.22), u(0.28, 0.36)
        pot_top = u(0.35, 0.5)
        return ([Cylinder(0, 0, pot, 0.0, pot_top, _PALETTE["terracotta"], kind),
                 Cylinder(0, 0, 0.02, pot_top, 0.7, _PALETTE["trunk"], kind),
                 Cylinder(0, 0, crown, 0.7, 1.3, _PALETTE["leaf"], kind)],
                2 * pot, 2 * pot, False)
    if kind == "sofa":
        lx, ly = u(1.6, 2.2), u(0.8, 0.95)
        color = _PALETTE[str(rng.choice(["teal", "slate", "rust"]))]
        return ([Box(0, 0, lx / 2, ly / 2, 0.0, 0.42, color=color, label=kind),
                 Box(0, -ly / 2 + 0.09, lx / 2, 0.09, 0.42, 0.82, color=color, label=kind)],
                lx, ly, True)
    if kind == "shelf":
        lx, ly = u(0.8, 1.6), u(0.30, 0.40)
        color = _PALETTE[str(rng.choice(["walnut", "white", "slate"]))]
        return [Box(0, 0, lx / 2, ly / 2, 0.0, 1.8, color=color, label=kind)], lx, ly, True
    if kind == "console":
        lx, ly = u(1.0, 1.6), u(0.35, 0.40)
        return ([Box(0, 0, lx / 2, ly / 2, 0.25, 0.65, color=_PALETTE["slate"], label=kind)],
                lx, ly, True)
    raise ValueError(f"unknown furniture kind '{kind}'")
