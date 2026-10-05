"""Exact ground-truth geometry: solids, their footprint on a grid, and distances to a path.

Scenes are unions of two analytic solids, both extruded along z: an oriented box and a vertical
cylinder. Ground truth never goes through Gaussians or sampling:

* a cell is occupied when its square and a solid's footprint overlap with positive area
  (separating-axis test for boxes, closest-point test for discs);
* the clearance of a path is the exact distance between its segments and the footprints
  (two convex sets that do not intersect are closest at a vertex of one of them).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from splat_navmap.grid import GridSpec

_OVERLAP_TOL = 1e-9  # footprints that only touch a cell edge do not occupy the cell
_TINY = 1e-12
Color = tuple[float, float, float]


@dataclass(frozen=True)
class Box:
    """Box centred at ``(cx, cy)``, half-extents ``(hx, hy)``, rotated by ``yaw`` about z."""

    cx: float
    cy: float
    hx: float
    hy: float
    z0: float
    z1: float
    yaw: float = 0.0
    color: Color = (0.8, 0.8, 0.8)
    label: str = "box"

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit vectors of the footprint's local x and y axes in the world frame."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([c, s]), np.array([-s, c])


@dataclass(frozen=True)
class Cylinder:
    """Vertical cylinder of given radius between heights ``z0`` and ``z1``."""

    cx: float
    cy: float
    radius: float
    z0: float
    z1: float
    color: Color = (0.8, 0.8, 0.8)
    label: str = "cylinder"


Solid = Box | Cylinder


def in_band(solid: Solid, z_lo: float, z_hi: float) -> bool:
    """True when the solid occupies some height strictly inside ``(z_lo, z_hi)``."""
    return solid.z0 < z_hi - _OVERLAP_TOL and solid.z1 > z_lo + _OVERLAP_TOL


def footprint_mask(solid: Solid, spec: GridSpec) -> np.ndarray:
    """Boolean grid of the cells whose square overlaps the solid's footprint."""
    xs, ys = spec.centers()
    dx, dy = xs[None, :] - solid.cx, ys[:, None] - solid.cy
    half = spec.resolution / 2
    if isinstance(solid, Cylinder):
        gap_x, gap_y = np.maximum(np.abs(dx) - half, 0), np.maximum(np.abs(dy) - half, 0)
        return gap_x**2 + gap_y**2 < (solid.radius - _OVERLAP_TOL) ** 2
    (ux, uy), (vx, vy) = solid.axes()
    reach = half * (abs(ux) + abs(uy))  # half-width of a cell square projected on a box axis
    return (
        (np.abs(dx) < half + solid.hx * abs(ux) + solid.hy * abs(vx) - _OVERLAP_TOL)
        & (np.abs(dy) < half + solid.hx * abs(uy) + solid.hy * abs(vy) - _OVERLAP_TOL)
        & (np.abs(dx * ux + dy * uy) < solid.hx + reach - _OVERLAP_TOL)
        & (np.abs(dx * vx + dy * vy) < solid.hy + reach - _OVERLAP_TOL)
    )


def rasterize(solids: Sequence[Solid], spec: GridSpec, z_lo: float, z_hi: float) -> np.ndarray:
    """Exact occupancy of the height band ``(z_lo, z_hi)``: union of the footprints in the band."""
    occupied = np.zeros(spec.shape, dtype=bool)
    for solid in solids:
        if in_band(solid, z_lo, z_hi):
            occupied |= footprint_mask(solid, spec)
    return occupied


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points ``p`` to segments ``a -> b`` (broadcast over leading axes)."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), _TINY), 0.0, 1.0)
    return np.linalg.norm(a + t[..., None] * ab - p, axis=-1)


def _segment_rectangle_distance(a: np.ndarray, b: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Distance from segments ``a -> b`` (M, 2) to the rectangle ``[-half, half]``."""
    d = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo, t_hi = (-half - a) / d, (half - a) / d
    parallel = np.abs(d) < _TINY
    inside_slab = np.abs(a) <= half
    t_near = np.where(parallel, np.where(inside_slab, -np.inf, np.inf), np.minimum(t_lo, t_hi))
    t_far = np.where(parallel, np.where(inside_slab, np.inf, -np.inf), np.maximum(t_lo, t_hi))
    crosses = np.maximum(t_near.max(axis=1), 0.0) <= np.minimum(t_far.min(axis=1), 1.0)

    def to_rectangle(p: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.maximum(np.abs(p) - half, 0.0), axis=1)

    corners = half * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    corner_gap = _point_segment_distance(corners[None], a[:, None], b[:, None]).min(axis=1)
    gap = np.minimum(np.minimum(to_rectangle(a), to_rectangle(b)), corner_gap)
    return np.where(crosses, 0.0, gap)


def segment_distance(solid: Solid, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact distance from segments ``a -> b`` (M, 2) to the solid's footprint (0 if they meet)."""
    centre = np.array([solid.cx, solid.cy])
    if isinstance(solid, Cylinder):
        return np.maximum(_point_segment_distance(centre, a, b) - solid.radius, 0.0)
    frame = np.stack(solid.axes(), axis=1)  # columns are the box axes: world -> box is `@ frame`
    return _segment_rectangle_distance((a - centre) @ frame, (b - centre) @ frame,
                                       np.array([solid.hx, solid.hy]))


def clearance(solids: Sequence[Solid], polyline: np.ndarray, z_lo: float, z_hi: float) -> float:
    """Smallest distance between a polyline (K, 2) and the solids present in the height band."""
    points = np.asarray(polyline, dtype=float)
    a, b = (points, points) if len(points) == 1 else (points[:-1], points[1:])
    gaps = [segment_distance(s, a, b).min() for s in solids if in_band(s, z_lo, z_hi)]
    return float(min(gaps, default=np.inf))
