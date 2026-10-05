"""2D grid frame and robot-radius inflation.

Conventions (same as ``nav_msgs/OccupancyGrid``): arrays are indexed ``[iy, ix]``, cell
``(ix, iy)`` covers ``origin + [ix, ix + 1] x [iy, iy + 1]`` times the resolution, and planners
address cells as ``(ix, iy)`` tuples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

DEFAULT_RESOLUTION = 0.05  # metres per cell
DEFAULT_MARGIN = 0.10  # border around the scene bounds, metres (inside the inflation radius)
_DISTANCE_TOL = 1e-9  # makes "exactly one radius away" deterministic (counted as free)
_SQUARE = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class GridSpec:
    """Placement of a grid in the world: lower-left corner, cell size, shape ``(ny, nx)``."""

    origin: tuple[float, float]
    resolution: float
    shape: tuple[int, int]

    @classmethod
    def covering(cls, bounds: tuple[float, float, float, float],
                 resolution: float = DEFAULT_RESOLUTION,
                 margin: float = DEFAULT_MARGIN) -> GridSpec:
        """Grid covering ``bounds = (xmin, ymin, xmax, ymax)`` plus a margin on each side."""
        xmin, ymin, xmax, ymax = bounds
        nx = math.ceil((xmax - xmin + 2 * margin) / resolution - 1e-9)
        ny = math.ceil((ymax - ymin + 2 * margin) / resolution - 1e-9)
        return cls((xmin - margin, ymin - margin), resolution, (ny, nx))

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """World coordinates of the cell centres along x (nx,) and along y (ny,)."""
        ny, nx = self.shape
        return (self.origin[0] + (np.arange(nx) + 0.5) * self.resolution,
                self.origin[1] + (np.arange(ny) + 0.5) * self.resolution)

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        """Integer cell indices ``(ix, iy)`` of world points (..., 2); may fall outside the grid."""
        return np.floor((np.asarray(xy, dtype=float) - self.origin) / self.resolution).astype(int)

    def cell_to_world(self, cells: np.ndarray) -> np.ndarray:
        """World coordinates of the centres of cells (..., 2) given as ``(ix, iy)``."""
        return np.asarray(self.origin) + (np.asarray(cells, dtype=float) + 0.5) * self.resolution


def inflate(occupied: np.ndarray, radius: float, resolution: float) -> np.ndarray:
    """Cells the centre of a disc robot must not enter, with a conservative guarantee.

    A cell is blocked when *any point of its square* is closer than ``radius`` to *any point of
    an occupied square*. The distance between two cell squares offset by ``(dx, dy)`` cells is
    ``resolution * hypot(max(|dx| - 1, 0), max(|dy| - 1, 0))``, which is exactly the Euclidean
    distance transform to the occupied set grown by one cell in the 8-neighbourhood.

    Consequence: if every true obstacle lies inside occupied cells, a robot centre anywhere
    inside unblocked cells keeps a true clearance of at least ``radius``. Paths planned on the
    ground-truth grid therefore never collide, so every collision measured on an extracted grid
    is caused by the map and not by discretisation.
    """
    grown = ndimage.binary_dilation(occupied, structure=_SQUARE)
    distance = ndimage.distance_transform_edt(~grown) * resolution
    return distance < radius - _DISTANCE_TOL


def reachable_components(blocked: np.ndarray) -> np.ndarray:
    """Label free cells by connected component (0 = blocked).

    4-connectivity is used on purpose: with corner cutting forbidden, a diagonal move needs both
    orthogonal neighbours free, so 8-connected planning reaches exactly the 4-connected component.
    """
    labels, _ = ndimage.label(~blocked)
    return labels
