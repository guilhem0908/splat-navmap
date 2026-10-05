"""Score an extracted grid against exact geometry and against the paths planned on it.

Two views of the same map:

* **cells**: precision, recall and IoU of the occupied cells. Cells strictly inside a solid
  (all eight neighbours occupied in the truth) are ignored, because a splat only models visible
  surfaces and nothing can be planned through them anyway.
* **paths**: for start-goal pairs that are feasible in the true geometry, plan on the extracted
  grid and classify the outcome as *unreachable* (no path, or an endpoint is blocked),
  *collision* (the robot disc swept along the path intersects the true solids), or *valid*.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from splat_navmap.geometry import clearance
from splat_navmap.grid import GridSpec, inflate, reachable_components
from splat_navmap.planners import Cell, Planner, path_length
from splat_navmap.scene import Robot, Scene

COLLISION_TOL = 1e-6  # a path collides when its true clearance is below radius - tol, metres
MIN_PAIR_DISTANCE = 2.0  # straight-line distance between sampled start and goal, metres
ENDPOINT_MARGIN = 0.15  # extra clearance of sampled starts and goals, metres
_SQUARE = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class MapScore:
    """Cell counts of an occupancy grid against the truth (occupied = positive class)."""

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def iou(self) -> float:
        return self.tp / max(self.tp + self.fp + self.fn, 1)


@dataclass(frozen=True)
class PathStats:
    """Outcome counts over a set of feasible start-goal pairs."""

    pairs: int
    unreachable: int
    endpoint_blocked: int  # subset of ``unreachable``: start or goal lies in a blocked cell
    collisions: int
    valid: int
    ratio_sum: float  # sum over valid paths of length / ground-truth-optimal length

    @property
    def mean_ratio(self) -> float:
        return self.ratio_sum / self.valid if self.valid else math.nan


def score_map(estimate: np.ndarray, truth: np.ndarray) -> MapScore:
    """Compare occupied cells, ignoring the unobservable interior of solids."""
    surface = ~ndimage.binary_erosion(truth, structure=_SQUARE, border_value=1)
    return MapScore(tp=int((estimate & truth & surface).sum()),
                    fp=int((estimate & ~truth).sum()),
                    fn=int((~estimate & truth & surface).sum()))


def remove_small_components(occupied: np.ndarray, min_cells: int) -> np.ndarray:
    """Drop 8-connected groups of occupied cells smaller than ``min_cells``."""
    labels, count = ndimage.label(occupied, structure=_SQUARE)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    return (sizes >= min_cells)[labels] & occupied


def sample_pairs(blocked: np.ndarray, resolution: float, rng: np.random.Generator, count: int,
                 min_distance: float = MIN_PAIR_DISTANCE,
                 margin: float = ENDPOINT_MARGIN) -> list[tuple[Cell, Cell]]:
    """Draw start-goal pairs inside the largest connected free region of ``blocked``.

    Endpoints keep ``margin`` metres from the blocked cells, so that a map which is merely one
    or two cells fatter than the truth does not turn a feasible pair into an unreachable one.
    """
    labels = reachable_components(blocked)
    largest = np.bincount(labels.ravel())[1:].argmax() + 1
    roomy = ndimage.distance_transform_edt(~blocked) * resolution >= margin
    iy, ix = np.nonzero((labels == largest) & roomy)
    pairs: list[tuple[Cell, Cell]] = []
    while len(pairs) < count:
        i, j = rng.integers(len(ix), size=2)
        if math.hypot(ix[i] - ix[j], iy[i] - iy[j]) * resolution >= min_distance:
            pairs.append(((int(ix[i]), int(iy[i])), (int(ix[j]), int(iy[j]))))
    return pairs


def path_clearance(scene: Scene, spec: GridSpec, path: Sequence[Cell], robot: Robot) -> float:
    """Exact distance between the path polyline and the solids at the robot's body heights."""
    return clearance(scene.solids, spec.cell_to_world(np.asarray(path)),
                     robot.ground_clearance, robot.height)


def collides(scene: Scene, spec: GridSpec, path: Sequence[Cell], robot: Robot) -> bool:
    """True when the robot disc swept along the path intersects the true geometry."""
    return path_clearance(scene, spec, path, robot) < robot.radius - COLLISION_TOL


def evaluate_paths(scene: Scene, spec: GridSpec, robot: Robot, occupied: np.ndarray,
                   pairs: Sequence[tuple[Cell, Cell]], reference_lengths: Sequence[float],
                   planner: Planner) -> PathStats:
    """Plan every pair on ``occupied`` inflated by the robot radius and classify the outcomes."""
    blocked = inflate(occupied, robot.radius, spec.resolution)
    unreachable = endpoint_blocked = collisions = valid = 0
    ratio_sum = 0.0
    for (start, goal), reference in zip(pairs, reference_lengths, strict=True):
        path = planner(blocked, start, goal)
        if path is None:
            unreachable += 1
            endpoint_blocked += bool(blocked[start[1], start[0]] or blocked[goal[1], goal[0]])
        elif collides(scene, spec, path, robot):
            collisions += 1
        else:
            valid += 1
            ratio_sum += path_length(path) / reference
    return PathStats(len(pairs), unreachable, endpoint_blocked, collisions, valid, ratio_sum)
