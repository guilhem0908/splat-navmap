"""The benchmark: which extraction settings give a map a planner can trust?

A *condition* fixes the defects of the reconstruction, the height band, the extraction method,
the opacity threshold, an optional small-component filter and the planner. Every condition is
evaluated on the same seeded scenes and the same start-goal pairs, which are feasible in the
true geometry by construction. One row per (condition, scene) is produced; aggregation happens
in ``scripts/reproduce.py`` so that the raw per-scene counts stay available.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from splat_navmap.evaluate import (
    collides,
    evaluate_paths,
    remove_small_components,
    sample_pairs,
    score_map,
)
from splat_navmap.extract import EXTRACTORS, Band
from splat_navmap.grid import inflate
from splat_navmap.planners import astar, path_length, theta_star
from splat_navmap.scene import Robot, generate_scene
from splat_navmap.synth import DEFECT_LEVELS, Defects, reconstruct

PLANNERS = {"astar": astar, "theta": theta_star}
THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BANDS = {
    "body": Band(0.02, 0.40),  # exactly the heights swept by the robot's body
    "guarded": Band(0.05, 0.40),  # body band with a 5 cm guard above the floor
    "floor": Band(0.00, 0.40),  # no guard at all
    "guard10": Band(0.10, 0.40),  # 10 cm guard: hides anything lower than 10 cm
    "slice": Band(0.18, 0.22),  # one thin slice at mid height
    "short": Band(0.05, 0.20),  # robot assumed half as tall as it is
    "tall": Band(0.05, 0.90),  # robot assumed more than twice as tall
}
DEFAULT_BAND = "guarded"
_SEVERE = DEFECT_LEVELS["severe"]
DEFECTS = {
    **DEFECT_LEVELS,
    "floaters": Defects(floater_density=_SEVERE.floater_density),
    "holes": Defects(hole_density=_SEVERE.hole_density),
    "weak": Defects(weak_fraction=_SEVERE.weak_fraction),
    "thickness": Defects(thickness=_SEVERE.thickness),
}
PAIR_STREAM = 7  # sub-stream of the scene seed used to draw start-goal pairs


@dataclass(frozen=True)
class Condition:
    """One benchmark setting. ``method == 'truth'`` plans on the exact occupancy (control)."""

    experiment: str
    defect: str
    method: str
    tau: float
    band: str = DEFAULT_BAND
    min_component: int = 1
    planner: str = "astar"


def evaluate_scene(seed: int, conditions: Sequence[Condition], pairs_per_scene: int,
                   robot: Robot | None = None) -> list[dict]:
    """Evaluate all conditions on the scene drawn from ``seed``; one result row per condition."""
    robot = robot or Robot()
    scene = generate_scene(seed, robot)
    spec = scene.grid()
    truth = scene.truth(spec, robot)
    truth_blocked = inflate(truth, robot.radius, spec.resolution)
    pairs = sample_pairs(truth_blocked, spec.resolution, np.random.default_rng([seed, PAIR_STREAM]),
                         pairs_per_scene)
    reference: dict[str, list[float]] = {}
    for name in sorted({c.planner for c in conditions}):
        paths = [PLANNERS[name](truth_blocked, start, goal) for start, goal in pairs]
        if any(collides(scene, spec, path, robot) for path in paths):
            raise AssertionError("a path planned on the exact grid collides: inflation is broken")
        reference[name] = [path_length(path) for path in paths]

    cloud_name, cloud = "", None  # only the latest reconstruction is kept in memory
    scores: dict[tuple[str, str, str], np.ndarray] = {}
    rows = []
    for condition in conditions:
        if condition.method == "truth":
            occupied = truth
        else:
            key = (condition.defect, condition.band, condition.method)
            if key not in scores:
                if condition.defect != cloud_name:
                    cloud_name = condition.defect
                    cloud = reconstruct(scene, seed, DEFECTS[cloud_name])
                scores[key] = EXTRACTORS[condition.method](cloud, spec, BANDS[condition.band])
            occupied = scores[key] >= condition.tau
            if condition.min_component > 1:
                occupied = remove_small_components(occupied, condition.min_component)
        cells = score_map(occupied, truth)
        paths = evaluate_paths(scene, spec, robot, occupied, pairs, reference[condition.planner],
                               PLANNERS[condition.planner])
        rows.append({**asdict(condition), "seed": seed, **asdict(cells), **asdict(paths)})
    return rows


def _worker(job: tuple[int, Sequence[Condition], int]) -> list[dict]:
    return evaluate_scene(*job)


def run(conditions: Sequence[Condition], seeds: Sequence[int], pairs_per_scene: int,
        jobs: int = 1) -> list[dict]:
    """Evaluate every condition on every seed; the result does not depend on ``jobs``."""
    work = [(seed, conditions, pairs_per_scene) for seed in seeds]
    if jobs <= 1:
        batches = [_worker(job) for job in work]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            batches = list(pool.map(_worker, work))
    return [row for batch in batches for row in batch]
