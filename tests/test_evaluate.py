"""Map scores, path outcomes, and the benchmark's own invariants."""

from __future__ import annotations

import numpy as np
import pytest

from splat_navmap.benchmark import Condition, evaluate_scene, run
from splat_navmap.evaluate import (
    ENDPOINT_MARGIN,
    MIN_PAIR_DISTANCE,
    collides,
    evaluate_paths,
    path_clearance,
    remove_small_components,
    sample_pairs,
    score_map,
)
from splat_navmap.geometry import Box
from splat_navmap.grid import GridSpec, inflate, reachable_components
from splat_navmap.planners import astar, path_length, theta_star
from splat_navmap.scene import Robot, Scene, generate_scene

ROBOT = Robot()


def walled_scene() -> tuple[Scene, GridSpec]:
    """Two 3 x 3 m rooms separated by a wall with a 1 m door at the top."""
    floor = Box(3.0, 1.5, 3.2, 1.7, -0.05, 0.0, label="floor")
    solids = (Box(3.0, 1.0, 0.05, 1.0, 0.0, 2.5), Box(3.0, -0.1, 3.2, 0.1, 0.0, 2.5),
              Box(3.0, 3.1, 3.2, 0.1, 0.0, 2.5), Box(-0.1, 1.5, 0.1, 1.7, 0.0, 2.5),
              Box(6.1, 1.5, 0.1, 1.7, 0.0, 2.5))
    scene = Scene((-0.2, -0.2, 6.2, 3.2), floor, solids, ((0, 0, 3, 3), (3, 0, 6, 3)))
    return scene, scene.grid()


def cell(spec: GridSpec, x: float, y: float) -> tuple[int, int]:
    ix, iy = spec.world_to_cell([x, y])
    return int(ix), int(iy)


def test_score_map_counts_cells_and_ignores_solid_interiors() -> None:
    truth = np.zeros((9, 9), dtype=bool)
    truth[2:7, 2:7] = True  # 5 x 5 block: 16 surface cells around a 3 x 3 interior
    shell = truth.copy()
    shell[3:6, 3:6] = False
    assert score_map(shell, truth).iou == 1.0
    assert score_map(truth, truth).iou == 1.0
    estimate = shell.copy()
    estimate[2, 2:5] = False  # three surface cells missed
    estimate[0, 0:2] = True  # two phantom cells
    score = score_map(estimate, truth)
    assert (score.tp, score.fp, score.fn) == (13, 2, 3)
    assert score.precision == pytest.approx(13 / 15)
    assert score.recall == pytest.approx(13 / 16)
    assert score.iou == pytest.approx(13 / 18)


def test_small_component_filter_drops_specks_and_keeps_walls() -> None:
    occupied = np.zeros((10, 10), dtype=bool)
    occupied[1, :] = True  # wall, 10 cells
    occupied[5, 5] = occupied[6, 6] = True  # diagonal pair: one 8-connected group of 2
    occupied[8, 1:4] = True  # group of 3
    assert remove_small_components(occupied, 3).sum() == 13
    assert remove_small_components(occupied, 4).sum() == 10
    np.testing.assert_array_equal(remove_small_components(occupied, 1), occupied)


def test_sampled_pairs_are_feasible_separated_and_reproducible() -> None:
    scene = generate_scene(3)
    spec = scene.grid()
    truth = scene.truth(spec, ROBOT)
    blocked = inflate(truth, ROBOT.radius, spec.resolution)
    pairs = sample_pairs(blocked, spec.resolution, np.random.default_rng(0), 50)
    assert pairs == sample_pairs(blocked, spec.resolution, np.random.default_rng(0), 50)
    labels = reachable_components(blocked)
    wider = inflate(truth, ROBOT.radius + ENDPOINT_MARGIN - spec.resolution, spec.resolution)
    for start, goal in pairs:
        assert labels[start[1], start[0]] == labels[goal[1], goal[0]] > 0
        assert not wider[start[1], start[0]] and not wider[goal[1], goal[0]]
        gap = np.hypot(start[0] - goal[0], start[1] - goal[1]) * spec.resolution
        assert gap >= MIN_PAIR_DISTANCE


@pytest.mark.parametrize("seed", range(3))
def test_paths_planned_on_the_exact_grid_never_collide(seed: int) -> None:
    scene = generate_scene(seed)
    spec = scene.grid()
    blocked = inflate(scene.truth(spec, ROBOT), ROBOT.radius, spec.resolution)
    pairs = sample_pairs(blocked, spec.resolution, np.random.default_rng(seed), 12)
    for planner in (astar, theta_star):
        for start, goal in pairs:
            path = planner(blocked, start, goal)
            assert path is not None
            assert path_clearance(scene, spec, path, ROBOT) >= ROBOT.radius - 1e-9


def test_a_missing_wall_in_the_map_turns_into_a_measured_collision() -> None:
    scene, spec = walled_scene()
    truth = scene.truth(spec, ROBOT)
    start, goal = cell(spec, 1.5, 0.8), cell(spec, 4.5, 0.8)
    honest = astar(inflate(truth, ROBOT.radius, spec.resolution), start, goal)
    assert not collides(scene, spec, honest, ROBOT)
    assert path_length(honest) * spec.resolution > 4.0  # detour through the door

    holed = truth.copy()
    wall_columns = np.flatnonzero(truth[start[1]])[1:-1]  # drop the two outer walls
    holed[4:30, wall_columns.min() : wall_columns.max() + 1] = False  # the map forgets the wall
    shortcut = astar(inflate(holed, ROBOT.radius, spec.resolution), start, goal)
    assert path_length(shortcut) * spec.resolution < 3.2
    assert path_clearance(scene, spec, shortcut, ROBOT) == 0.0
    assert collides(scene, spec, shortcut, ROBOT)

    stats = evaluate_paths(scene, spec, ROBOT, holed, [(start, goal)], [path_length(honest)],
                           astar)
    assert (stats.collisions, stats.valid, stats.unreachable) == (1, 0, 0)


def test_a_phantom_obstacle_in_the_map_turns_into_an_unreachable_pair() -> None:
    scene, spec = walled_scene()
    truth = scene.truth(spec, ROBOT)
    start, goal = cell(spec, 1.5, 0.8), cell(spec, 4.5, 0.8)
    reference = [path_length(astar(inflate(truth, ROBOT.radius, spec.resolution), start, goal))]
    closed_door = truth.copy()
    closed_door[:, cell(spec, 3.0, 0.0)[0]] = True
    stats = evaluate_paths(scene, spec, ROBOT, closed_door, [(start, goal)], reference, astar)
    assert (stats.unreachable, stats.endpoint_blocked, stats.collisions) == (1, 0, 0)
    on_goal = truth.copy()
    on_goal[goal[1], goal[0]] = True
    stats = evaluate_paths(scene, spec, ROBOT, on_goal, [(start, goal)], reference, astar)
    assert (stats.unreachable, stats.endpoint_blocked) == (1, 1)
    exact = evaluate_paths(scene, spec, ROBOT, truth, [(start, goal)], reference, astar)
    assert (exact.valid, exact.mean_ratio) == (1, 1.0)


@pytest.mark.slow
def test_benchmark_rows_are_consistent_and_independent_of_the_worker_count() -> None:
    conditions = [Condition("demo", "clean", "truth", 0.0),
                  Condition("demo", "clean", "truth", 0.0, planner="theta"),
                  Condition("demo", "moderate", "centres", 0.3),
                  Condition("demo", "moderate", "footprint", 0.5, min_component=4)]
    serial = run(conditions, seeds=[0, 1], pairs_per_scene=8, jobs=1)
    assert serial == run(conditions, seeds=[0, 1], pairs_per_scene=8, jobs=2)
    assert serial[:4] == evaluate_scene(0, conditions, 8)
    assert len(serial) == 8
    for row in serial:
        assert row["pairs"] == 8 == row["unreachable"] + row["collisions"] + row["valid"]
        assert row["endpoint_blocked"] <= row["unreachable"]
        if row["method"] == "truth":
            assert (row["fp"], row["fn"], row["valid"]) == (0, 0, 8)
            assert row["ratio_sum"] == pytest.approx(8.0)
