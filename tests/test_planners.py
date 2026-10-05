"""Planner properties checked against independent brute-force references on small grids."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from splat_navmap.geometry import Box, segment_distance
from splat_navmap.planners import astar, line_of_sight, path_length, theta_star

SEEDS = range(12)
SIZE = 14
OBSTACLE_RATE = 0.28


def random_grid(seed: int, size: int = SIZE) -> np.ndarray:
    return np.random.default_rng(seed).random((size, size)) < OBSTACLE_RATE


def free_cells(blocked: np.ndarray) -> list[tuple[int, int]]:
    iy, ix = np.nonzero(~blocked)
    return list(zip(ix.tolist(), iy.tolist(), strict=True))


def grid_graph_distances(blocked: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Dijkstra on an explicitly built 8-connected graph with the no-corner-cutting rule."""
    ny, nx = blocked.shape
    rows, cols, weights = [], [], []
    for iy in range(ny):
        for ix in range(nx):
            if blocked[iy, ix]:
                continue
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < nx and 0 <= jy < ny) or blocked[jy, jx]:
                    continue
                if dx and dy and (blocked[iy, jx] or blocked[jy, ix]):
                    continue
                rows.append(iy * nx + ix)
                cols.append(jy * nx + jx)
                weights.append(math.hypot(dx, dy))
    graph = coo_matrix((weights, (rows, cols)), shape=(ny * nx, ny * nx))
    return dijkstra(graph, directed=False, indices=source[1] * nx + source[0]).reshape(ny, nx)


def visibility_graph_distances(blocked: np.ndarray, source: tuple[int, int]) -> dict:
    """Shortest paths over the complete visibility graph of the free cell centres."""
    cells = free_cells(blocked)
    n = len(cells)
    weights = np.zeros((n, n))
    for i, a in enumerate(cells):
        for j in range(i + 1, n):
            if line_of_sight(blocked, a, cells[j]):
                weights[i, j] = math.hypot(a[0] - cells[j][0], a[1] - cells[j][1])
    dist = dijkstra(coo_matrix(weights), directed=False, indices=cells.index(source))
    return dict(zip(cells, dist.tolist(), strict=True))


@pytest.mark.parametrize("seed", SEEDS)
def test_astar_cost_equals_dijkstra_on_random_grids(seed: int) -> None:
    blocked = random_grid(seed)
    cells = free_cells(blocked)
    source = cells[0]
    reference = grid_graph_distances(blocked, source)
    for goal in cells[:: max(1, len(cells) // 25)]:
        path = astar(blocked, source, goal)
        if math.isinf(reference[goal[1], goal[0]]):
            assert path is None
        else:
            assert path[0] == source and path[-1] == goal
            assert path_length(path) == pytest.approx(reference[goal[1], goal[0]], abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_astar_moves_one_cell_at_a_time_and_never_cuts_a_corner(seed: int) -> None:
    blocked = random_grid(seed)
    cells = free_cells(blocked)
    path = astar(blocked, cells[0], cells[-1])
    if path is None:
        return
    for (ax, ay), (bx, by) in itertools.pairwise(path):
        assert max(abs(bx - ax), abs(by - ay)) == 1
        assert not blocked[by, bx]
        assert not blocked[ay, bx] and not blocked[by, ax]


def test_diagonal_gap_between_two_obstacles_is_not_traversable() -> None:
    blocked = np.zeros((2, 2), dtype=bool)
    blocked[0, 1] = blocked[1, 0] = True
    assert astar(blocked, (0, 0), (1, 1)) is None
    assert theta_star(blocked, (0, 0), (1, 1)) is None
    assert not line_of_sight(blocked, (0, 0), (1, 1))


def test_blocked_or_outside_endpoints_give_no_path() -> None:
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[2, 2] = True
    for planner in (astar, theta_star):
        assert planner(blocked, (2, 2), (4, 4)) is None
        assert planner(blocked, (0, 0), (5, 0)) is None
        assert planner(blocked, (0, 0), (0, 0)) == [(0, 0)]


@pytest.mark.parametrize("seed", SEEDS)
def test_line_of_sight_matches_exact_segment_square_intersection(seed: int) -> None:
    blocked = random_grid(seed)
    rng = np.random.default_rng(1000 + seed)
    cells = free_cells(blocked)
    obstacles = [Box(ix + 0.5, iy + 0.5, 0.5, 0.5, 0.0, 1.0)
                 for iy, ix in zip(*np.nonzero(blocked), strict=True)]
    for _ in range(60):
        a, b = (cells[i] for i in rng.integers(len(cells), size=2))
        pa, pb = np.array([a], dtype=float) + 0.5, np.array([b], dtype=float) + 0.5
        touches = any(segment_distance(box, pa, pb)[0] < 1e-9 for box in obstacles)
        assert line_of_sight(blocked, a, b) == (not touches)
        assert line_of_sight(blocked, a, b) == line_of_sight(blocked, b, a)


@pytest.mark.parametrize("seed", SEEDS)
def test_theta_star_lies_between_visibility_graph_optimum_and_astar(seed: int) -> None:
    blocked = random_grid(seed)
    cells = free_cells(blocked)
    source = cells[0]
    optimum = visibility_graph_distances(blocked, source)
    for goal in cells[:: max(1, len(cells) // 20)]:
        any_angle, lattice = theta_star(blocked, source, goal), astar(blocked, source, goal)
        assert (any_angle is None) == (lattice is None) == math.isinf(optimum[goal])
        if any_angle is None:
            continue
        assert all(line_of_sight(blocked, p, q)
                   for p, q in itertools.pairwise(any_angle))
        assert optimum[goal] - 1e-9 <= path_length(any_angle) <= path_length(lattice) + 1e-9


def test_theta_star_is_near_optimal_in_a_two_room_map() -> None:
    """Basic Theta* is not optimal, but on room-like maps it stays within a few percent."""
    blocked = np.zeros((18, 24), dtype=bool)
    blocked[:, 11:13] = True
    blocked[7:10, 11:13] = False  # door
    blocked[3:6, 4:7] = blocked[12:15, 16:20] = True  # furniture
    source = (1, 1)
    optimum = visibility_graph_distances(blocked, source)
    goals = [goal for goal in free_cells(blocked)[::5] if goal != source]
    any_angle = np.array([path_length(theta_star(blocked, source, g)) / optimum[g] for g in goals])
    lattice = np.array([path_length(astar(blocked, source, g)) / optimum[g] for g in goals])
    assert any_angle.min() >= 1.0 - 1e-9
    assert any_angle.mean() < 1.01 and any_angle.max() < 1.05
    assert lattice.mean() > 1.04  # the 8-connected optimum is measurably longer


def test_theta_star_is_a_straight_line_in_free_space() -> None:
    blocked = np.zeros((30, 40), dtype=bool)
    path = theta_star(blocked, (2, 3), (37, 20))
    assert path == [(2, 3), (37, 20)]
    assert path_length(path) == pytest.approx(math.hypot(35, 17))
    assert path_length(astar(blocked, (2, 3), (37, 20))) == pytest.approx(18 + 17 * math.sqrt(2))
