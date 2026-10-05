"""Exact geometry kernel and conservative inflation, each against a brute-force reference."""

from __future__ import annotations

import math

import numpy as np
import pytest

from splat_navmap.geometry import (
    Box,
    Cylinder,
    clearance,
    footprint_mask,
    rasterize,
    segment_distance,
)
from splat_navmap.grid import GridSpec, inflate

SPEC = GridSpec(origin=(-1.0, -0.5), resolution=0.05, shape=(50, 60))


def random_solid(rng: np.random.Generator, shrink: float = 0.0):
    cx, cy = rng.uniform(0.0, 1.2), rng.uniform(0.3, 1.5)
    if rng.random() < 0.5:
        return Cylinder(cx, cy, rng.uniform(0.05, 0.4) - shrink, 0.0, 1.0)
    return Box(cx, cy, rng.uniform(0.05, 0.5) - shrink, rng.uniform(0.03, 0.3) - shrink, 0.0, 1.0,
               yaw=rng.uniform(0.0, math.pi))


def contains(solid, points: np.ndarray) -> np.ndarray:
    """Point-in-footprint test written independently of the module under test."""
    rel = points - np.array([solid.cx, solid.cy])
    if isinstance(solid, Cylinder):
        return (rel**2).sum(-1) < solid.radius**2
    c, s = math.cos(solid.yaw), math.sin(solid.yaw)
    local_x, local_y = rel[..., 0] * c + rel[..., 1] * s, -rel[..., 0] * s + rel[..., 1] * c
    return (np.abs(local_x) < solid.hx) & (np.abs(local_y) < solid.hy)


@pytest.mark.parametrize("seed", range(20))
def test_footprint_mask_matches_dense_point_sampling(seed: int) -> None:
    """Cells hit by a dense point lattice are occupied; cells occupied by a slightly shrunk
    solid are hit by the lattice. The exact mask is squeezed between the two."""
    lattice = 12
    shrink = SPEC.resolution / lattice
    rng = np.random.default_rng(seed)
    solid = random_solid(rng)
    rng = np.random.default_rng(seed)
    smaller = random_solid(rng, shrink=shrink)
    xs, ys = SPEC.centers()
    sub = (np.arange(lattice) + 0.5) / lattice - 0.5
    px = (xs[:, None] + sub * SPEC.resolution).ravel()
    py = (ys[:, None] + sub * SPEC.resolution).ravel()
    inside = contains(solid, np.stack(np.meshgrid(px, py), axis=-1))
    sampled = inside.reshape(len(ys), lattice, len(xs), lattice).any(axis=(1, 3))
    exact = footprint_mask(solid, SPEC)
    assert exact.sum() > 0
    assert not (sampled & ~exact).any()
    assert not (footprint_mask(smaller, SPEC) & ~sampled).any()


@pytest.mark.parametrize("seed", range(20))
def test_segment_distance_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    solid = random_solid(rng)
    a, b = rng.uniform(-0.5, 2.0, (40, 2)), rng.uniform(-0.5, 2.0, (40, 2))
    exact = segment_distance(solid, a, b)
    # Brute force: dense points on each segment against dense points of the footprint.
    grid = np.stack(np.meshgrid(np.linspace(-0.6, 2.1, 541), np.linspace(-0.6, 2.1, 541)), -1)
    body = grid[contains(solid, grid)]
    t = np.linspace(0.0, 1.0, 81)[:, None]
    for k in range(0, 40, 4):
        points = a[k] + t * (b[k] - a[k])
        brute = np.sqrt(((points[:, None, :] - body[None]) ** 2).sum(-1).min())
        if contains(solid, points).any():
            assert exact[k] == 0.0
        else:
            assert exact[k] == pytest.approx(brute, abs=0.02)
            assert exact[k] <= brute + 1e-9  # the sampled distance can only overestimate


def test_segment_touching_a_box_corner_has_zero_distance() -> None:
    box = Box(0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    a, b = np.array([[0.0, 2.0], [0.0, 2.1]]), np.array([[2.0, 0.0], [2.1, 0.0]])
    np.testing.assert_allclose(segment_distance(box, a, b), [0.0, 0.1 / math.sqrt(2)], atol=1e-12)


def test_only_solids_inside_the_height_band_count() -> None:
    table_top = Box(1.0, 1.0, 0.6, 0.4, 0.72, 0.76)
    leg = Box(0.5, 0.7, 0.025, 0.025, 0.0, 0.72)
    rug = Box(1.0, 1.0, 0.8, 0.8, 0.0, 0.01)
    solids = [table_top, leg, rug]
    spec = GridSpec((0.0, 0.0), 0.05, (40, 40))
    under_the_top = rasterize(solids, spec, 0.02, 0.40)
    assert under_the_top.sum() == footprint_mask(leg, spec).sum() > 0
    assert rasterize(solids, spec, 0.02, 0.90).sum() == footprint_mask(table_top, spec).sum()
    path = np.array([[0.2, 1.0], [1.8, 1.0]])  # drives under the table, between the legs
    assert clearance(solids, path, 0.02, 0.40) == pytest.approx(0.275)
    assert clearance(solids, path, 0.02, 0.90) == 0.0
    assert clearance([], path, 0.0, 1.0) == math.inf


def test_grid_frame_round_trip_and_cover() -> None:
    spec = GridSpec.covering((0.0, 0.0, 10.0, 7.03), resolution=0.05, margin=0.1)
    assert spec.shape == (145, 204)
    cells = np.array([[0, 0], [203, 144], [17, 99]])
    np.testing.assert_array_equal(spec.world_to_cell(spec.cell_to_world(cells)), cells)
    np.testing.assert_allclose(spec.cell_to_world([0, 0]), [-0.075, -0.075])


@pytest.mark.parametrize("seed", range(6))
def test_inflation_equals_brute_force_distance_between_cell_squares(seed: int) -> None:
    rng = np.random.default_rng(seed)
    occupied = rng.random((18, 22)) < 0.04
    resolution, radius = 0.05, 0.2
    blocked = inflate(occupied, radius, resolution)
    oy, ox = np.nonzero(occupied)
    for iy in range(occupied.shape[0]):
        for ix in range(occupied.shape[1]):
            gap = np.hypot(np.maximum(np.abs(ox - ix) - 1, 0), np.maximum(np.abs(oy - iy) - 1, 0))
            assert blocked[iy, ix] == (gap.min() * resolution < radius - 1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_any_point_of_an_unblocked_cell_keeps_the_true_clearance(seed: int) -> None:
    rng = np.random.default_rng(200 + seed)
    solids = [random_solid(rng) for _ in range(4)]
    radius = 0.2
    blocked = inflate(rasterize(solids, SPEC, 0.0, 1.0), radius, SPEC.resolution)
    free_y, free_x = np.nonzero(~blocked)
    assert free_x.size > 100
    pick = rng.integers(free_x.size, size=400)
    # Random points anywhere inside free cells, corners included.
    offsets = rng.choice([0.0, 0.5, 1.0], size=(400, 2), p=[0.25, 0.5, 0.25])
    corners = np.column_stack([free_x[pick], free_y[pick]])
    points = np.asarray(SPEC.origin) + (corners + offsets) * SPEC.resolution
    gaps = np.min([segment_distance(s, points, points) for s in solids], axis=0)
    assert gaps.min() >= radius - 1e-9
