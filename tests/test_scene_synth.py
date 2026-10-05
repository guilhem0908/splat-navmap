"""Procedural scenes and their Gaussian reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from splat_navmap.gaussians import quat_to_rotmat
from splat_navmap.geometry import Box, Cylinder
from splat_navmap.grid import inflate, reachable_components
from splat_navmap.scene import WALL_HEIGHT, Robot, Scene, generate_scene
from splat_navmap.synth import (
    DEFECT_LEVELS,
    SURFACE_OPACITY,
    WEAK_OPACITY,
    Defects,
    reconstruct,
)

ROBOT = Robot()


def one_box_scene() -> Scene:
    """A 1.0 x 0.6 x 0.5 m box rotated by 30 degrees on a 4 x 4 m floor."""
    floor = Box(2.0, 2.0, 2.0, 2.0, -0.05, 0.0, label="floor")
    box = Box(2.0, 2.0, 0.5, 0.3, 0.0, 0.5, yaw=np.pi / 6)
    return Scene((0.0, 0.0, 4.0, 4.0), floor, (box,), ((0.0, 0.0, 4.0, 4.0),))


def test_same_seed_gives_the_same_scene_and_different_seeds_differ() -> None:
    assert generate_scene(5) == generate_scene(5)
    assert generate_scene(5) != generate_scene(6)


@pytest.mark.parametrize("seed", range(6))
def test_generated_flat_is_closed_furnished_and_connected(seed: int) -> None:
    scene = generate_scene(seed)
    spec = scene.grid()
    truth = scene.truth(spec, ROBOT)
    assert 4 <= len(scene.rooms) <= 6
    labels = {solid.label for solid in scene.solids}
    assert {"wall", "rug"} <= labels and len(labels) >= 5
    # The outer walls close the flat: its border cells are all occupied.
    ring = np.zeros_like(truth)
    ring[2, 2:-2] = ring[-3, 2:-2] = ring[2:-2, 2] = ring[2:-2, -3] = True
    assert truth[ring].all()
    sizes = np.bincount(reachable_components(
        inflate(truth, ROBOT.radius, spec.resolution)).ravel())[1:]
    assert sizes.max() >= 0.97 * sizes.sum()
    # Rooms tile the interior exactly.
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in scene.rooms)
    width, depth = scene.floor.hx * 2 - 0.32, scene.floor.hy * 2 - 0.32
    assert area == pytest.approx(width * depth)


def test_truth_ignores_what_the_robot_passes_under_or_over() -> None:
    scene = generate_scene(0)
    spec = scene.grid()
    truth = scene.truth(spec, ROBOT)
    tall = scene.truth(spec, Robot(height=0.9))
    kinds = {"under": [s for s in scene.solids if s.z0 >= ROBOT.height],
             "over": [s for s in scene.solids if s.z1 <= ROBOT.ground_clearance]}
    assert kinds["under"] and kinds["over"]
    assert tall.sum() > truth.sum()
    assert not (truth & ~tall).any()


def test_clean_surfels_lie_on_the_surfaces_with_their_thin_axis_along_the_normal() -> None:
    scene = one_box_scene()
    cloud = reconstruct(scene, seed=1)
    box = scene.solids[0]
    (ux, uy), (vx, vy) = box.axes()
    rel = cloud.means[:, :2] - [box.cx, box.cy]
    local = np.column_stack([rel @ [ux, uy], rel @ [vx, vy], cloud.means[:, 2]])
    # The third rotation column is the surface normal and carries the smallest scale.
    normals = quat_to_rotmat(cloud.quats)[:, :, 2]
    assert np.all(cloud.scales[:, 2] < cloud.scales[:, :2].min(axis=1))
    horizontal = np.abs(normals[:, 2]) > 1 - 1e-9
    facing_u = np.abs(normals @ [ux, uy, 0.0]) > 1 - 1e-9
    facing_v = np.abs(normals @ [vx, vy, 0.0]) > 1 - 1e-9
    assert (horizontal | facing_u | facing_v).all()
    assert min(horizontal.sum(), facing_u.sum(), facing_v.sum()) > 100
    tol = 0.02  # a few times the clean thickness
    heights = local[horizontal, 2]
    assert np.all(np.minimum(np.abs(heights), np.abs(heights - box.z1)) < tol)
    assert np.all(np.abs(np.abs(local[facing_u, 0]) - box.hx) < tol)
    assert np.all(np.abs(local[facing_u, 1]) < box.hy + tol)
    assert np.all(np.abs(np.abs(local[facing_v, 1]) - box.hy) < tol)
    assert np.all((local[~horizontal, 2] > -tol) & (local[~horizontal, 2] < box.z1 + tol))
    assert cloud.opacities.min() >= SURFACE_OPACITY[0]
    assert cloud.colors.min() >= 0.0 and cloud.colors.max() <= 1.0


def test_surface_density_follows_the_requested_density() -> None:
    scene = one_box_scene()
    box = scene.solids[0]
    area = 16.0 + 4 * box.hx * box.hy + 2 * (2 * box.hx + 2 * box.hy) * box.z1
    for density in (100.0, 400.0):
        assert len(reconstruct(scene, 0, density=density)) == pytest.approx(density * area, rel=0.1)


def test_cylinder_surfels_lie_on_the_cylinder() -> None:
    floor = Box(1.0, 1.0, 1.0, 1.0, -0.05, 0.0, label="floor")
    pot = Cylinder(1.0, 1.0, 0.3, 0.0, 0.6)
    cloud = reconstruct(Scene((0, 0, 2, 2), floor, (pot,), ((0, 0, 2, 2),)), seed=0)
    raised = cloud.subset(cloud.means[:, 2] > 0.05)
    radial = np.linalg.norm(raised.means[:, :2] - 1.0, axis=1)
    on_side = np.abs(radial - 0.3) < 0.02
    on_cap = (np.abs(raised.means[:, 2] - 0.6) < 0.02) & (radial < 0.3 + 0.02)
    assert (on_side | on_cap).all() and on_side.sum() > 100 and on_cap.sum() > 50


def test_thickness_sets_the_spread_of_the_density_across_the_surface() -> None:
    scene = one_box_scene()
    for thickness in (0.01, 0.03):
        cloud = reconstruct(scene, 2, Defects(thickness=thickness))
        floor = cloud.subset((np.abs(cloud.means[:, 2]) < 0.15)
                             & (np.linalg.norm(cloud.means[:, :2] - 2.0, axis=1) > 1.0))
        scatter = floor.means[:, 2].std()
        extent = floor.scales[:, 2]
        np.testing.assert_allclose(extent, thickness / np.sqrt(2))
        assert scatter == pytest.approx(thickness / np.sqrt(2), rel=0.05)
        assert np.sqrt(scatter**2 + extent[0] ** 2) == pytest.approx(thickness, rel=0.05)


def test_each_defect_has_its_own_random_stream() -> None:
    scene = generate_scene(1)
    clean = reconstruct(scene, 1)
    severe = DEFECT_LEVELS["severe"]

    floaters = reconstruct(scene, 1, Defects(floater_density=severe.floater_density))
    extra = len(floaters) - len(clean)
    xmin, ymin, xmax, ymax = scene.bounds
    assert extra == round(severe.floater_density * (xmax - xmin) * (ymax - ymin) * WALL_HEIGHT)
    np.testing.assert_array_equal(floaters.means[: len(clean)], clean.means)
    assert floaters.opacities[len(clean):].max() < 0.62
    assert 0.1 < floaters.opacities[len(clean):].mean() < 0.25

    weak = reconstruct(scene, 1, Defects(weak_fraction=severe.weak_fraction))
    np.testing.assert_array_equal(weak.means, clean.means)
    lowered = weak.opacities < SURFACE_OPACITY[0]
    assert lowered.mean() == pytest.approx(severe.weak_fraction, abs=1e-3)
    assert weak.opacities[lowered].max() <= WEAK_OPACITY[1]
    np.testing.assert_array_equal(weak.opacities[~lowered], clean.opacities[~lowered])

    holes = reconstruct(scene, 1, Defects(hole_density=severe.hole_density))
    assert 0.9 * len(clean) < len(holes) < len(clean)
    kept = np.isin(clean.means.view([("", clean.means.dtype)] * 3).ravel(),
                   holes.means.view([("", holes.means.dtype)] * 3).ravel())
    assert kept.sum() == len(holes)

    everything = reconstruct(scene, 1, severe)
    np.testing.assert_array_equal(everything.means[-extra:], floaters.means[-extra:])


def test_reconstruction_is_deterministic() -> None:
    scene = generate_scene(2)
    first = reconstruct(scene, 2, DEFECT_LEVELS["moderate"], density=100.0)
    second = reconstruct(scene, 2, DEFECT_LEVELS["moderate"], density=100.0)
    for name in ("means", "scales", "quats", "opacities", "colors"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
