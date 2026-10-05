"""Extraction: the closed-form band-clipped footprint against numerical integration."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from splat_navmap.extract import (
    DILATION,
    MIN_CONTRIBUTION,
    SUPERSAMPLE,
    Band,
    band_mass,
    centre_score,
    footprint_score,
)
from splat_navmap.gaussians import GaussianCloud
from splat_navmap.grid import GridSpec

SPEC = GridSpec(origin=(0.0, 0.0), resolution=0.05, shape=(10, 12))
BAND = Band(0.05, 0.40)
IDENTITY = np.array([[1.0, 0.0, 0.0, 0.0]])


def random_cloud(n: int, seed: int) -> GaussianCloud:
    rng = np.random.default_rng(seed)
    quats = rng.standard_normal((n, 4))
    return GaussianCloud(
        means=np.column_stack([rng.uniform(0.0, 0.6, n), rng.uniform(0.0, 0.5, n),
                               rng.uniform(-0.1, 0.55, n)]),
        scales=np.exp(rng.uniform(np.log(0.01), np.log(0.06), (n, 3))),
        quats=quats / np.linalg.norm(quats, axis=1, keepdims=True),
        opacities=rng.uniform(0.05, 0.95, n),
        colors=np.full((n, 3), 0.5),
    )


def single(mean, scales, quat=IDENTITY, opacity=0.9) -> GaussianCloud:
    return GaussianCloud(np.array([mean], dtype=float), np.array([scales], dtype=float),
                         np.asarray(quat, dtype=float), np.array([opacity]), np.full((1, 3), 0.5))


def ray_opacity_by_quadrature(cloud: GaussianCloud, band: Band, x: float, y: float) -> float:
    """Accumulated opacity of one vertical ray, from the 3D density integrated along z.

    Uses only the 3x3 precision matrix: the opacity of a Gaussian along a full ray is
    ``alpha * exp(-0.5 d^T S_xy^-1 d)``, i.e. its line integral divided by ``sqrt(2 pi / P_zz)``.
    """
    transmittance = 1.0
    rotations = Rotation.from_quat(np.roll(cloud.quats, -1, axis=1)).as_matrix()
    z = np.linspace(band.z_lo, band.z_hi, 4001)
    for mean, scale, rot, alpha in zip(cloud.means, cloud.scales, rotations, cloud.opacities,
                                       strict=True):
        precision = rot @ np.diag(1.0 / scale**2) @ rot.T
        q = np.stack([np.full_like(z, x - mean[0]), np.full_like(z, y - mean[1]), z - mean[2]])
        density = np.exp(-0.5 * np.einsum("iz,ij,jz->z", q, precision, q))
        line_integral = np.sum((density[1:] + density[:-1]) / 2) * (z[1] - z[0])
        transmittance *= 1.0 - alpha * line_integral * math.sqrt(precision[2, 2] / (2 * math.pi))
    return 1.0 - transmittance


@pytest.mark.parametrize("seed", range(3))
def test_footprint_equals_quadrature_of_the_3d_density_on_every_ray(seed: int) -> None:
    cloud = random_cloud(25, seed)
    score = footprint_score(cloud, SPEC, BAND, supersample=1, dilation=0.0, cutoff=7.0)
    xs, ys = SPEC.centers()
    # Gaussians with almost no density in the band are skipped by the extractor.
    kept = cloud.subset(cloud.opacities * band_mass(cloud, BAND) > MIN_CONTRIBUTION)
    assert 10 < len(kept) < len(cloud)
    brute = np.array([[ray_opacity_by_quadrature(kept, BAND, x, y) for x in xs] for y in ys])
    np.testing.assert_allclose(score, brute, atol=2e-5)
    assert brute.max() > 0.5
    everything = np.array([[ray_opacity_by_quadrature(cloud, BAND, x, y) for x in xs] for y in ys])
    np.testing.assert_allclose(score, everything, atol=2e-3)


def test_cell_score_is_the_most_opaque_of_its_rays() -> None:
    cloud = random_cloud(25, 7)
    fine = GridSpec(SPEC.origin, SPEC.resolution / 3, (30, 36))
    per_ray = footprint_score(cloud, fine, BAND, supersample=1, dilation=DILATION)
    per_cell = footprint_score(cloud, SPEC, BAND, supersample=3, dilation=DILATION)
    # Same rays and same dilation in metres: only the pooling differs.
    np.testing.assert_allclose(per_cell, per_ray.reshape(10, 3, 12, 3).max(axis=(1, 3)), atol=1e-12)


def test_band_mass_matches_monte_carlo_sampling() -> None:
    cloud = random_cloud(6, 3)
    rng = np.random.default_rng(0)
    rotations = Rotation.from_quat(np.roll(cloud.quats, -1, axis=1)).as_matrix()
    for k in range(len(cloud)):
        local = rng.standard_normal((200_000, 3)) * cloud.scales[k]
        samples = cloud.means[k] + local @ rotations[k].T
        share = np.mean((samples[:, 2] > BAND.z_lo) & (samples[:, 2] < BAND.z_hi))
        assert band_mass(cloud, BAND)[k] == pytest.approx(share, abs=4e-3)


def test_gaussian_outside_the_band_is_ignored_and_inside_keeps_its_opacity() -> None:
    above = single([0.3, 0.25, 1.5], [0.05, 0.05, 0.05])
    inside = single([0.325, 0.275, 0.2], [0.05, 0.05, 0.02], opacity=0.7)
    assert footprint_score(above, SPEC, BAND).max() == 0.0
    assert centre_score(above, SPEC, BAND).max() == 0.0
    # The mean sits on a ray (cell 6, 5 has a ray at its centre for an odd supersampling).
    assert SUPERSAMPLE % 2 == 1
    assert footprint_score(inside, SPEC, BAND)[5, 6] == pytest.approx(0.7, abs=1e-6)
    assert centre_score(inside, SPEC, BAND)[5, 6] == 0.7


def test_centre_score_is_the_largest_in_band_opacity_per_cell() -> None:
    cloud = random_cloud(400, 11)
    expected = np.zeros(SPEC.shape)
    for mean, alpha in zip(cloud.means, cloud.opacities, strict=True):
        ix, iy = int(mean[0] // 0.05), int(mean[1] // 0.05)
        if BAND.z_lo <= mean[2] <= BAND.z_hi and 0 <= ix < 12 and 0 <= iy < 10:
            expected[iy, ix] = max(expected[iy, ix], alpha)
    np.testing.assert_array_equal(centre_score(cloud, SPEC, BAND), expected)


@pytest.mark.parametrize("offset", np.linspace(0.0, 0.05, 11))
def test_thin_opaque_wall_is_seen_wherever_it_falls_between_rays(offset: float) -> None:
    """A wall 2 mm thick, much thinner than the ray spacing, never slips through."""
    alpha = 0.95
    wall = GaussianCloud(
        means=np.column_stack([np.full(12, 0.2 + offset), np.linspace(0.0, 0.5, 12),
                               np.full(12, 0.2)]),
        scales=np.tile([0.001, 0.03, 0.03], (12, 1)), quats=np.tile(IDENTITY, (12, 1)),
        opacities=np.full(12, alpha), colors=np.full((12, 3), 0.5))
    score = footprint_score(wall, SPEC, BAND)
    column = int((0.2 + offset) // 0.05) if offset < 0.05 else 4
    hit = np.maximum(score[2:8, column], score[2:8, min(column + 1, 11)])
    assert hit.min() >= math.exp(-0.5) * alpha
    assert centre_score(wall, SPEC, BAND).max() == alpha


def test_widening_the_band_never_lowers_a_score() -> None:
    cloud = random_cloud(60, 5)
    narrow = footprint_score(cloud, SPEC, Band(0.15, 0.25))
    wide = footprint_score(cloud, SPEC, Band(0.05, 0.40))
    assert np.all(wide >= narrow - 1e-12)
    assert np.all(centre_score(cloud, SPEC, Band(0.05, 0.40))
                  >= centre_score(cloud, SPEC, Band(0.15, 0.25)))


def test_shifting_the_scene_by_whole_cells_shifts_the_map() -> None:
    cloud = random_cloud(40, 9)
    shift = np.array([2 * 0.05, 1 * 0.05, 0.0])
    moved = GaussianCloud(cloud.means + shift, cloud.scales, cloud.quats, cloud.opacities,
                          cloud.colors)
    wide = GridSpec((-0.5, -0.5), 0.05, (30, 34))
    for extractor in (centre_score, footprint_score):
        before, after = extractor(cloud, wide, BAND), extractor(moved, wide, BAND)
        np.testing.assert_allclose(after[1:, 2:], before[:-1, :-2], atol=1e-9)


def test_tilted_gaussian_is_clipped_where_it_leaves_the_band() -> None:
    """A long needle tilted 45 degrees crosses the band only over part of its footprint."""
    tilt = Rotation.from_euler("y", -45, degrees=True).as_quat()
    needle = single([0.3, 0.275, 0.3], [0.15, 0.005, 0.005], quat=[np.roll(tilt, 1)], opacity=0.9)
    whole = footprint_score(needle, SPEC, Band(-5.0, 5.0))[5]
    clipped = footprint_score(needle, SPEC, Band(0.25, 0.35))[5]
    assert whole[3] > 0.4 and whole[8] > 0.4  # one sigma away, the full footprint is opaque
    assert clipped[6] > 0.8  # the middle, at height 0.3, is inside the band
    assert clipped[3] < 0.01 and clipped[8] < 0.01  # there the needle is below / above the band
