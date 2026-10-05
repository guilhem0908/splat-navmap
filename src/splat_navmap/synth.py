"""Turn an analytic scene into a Gaussian "reconstruction" with controllable defects.

Every visible face is covered with flat, surface-aligned Gaussians (surfels): two tangential
standard deviations proportional to the sampling step and a small one along the normal. This is
a synthetic stand-in for a trained scene, not a trained scene. The defects reproduce, with
explicit knobs, four ways a real reconstruction departs from the geometry:

* **floaters**: semi-transparent blobs in free space;
* **holes**: balls in which nothing was reconstructed (regions no camera saw);
* **weak surfaces**: spatially coherent patches whose Gaussians have low opacity;
* **thickness**: surfaces that are fuzzy along their normal. The density profile across a
  surface has standard deviation ``t``; half of that variance comes from the extent of each
  Gaussian along the normal and half from scattering the centres, so neither a centre-based
  nor an extent-based extractor is favoured by construction.

Each defect draws from its own random stream, so switching one on or off leaves the others
(and the underlying surface sample) unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from splat_navmap.gaussians import GaussianCloud, rotmat_to_quat
from splat_navmap.geometry import Cylinder, Solid, rasterize
from splat_navmap.scene import WALL_HEIGHT, Scene

DEFAULT_DENSITY = 400.0  # surfels per square metre
TANGENT_SCALE = 0.5  # tangential std as a fraction of the sampling step
JITTER = 0.3  # stratified jitter, as a fraction of the sampling step
SURFACE_OPACITY = (0.90, 0.99)
WEAK_OPACITY = (0.05, 0.45)
WEAK_LENGTH = 0.8  # correlation length of the weak-surface field, metres
WEAK_FEATURES = 32
FLOATER_STD = (0.02, 0.12)  # range of the per-axis std of a floater, metres
FLOATER_OPACITY_BETA = (1.2, 3.5)
FLOATER_OPACITY_RANGE = (0.02, 0.62)
HOLE_RADIUS = (0.15, 0.45)
HOLE_MAX_HEIGHT = 1.0  # holes are centred on surface points below this height
LIGHT = np.array([0.35, 0.25, 0.90]) / np.linalg.norm([0.35, 0.25, 0.90])
AMBIENT, DIFFUSE = 0.55, 0.45
CONTACT_SHADOW = (0.55, 0.30)  # darkest factor, and distance over which it fades, metres
WALL_CUT_COLOR = (0.22, 0.23, 0.26)
PLANK = (1.2, 0.12)  # floor plank length and width, metres


@dataclass(frozen=True)
class Defects:
    """Defect magnitudes. The default is a clean reconstruction."""

    floater_density: float = 0.0  # floaters per cubic metre of the building
    hole_density: float = 0.0  # unobserved balls per square metre of floor
    weak_fraction: float = 0.0  # share of the surfaces reconstructed with low opacity
    thickness: float = 0.004  # std of the density profile across a surface, metres


DEFECT_LEVELS = {
    "clean": Defects(),
    "mild": Defects(floater_density=1.0, hole_density=0.03, weak_fraction=0.05, thickness=0.008),
    "moderate": Defects(floater_density=3.0, hole_density=0.08, weak_fraction=0.15,
                        thickness=0.015),
    "severe": Defects(floater_density=8.0, hole_density=0.15, weak_fraction=0.30,
                      thickness=0.025),
}


def reconstruct(scene: Scene, seed: int, defects: Defects | None = None,
                density: float = DEFAULT_DENSITY) -> GaussianCloud:
    """Sample the scene's surfaces into Gaussians and apply the requested defects."""
    defects = defects or Defects()
    rng = np.random.default_rng([seed, 0])
    step = 1.0 / math.sqrt(density)
    faces = [_sample_face(rng, face, step) for solid in (scene.floor, *scene.solids)
             for face in _faces(solid)]
    points, frames, spacing, albedo = (np.concatenate(x) for x in zip(*faces, strict=True))
    normals = frames[:, :, 2]
    n = len(points)

    across = defects.thickness / math.sqrt(2.0)  # equal split between scatter and extent
    means = points + (across * rng.standard_normal(n))[:, None] * normals
    tangential = TANGENT_SCALE * spacing[:, None] * np.exp(0.15 * rng.standard_normal((n, 2)))
    scales = np.column_stack([tangential, np.full(n, across)])
    opacities = rng.uniform(*SURFACE_OPACITY, size=n)
    colors = _shade(scene, points, normals, albedo, rng)

    if defects.weak_fraction > 0:
        weak_rng = np.random.default_rng([seed, 1])
        field = _smooth_field(weak_rng, points)
        weak = field < np.quantile(field, defects.weak_fraction)
        opacities = np.where(weak, weak_rng.uniform(*WEAK_OPACITY, size=n), opacities)
    surface = GaussianCloud(means, scales, rotmat_to_quat(frames), opacities, colors)

    xmin, ymin, xmax, ymax = scene.bounds
    if defects.hole_density > 0:
        hole_rng = np.random.default_rng([seed, 2])
        low = np.flatnonzero((points[:, 2] > 0.0) & (points[:, 2] < HOLE_MAX_HEIGHT))
        count = round(defects.hole_density * (xmax - xmin) * (ymax - ymin))
        centres = points[hole_rng.choice(low, size=count, replace=False)]
        radii = hole_rng.uniform(*HOLE_RADIUS, size=count)
        gaps = np.linalg.norm(means[:, None, :] - centres[None], axis=2)
        surface = surface.subset(~(gaps < radii).any(axis=1))

    clouds = [surface]
    if defects.floater_density > 0:
        float_rng = np.random.default_rng([seed, 3])
        count = round(defects.floater_density * (xmax - xmin) * (ymax - ymin) * WALL_HEIGHT)
        clouds.append(_floaters(float_rng, scene, count))
    return GaussianCloud.concatenate(clouds)


def _faces(solid: Solid) -> list[tuple]:
    """Describe the visible faces of a solid; faces resting on the floor are skipped."""
    lifted = solid.z0 > 0.02
    if isinstance(solid, Cylinder):
        faces = [("side", solid), ("cap", solid, solid.z1, 1.0)]
        return [*faces, ("cap", solid, solid.z0, -1.0)] if lifted else faces
    u, v = (np.append(axis, 0.0) for axis in solid.axes())
    up = np.array([0.0, 0.0, 1.0])
    centre = np.array([solid.cx, solid.cy, (solid.z0 + solid.z1) / 2])
    half_z = (solid.z1 - solid.z0) / 2
    faces = [("plane", solid, centre + half_z * up, u, solid.hx, solid.hy, up)]
    if solid.label == "floor":
        return faces
    if lifted:
        faces.append(("plane", solid, centre - half_z * up, v, solid.hy, solid.hx, -up))
    for axis, half, half_other in ((u, solid.hx, solid.hy), (v, solid.hy, solid.hx)):
        for sign in (1.0, -1.0):
            normal = sign * axis
            tangent = np.cross(up, normal)
            faces.append(("plane", solid, centre + half * normal, tangent, half_other, half_z,
                          normal))
    return faces


def _lattice(rng: np.random.Generator, half_a: float, half_b: float, step: float,
             ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Jittered lattice on ``[-half_a, half_a] x [-half_b, half_b]`` with at least 2 x 2 points."""
    na = max(2, math.ceil(2 * half_a / step))
    nb = max(2, math.ceil(2 * half_b / step))
    da, db = 2 * half_a / na, 2 * half_b / nb
    ia, ib = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
    a = (ia.ravel() + 0.5 + rng.uniform(-JITTER, JITTER, na * nb)) * da - half_a
    b = (ib.ravel() + 0.5 + rng.uniform(-JITTER, JITTER, na * nb)) * db - half_b
    return a, b, da, db


def _sample_face(rng: np.random.Generator, face: tuple, step: float,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return surfel positions (K, 3), frames (K, 3, 3), sampling step (K,), albedo (K, 3)."""
    kind, solid = face[0], face[1]
    up = np.array([0.0, 0.0, 1.0])
    if kind == "plane":
        _, _, centre, tangent, half_a, half_b, normal = face
        a, b, da, db = _lattice(rng, half_a, half_b, step)
        bitangent = np.cross(normal, tangent)
        points = centre + a[:, None] * tangent + b[:, None] * bitangent
        normals = np.broadcast_to(normal, points.shape)
        tangents = np.broadcast_to(tangent, points.shape)
    elif kind == "side":
        half_z = (solid.z1 - solid.z0) / 2
        a, b, da, db = _lattice(rng, math.pi * solid.radius, half_z, step)
        theta = a / solid.radius
        normals = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        tangents = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
        points = (np.array([solid.cx, solid.cy, solid.z0 + half_z]) + solid.radius * normals
                  + b[:, None] * up)
    else:  # disc cap
        _, _, z, sign = face
        a, b, da, db = _lattice(rng, solid.radius, solid.radius, step)
        keep = a**2 + b**2 <= solid.radius**2
        a, b = a[keep], b[keep]
        points = np.column_stack([solid.cx + a, solid.cy + sign * b, np.full(a.shape, z)])
        normals = np.broadcast_to(sign * up, points.shape)
        tangents = np.broadcast_to(np.array([1.0, 0.0, 0.0]), points.shape)
    # Random in-plane orientation: frame columns are (t1, t2, n) with t1 x t2 = n.
    phi = rng.uniform(0.0, math.pi, len(points))
    bitangents = np.cross(normals, tangents)
    t1 = np.cos(phi)[:, None] * tangents + np.sin(phi)[:, None] * bitangents
    frames = np.stack([t1, np.cross(normals, t1), normals], axis=2)
    spacing = np.full(len(points), min(da, db))
    is_wall_cut = solid.label == "wall" and kind == "plane" and face[6][2] > 0.5
    albedo = np.broadcast_to(WALL_CUT_COLOR if is_wall_cut else solid.color, points.shape)
    if solid.label == "floor":
        albedo = albedo * _plank_tone(points)[:, None]
    return points, frames, spacing, albedo


def _plank_tone(points: np.ndarray) -> np.ndarray:
    """Per-plank brightness in [0.82, 1.12] from a hash of the plank index."""
    row = np.floor(points[:, 1] / PLANK[1])
    column = np.floor((points[:, 0] + 0.37 * PLANK[0] * (row % 3)) / PLANK[0])
    noise = np.modf(np.abs(np.sin(row * 12.9898 + column * 78.233)) * 43758.5453)[0]
    return 0.82 + 0.30 * noise


def _shade(scene: Scene, points: np.ndarray, normals: np.ndarray, albedo: np.ndarray,
           rng: np.random.Generator) -> np.ndarray:
    """Lambert shading with a soft contact shadow on the floor and at the foot of the walls."""
    light = AMBIENT + DIFFUSE * np.clip(normals @ LIGHT, 0.0, None)
    spec = scene.grid()
    darkest, fade = CONTACT_SHADOW
    low_solids = rasterize(scene.solids, spec, 0.02, 1.0)
    distance = ndimage.distance_transform_edt(~low_solids) * spec.resolution
    cells = np.clip(spec.world_to_cell(points[:, :2]), 0, np.array(spec.shape[::-1]) - 1)
    on_floor = normals[:, 2] > 0.5
    near = np.where(on_floor & (points[:, 2] < 0.02), distance[cells[:, 1], cells[:, 0]],
                    np.where(on_floor, fade, points[:, 2]))
    shadow = darkest + (1.0 - darkest) * np.clip(near / fade, 0.0, 1.0) ** 0.7
    grain = 1.0 + 0.03 * rng.standard_normal((len(points), 1))
    return np.clip(albedo * (light * shadow)[:, None] * grain, 0.0, 1.0)


def _smooth_field(rng: np.random.Generator, points: np.ndarray) -> np.ndarray:
    """Random Fourier features of a squared-exponential kernel (Rahimi & Recht, 2007)."""
    frequencies = rng.standard_normal((WEAK_FEATURES, 3)) / WEAK_LENGTH
    phases = rng.uniform(0.0, 2 * math.pi, WEAK_FEATURES)
    return np.cos(points @ frequencies.T + phases).sum(axis=1)


def _floaters(rng: np.random.Generator, scene: Scene, count: int) -> GaussianCloud:
    """Random semi-transparent anisotropic blobs anywhere inside the building."""
    xmin, ymin, xmax, ymax = scene.bounds
    means = rng.uniform([xmin, ymin, 0.0], [xmax, ymax, WALL_HEIGHT], size=(count, 3))
    scales = np.exp(rng.uniform(*np.log(FLOATER_STD), size=(count, 3)))
    quats = rng.standard_normal((count, 4))
    lo, hi = FLOATER_OPACITY_RANGE
    opacities = lo + (hi - lo) * rng.beta(*FLOATER_OPACITY_BETA, size=count)
    colors = np.clip(0.75 + 0.12 * rng.standard_normal((count, 1))
                     + 0.04 * rng.standard_normal((count, 3)), 0.0, 1.0)
    return GaussianCloud(means, scales, quats / np.linalg.norm(quats, axis=1, keepdims=True),
                         opacities, colors)

