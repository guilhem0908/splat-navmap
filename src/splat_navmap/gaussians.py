"""Container and algebra for a set of 3D Gaussians.

A 3D Gaussian Splatting primitive is an unnormalised Gaussian
``exp(-0.5 (p - mu)^T Sigma^-1 (p - mu))`` scaled by an opacity ``alpha`` in [0, 1].
Its covariance is stored factorised as ``Sigma = R S S^T R^T`` with a rotation ``R``
(unit quaternion, scalar first) and per-axis standard deviations ``S = diag(s)``
(Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", 2023, eq. 6).

Everything here is expressed in *activated* units: scales in metres, opacities in [0, 1],
colours in [0, 1]. The log / logit / spherical-harmonics encodings of the on-disk format live
in :mod:`splat_navmap.ply_io`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class GaussianCloud:
    """``N`` Gaussians in activated units.

    ``means`` (N, 3) and ``scales`` (N, 3) in metres, ``quats`` (N, 4) as (w, x, y, z),
    ``opacities`` (N,) and ``colors`` (N, 3) in [0, 1].
    """

    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray

    def __post_init__(self) -> None:
        n = self.means.shape[0]
        expected = {"means": (n, 3), "scales": (n, 3), "quats": (n, 4), "opacities": (n,),
                    "colors": (n, 3)}
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")

    def __len__(self) -> int:
        return self.means.shape[0]

    def covariances(self) -> np.ndarray:
        """Return the (N, 3, 3) covariances ``R diag(s^2) R^T``."""
        rot = quat_to_rotmat(self.quats)
        return np.einsum("nij,nj,nkj->nik", rot, self.scales**2, rot)

    def subset(self, mask: np.ndarray) -> GaussianCloud:
        """Return the Gaussians selected by a boolean mask or an index array."""
        return GaussianCloud(self.means[mask], self.scales[mask], self.quats[mask],
                             self.opacities[mask], self.colors[mask])

    @staticmethod
    def concatenate(clouds: Sequence[GaussianCloud]) -> GaussianCloud:
        """Stack several clouds into one."""
        return GaussianCloud(*(np.concatenate([getattr(c, name) for c in clouds])
                               for name in ("means", "scales", "quats", "opacities", "colors")))


def quat_to_rotmat(quats: np.ndarray) -> np.ndarray:
    """Convert (N, 4) quaternions (w, x, y, z), normalised here, to (N, 3, 3) rotations."""
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def rotmat_to_quat(rotations: np.ndarray) -> np.ndarray:
    """Convert (N, 3, 3) proper rotations to (N, 4) unit quaternions (w, x, y, z) with w >= 0."""
    xyzw = Rotation.from_matrix(rotations).as_quat()
    wxyz = np.roll(xyzw, 1, axis=1)
    return wxyz * np.where(wxyz[:, :1] < 0, -1.0, 1.0)
