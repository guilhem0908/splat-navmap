"""From Gaussians to an occupancy score per cell, for a robot-height band ``[z_lo, z_hi]``.

Both extractors return a score in [0, 1] per cell; the occupancy grid is ``score >= tau``.

**centres** treats the scene as a point cloud: the score of a cell is the largest opacity among
the Gaussians whose *centre* lies in the cell and in the band. ``score >= tau`` is then
"at least one centre with opacity >= tau", the usual first attempt.

**footprint** renders the band from above with the 3DGS image-formation model. Along the
vertical ray through ``(x, y)``, a Gaussian with opacity ``alpha`` contributes

    a(x, y) = alpha * exp(-0.5 d^T S_xy^-1 d) * w(x, y),          d = (x, y) - mu_xy

where ``S_xy`` is the 2x2 marginal covariance (the orthographic EWA splat of Zwicker et al.,
2001) and ``w`` is the share of the ray's density that falls inside the band. Conditioning the
3D Gaussian on ``(x, y)`` (Schur complement) gives a 1D Gaussian in z with

    m(x, y) = mu_z + S_z,xy S_xy^-1 d,      s^2 = S_zz - S_z,xy S_xy^-1 S_xy,z

so ``w = Phi((z_hi - m) / s) - Phi((z_lo - m) / s)``, exactly. Rays are cast on a lattice
``SUPERSAMPLE`` times finer than the grid and accumulate opacity as ``1 - prod_i (1 - a_i)``
(order-independent, since only the total is needed). A cell takes the value of its most opaque
ray: it is occupied when some vertical ray through it is at least ``tau`` opaque. So that a
surface thinner than the ray spacing cannot fall between two rays, every footprint is widened
by half that spacing (``S_xy + (step / 2)^2 I``), the same low-pass dilation the reference
rasteriser applies in screen space; a thin wall midway between two rays keeps
``exp(-1/2) = 61 %`` of its opacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from splat_navmap.gaussians import GaussianCloud
from splat_navmap.grid import GridSpec

SUPERSAMPLE = 3  # vertical rays per cell side
DILATION = 0.5  # std added to every footprint, as a fraction of the ray spacing
CUTOFF_SIGMA = 3.0  # footprints are evaluated up to this many standard deviations
MAX_PATCH_SAMPLES = 48  # half-size limit of a footprint patch, in rays
MIN_CONTRIBUTION = 1e-3  # Gaussians that cannot add more than this to any score are skipped
_CHUNK_SAMPLES = 250_000  # bound on the temporaries of one vectorised batch (about 2 MB each)
_ALPHA_MAX = 1.0 - 1e-6  # keeps log(1 - a) finite
_PHI_CLIP = 8.0  # Phi(+-8) equals 0 or 1 to 1e-15; clipping avoids the slow tails of ndtr


@dataclass(frozen=True)
class Band:
    """Height interval ``[z_lo, z_hi]`` in metres, measured from the floor."""

    z_lo: float
    z_hi: float


def band_mass(cloud: GaussianCloud, band: Band) -> np.ndarray:
    """Share of each Gaussian's density inside the band, from its marginal ``N(mu_z, S_zz)``."""
    sigma_z = np.sqrt(cloud.covariances()[:, 2, 2])
    return ndtr((band.z_hi - cloud.means[:, 2]) / sigma_z) - ndtr(
        (band.z_lo - cloud.means[:, 2]) / sigma_z)


def centre_score(cloud: GaussianCloud, spec: GridSpec, band: Band) -> np.ndarray:
    """Per cell: the largest opacity among the Gaussian centres inside the cell and the band."""
    ny, nx = spec.shape
    cells = spec.world_to_cell(cloud.means[:, :2])
    keep = ((cloud.means[:, 2] >= band.z_lo) & (cloud.means[:, 2] <= band.z_hi)
            & (cells[:, 0] >= 0) & (cells[:, 0] < nx) & (cells[:, 1] >= 0) & (cells[:, 1] < ny))
    score = np.zeros(ny * nx)
    np.maximum.at(score, cells[keep, 1] * nx + cells[keep, 0], cloud.opacities[keep])
    return score.reshape(ny, nx)


def footprint_score(cloud: GaussianCloud, spec: GridSpec, band: Band, *,
                    supersample: int = SUPERSAMPLE, dilation: float = DILATION,
                    cutoff: float = CUTOFF_SIGMA) -> np.ndarray:
    """Per cell: largest accumulated opacity of the band seen from above (module docstring)."""
    ny, nx = spec.shape
    rows, columns = ny * supersample, nx * supersample
    step = spec.resolution / supersample
    keep = cloud.opacities * band_mass(cloud, band) > MIN_CONTRIBUTION
    means, alpha, cov = cloud.means[keep], cloud.opacities[keep], cloud.covariances()[keep]

    cov_xy = cov[:, :2, :2]
    filtered = cov_xy + (dilation * step) ** 2 * np.eye(2)
    precision = np.linalg.inv(filtered)
    slope = np.einsum("ni,nij->nj", cov[:, 2, :2], np.linalg.inv(cov_xy))
    sigma_cond = np.sqrt(np.maximum(cov[:, 2, 2] - (slope * cov[:, 2, :2]).sum(axis=1), 1e-12))

    radius = cutoff * np.sqrt(np.linalg.eigvalsh(filtered)[:, 1])
    half_width = np.minimum(np.ceil(radius / step).astype(int), MAX_PATCH_SAMPLES)
    centre = np.floor((means[:, :2] - spec.origin) / step).astype(int)  # sample index (ix, iy)

    log_transmittance = np.zeros(rows * columns)
    for half in np.unique(half_width):
        members = np.flatnonzero(half_width == half)
        offsets = np.arange(-half, half + 1)
        chunk = max(1, _CHUNK_SAMPLES // offsets.size**2)
        for start in range(0, members.size, chunk):
            idx = members[start : start + chunk]
            ix = centre[idx, 0, None] + offsets  # (n, side) sample indices along x
            iy = centre[idx, 1, None] + offsets
            dx = (spec.origin[0] + (ix + 0.5) * step - means[idx, 0, None])[:, None, :]
            dy = (spec.origin[1] + (iy + 0.5) * step - means[idx, 1, None])[:, :, None]
            p = precision[idx]
            maha = (p[:, 0, 0, None, None] * dx * dx + 2 * p[:, 0, 1, None, None] * dx * dy
                    + p[:, 1, 1, None, None] * dy * dy)
            z_mean = (means[idx, 2, None, None] + slope[idx, 0, None, None] * dx
                      + slope[idx, 1, None, None] * dy)
            s = sigma_cond[idx, None, None]
            upper = np.clip((band.z_hi - z_mean) / s, -_PHI_CLIP, _PHI_CLIP)
            lower = np.clip((band.z_lo - z_mean) / s, -_PHI_CLIP, _PHI_CLIP)
            a = alpha[idx, None, None] * np.exp(-0.5 * maha) * (ndtr(upper) - ndtr(lower))
            gx, gy = np.broadcast_arrays(ix[:, None, :], iy[:, :, None])
            valid = (gx >= 0) & (gx < columns) & (gy >= 0) & (gy < rows)
            log_transmittance += np.bincount(
                (gy * columns + gx)[valid], weights=np.log1p(-np.minimum(a[valid], _ALPHA_MAX)),
                minlength=rows * columns)
    opacity = 1.0 - np.exp(log_transmittance)
    return opacity.reshape(ny, supersample, nx, supersample).max(axis=(1, 3))


EXTRACTORS = {"centres": centre_score, "footprint": footprint_score}
