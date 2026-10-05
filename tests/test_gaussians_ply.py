"""Gaussian algebra and the INRIA .ply encoding."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from splat_navmap.gaussians import GaussianCloud, quat_to_rotmat, rotmat_to_quat
from splat_navmap.ply_io import SH_C0, read_ply, write_ply

INRIA_FIELDS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def random_cloud(n: int, seed: int = 0) -> GaussianCloud:
    rng = np.random.default_rng(seed)
    quats = rng.standard_normal((n, 4))
    return GaussianCloud(
        means=rng.uniform(-5, 5, (n, 3)),
        scales=np.exp(rng.uniform(np.log(0.005), np.log(0.3), (n, 3))),
        quats=quats / np.linalg.norm(quats, axis=1, keepdims=True),
        opacities=rng.uniform(0.01, 0.99, n),
        colors=rng.uniform(0.0, 1.0, (n, 3)),
    )


def test_quaternion_to_matrix_matches_scipy_convention() -> None:
    quats = np.random.default_rng(1).standard_normal((200, 4))
    reference = Rotation.from_quat(np.roll(quats, -1, axis=1)).as_matrix()  # scipy is (x, y, z, w)
    np.testing.assert_allclose(quat_to_rotmat(quats), reference, atol=1e-12)


def test_matrix_to_quaternion_round_trip_keeps_the_rotation() -> None:
    rotations = quat_to_rotmat(np.random.default_rng(2).standard_normal((200, 4)))
    quats = rotmat_to_quat(rotations)
    assert np.all(quats[:, 0] >= 0)
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(quat_to_rotmat(quats), rotations, atol=1e-12)


def test_covariance_has_the_squared_scales_as_eigenvalues() -> None:
    cloud = random_cloud(100)
    cov = cloud.covariances()
    np.testing.assert_allclose(cov, np.swapaxes(cov, 1, 2), atol=1e-15)
    np.testing.assert_allclose(np.linalg.eigvalsh(cov), np.sort(cloud.scales**2, axis=1),
                               rtol=1e-9)
    # The first rotation column is the direction of the first scale: Sigma r_0 = s_0^2 r_0.
    r0 = quat_to_rotmat(cloud.quats)[:, :, 0]
    np.testing.assert_allclose(np.einsum("nij,nj->ni", cov, r0), cloud.scales[:, :1] ** 2 * r0,
                               atol=1e-12)


def test_ply_round_trip_preserves_every_attribute_to_float32_precision(tmp_path) -> None:
    cloud = random_cloud(500)
    write_ply(tmp_path / "scene.ply", cloud)
    loaded = read_ply(tmp_path / "scene.ply")
    np.testing.assert_allclose(loaded.means, cloud.means, atol=1e-6)
    np.testing.assert_allclose(loaded.scales, cloud.scales, rtol=1e-6)
    np.testing.assert_allclose(loaded.opacities, cloud.opacities, atol=1e-6)
    np.testing.assert_allclose(loaded.colors, cloud.colors, atol=1e-6)
    np.testing.assert_allclose(loaded.covariances(), cloud.covariances(), atol=1e-7)


def test_ply_file_uses_the_inria_layout_and_encodings(tmp_path) -> None:
    cloud = random_cloud(50)
    path = tmp_path / "scene.ply"
    write_ply(path, cloud)
    raw = path.read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    lines = header.decode("ascii").splitlines()
    assert lines[:3] == ["ply", "format binary_little_endian 1.0", "element vertex 50"]
    assert [line.split()[2] for line in lines[3:]] == INRIA_FIELDS
    assert all(line.split()[1] == "float" for line in lines[3:])
    table = np.frombuffer(body, dtype="<f4").reshape(50, len(INRIA_FIELDS)).astype(np.float64)
    np.testing.assert_allclose(table[:, 9], np.log(cloud.opacities / (1 - cloud.opacities)),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(table[:, 10:13], np.log(cloud.scales), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(0.5 + SH_C0 * table[:, 6:9], cloud.colors, atol=1e-6)
    np.testing.assert_allclose(table[:, 13:17], cloud.quats, atol=1e-6)


def test_reader_skips_higher_order_harmonics_and_normalises_quaternions(tmp_path) -> None:
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "f_rest_1", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    row = np.array([[1.0, 2.0, 3.0, 0.5, 0.0, -0.5, 9.0, 9.0, 0.0, np.log(0.1), np.log(0.2),
                     np.log(0.3), 2.0, 0.0, 0.0, 0.0]], dtype="<f4")
    header = "\n".join(["ply", "format binary_little_endian 1.0", "element vertex 1",
                        *(f"property float {name}" for name in names), "end_header\n"])
    path = tmp_path / "trained.ply"
    path.write_bytes(header.encode("ascii") + row.tobytes())
    cloud = read_ply(path)
    np.testing.assert_allclose(cloud.means, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(cloud.scales, [[0.1, 0.2, 0.3]], rtol=1e-6)
    np.testing.assert_allclose(cloud.opacities, [0.5])
    np.testing.assert_allclose(cloud.quats, [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(cloud.colors, 0.5 + SH_C0 * np.array([[0.5, 0.0, -0.5]]), rtol=1e-6)


def test_reader_rejects_ascii_ply(tmp_path) -> None:
    path = tmp_path / "ascii.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError, match="unsupported PLY format"):
        read_ply(path)
