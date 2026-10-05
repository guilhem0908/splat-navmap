"""Read and write the INRIA 3D Gaussian Splatting ``.ply`` layout.

The reference implementation (github.com/graphdeco-inria/gaussian-splatting, ``save_ply``)
stores one binary little-endian vertex per Gaussian with float32 properties::

    x y z  nx ny nz  f_dc_0..2  [f_rest_*]  opacity  scale_0..2  rot_0..3

with these encodings, inverted on load:

* ``scale_k = log(s_k)``                        (activated with ``exp``)
* ``opacity = logit(alpha)``                    (activated with the sigmoid)
* ``f_dc = (rgb - 0.5) / C0``, ``C0 = 1 / (2 sqrt(pi))`` the degree-0 spherical harmonic
* ``rot`` = quaternion (w, x, y, z), normalised on load

Higher-order ``f_rest_*`` coefficients (view-dependent colour) are skipped on load and not
written: occupancy extraction only needs geometry and opacity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from splat_navmap.gaussians import GaussianCloud

SH_C0 = 0.28209479177387814  # 1 / (2 sqrt(pi)), degree-0 real spherical harmonic
OPACITY_EPS = 1e-6  # keeps logit(alpha) finite for alpha in {0, 1}

_PLY_TO_NUMPY = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
    "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2",
    "uint16": "<u2", "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8",
}
_WRITE_FIELDS = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                 "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


def write_ply(path: str | Path, cloud: GaussianCloud) -> None:
    """Write ``cloud`` as a degree-0 INRIA 3DGS ``.ply`` (17 float32 per Gaussian)."""
    alpha = np.clip(cloud.opacities, OPACITY_EPS, 1.0 - OPACITY_EPS)
    columns = np.column_stack([
        cloud.means,
        np.zeros((len(cloud), 3)),
        (cloud.colors - 0.5) / SH_C0,
        np.log(alpha / (1.0 - alpha)),
        np.log(cloud.scales),
        cloud.quats,
    ]).astype("<f4")
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(cloud)}"]
    header += [f"property float {name}" for name in _WRITE_FIELDS]
    header.append("end_header\n")
    with open(path, "wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(columns.tobytes())


def read_ply(path: str | Path) -> GaussianCloud:
    """Load a binary little-endian 3DGS ``.ply``; extra vertex properties are ignored."""
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        count, fields, in_vertex = 0, [], False
        while (line := handle.readline().decode("ascii").strip()) != "end_header":
            tokens = line.split()
            if tokens[0] == "format" and tokens[1] != "binary_little_endian":
                raise ValueError(f"unsupported PLY format '{tokens[1]}'")
            if tokens[0] == "element":
                in_vertex = tokens[1] == "vertex"
                count = int(tokens[2]) if in_vertex else count
            elif tokens[0] == "property" and in_vertex:
                if tokens[1] == "list":
                    raise ValueError("list properties are not supported on the vertex element")
                fields.append((tokens[2], _PLY_TO_NUMPY[tokens[1]]))
        dtype = np.dtype(fields)
        vertex = np.frombuffer(handle.read(count * dtype.itemsize), dtype=dtype, count=count)

    def stack(*names: str) -> np.ndarray:
        return np.column_stack([vertex[name] for name in names]).astype(np.float64)

    quats = stack("rot_0", "rot_1", "rot_2", "rot_3")
    return GaussianCloud(
        means=stack("x", "y", "z"),
        scales=np.exp(stack("scale_0", "scale_1", "scale_2")),
        quats=quats / np.linalg.norm(quats, axis=1, keepdims=True),
        opacities=1.0 / (1.0 + np.exp(-vertex["opacity"].astype(np.float64))),
        colors=np.clip(0.5 + SH_C0 * stack("f_dc_0", "f_dc_1", "f_dc_2"), 0.0, 1.0),
    )
