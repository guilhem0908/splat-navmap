"""Export / import an occupancy grid in the ROS ``map_server`` format used by Nav2.

A map is a ``.pgm`` image plus a ``.yaml`` file (https://docs.nav2.org, "Map Server"). With
``negate: 0`` a pixel ``v`` means occupancy probability ``(255 - v) / 255``: above
``occupied_thresh`` the cell is occupied, below ``free_thresh`` it is free. The first image row
is the *top* of the map (largest y) and ``origin`` is the world pose of the lower-left pixel.
The raw (non-inflated) grid is exported: Nav2 inflates obstacles itself in its costmap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from splat_navmap.grid import GridSpec

PIXEL_OCCUPIED = 0
PIXEL_FREE = 254
OCCUPIED_THRESH = 0.65
FREE_THRESH = 0.196


def write_map(stem: str | Path, occupied: np.ndarray, spec: GridSpec) -> tuple[Path, Path]:
    """Write ``<stem>.pgm`` and ``<stem>.yaml``; return both paths."""
    stem = Path(stem)
    pgm, yaml = stem.with_suffix(".pgm"), stem.with_suffix(".yaml")
    pixels = np.where(occupied, PIXEL_OCCUPIED, PIXEL_FREE).astype(np.uint8)[::-1]
    ny, nx = occupied.shape
    pgm.write_bytes(f"P5\n{nx} {ny}\n255\n".encode("ascii") + pixels.tobytes())
    yaml.write_text(
        f"image: {pgm.name}\n"
        "mode: trinary\n"
        f"resolution: {spec.resolution:.6g}\n"
        f"origin: [{spec.origin[0]:.6f}, {spec.origin[1]:.6f}, 0.0]\n"
        "negate: 0\n"
        f"occupied_thresh: {OCCUPIED_THRESH}\n"
        f"free_thresh: {FREE_THRESH}\n", encoding="ascii")
    return pgm, yaml


def read_map(yaml_path: str | Path) -> tuple[np.ndarray, GridSpec]:
    """Load a map written by :func:`write_map` (or any binary P5 map with a flat yaml)."""
    yaml_path = Path(yaml_path)
    fields = dict(line.split(":", 1) for line in yaml_path.read_text().splitlines() if ":" in line)
    data = (yaml_path.parent / fields["image"].strip()).read_bytes()
    tokens, offset = [], 0
    while len(tokens) < 4:  # magic, width, height, maxval; '#' starts a comment line
        end = offset
        while data[end : end + 1].isspace():
            end += 1
        if data[end : end + 1] == b"#":
            offset = data.index(b"\n", end) + 1
            continue
        offset = end
        while not data[offset : offset + 1].isspace():
            offset += 1
        tokens.append(data[end:offset])
    if tokens[0] != b"P5" or int(tokens[3]) != 255:
        raise ValueError(f"{fields['image'].strip()} is not an 8-bit binary PGM")
    nx, ny = int(tokens[1]), int(tokens[2])
    pixels = np.frombuffer(data, dtype=np.uint8, count=nx * ny, offset=offset + 1)
    probability = (255.0 - pixels.reshape(ny, nx)[::-1]) / 255.0
    if int(fields["negate"]):
        probability = 1.0 - probability
    origin = [float(v) for v in fields["origin"].strip(" []").split(",")]
    spec = GridSpec((origin[0], origin[1]), float(fields["resolution"]), (ny, nx))
    return probability > float(fields["occupied_thresh"]), spec
