"""Nav2 map export and the command line, end to end on a small scene."""

from __future__ import annotations

import numpy as np

from splat_navmap import nav2
from splat_navmap.cli import main
from splat_navmap.grid import GridSpec
from splat_navmap.ply_io import read_ply

PGM_HEADER = b"P5\n11 7\n255\n"


def test_map_round_trip_and_file_layout(tmp_path) -> None:
    rng = np.random.default_rng(0)
    occupied = rng.random((7, 11)) < 0.3
    occupied[0, 0], occupied[6, 10] = True, False
    spec = GridSpec((-1.25, 2.5), 0.05, occupied.shape)
    pgm, yaml = nav2.write_map(tmp_path / "flat", occupied, spec)

    raw = pgm.read_bytes()
    assert raw.startswith(PGM_HEADER)
    pixels = np.frombuffer(raw[len(PGM_HEADER):], dtype=np.uint8).reshape(7, 11)
    assert pixels[6, 0] == 0  # cell (0, 0) is the bottom-left pixel: last image row, occupied
    assert pixels[0, 10] == 254  # cell (10, 6) is the top-right pixel, free
    assert set(np.unique(pixels).tolist()) == {0, 254}

    text = yaml.read_text()
    assert "image: flat.pgm" in text and "negate: 0" in text
    assert "resolution: 0.05" in text and "origin: [-1.250000, 2.500000, 0.0]" in text

    loaded, loaded_spec = nav2.read_map(yaml)
    np.testing.assert_array_equal(loaded, occupied)
    assert loaded_spec == spec


def test_reader_follows_map_server_thresholds_negate_and_comments(tmp_path) -> None:
    pixels = np.array([[254, 205, 100], [0, 60, 255]], dtype=np.uint8)
    (tmp_path / "m.pgm").write_bytes(b"P5\n# saved by a mapper\n3 2\n255\n" + pixels.tobytes())
    yaml = tmp_path / "m.yaml"
    yaml.write_text("image: m.pgm\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\nnegate: 0\n"
                    "occupied_thresh: 0.65\nfree_thresh: 0.196\n")
    occupied, spec = nav2.read_map(yaml)
    # (255 - v) / 255 > 0.65 only for v in {0, 60}; the second image row is row 0 of the map.
    np.testing.assert_array_equal(occupied, [[True, True, False], [False, False, False]])
    assert spec == GridSpec((0.0, 0.0), 0.1, (2, 3))
    yaml.write_text(yaml.read_text().replace("negate: 0", "negate: 1"))
    np.testing.assert_array_equal(nav2.read_map(yaml)[0],
                                  [[False, False, True], [True, True, False]])


def test_command_line_generates_extracts_and_plans(tmp_path, capsys) -> None:
    ply, stem = tmp_path / "flat.ply", tmp_path / "flat"
    assert main(["generate", "--seed", "3", "--defects", "mild", "--density", "150",
                 "--out", str(ply), "--truth", str(tmp_path / "truth")]) == 0
    assert len(read_ply(ply)) > 30_000
    assert main(["extract", str(ply), "--method", "footprint", "--tau", "0.5",
                 "--out", str(stem)]) == 0
    occupied, spec = nav2.read_map(stem.with_suffix(".yaml"))
    truth, truth_spec = nav2.read_map(tmp_path / "truth.yaml")
    assert abs(spec.shape[0] - truth_spec.shape[0]) <= 4
    assert occupied.sum() > 0.5 * truth.sum()

    plan = ["plan", str(stem.with_suffix(".yaml")), "--start", "1.0", "1.0",
            "--goal", "8.0", "6.0"]
    assert main([*plan, "--preview", str(tmp_path / "path.png")]) == 0
    assert main([*plan, "--planner", "theta"]) == 0
    lines = capsys.readouterr().out.splitlines()
    lattice, any_angle = (float(line.split(", ")[1].split()[0]) for line in lines[-2:])
    assert 7.0 < any_angle < lattice < 14.0
    assert (tmp_path / "path.png").stat().st_size > 1000
    assert main(["plan", str(stem.with_suffix(".yaml")), "--start", "-5", "-5",
                 "--goal", "8.0", "6.0"]) == 1
