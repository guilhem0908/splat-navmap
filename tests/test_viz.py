"""Figures: the colour code of the map pictures and the path-collision markers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from splat_navmap.benchmark import BANDS, DEFAULT_BAND
from splat_navmap.extract import EXTRACTORS
from splat_navmap.geometry import Box
from splat_navmap.grid import GridSpec
from splat_navmap.planners import astar
from splat_navmap.scene import Robot, Scene
from splat_navmap.synth import DEFECT_LEVELS, reconstruct
from splat_navmap.viz import (
    CORRECT,
    GIF_SIZE,
    MISSING,
    PHANTOM,
    _rgb,
    collision_points,
    error_image,
    hero_frame,
    save_gif,
)

ROBOT = Robot()


def two_rooms() -> tuple[Scene, GridSpec]:
    """Two 3 x 3 m rooms; the wall between them has no door, so any crossing is a collision."""
    floor = Box(3.0, 1.5, 3.2, 1.7, -0.05, 0.0, label="floor")
    solids = (Box(3.0, 1.5, 0.05, 1.5, 0.0, 2.5), Box(3.0, -0.1, 3.2, 0.1, 0.0, 2.5),
              Box(3.0, 3.1, 3.2, 0.1, 0.0, 2.5), Box(-0.1, 1.5, 0.1, 1.7, 0.0, 2.5),
              Box(6.1, 1.5, 0.1, 1.7, 0.0, 2.5))
    scene = Scene((-0.2, -0.2, 6.2, 3.2), floor, solids, ((0, 0, 3, 3), (3, 0, 6, 3)))
    return scene, scene.grid()


def cell(spec: GridSpec, x: float, y: float) -> tuple[int, int]:
    ix, iy = spec.world_to_cell([x, y])
    return int(ix), int(iy)


def test_error_image_colours_correct_phantom_and_missing_cells():
    truth = np.zeros((7, 7), dtype=bool)
    truth[1:6, 1:6] = True  # the centre cell (3, 3) is interior and not scored
    estimate = truth.copy()
    estimate[1, 1] = False  # missing, on the boundary
    estimate[0, 0] = True  # phantom
    image = error_image(estimate, truth)
    assert np.allclose(image[0, 0], _rgb(PHANTOM))
    assert np.allclose(image[1, 1], _rgb(MISSING))
    assert np.allclose(image[1, 2], _rgb(CORRECT))


def test_collision_markers_lie_where_a_path_crosses_a_wall():
    scene, spec = two_rooms()
    left, right = (int(v) for v in spec.world_to_cell([1.0, 1.5])), \
        (int(v) for v in spec.world_to_cell([5.0, 1.5]))
    start, goal = tuple(left), tuple(right)
    through_the_wall = astar(np.zeros(spec.shape, dtype=bool), start, goal)
    hits = collision_points(scene, spec, through_the_wall, ROBOT)
    assert len(hits) > 0
    assert np.all(np.abs(hits[:, 0] - 3.0) < 0.05 + ROBOT.radius)
    along_the_room = astar(np.zeros(spec.shape, dtype=bool), start,
                           tuple(int(v) for v in spec.world_to_cell([1.0, 2.5])))
    assert len(collision_points(scene, spec, along_the_room, ROBOT)) == 0


def test_hero_frame_has_the_gif_size_and_the_gif_keeps_every_frame(tmp_path):
    scene, spec = two_rooms()
    band = BANDS[DEFAULT_BAND]
    cloud = reconstruct(scene, 0, DEFECT_LEVELS["mild"], density=100.0)
    scores = {name: extractor(cloud, spec, band) for name, extractor in EXTRACTORS.items()}
    start, goal = cell(spec, 1.0, 1.0), cell(spec, 1.0, 2.5)
    frames = [hero_frame(scene, spec, ROBOT, cloud, band, scores, tau, start, goal, 60.0)
              for tau in (0.2, 0.6)]
    assert all(frame.size == GIF_SIZE for frame in frames)
    assert np.asarray(frames[0]).std() > 10  # not a blank canvas
    save_gif(frames, tmp_path / "sweep.gif")
    with Image.open(tmp_path / "sweep.gif") as gif:
        assert gif.n_frames == 2
        assert gif.size == GIF_SIZE
