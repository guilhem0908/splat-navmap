"""Command line: generate a scene, slice a ``.ply`` into a Nav2 map, plan on a map."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from splat_navmap import nav2
from splat_navmap.evaluate import remove_small_components
from splat_navmap.extract import EXTRACTORS, Band
from splat_navmap.grid import DEFAULT_RESOLUTION, GridSpec, inflate
from splat_navmap.planners import astar, path_length, theta_star
from splat_navmap.ply_io import read_ply, write_ply
from splat_navmap.scene import Robot, generate_scene
from splat_navmap.synth import DEFAULT_DENSITY, DEFECT_LEVELS, reconstruct

PREVIEW_SCALE = 4  # pixels per cell in the preview image
_COLORS = {"free": (255, 255, 255), "inflated": (214, 222, 235), "occupied": (30, 34, 42),
           "path": (220, 60, 50), "start": (40, 160, 90), "goal": (50, 100, 220)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="splat-navmap", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    gen = commands.add_parser("generate", help="write a procedural scene as a 3DGS .ply")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--defects", choices=list(DEFECT_LEVELS), default="moderate")
    gen.add_argument("--density", type=float, default=DEFAULT_DENSITY, help="surfels per m^2")
    gen.add_argument("--out", type=Path, required=True, help="output .ply")
    gen.add_argument("--truth", type=Path, help="also write the exact occupancy map (stem)")

    ext = commands.add_parser("extract", help="slice a 3DGS .ply (z up, metres) into a Nav2 map")
    ext.add_argument("ply", type=Path)
    ext.add_argument("--method", choices=list(EXTRACTORS), default="footprint")
    ext.add_argument("--tau", type=float, default=0.5, help="opacity threshold")
    ext.add_argument("--band", type=float, nargs=2, default=(0.05, 0.40), metavar=("Z_LO", "Z_HI"))
    ext.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    ext.add_argument("--min-component", type=int, default=1,
                     help="drop occupied groups smaller than this many cells")
    ext.add_argument("--out", type=Path, required=True, help="output stem for .pgm / .yaml")

    plan = commands.add_parser("plan", help="plan a path on a Nav2 map")
    plan.add_argument("map", type=Path, help="map .yaml")
    plan.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"))
    plan.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    plan.add_argument("--planner", choices=["astar", "theta"], default="astar")
    plan.add_argument("--radius", type=float, default=Robot().radius, help="robot radius, metres")
    plan.add_argument("--preview", type=Path, help="write a .png of the map and the path")

    args = parser.parse_args(argv)
    return {"generate": _generate, "extract": _extract, "plan": _plan}[args.command](args)


def _generate(args: argparse.Namespace) -> int:
    scene = generate_scene(args.seed)
    cloud = reconstruct(scene, args.seed, DEFECT_LEVELS[args.defects], args.density)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.out, cloud)
    print(f"{args.out}: {len(cloud)} Gaussians, {len(scene.rooms)} rooms, "
          f"defects '{args.defects}'")
    if args.truth:
        spec = scene.grid()
        pgm, _ = nav2.write_map(args.truth, scene.truth(spec, Robot()), spec)
        print(f"{pgm}: exact occupancy, {spec.shape[1]} x {spec.shape[0]} cells")
    return 0


def _extract(args: argparse.Namespace) -> int:
    cloud = read_ply(args.ply)
    band = Band(*args.band)
    in_band = (cloud.means[:, 2] >= band.z_lo) & (cloud.means[:, 2] <= band.z_hi)
    if not in_band.any():
        print(f"no Gaussian centre between z = {band.z_lo} and {band.z_hi}")
        return 1
    xy = cloud.means[in_band, :2]
    spec = GridSpec.covering((*xy.min(axis=0), *xy.max(axis=0)), args.resolution)
    occupied = EXTRACTORS[args.method](cloud, spec, band) >= args.tau
    if args.min_component > 1:
        occupied = remove_small_components(occupied, args.min_component)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pgm, yaml = nav2.write_map(args.out, occupied, spec)
    print(f"{pgm} + {yaml.name}: {spec.shape[1]} x {spec.shape[0]} cells at "
          f"{spec.resolution} m, {int(occupied.sum())} occupied ({args.method}, tau {args.tau})")
    return 0


def _plan(args: argparse.Namespace) -> int:
    occupied, spec = nav2.read_map(args.map)
    blocked = inflate(occupied, args.radius, spec.resolution)
    start, goal = (tuple(int(v) for v in spec.world_to_cell(p)) for p in (args.start, args.goal))
    path = (theta_star if args.planner == "theta" else astar)(blocked, start, goal)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        _preview(occupied, blocked, path, start, goal).save(args.preview)
    if path is None:
        print("no path: an endpoint is blocked or the goal is not reachable")
        return 1
    print(f"{args.planner}: {len(path)} waypoints, {path_length(path) * spec.resolution:.2f} m")
    return 0


def _preview(occupied: np.ndarray, blocked: np.ndarray, path: list | None, start: tuple,
             goal: tuple) -> Image.Image:
    """Top-down picture: occupied cells, inflation, path and endpoints (y up)."""
    pixels = np.full((*occupied.shape, 3), _COLORS["free"], dtype=np.uint8)
    pixels[blocked] = _COLORS["inflated"]
    pixels[occupied] = _COLORS["occupied"]
    ny = occupied.shape[0]
    image = Image.fromarray(pixels[::-1]).resize(
        (occupied.shape[1] * PREVIEW_SCALE, ny * PREVIEW_SCALE), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    def pixel(cell: tuple) -> tuple[float, float]:
        return (cell[0] + 0.5) * PREVIEW_SCALE, (ny - cell[1] - 0.5) * PREVIEW_SCALE

    if path:
        draw.line([pixel(c) for c in path], fill=_COLORS["path"], width=PREVIEW_SCALE)
    for cell, name in ((start, "start"), (goal, "goal")):
        x, y = pixel(cell)
        radius = 1.5 * PREVIEW_SCALE
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=_COLORS[name])
    return image
