"""Regenerate ``results/`` and the figures under ``docs/`` from fixed seeds.

    python scripts/reproduce.py --jobs 4     # full benchmark, then figures
    python scripts/reproduce.py --figures    # redraw the figures from the committed results
    python scripts/reproduce.py --quick      # 2 scenes x 10 pairs into out/quick (smoke test)
    python scripts/reproduce.py --cache out/cache   # keep each finished scene, resume after a crash
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from splat_navmap.benchmark import (
    BANDS,
    DEFAULT_BAND,
    DEFECTS,
    THRESHOLDS,
    Condition,
    evaluate_scene,
    run,
)
from splat_navmap.evaluate import sample_pairs
from splat_navmap.extract import EXTRACTORS
from splat_navmap.grid import inflate
from splat_navmap.planners import astar, path_length, theta_star
from splat_navmap.scene import SHOWCASE_SEED, Robot, generate_scene, showcase_scene
from splat_navmap.synth import DEFECT_LEVELS, reconstruct
from splat_navmap.viz import hero_frame, plot_tradeoff, save_gif

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(10))
PAIRS_PER_SCENE = 100
PROBE_TAUS = (0.3, 0.5, 0.7)  # thresholds used by the band, defect and filter experiments
OPERATING_POINTS = (("centres", 0.3), ("footprint", 0.5))  # used by the planner comparison
FILTER_SIZES = (4, 10)  # small-component filter, in cells
SINGLE_DEFECTS = ("floaters", "holes", "weak", "thickness")
HERO = {"seed": 4, "defect": "moderate", "start": (1.62, 0.47), "goal": (10.22, 5.76)}
HERO_TAUS = tuple(round(0.05 * k, 2) for k in range(1, 20))
KEY = ("experiment", "defect", "band", "method", "tau", "min_component", "planner")
COUNTS = ("tp", "fp", "fn", "pairs", "unreachable", "endpoint_blocked", "collisions", "valid")


def build_conditions(thresholds: tuple[float, ...]) -> list[Condition]:
    """Every benchmark setting, grouped by defect so that workers reuse reconstructions."""
    conditions = [Condition("control", "clean", "truth", 0.0),
                  Condition("control", "clean", "truth", 0.0, planner="theta")]
    for defect in DEFECT_LEVELS:
        conditions += [Condition("threshold", defect, method, tau)
                       for method in EXTRACTORS for tau in thresholds]
        if defect in ("clean", "moderate"):
            conditions += [Condition("planner", defect, method, tau, planner="theta")
                           for method, tau in OPERATING_POINTS]
        if defect == "moderate":
            conditions += [Condition("band", defect, method, tau, band=band)
                           for band in BANDS if band != DEFAULT_BAND
                           for method in EXTRACTORS for tau in PROBE_TAUS]
            conditions += [Condition("filter", defect, method, tau, min_component=size)
                           for size in FILTER_SIZES for method in EXTRACTORS for tau in PROBE_TAUS]
    conditions += [Condition("defect", defect, method, tau) for defect in SINGLE_DEFECTS
                   for method in EXTRACTORS for tau in PROBE_TAUS]
    return conditions


def run_cached(conditions: list[Condition], seeds: tuple[int, ...], pairs: int, jobs: int,
               cache: Path) -> list[dict]:
    """Like ``benchmark.run`` but each finished scene is kept in ``cache`` and reused on a rerun.

    The cache folder is named after a hash of the settings; it must be emptied after a code change.
    """
    cache = cache / hashlib.sha1(repr((conditions, pairs)).encode()).hexdigest()[:12]
    cache.mkdir(parents=True, exist_ok=True)
    pending = [seed for seed in seeds if not (cache / f"scene{seed}.json").exists()]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(evaluate_scene, seed, conditions, pairs): seed for seed in pending}
        for future in as_completed(futures):
            (cache / f"scene{futures[future]}.json").write_text(json.dumps(future.result()))
            print(f"scene {futures[future]} done", flush=True)
    return [row for seed in seeds for row in json.loads((cache / f"scene{seed}.json").read_text())]


def aggregate(rows: list[dict]) -> list[dict]:
    """Sum the per-scene counts of each condition and derive rates and per-scene spreads."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in KEY)].append(row)
    summary = []
    for key, members in groups.items():
        total = {name: sum(int(m[name]) for m in members) for name in COUNTS}
        ratio_sum = sum(float(m["ratio_sum"]) for m in members)
        out = dict(zip(KEY, key, strict=True)) | {"scenes": len(members)} | total
        out["precision"] = f"{total['tp'] / max(total['tp'] + total['fp'], 1):.4f}"
        out["recall"] = f"{total['tp'] / max(total['tp'] + total['fn'], 1):.4f}"
        out["iou"] = f"{total['tp'] / max(total['tp'] + total['fp'] + total['fn'], 1):.4f}"
        for name, count in (("collision", "collisions"), ("unreachable", "unreachable")):
            per_scene = [100.0 * int(m[count]) / int(m["pairs"]) for m in members]
            out[f"{name}_pct"] = f"{100.0 * total[count] / total['pairs']:.2f}"
            out[f"{name}_pct_min"] = f"{min(per_scene):.2f}"
            out[f"{name}_pct_max"] = f"{max(per_scene):.2f}"
        out["mean_ratio"] = f"{ratio_sum / total['valid']:.4f}" if total["valid"] else ""
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def measure_timings(benchmark_seconds: float, jobs: int) -> dict:
    """CPU timings on the showcase scene (single process, median of three runs)."""
    robot = Robot()
    scene = showcase_scene()
    spec = scene.grid()

    def median_seconds(function) -> float:
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            function()
            samples.append(time.perf_counter() - start)
        return statistics.median(samples)

    cloud = reconstruct(scene, SHOWCASE_SEED, DEFECT_LEVELS["mild"])
    blocked = inflate(scene.truth(spec, robot), robot.radius, spec.resolution)
    pairs = sample_pairs(blocked, spec.resolution, np.random.default_rng(0), 50)
    plan_ms = {}
    for name, planner in (("astar", astar), ("theta", theta_star)):
        samples = []
        for start_cell, goal_cell in pairs:
            start = time.perf_counter()
            planner(blocked, start_cell, goal_cell)
            samples.append(1000.0 * (time.perf_counter() - start))
        plan_ms[name] = round(statistics.median(samples), 1)
    return {
        "machine": platform.processor(), "python": platform.python_version(),
        "showcase_gaussians": len(cloud), "grid_cells": int(spec.shape[0] * spec.shape[1]),
        "reconstruct_s": round(median_seconds(
            lambda: reconstruct(scene, SHOWCASE_SEED, DEFECT_LEVELS["mild"])), 2),
        "extract_centres_s": round(median_seconds(
            lambda: EXTRACTORS["centres"](cloud, spec, BANDS[DEFAULT_BAND])), 3),
        "extract_footprint_s": round(median_seconds(
            lambda: EXTRACTORS["footprint"](cloud, spec, BANDS[DEFAULT_BAND])), 2),
        "astar_ms_median": plan_ms["astar"], "theta_ms_median": plan_ms["theta"],
        "benchmark_minutes": round(benchmark_seconds / 60.0, 1), "benchmark_processes": jobs,
    }


def draw_hero(path: Path) -> None:
    """Threshold sweep on one benchmark scene, both extractors, one start-goal pair."""
    robot = Robot()
    scene = generate_scene(HERO["seed"], robot)
    spec = scene.grid()
    band = BANDS[DEFAULT_BAND]
    cloud = reconstruct(scene, HERO["seed"], DEFECTS[HERO["defect"]])
    scores = {name: extractor(cloud, spec, band) for name, extractor in EXTRACTORS.items()}
    start, goal = (tuple(int(v) for v in spec.world_to_cell(HERO[k])) for k in ("start", "goal"))
    truth_blocked = inflate(scene.truth(spec, robot), robot.radius, spec.resolution)
    reference = path_length(astar(truth_blocked, start, goal))
    frames = [hero_frame(scene, spec, robot, cloud, band, scores, tau, start, goal, reference)
              for tau in HERO_TAUS]
    save_gif(frames, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="tiny run into out/quick")
    parser.add_argument("--figures", action="store_true", help="only redraw the figures")
    parser.add_argument("--cache", type=Path, help="folder that keeps finished scenes (resume)")
    parser.add_argument("--jobs", type=int,
                        help="worker processes (default: one per 4 CPUs, at most 10)",
                        default=min(len(SEEDS), max(1, (os.cpu_count() or 2) // 4)))
    args = parser.parse_args()

    results = ROOT / ("out/quick" if args.quick else "results")
    figures = ROOT / ("out/quick" if args.quick else "docs")
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    if not args.figures:
        seeds, pairs = (SEEDS[:2], 10) if args.quick else (SEEDS, PAIRS_PER_SCENE)
        thresholds = PROBE_TAUS if args.quick else THRESHOLDS
        conditions = build_conditions(thresholds)
        print(f"{len(conditions)} conditions x {len(seeds)} scenes x {pairs} pairs, "
              f"{args.jobs} processes")
        start = time.perf_counter()
        rows = (run_cached(conditions, seeds, pairs, args.jobs, args.cache) if args.cache
                else run(conditions, seeds, pairs, jobs=args.jobs))
        elapsed = time.perf_counter() - start
        write_csv(results / "per_scene.csv", rows)
        write_csv(results / "summary.csv", aggregate(rows))
        config = {"seeds": list(seeds), "pairs_per_scene": pairs, "robot": vars(Robot()),
                  "thresholds": list(thresholds),
                  "bands": {name: [band.z_lo, band.z_hi] for name, band in BANDS.items()},
                  "defects": {name: vars(defects) for name, defects in DEFECTS.items()}}
        (results / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        timings = measure_timings(elapsed, args.jobs)
        (results / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        print(f"benchmark done in {elapsed / 60:.1f} min")

    plot_tradeoff(read_csv(results / "summary.csv"), figures / "tradeoff.png")
    draw_hero(figures / "hero.gif")
    print(f"results in {results.relative_to(ROOT)}, figures in {figures.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
