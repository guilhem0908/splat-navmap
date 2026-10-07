"""Fail when a number in README.md differs from the committed results.

    python scripts/check_readme.py            # check; exit code 1 on any mismatch
    python scripts/check_readme.py --write    # rewrite the generated tables in place

Two kinds of numbers are covered. Tables between ``<!-- table:NAME -->`` and
``<!-- /table:NAME -->`` are rendered from ``results/summary.csv`` and must match line by line.
Every number quoted in the prose is listed in :func:`claims`; each rendered string must appear
verbatim in the README.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from splat_navmap.synth import FLOATER_OPACITY_RANGE, SURFACE_OPACITY, WEAK_OPACITY

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DEFAULT_BAND = "guarded"
CENTRES_POINT, FOOTPRINT_POINT = 0.3, 0.5  # operating points, as in scripts/reproduce.py
METHODS = (("centres", "centre counting"), ("footprint", "footprint accumulation"))
Rows = list[dict]


class Results:
    """The committed benchmark summary, with lookup by experiment settings."""

    def __init__(self) -> None:
        with open(ROOT / "results" / "summary.csv", newline="", encoding="utf-8") as handle:
            self.rows: Rows = list(csv.DictReader(handle))
        self.config = json.loads((ROOT / "results" / "config.json").read_text())
        self.timings = json.loads((ROOT / "results" / "timings.json").read_text())

    def get(self, experiment: str, defect: str, method: str, tau: float, *,
            band: str = DEFAULT_BAND, planner: str = "astar", min_component: int = 1) -> dict:
        wanted = {"experiment": experiment, "defect": defect, "method": method, "band": band,
                  "planner": planner, "min_component": str(min_component)}
        for row in self.rows:
            if all(row[key] == value for key, value in wanted.items()) and (
                    abs(float(row["tau"]) - tau) < 1e-9):
                return row
        raise KeyError(f"no result for {wanted} tau={tau}")

    def threshold(self, defect: str, method: str, tau: float) -> dict:
        return self.get("threshold", defect, method, tau)

    def point(self, experiment: str, defect: str, method: str, **settings) -> dict:
        """The row of a method at its operating point."""
        tau = CENTRES_POINT if method == "centres" else FOOTPRINT_POINT
        return self.get(experiment, defect, method, tau, **settings)


def count(row: dict, name: str) -> str:
    """``'76 (7.6 %)'``: raw count and percentage of the pairs of a result row."""
    return f"{int(row[name]):,} ({100 * int(row[name]) / int(row['pairs']):.1f} %)"


def pct(row: dict, name: str) -> str:
    return f"{100 * int(row[name]) / int(row['pairs']):.1f} %"


def ratio(row: dict) -> str:
    return f"{float(row['mean_ratio']):.3f}" if row["mean_ratio"] else "n/a"


def table(header: list[str], body: list[list[str]], left: int = 1) -> list[str]:
    """Markdown table; the first ``left`` columns are left-aligned, the others right-aligned."""
    align = [":---" if k < left else "---:" for k in range(len(header))]
    lines = [header, align, *body]
    return [f"| {' | '.join(line)} |" for line in lines]


def sweep_table(results: Results) -> list[str]:
    """Threshold sweep with moderate defects: map quality and planning outcome."""
    header = ["method", "τ", "precision", "recall", "IoU", "paths through real geometry",
              "feasible pairs unreachable", "length / optimum"]
    body = []
    for method, label in METHODS:
        for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
            row = results.threshold("moderate", method, tau)
            body.append([label, f"{tau:.1f}", f"{float(row['precision']):.3f}",
                         f"{float(row['recall']):.3f}", f"{float(row['iou']):.3f}",
                         count(row, "collisions"), count(row, "unreachable"), ratio(row)])
    return table(header, body, left=2)


def defect_table(results: Results) -> list[str]:
    """Defect level, then each defect alone at its severe magnitude, at the operating points."""
    header = ["defects"] + [f"{label}, τ = {tau}: {kind}" for label, tau in
                            (("centres", CENTRES_POINT), ("footprint", FOOTPRINT_POINT))
                            for kind in ("collide", "unreachable")]
    rows = [("threshold", level, level) for level in ("clean", "mild", "moderate", "severe")]
    rows += [("defect", name, f"{label} only") for name, label in (
        ("floaters", "floaters"), ("holes", "holes"), ("weak", "weak patches"),
        ("thickness", "fuzzy surfaces"))]
    body = []
    for experiment, defect, label in rows:
        cells = []
        for method, _ in METHODS:
            row = results.point(experiment, defect, method)
            cells += [pct(row, "collisions"), pct(row, "unreachable")]
        body.append([label, *cells])
    return table(header, body)


def band_table(results: Results) -> list[str]:
    """Robot-height band assumptions with moderate defects."""
    header = ["height band (m)"] + [f"{label}: {kind}" for label in ("centres", "footprint")
                                    for kind in ("collide", "unreachable")]
    bands = results.config["bands"]
    body = []
    for name in ("body", "floor", "guarded", "guard10", "short", "tall", "slice"):
        lo, hi = bands[name]
        cells = []
        for method, _ in METHODS:
            row = (results.point("threshold", "moderate", method) if name == DEFAULT_BAND
                   else results.point("band", "moderate", method, band=name))
            cells += [pct(row, "collisions"), pct(row, "unreachable")]
        body.append([f"{lo:.2f} to {hi:.2f}", *cells])
    return table(header, body)


def variant_table(results: Results) -> list[str]:
    """Planner and small-component filter at the operating points, moderate defects."""
    header = ["variant"] + [f"{label}: {kind}" for label in ("centres", "footprint")
                            for kind in ("collide", "unreachable")]
    variants = (("A*, no filter", "threshold", {}), ("Theta*, no filter", "planner",
                                                     {"planner": "theta"}),
                ("A*, drop groups under 4 cells", "filter", {"min_component": 4}),
                ("A*, drop groups under 10 cells", "filter", {"min_component": 10}))
    body = []
    for name, experiment, settings in variants:
        cells = []
        for method, _ in METHODS:
            row = results.point(experiment, "moderate", method, **settings)
            cells += [pct(row, "collisions"), pct(row, "unreachable")]
        body.append([name, *cells])
    return table(header, body)


TABLES: dict[str, Callable[[Results], list[str]]] = {
    "sweep": sweep_table, "defects": defect_table, "bands": band_table,
    "variants": variant_table,
}


def claims(results: Results) -> dict[str, str]:
    """Numbers quoted in the prose: name -> the exact string the README must contain."""
    config, timings = results.config, results.timings
    pairs = len(config["seeds"]) * config["pairs_per_scene"]
    robot = config["robot"]
    guarded_lo, guarded_hi = config["bands"][DEFAULT_BAND]
    sweep = {m: {tau: results.threshold("moderate", m, tau) for tau in (0.3, 0.5)}
             for m, _ in METHODS}
    severe = {m: results.point("threshold", "severe", m) for m, _ in METHODS}
    single = {(d, m): results.point("defect", d, m) for d in ("floaters", "thickness")
              for m, _ in METHODS}
    band = {name: {m: results.point("band", "moderate", m, band=name) for m, _ in METHODS}
            for name in ("floor", "slice", "tall")}
    filtered = {(size, m): results.point("filter", "moderate", m, min_component=size)
                for size in (4, 10) for m, _ in METHODS}
    centres, footprint = sweep["centres"][0.3], sweep["footprint"][0.5]
    centres_half = sweep["centres"][0.5]
    return {
        "scenes": f"{len(config['seeds'])} scenes",
        "pairs": f"{pairs:,} start-goal pairs",
        "robot_radius": f"radius {robot['radius']:.2f} m",
        "robot_body": f"body from {robot['ground_clearance']:.2f} to {robot['height']:.2f} m",
        "band": f"band {guarded_lo:.2f} to {guarded_hi:.2f} m",
        "centres_0.3": f"τ = 0.3 leaves {pct(centres, 'unreachable')} of feasible pairs "
                       "unreachable",
        "centres_0.5": f"τ = 0.5 sends {pct(centres_half, 'collisions')} of paths through real "
                       "geometry",
        "footprint_0.5": f"at τ = 0.5 leaves {pct(footprint, 'unreachable')} unreachable and "
                         f"{pct(footprint, 'collisions')} colliding",
        "footprint_0.5_spread": f"collides on {pct(footprint, 'collisions')} of pairs overall and "
                                f"on up to {float(footprint['collision_pct_max']):.1f} % in the "
                                "worst scene",
        "iou_footprint_0.5": f"IoU {float(footprint['iou']):.3f}",
        "iou_centres_0.3": f"{float(centres['iou']):.3f} for centres",
        "severe_unreachable": f"({pct(severe['centres'], 'unreachable')} and "
                              f"{pct(severe['footprint'], 'unreachable')} of pairs unreachable)",
        "floaters_alone": f"{pct(single['floaters', 'centres'], 'unreachable')} unreachable) and "
                          f"hardly for footprint accumulation "
                          f"({pct(single['floaters', 'footprint'], 'unreachable')})",
        "thickness_alone": f"{pct(single['thickness', 'centres'], 'unreachable')} and "
                           f"{pct(single['thickness', 'footprint'], 'unreachable')} unreachable",
        "floor_band": f"({pct(band['floor']['centres'], 'unreachable')} unreachable)",
        "slice_band": f"({pct(band['slice']['centres'], 'collisions')} and "
                      f"{pct(band['slice']['footprint'], 'collisions')} collide)",
        "tall_band": f"({pct(band['tall']['centres'], 'unreachable')} and "
                     f"{pct(band['tall']['footprint'], 'unreachable')} unreachable)",
        "filter_4": f"({pct(centres, 'unreachable')} to "
                    f"{pct(filtered[4, 'centres'], 'unreachable')} unreachable for centre "
                    "counting)",
        "filter_10": f"collisions jump to {pct(filtered[10, 'centres'], 'collisions')} and "
                     f"{pct(filtered[10, 'footprint'], 'collisions')}",
        "surface_opacity": f"opacity {SURFACE_OPACITY[0]:.2f} to {SURFACE_OPACITY[1]:.2f}",
        "weak_patches": f"weak patches ({WEAK_OPACITY[0]:.2f} to {WEAK_OPACITY[1]:.2f})",
        "floater_opacity": f"floaters (up to {FLOATER_OPACITY_RANGE[1]:.2f})",
        "python": f"Python {timings['python'].rsplit('.', 1)[0]}",
        "grid_cells": f"grid of {timings['grid_cells']:,} cells",
        "showcase": f"{timings['showcase_gaussians']:,} Gaussians",
        "extract_footprint": f"footprint extraction {timings['extract_footprint_s']:.1f} s",
        "extract_centres": f"centre counting {1000 * timings['extract_centres_s']:.0f} ms",
        "astar": f"A* {timings['astar_ms_median']:.0f} ms",
        "theta": f"Theta* {timings['theta_ms_median']:.0f} ms",
        "benchmark_minutes": f"{timings['benchmark_minutes']:.0f} min on "
                             f"{timings['benchmark_processes']} processes",
    }


def render_block(name: str, lines: list[str]) -> str:
    return f"<!-- table:{name} -->\n" + "\n".join(lines) + f"\n<!-- /table:{name} -->"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite the generated tables")
    args = parser.parse_args()

    results = Results()
    text = README.read_text(encoding="utf-8")
    problems: list[str] = []
    for name, build in TABLES.items():
        pattern = re.compile(rf"<!-- table:{name} -->\n(?:.*?\n)?<!-- /table:{name} -->",
                             re.DOTALL)
        expected = render_block(name, build(results))
        found = pattern.search(text)
        if found is None:
            problems.append(f"table '{name}' is missing from the README")
        elif found.group(0) != expected:
            if args.write:
                text = text[: found.start()] + expected + text[found.end():]
            else:
                problems.append(f"table '{name}' differs from results/summary.csv")
    if args.write:
        README.write_text(text, encoding="utf-8", newline="\n")
    prose = re.sub(r"<!-- table:(\w+) -->.*?<!-- /table:\1 -->", "", text, flags=re.DOTALL)
    problems += [f"prose lacks {name} = '{value}'" for name, value in claims(results).items()
                 if value not in prose]
    for problem in problems:
        print(f"README mismatch: {problem}", file=sys.stderr)
    if not problems:
        print(f"README numbers match results/ ({len(TABLES)} tables, {len(claims(results))} "
              "quoted values)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
