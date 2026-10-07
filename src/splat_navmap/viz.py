"""Figures: the threshold-sweep animation and the safety / reachability curves.

Colours follow one rule per job: the two extraction methods are the two categorical series
(blue, orange); in the maps, a phantom cell is orange, a missing cell is blue, and a true
collision is a red cross (status colour plus a shape, never colour alone).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy import ndimage

from splat_navmap.evaluate import COLLISION_TOL, score_map
from splat_navmap.extract import Band
from splat_navmap.gaussians import GaussianCloud
from splat_navmap.geometry import in_band, segment_distance
from splat_navmap.grid import GridSpec, inflate
from splat_navmap.planners import Cell, astar, path_length
from splat_navmap.scene import Robot, Scene

SURFACE, INK, SECONDARY, MUTED, GRIDLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"centres": "#2a78d6", "footprint": "#eb6834"}
SERIES_LABEL = {"centres": "centre counting", "footprint": "footprint accumulation"}
PHANTOM, MISSING, CRITICAL, GOOD = "#eb6834", "#2a78d6", "#d03b3b", "#0ca30c"
CORRECT, INTERIOR, HALO, REJECTED = "#2b2d33", "#c9c8c2", "#e6e9ef", "#cfd3da"
DEFECT_ORDER = ("clean", "mild", "moderate", "severe")
GIF_SIZE = (900, 600)  # pixels
GIF_FRAME_MS = 450
GIF_COLORS = 48

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": SECONDARY, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _rgb(color: str) -> np.ndarray:
    return np.array(matplotlib.colors.to_rgb(color))


def _extent(spec: GridSpec) -> tuple[float, float, float, float]:
    ny, nx = spec.shape
    return (spec.origin[0], spec.origin[0] + nx * spec.resolution,
            spec.origin[1], spec.origin[1] + ny * spec.resolution)


def error_image(estimate: np.ndarray, truth: np.ndarray, blocked: np.ndarray | None = None,
                ) -> np.ndarray:
    """RGB picture of a grid: correct, phantom and missing cells, optional inflation halo."""
    interior = ndimage.binary_erosion(truth, structure=np.ones((3, 3), dtype=bool), border_value=1)
    image = np.tile(_rgb(SURFACE), (*truth.shape, 1))
    if blocked is not None:
        image[blocked] = _rgb(HALO)
    image[interior & ~estimate] = _rgb(INTERIOR)
    image[estimate & truth] = _rgb(CORRECT)
    image[estimate & ~truth] = _rgb(PHANTOM)
    image[~estimate & truth & ~interior] = _rgb(MISSING)
    return image


def draw_ellipses(ax: plt.Axes, cloud: GaussianCloud, band: Band, tau: float) -> None:
    """One-sigma top-down ellipses of the Gaussians centred in the band; faint below ``tau``."""
    inside = (cloud.means[:, 2] >= band.z_lo) & (cloud.means[:, 2] <= band.z_hi)
    part = cloud.subset(inside)
    variances, vectors = np.linalg.eigh(part.covariances()[:, :2, :2])
    angles = np.degrees(np.arctan2(vectors[:, 1, 1], vectors[:, 0, 1]))
    for keep, color, alpha in ((part.opacities < tau, REJECTED, 0.55),
                               (part.opacities >= tau, CORRECT, 0.75)):
        ax.add_collection(EllipseCollection(
            2 * np.sqrt(variances[keep, 1]), 2 * np.sqrt(variances[keep, 0]), angles[keep],
            units="xy", offsets=part.means[keep, :2], offset_transform=ax.transData,
            facecolors=color, edgecolors="none", alpha=alpha))


def collision_points(scene: Scene, spec: GridSpec, path: Sequence[Cell], robot: Robot,
                     ) -> np.ndarray:
    """World positions along a path where the robot disc intersects the true geometry."""
    points = spec.cell_to_world(np.asarray(path, dtype=float))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    samples = np.concatenate([np.linspace(p, q, max(2, int(d / (spec.resolution / 2)) + 1))
                              for p, q, d in zip(points[:-1], points[1:], lengths, strict=True)])
    solids = [s for s in scene.solids if in_band(s, robot.ground_clearance, robot.height)]
    gaps = np.min([segment_distance(s, samples, samples) for s in solids], axis=0)
    return samples[gaps < robot.radius - COLLISION_TOL]


def _panel(ax: plt.Axes, spec: GridSpec, title: str, color: str = SECONDARY,
           weight: str = "normal") -> None:
    xmin, xmax, ymin, ymax = _extent(spec)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", xticks=[], yticks=[])
    ax.set_title(title, fontsize=8.5, color=color, weight=weight, loc="left", pad=3)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)


def _legend(fig: plt.Figure) -> None:
    """Colour key for the maps, drawn once under the panels."""
    handles = [
        Patch(facecolor=CORRECT, label="occupied, correct"),
        Patch(facecolor=PHANTOM, label="phantom cell (free in reality)"),
        Patch(facecolor=MISSING, label="missing cell (occupied in reality)"),
        Patch(facecolor=HALO, label="robot-radius inflation"),
        Line2D([], [], color=CRITICAL, marker="x", linestyle="", markersize=6,
               markeredgewidth=1.6, label="path inside real geometry"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=5, frameon=False, fontsize=7.2,
               handlelength=1.1, handletextpad=0.5, columnspacing=1.3,
               bbox_to_anchor=(0.5, 0.0))


def hero_frame(scene: Scene, spec: GridSpec, robot: Robot, cloud: GaussianCloud, band: Band,
               scores: dict[str, np.ndarray], tau: float, start: Cell, goal: Cell,
               reference: float) -> Image.Image:
    """One frame of the animation: both methods at one threshold."""
    truth = scene.truth(spec, robot)
    width, height = GIF_SIZE
    fig, axes = plt.subplots(2, 3, figsize=(width / 100, height / 100), dpi=100)
    fig.subplots_adjust(left=0.035, right=0.992, top=0.85, bottom=0.075, wspace=0.025,
                        hspace=0.17)
    fig.text(0.012, 0.958, "When does a Gaussian-splat map lie to the planner?", fontsize=13,
             weight="bold", va="center")
    fig.text(0.992, 0.958, f"opacity threshold  τ = {tau:.2f}", fontsize=13, weight="bold",
             ha="right", va="center")
    bar = fig.add_axes((0.012, 0.912, 0.98, 0.010))
    bar.barh(0, 1, color=GRIDLINE, height=1)
    bar.barh(0, tau, color=INK, height=1)
    bar.set(xlim=(0, 1), ylim=(-0.5, 0.5))
    bar.axis("off")
    for row, method in enumerate(("centres", "footprint")):
        estimate = scores[method] >= tau
        blocked = inflate(estimate, robot.radius, spec.resolution)
        cells = score_map(estimate, truth)
        left, middle, right = axes[row]
        label_y = 0.5 * (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1)
        fig.text(0.011, label_y, SERIES_LABEL[method], rotation=90, fontsize=9.5, weight="bold",
                 color=INK, ha="center", va="center")
        if method == "centres":
            _panel(left, spec, "Gaussians in the band, dark if opacity ≥ τ")
            draw_ellipses(left, cloud, band, tau)
        else:
            _panel(left, spec, "band opacity seen from above, line at τ")
            left.imshow(scores[method], origin="lower", extent=_extent(spec), cmap="Blues",
                        vmin=0, vmax=1, interpolation="nearest")
            left.contour(scores[method], levels=[tau], colors=INK, linewidths=0.5, origin="lower",
                         extent=_extent(spec))
        _panel(middle, spec, f"IoU {cells.iou:.2f}  ·  {cells.fp} phantom  ·  "
                             f"{cells.fn} missing")
        middle.imshow(error_image(estimate, truth), origin="lower", extent=_extent(spec),
                      interpolation="nearest")
        path = astar(blocked, start, goal)
        right.imshow(error_image(estimate, truth, blocked), origin="lower", extent=_extent(spec),
                     interpolation="nearest")
        ends = spec.cell_to_world(np.array([start, goal]))
        if path is None:
            verdict, color = "✕  goal declared unreachable", SECONDARY
        else:
            line = spec.cell_to_world(np.asarray(path))
            hits = collision_points(scene, spec, path, robot)
            right.plot(line[:, 0], line[:, 1], color=INK, linewidth=1.8, solid_capstyle="round")
            if len(hits):
                right.plot(hits[::4, 0], hits[::4, 1], "x", color=CRITICAL, markersize=7,
                           markeredgewidth=1.8)
                verdict, color = "✕  path drives through real geometry", CRITICAL
            else:
                ratio = path_length(path) / reference
                verdict, color = f"✓  safe path, {ratio:.2f} x the optimum", "#006300"
        right.plot(*ends[0], "o", color=GOOD, markersize=7, markeredgecolor=SURFACE,
                   markeredgewidth=1.2)
        right.plot(*ends[1], "s", color=INK, markersize=6.5, markeredgecolor=SURFACE,
                   markeredgewidth=1.2)
        _panel(right, spec, verdict, color, "bold")
    _legend(fig)
    fig.canvas.draw()
    frame = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
    plt.close(fig)
    return frame.convert("RGB")


def save_gif(frames: Sequence[Image.Image], path: Path) -> None:
    """Write an animated GIF with one shared adaptive palette."""
    sheet = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for k, frame in enumerate(frames):
        sheet.paste(frame, (0, k * frame.height))
    palette = sheet.quantize(GIF_COLORS, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=GIF_FRAME_MS,
                    loop=0, optimize=False, disposal=1)


def plot_tradeoff(summary: Sequence[dict], path: Path) -> None:
    """Small multiples: collision and false-unreachable rates against the threshold."""
    rows = [r for r in summary if r["experiment"] == "threshold"]
    metrics = (("collision", "planned paths that collide"),
               ("unreachable", "feasible pairs declared unreachable"))
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.9), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.11, wspace=0.08, hspace=0.22)
    for col, defect in enumerate(DEFECT_ORDER):
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for method, color in SERIES.items():
                points = sorted((r for r in rows if r["defect"] == defect
                                 and r["method"] == method), key=lambda r: float(r["tau"]))
                tau = [float(r["tau"]) for r in points]
                ax.fill_between(tau, [float(r[f"{metric}_pct_min"]) for r in points],
                                [float(r[f"{metric}_pct_max"]) for r in points], color=color,
                                alpha=0.10, linewidth=0)
                ax.plot(tau, [float(r[f"{metric}_pct"]) for r in points], color=color,
                        linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE,
                        markeredgewidth=1, label=SERIES_LABEL[method], solid_capstyle="round")
            ax.set_ylim(-3, 103)
            ax.set_xlim(0, 1)
            ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(f"{defect} defects", fontsize=9.5, color=INK, loc="left")
            else:
                ax.set_xlabel("opacity threshold τ")
            if col == 0:
                ax.set_ylabel(f"{label}\n(% of pairs)", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.955), ncols=2,
               frameon=False, fontsize=9)
    fig.text(0.065, 0.955, "A low threshold blocks the robot, a high one sends it through walls",
             fontsize=12, weight="bold", va="center")
    fig.text(0.065, 0.895, "Mean over the seeded scenes; the band spans the best and the worst "
             "scene. Height band 0.05 to 0.40 m, robot radius 0.20 m.", fontsize=8.5,
             color=SECONDARY, va="center")
    fig.savefig(path, dpi=150)
    plt.close(fig)
