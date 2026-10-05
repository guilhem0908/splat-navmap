"""Grid planners: A* (8-connected) and Theta* (any-angle), both without corner cutting.

Both take a boolean ``blocked`` array indexed ``[iy, ix]`` and cells as ``(ix, iy)`` and return
the list of cells from start to goal, or ``None`` when the goal cannot be reached.

* A* (Hart, Nilsson, Raphael 1968) with the octile heuristic, which is consistent on an
  8-connected grid, so the first expansion of the goal is optimal. A diagonal move is allowed
  only when the two orthogonal neighbours are free.
* Basic Theta* (Daniel, Nash, Koenig, Felner, JAIR 2010): identical search, but a neighbour
  ``n`` of the expanded cell ``c`` is connected straight to ``parent(c)`` whenever the two are in
  line of sight ("path 2"), which removes the 45-degree heading constraint.

Line of sight is the supercover of the segment between two cell centres: every cell whose
closed square meets the segment must be free. A segment between cell centres can touch a
square without entering it only at a corner, where the same rule as for diagonal moves applies.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np

Cell = tuple[int, int]
Planner = Callable[[np.ndarray, Cell, Cell], "list[Cell] | None"]
_SQRT2 = math.sqrt(2.0)


def astar(blocked: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Shortest 8-connected path (every cell listed), or ``None``."""
    return _search(blocked, start, goal, any_angle=False)


def theta_star(blocked: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Any-angle path (turning points only, consecutive ones in line of sight), or ``None``."""
    return _search(blocked, start, goal, any_angle=True)


def line_of_sight(blocked: np.ndarray, a: Cell, b: Cell) -> bool:
    """True when the segment between the centres of ``a`` and ``b`` meets no blocked cell."""
    occ, width = _pad(blocked)
    return _visible(occ, width, a[0] + 1, a[1] + 1, b[0] + 1, b[1] + 1)


def path_length(path: Sequence[Cell]) -> float:
    """Euclidean length of a cell path, in cells."""
    return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in itertools.pairwise(path))


def _pad(blocked: np.ndarray) -> tuple[list[bool], int]:
    """Flatten the grid inside a blocked border so that neighbour offsets never leave it."""
    padded = np.pad(blocked.astype(bool), 1, constant_values=True)
    return padded.ravel().tolist(), padded.shape[1]


def _visible(occ: list[bool], width: int, x0: int, y0: int, x1: int, y1: int) -> bool:
    """Integer supercover walk from (x0, y0) to (x1, y1) on the padded grid.

    ``err`` is twice the signed distance (scaled by the segment length) between the current
    cell centre and the line; its sign tells which cell edge the segment leaves through, and
    ``err == 0`` means it leaves exactly through a corner.
    """
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    step_x = 1 if x1 > x0 else -1
    step_y = width if y1 > y0 else -width
    cell, target = y0 * width + x0, y1 * width + x1
    err, dx2, dy2 = dx - dy, 2 * dx, 2 * dy
    while cell != target:
        if err > 0:
            cell += step_x
            err -= dy2
        elif err < 0:
            cell += step_y
            err += dx2
        else:
            if occ[cell + step_x] or occ[cell + step_y]:
                return False
            cell += step_x + step_y
            err += dx2 - dy2
        if occ[cell]:
            return False
    return True


def _search(blocked: np.ndarray, start: Cell, goal: Cell, *, any_angle: bool) -> list[Cell] | None:
    ny, nx = blocked.shape
    for ix, iy in (start, goal):
        if not (0 <= ix < nx and 0 <= iy < ny) or blocked[iy, ix]:
            return None
    occ, width = _pad(blocked)
    source = (start[1] + 1) * width + start[0] + 1
    target = (goal[1] + 1) * width + goal[0] + 1
    goal_x, goal_y = goal[0] + 1, goal[1] + 1
    # (flat offset, step cost, x part of the offset, y part of the offset)
    moves = [(sy * width + sx, _SQRT2 if sx and sy else 1.0, sx, sy * width)
             for sy in (-1, 0, 1) for sx in (-1, 0, 1) if sx or sy]

    def heuristic(cell: int) -> float:
        y, x = divmod(cell, width)
        ax, ay = abs(x - goal_x), abs(y - goal_y)
        if any_angle:
            return math.hypot(ax, ay)
        return ax + ay + (_SQRT2 - 2.0) * min(ax, ay)

    cost = {source: 0.0}
    parent = {source: source}
    closed = bytearray(len(occ))
    frontier = [(heuristic(source), source)]
    while frontier:
        _, cell = heapq.heappop(frontier)
        if closed[cell]:
            continue
        closed[cell] = 1
        if cell == target:
            break
        cost_cell, anchor = cost[cell], parent[cell]
        anchor_y, anchor_x = divmod(anchor, width)
        for offset, step, off_x, off_y in moves:
            nxt = cell + offset
            if occ[nxt] or closed[nxt]:
                continue
            if off_x and off_y and (occ[cell + off_x] or occ[cell + off_y]):
                continue
            best = cost.get(nxt, math.inf)
            via, candidate = cell, cost_cell + step
            if any_angle:
                # Path 2 (straight from the parent) is never longer than path 1 by the triangle
                # inequality, so when it cannot improve the neighbour nothing can.
                nxt_y, nxt_x = divmod(nxt, width)
                shortcut = cost[anchor] + math.hypot(nxt_x - anchor_x, nxt_y - anchor_y)
                if shortcut >= best:
                    continue
                if _visible(occ, width, anchor_x, anchor_y, nxt_x, nxt_y):
                    via, candidate = anchor, shortcut
            if candidate < best:
                cost[nxt], parent[nxt] = candidate, via
                heapq.heappush(frontier, (candidate + heuristic(nxt), nxt))
    if not closed[target]:
        return None
    cells = [target]
    while cells[-1] != source:
        cells.append(parent[cells[-1]])
    return [(c % width - 1, c // width - 1) for c in reversed(cells)]
