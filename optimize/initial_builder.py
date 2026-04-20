"""Stage C: initial network generation from scored corridors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from optimize.corridor_scoring import Corridor


@dataclass
class NetworkLine:
    id: str
    points: list[tuple[int, int]]


@dataclass
class NetworkPlan:
    lines: list[NetworkLine]
    metadata: dict[str, Any] = field(default_factory=dict)


def _dedupe_adjacent(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        if p != out[-1]:
            out.append(p)
    return out


def _pick_non_overlapping_corridor(
    corridors: list[Corridor],
    used_points: set[tuple[int, int]],
    *,
    min_novel_ratio: float = 0.35,
) -> Corridor | None:
    for cor in corridors:
        pts = set(cor.path)
        if not pts:
            continue
        novel_ratio = len(pts - used_points) / max(1, len(pts))
        if novel_ratio >= min_novel_ratio:
            return cor
    return corridors[0] if corridors else None


def _clip_to_grid(points: list[tuple[int, int]], grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = grid_shape
    clipped: list[tuple[int, int]] = []
    for x, y in points:
        cx = int(np.clip(x, 0, cols - 1))
        cy = int(np.clip(y, 0, rows - 1))
        clipped.append((cx, cy))
    return _dedupe_adjacent(clipped)


def build_initial_network(
    *,
    line_count: int,
    corridors: list[Corridor],
    grid_shape: tuple[int, int],
    rng: np.random.Generator,
    warm_start: NetworkPlan | None = None,
) -> NetworkPlan:
    """Build an initial network using warm-start chaining plus corridor seeding."""
    if line_count <= 0:
        raise ValueError("line_count must be positive")

    lines: list[NetworkLine] = []
    used_points: set[tuple[int, int]] = set()

    if warm_start is not None:
        for line in warm_start.lines[:line_count]:
            points = _clip_to_grid(line.points, grid_shape)
            if len(points) >= 2:
                lines.append(NetworkLine(id=f"L{len(lines)+1}", points=points))
                used_points.update(points)

    ranked = sorted(corridors, key=lambda c: c.score, reverse=True)
    while len(lines) < line_count:
        picked = _pick_non_overlapping_corridor(ranked, used_points)
        if picked is None:
            rows, cols = grid_shape
            x0, y0 = int(rng.integers(0, cols)), int(rng.integers(0, rows))
            x1, y1 = int(rng.integers(0, cols)), int(rng.integers(0, rows))
            points = _clip_to_grid([(x0, y0), (x1, y1)], grid_shape)
        else:
            points = _clip_to_grid(picked.path, grid_shape)
            if points and points[-1] in used_points:
                points = list(reversed(points))

        if len(points) < 2:
            continue
        lines.append(NetworkLine(id=f"L{len(lines)+1}", points=points))
        used_points.update(points)

    return NetworkPlan(
        lines=lines,
        metadata={
            "stage": "initial_builder",
            "line_count": line_count,
            "warm_started": warm_start is not None,
        },
    )
