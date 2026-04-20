"""Stage B: corridor scoring between extracted anchors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimize.anchors import Anchor


@dataclass(frozen=True)
class Corridor:
    """Anchor-to-anchor candidate corridor ranked by OD strength and distance."""

    anchor_a: str
    anchor_b: str
    score: float
    od_flow: float
    distance: float
    path: list[tuple[int, int]]


def _orthogonal_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Simple L-shaped path used for seed construction."""
    x0, y0 = start
    x1, y1 = end
    path: list[tuple[int, int]] = [(x0, y0)]

    sx = 1 if x1 >= x0 else -1
    for x in range(x0 + sx, x1 + sx, sx):
        path.append((x, y0))

    sy = 1 if y1 >= y0 else -1
    for y in range(y0 + sy, y1 + sy, sy):
        if path[-1] != (x1, y):
            path.append((x1, y))
    return path


def score_corridors(
    *,
    anchors: list[Anchor],
    od_matrix: np.ndarray,
    top_k: int = 120,
    distance_penalty: float = 1.15,
) -> list[Corridor]:
    """Score pairwise corridors using OD demand and geometric compactness."""
    if len(anchors) < 2:
        return []

    od = np.asarray(od_matrix, dtype=float)
    n = od.shape[0]
    out: list[Corridor] = []

    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            a = anchors[i]
            b = anchors[j]
            if a.tile_index >= n or b.tile_index >= n:
                continue
            flow_ab = float(od[a.tile_index, b.tile_index] + od[b.tile_index, a.tile_index])
            dist = float(np.hypot(a.x - b.x, a.y - b.y))
            score = flow_ab / ((1.0 + dist) ** distance_penalty)
            path = _orthogonal_path((a.x, a.y), (b.x, b.y))
            out.append(
                Corridor(
                    anchor_a=a.id,
                    anchor_b=b.id,
                    score=score,
                    od_flow=flow_ab,
                    distance=dist,
                    path=path,
                )
            )

    out.sort(key=lambda c: c.score, reverse=True)
    return out[:top_k]
