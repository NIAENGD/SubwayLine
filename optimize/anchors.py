"""Stage A: anchor extraction from jobs, population, and OD demand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


AnchorKind = Literal["job_hub", "residential_cluster", "transfer_opportunity"]


@dataclass(frozen=True)
class Anchor:
    """Important node used to seed corridor and line construction."""

    id: str
    kind: AnchorKind
    x: int
    y: int
    weight: float
    tile_index: int


def _top_tiles(field: np.ndarray, count: int, *, min_separation: float = 4.0) -> list[tuple[int, int, float]]:
    """Select top tiles with greedy spacing filter to avoid near-duplicates."""
    flat = np.asarray(field, dtype=float).reshape(-1)
    order = np.argsort(flat)[::-1]
    cols = field.shape[1]

    selected: list[tuple[int, int, float]] = []
    for idx in order:
        y, x = divmod(int(idx), cols)
        score = float(flat[idx])
        if score <= 0:
            break
        point = np.array([x, y], dtype=float)
        if any(np.linalg.norm(point - np.array([sx, sy], dtype=float)) < min_separation for sy, sx, _ in selected):
            continue
        selected.append((y, x, score))
        if len(selected) >= count:
            break
    return selected


def extract_anchors(
    *,
    population: np.ndarray,
    jobs: np.ndarray,
    od_matrix: np.ndarray,
    job_anchor_count: int = 10,
    residential_anchor_count: int = 10,
    transfer_anchor_count: int = 8,
) -> list[Anchor]:
    """Extract job, residential, and transfer anchors for staged optimization.

    The transfer opportunity field is based on total OD throughput by tile
    (`origin_flow + destination_flow`).
    """
    pop = np.asarray(population, dtype=float)
    jobs_arr = np.asarray(jobs, dtype=float)
    if pop.shape != jobs_arr.shape:
        raise ValueError("population and jobs must have the same shape")

    rows, cols = pop.shape
    n_tiles = rows * cols
    od = np.asarray(od_matrix, dtype=float)
    if od.shape != (n_tiles, n_tiles):
        raise ValueError("od_matrix shape must be (grid_tiles, grid_tiles)")

    origin = od.sum(axis=1).reshape(rows, cols)
    destination = od.sum(axis=0).reshape(rows, cols)
    transfer_field = origin + destination

    anchors: list[Anchor] = []
    next_id = 1

    for y, x, score in _top_tiles(jobs_arr, job_anchor_count):
        tile_idx = y * cols + x
        anchors.append(Anchor(id=f"A{next_id:03d}", kind="job_hub", x=x, y=y, weight=score, tile_index=tile_idx))
        next_id += 1

    for y, x, score in _top_tiles(pop, residential_anchor_count):
        tile_idx = y * cols + x
        anchors.append(
            Anchor(
                id=f"A{next_id:03d}",
                kind="residential_cluster",
                x=x,
                y=y,
                weight=score,
                tile_index=tile_idx,
            )
        )
        next_id += 1

    for y, x, score in _top_tiles(transfer_field, transfer_anchor_count, min_separation=3.0):
        tile_idx = y * cols + x
        anchors.append(
            Anchor(
                id=f"A{next_id:03d}",
                kind="transfer_opportunity",
                x=x,
                y=y,
                weight=score,
                tile_index=tile_idx,
            )
        )
        next_id += 1

    return anchors
