"""Rasterization helpers for line segments and shared-track overlap penalties."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from network.stations import Line, Station


@dataclass(frozen=True)
class OverlapSummary:
    segment_overlap_counts: dict[tuple[str, str], int]
    line_overlap_penalty: dict[str, float]


def rasterize_segment(a: tuple[float, float], b: tuple[float, float]) -> list[tuple[int, int]]:
    """Rasterize segment using integer Bresenham on tile-center coordinates (x, y)."""
    x0, y0 = int(round(a[0])), int(round(a[1]))
    x1, y1 = int(round(b[0])), int(round(b[1]))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    out: list[tuple[int, int]] = []
    while True:
        out.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return out


def _canon_tile_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def build_segment_tile_edges(
    lines: list[Line],
    stations_by_id: dict[str, Station],
) -> dict[tuple[str, str], set[tuple[tuple[int, int], tuple[int, int]]]]:
    """Map (line_id, segment_key) -> canonical tile-edge set."""
    edges_by_segment: dict[tuple[str, str], set[tuple[tuple[int, int], tuple[int, int]]]] = {}

    for line in lines:
        for i in range(len(line.stations) - 1):
            s0 = line.stations[i]
            s1 = line.stations[i + 1]
            p0 = stations_by_id[s0]
            p1 = stations_by_id[s1]
            tiles = rasterize_segment((p0.x, p0.y), (p1.x, p1.y))
            seg_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for k in range(len(tiles) - 1):
                seg_edges.add(_canon_tile_edge(tiles[k], tiles[k + 1]))
            seg_key = f"{s0}->{s1}"
            edges_by_segment[(line.id, seg_key)] = seg_edges

    return edges_by_segment


def compute_overlap_penalties(
    lines: list[Line],
    stations_by_id: dict[str, Station],
    *,
    penalty_per_shared_edge: float = 1.0,
) -> OverlapSummary:
    """Compute overlap penalties for shared stations/track edges across lines."""
    segment_edges = build_segment_tile_edges(lines, stations_by_id)
    edge_usage: defaultdict[tuple[tuple[int, int], tuple[int, int]], set[tuple[str, str]]] = defaultdict(set)

    for seg_ref, edges in segment_edges.items():
        for edge in edges:
            edge_usage[edge].add(seg_ref)

    overlap_counts: dict[tuple[str, str], int] = {seg_ref: 0 for seg_ref in segment_edges}
    for refs in edge_usage.values():
        if len(refs) <= 1:
            continue
        for seg_ref in refs:
            overlap_counts[seg_ref] += len(refs) - 1

    line_penalty: defaultdict[str, float] = defaultdict(float)
    for (line_id, seg_key), count in overlap_counts.items():
        if count <= 0:
            continue
        line_penalty[line_id] += penalty_per_shared_edge * float(count)

    return OverlapSummary(
        segment_overlap_counts=overlap_counts,
        line_overlap_penalty=dict(line_penalty),
    )
