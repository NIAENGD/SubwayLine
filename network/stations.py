"""Transit network station and line data model + candidate generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Station:
    """Transit station candidate or selected station."""

    id: str
    x: float
    y: float
    name: str
    score: float


@dataclass(frozen=True)
class Line:
    """Ordered station sequence for a transit line."""

    id: str
    stations: list[str]
    mode: str
    headway_min: float


@dataclass(frozen=True)
class Segment:
    """Directed segment between adjacent stations."""

    from_station: str
    to_station: str
    geometry: list[tuple[float, float]]
    length_miles: float


@dataclass(frozen=True)
class CandidateConfig:
    """Configuration for station candidate extraction."""

    min_spacing_tiles: float = 2.0
    max_candidates: int = 120
    center_bonus: float = 1.0
    residential_weight: float = 0.9
    jobs_population_weight: float = 1.0
    throughput_weight: float = 1.1


def _normalize(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    lo = float(np.min(field))
    hi = float(np.max(field))
    if hi - lo <= 1e-12:
        return np.zeros_like(field, dtype=float)
    return (field - lo) / (hi - lo)


def _local_maxima(field: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Return local maxima (y, x, value) above threshold in an 8-neighborhood."""
    arr = np.asarray(field, dtype=float)
    rows, cols = arr.shape
    out: list[tuple[int, int, float]] = []
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            v = arr[y, x]
            if v < threshold:
                continue
            neighborhood = arr[y - 1 : y + 2, x - 1 : x + 2]
            if v >= float(np.max(neighborhood)):
                out.append((y, x, float(v)))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _greedy_spacing_filter(
    candidates: list[tuple[int, int, float]],
    *,
    min_spacing_tiles: float,
    max_candidates: int,
) -> list[tuple[int, int, float]]:
    selected: list[tuple[int, int, float]] = []
    for y, x, score in candidates:
        if len(selected) >= max_candidates:
            break
        p = (float(x), float(y))
        if any(_distance((float(px), float(py)), p) < min_spacing_tiles for py, px, _ in selected):
            continue
        selected.append((y, x, score))
    return selected


def generate_station_candidates(
    *,
    population: np.ndarray,
    jobs: np.ndarray,
    od_throughput: np.ndarray,
    center_tiles: list[tuple[int, int]],
    config: CandidateConfig | None = None,
) -> list[Station]:
    """Generate station candidates from center tiles, clusters, and high-demand tiles."""
    cfg = config or CandidateConfig()
    pop_n = _normalize(population)
    jobs_n = _normalize(jobs)
    throughput_n = _normalize(od_throughput)

    combined = cfg.jobs_population_weight * (0.55 * jobs_n + 0.45 * pop_n)
    residential = pop_n * (1.0 - jobs_n)
    residential_n = _normalize(residential)

    score_field = (
        combined
        + cfg.residential_weight * residential_n
        + cfg.throughput_weight * throughput_n
    )

    for cy, cx in center_tiles:
        if 0 <= cy < score_field.shape[0] and 0 <= cx < score_field.shape[1]:
            score_field[cy, cx] += cfg.center_bonus

    hot_threshold = float(np.quantile(score_field, 0.90))
    res_threshold = float(np.quantile(residential_n, 0.93))
    thr_threshold = float(np.quantile(throughput_n, 0.93))

    selected_tiles: dict[tuple[int, int], float] = {}

    def absorb(cands: list[tuple[int, int, float]], base_boost: float = 0.0) -> None:
        for y, x, value in cands:
            selected_tiles[(y, x)] = max(selected_tiles.get((y, x), -1e18), value + base_boost)

    absorb(_local_maxima(score_field, hot_threshold))
    absorb(_local_maxima(residential_n, res_threshold), base_boost=0.1)
    absorb(_local_maxima(throughput_n, thr_threshold), base_boost=0.12)

    for cy, cx in center_tiles:
        if 0 <= cy < score_field.shape[0] and 0 <= cx < score_field.shape[1]:
            selected_tiles[(cy, cx)] = max(selected_tiles.get((cy, cx), -1e18), float(score_field[cy, cx]) + 0.15)

    ranked = sorted(((y, x, s) for (y, x), s in selected_tiles.items()), key=lambda t: t[2], reverse=True)
    spaced = _greedy_spacing_filter(
        ranked,
        min_spacing_tiles=cfg.min_spacing_tiles,
        max_candidates=cfg.max_candidates,
    )

    stations: list[Station] = []
    for idx, (y, x, score) in enumerate(spaced, start=1):
        stations.append(
            Station(
                id=f"S{idx:03d}",
                x=float(x),
                y=float(y),
                name=f"Station {idx:03d}",
                score=float(score),
            )
        )
    return stations
