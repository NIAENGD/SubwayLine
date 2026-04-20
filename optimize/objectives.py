"""Objective metrics and composite scoring profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from optimize.initial_builder import NetworkPlan


PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "ridership_max": {
        "ridership_capture": 0.55,
        "accessibility_improvement": 0.15,
        "population_coverage": 0.10,
        "directness": 0.15,
        "operating_cost": -0.04,
        "equity_penalty": -0.01,
    },
    "accessibility_max": {
        "ridership_capture": 0.20,
        "accessibility_improvement": 0.50,
        "population_coverage": 0.12,
        "directness": 0.12,
        "operating_cost": -0.04,
        "equity_penalty": -0.02,
    },
    "balanced": {
        "ridership_capture": 0.35,
        "accessibility_improvement": 0.25,
        "population_coverage": 0.15,
        "directness": 0.10,
        "operating_cost": -0.10,
        "equity_penalty": -0.05,
    },
    "low_cost": {
        "ridership_capture": 0.22,
        "accessibility_improvement": 0.18,
        "population_coverage": 0.10,
        "directness": 0.10,
        "operating_cost": -0.35,
        "equity_penalty": -0.05,
    },
    "coverage_max": {
        "ridership_capture": 0.20,
        "accessibility_improvement": 0.15,
        "population_coverage": 0.45,
        "directness": 0.10,
        "operating_cost": -0.05,
        "equity_penalty": -0.05,
    },
}


@dataclass(frozen=True)
class NetworkMetrics:
    ridership_capture: float
    accessibility_improvement: float
    population_coverage: float
    directness: float
    operating_cost: float
    equity_penalty: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreSummary:
    profile: str
    score: float
    metrics: NetworkMetrics
    weighted_terms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "score": self.score,
            "metrics": self.metrics.to_dict(),
            "weighted_terms": dict(self.weighted_terms),
        }


def _line_length(points: list[tuple[int, int]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        sum(np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]))
    )


def _all_station_points(plan: NetworkPlan) -> np.ndarray:
    pts: list[tuple[int, int]] = []
    for line in plan.lines:
        pts.extend(line.points)
    if not pts:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)


def evaluate_metrics(
    *,
    plan: NetworkPlan,
    population: np.ndarray,
    jobs: np.ndarray,
    od_matrix: np.ndarray,
    access_radius_tiles: float = 2.0,
) -> NetworkMetrics:
    """Compute normalized metrics for objective scoring."""
    pop = np.asarray(population, dtype=float)
    jobs_arr = np.asarray(jobs, dtype=float)
    rows, cols = pop.shape

    stations = _all_station_points(plan)
    yy, xx = np.indices((rows, cols), dtype=float)
    if stations.size == 0:
        min_dist = np.full((rows, cols), np.inf)
    else:
        dists = np.sqrt((xx[..., None] - stations[:, 0]) ** 2 + (yy[..., None] - stations[:, 1]) ** 2)
        min_dist = dists.min(axis=2)

    covered = min_dist <= access_radius_tiles
    total_pop = max(float(pop.sum()), 1e-9)
    population_coverage = float(pop[covered].sum() / total_pop)

    reachable_jobs = float(jobs_arr[covered].sum())
    accessibility_improvement = min(1.0, reachable_jobs / max(float(jobs_arr.sum()), 1e-9))

    tile_count = rows * cols
    if od_matrix.shape == (tile_count, tile_count):
        od = od_matrix.reshape(rows, cols, rows, cols)
        mask_o = covered.reshape(rows, cols, 1, 1)
        mask_d = covered.reshape(1, 1, rows, cols)
        captured = float(od[mask_o & mask_d].sum())
        ridership_capture = captured / max(float(od_matrix.sum()), 1e-9)
    else:
        ridership_capture = 0.0

    total_length = sum(_line_length(line.points) for line in plan.lines)
    directness = 1.0 / (1.0 + total_length / max(1, len(plan.lines)))
    operating_cost = total_length * (1.0 + 0.15 * len(plan.lines))

    if covered.any():
        neighborhood_access = covered.astype(float)
        equity_penalty = float(np.std(neighborhood_access))
    else:
        equity_penalty = 1.0

    return NetworkMetrics(
        ridership_capture=ridership_capture,
        accessibility_improvement=accessibility_improvement,
        population_coverage=population_coverage,
        directness=directness,
        operating_cost=operating_cost,
        equity_penalty=equity_penalty,
    )


def score_network(
    *,
    plan: NetworkPlan,
    population: np.ndarray,
    jobs: np.ndarray,
    od_matrix: np.ndarray,
    profile: str = "balanced",
) -> ScoreSummary:
    """Compute weighted objective score using one of the required profiles."""
    weights = PROFILE_WEIGHTS.get(profile)
    if weights is None:
        valid = ", ".join(sorted(PROFILE_WEIGHTS))
        raise ValueError(f"Unknown objective profile '{profile}'. Valid: {valid}")

    metrics = evaluate_metrics(plan=plan, population=population, jobs=jobs, od_matrix=od_matrix)

    weighted_terms: dict[str, float] = {}
    for name, weight in weights.items():
        value = getattr(metrics, name)
        if name == "operating_cost":
            value = value / 500.0
        weighted_terms[name] = weight * value

    score = float(sum(weighted_terms.values()))
    return ScoreSummary(profile=profile, score=score, metrics=metrics, weighted_terms=weighted_terms)
