"""Network metric component calculations for optimization explainability."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from optimize.initial_builder import NetworkPlan


@dataclass(frozen=True)
class MetricComponents:
    """Component-level objective metrics for explainable scoring."""

    ridership_capture: float
    accessibility_improvement: float
    coverage: float
    directness_vs_car: float
    operating_cost: float
    equity_penalty: float

    @property
    def population_coverage(self) -> float:
        """Backward-compatible alias for coverage."""
        return self.coverage

    @property
    def directness(self) -> float:
        """Backward-compatible alias for directness_vs_car."""
        return self.directness_vs_car

    def to_dict(self) -> dict[str, float]:
        out = asdict(self)
        # Include compatibility aliases for older callers and serialized outputs.
        out["population_coverage"] = self.coverage
        out["directness"] = self.directness_vs_car
        return out


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


def compute_metric_components(
    *,
    plan: NetworkPlan,
    population: np.ndarray,
    jobs: np.ndarray,
    od_matrix: np.ndarray,
    access_radius_tiles: float = 2.0,
) -> MetricComponents:
    """Compute objective component metrics used by optimization profiles."""
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
    coverage = float(pop[covered].sum() / total_pop)

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
    avg_line_length = total_length / max(1, len(plan.lines))
    directness_vs_car = 1.0 / (1.0 + avg_line_length)
    operating_cost = total_length * (1.0 + 0.15 * len(plan.lines))

    if covered.any():
        neighborhood_access = covered.astype(float)
        equity_penalty = float(np.std(neighborhood_access))
    else:
        equity_penalty = 1.0

    return MetricComponents(
        ridership_capture=ridership_capture,
        accessibility_improvement=accessibility_improvement,
        coverage=coverage,
        directness_vs_car=directness_vs_car,
        operating_cost=operating_cost,
        equity_penalty=equity_penalty,
    )
