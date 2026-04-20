"""Stage D: simulated annealing refinement for transit network plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimize.initial_builder import NetworkPlan
from optimize.mutations import apply_random_mutation
from optimize.objectives import ScoreSummary, score_network


@dataclass(frozen=True)
class AnnealingResult:
    best_plan: NetworkPlan
    best_summary: ScoreSummary
    iterations: int
    accepted_moves: int


def _temperature_at(i: int, total: int, t0: float, t1: float) -> float:
    if total <= 1:
        return t1
    frac = i / (total - 1)
    return t0 * ((t1 / t0) ** frac)


def refine_with_annealing(
    *,
    initial_plan: NetworkPlan,
    population: np.ndarray,
    jobs: np.ndarray,
    od_matrix: np.ndarray,
    objective_profile: str,
    line_count: int,
    grid_shape: tuple[int, int],
    iterations: int,
    temperature_schedule: tuple[float, float],
    rng: np.random.Generator,
) -> AnnealingResult:
    """Run simulated annealing with required mutation operators."""
    t0, t1 = temperature_schedule
    if t0 <= 0 or t1 <= 0:
        raise ValueError("temperature_schedule must contain positive values")

    current = initial_plan
    current_summary = score_network(
        plan=current,
        population=population,
        jobs=jobs,
        od_matrix=od_matrix,
        profile=objective_profile,
    )

    best = current
    best_summary = current_summary
    accepted = 0

    for i in range(iterations):
        temp = _temperature_at(i, iterations, t0, t1)
        candidate = apply_random_mutation(
            plan=current,
            rng=rng,
            grid_shape=grid_shape,
            target_line_count=line_count,
        )
        cand_summary = score_network(
            plan=candidate,
            population=population,
            jobs=jobs,
            od_matrix=od_matrix,
            profile=objective_profile,
        )

        delta = cand_summary.score - current_summary.score
        accept = delta >= 0 or float(rng.random()) < np.exp(delta / max(temp, 1e-9))
        if accept:
            current = candidate
            current_summary = cand_summary
            accepted += 1

        if current_summary.score > best_summary.score:
            best = current
            best_summary = current_summary

    return AnnealingResult(
        best_plan=best,
        best_summary=best_summary,
        iterations=iterations,
        accepted_moves=accepted,
    )
