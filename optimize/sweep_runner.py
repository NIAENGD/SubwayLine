"""Run staged optimization sweeps with warm-start chaining (1..N lines)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from optimize.anchors import Anchor, extract_anchors
from optimize.annealing import AnnealingResult, refine_with_annealing
from optimize.corridor_scoring import Corridor, score_corridors
from optimize.initial_builder import NetworkPlan, build_initial_network


@dataclass(frozen=True)
class LineCountResult:
    line_count: int
    best_plan: NetworkPlan
    score_summary: dict[str, Any]
    metrics: dict[str, float]
    geometry: list[dict[str, Any]]


def _serialize_plan(plan: NetworkPlan) -> dict[str, Any]:
    return {
        "lines": [{"id": line.id, "points": [[x, y] for x, y in line.points]} for line in plan.lines],
        "metadata": dict(plan.metadata),
    }


def _geometry_output(plan: NetworkPlan) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in plan.lines:
        out.append({"line_id": line.id, "polyline": [[x, y] for x, y in line.points]})
    return out


def _save_line_count_outputs(base_dir: Path, result: LineCountResult) -> None:
    run_dir = base_dir / f"lines_{result.line_count}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "geometry.json").write_text(json.dumps(result.geometry, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    (run_dir / "score_summary.json").write_text(json.dumps(result.score_summary, indent=2), encoding="utf-8")
    (run_dir / "best_network.json").write_text(
        json.dumps(_serialize_plan(result.best_plan), indent=2),
        encoding="utf-8",
    )


def run_staged_line_sweep(
    *,
    population: np.ndarray,
    jobs: np.ndarray,
    od_matrix: np.ndarray,
    max_lines: int = 8,
    objective_profile: str = "balanced",
    iterations: int = 250,
    temperature_schedule: tuple[float, float] = (1.0, 0.01),
    output_dir: str | Path = "outputs/optimization_sweep",
    random_seed: int = 0,
) -> dict[int, LineCountResult]:
    """Solve 1-line through N-line cases using warm-start chaining.

    Stages:
      A) anchor extraction
      B) corridor scoring
      C) initial build
      D) annealing refinement
    """
    if max_lines <= 0:
        raise ValueError("max_lines must be positive")

    pop = np.asarray(population, dtype=float)
    jobs_arr = np.asarray(jobs, dtype=float)
    rng = np.random.default_rng(random_seed)

    anchors: list[Anchor] = extract_anchors(population=pop, jobs=jobs_arr, od_matrix=od_matrix)
    corridors: list[Corridor] = score_corridors(anchors=anchors, od_matrix=od_matrix)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, LineCountResult] = {}
    warm_start: NetworkPlan | None = None

    for line_count in range(1, max_lines + 1):
        initial = build_initial_network(
            line_count=line_count,
            corridors=corridors,
            grid_shape=pop.shape,
            rng=rng,
            warm_start=warm_start,
        )

        annealed: AnnealingResult = refine_with_annealing(
            initial_plan=initial,
            population=pop,
            jobs=jobs_arr,
            od_matrix=od_matrix,
            objective_profile=objective_profile,
            line_count=line_count,
            grid_shape=pop.shape,
            iterations=iterations,
            temperature_schedule=temperature_schedule,
            rng=rng,
        )

        warm_start = annealed.best_plan
        summary = annealed.best_summary.to_dict()
        metrics = annealed.best_summary.metrics.to_dict()
        geometry = _geometry_output(annealed.best_plan)

        result = LineCountResult(
            line_count=line_count,
            best_plan=annealed.best_plan,
            score_summary=summary,
            metrics=metrics,
            geometry=geometry,
        )
        results[line_count] = result
        _save_line_count_outputs(out_dir, result)

    manifest = {
        "max_lines": max_lines,
        "objective_profile": objective_profile,
        "random_seed": random_seed,
        "line_counts": list(range(1, max_lines + 1)),
        "results": {
            str(k): {
                "line_count": v.line_count,
                "score": v.score_summary.get("score"),
                "metrics": v.metrics,
            }
            for k, v in results.items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results
