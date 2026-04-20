"""Objective metrics and composite scoring profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from network.metrics import MetricComponents, compute_metric_components
from optimize.initial_builder import NetworkPlan

REQUIRED_PROFILES: tuple[str, ...] = (
    "ridership-max",
    "accessibility-max",
    "balanced",
    "low-cost",
    "coverage-max",
)

# Hyphenated required profile IDs with balanced as the default weighting profile.
PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "ridership-max": {
        "ridership_capture": 0.55,
        "accessibility_improvement": 0.15,
        "coverage": 0.10,
        "directness_vs_car": 0.15,
        "operating_cost": -0.04,
        "equity_penalty": -0.01,
    },
    "accessibility-max": {
        "ridership_capture": 0.20,
        "accessibility_improvement": 0.50,
        "coverage": 0.12,
        "directness_vs_car": 0.12,
        "operating_cost": -0.04,
        "equity_penalty": -0.02,
    },
    "balanced": {
        "ridership_capture": 0.35,
        "accessibility_improvement": 0.25,
        "coverage": 0.15,
        "directness_vs_car": 0.10,
        "operating_cost": -0.10,
        "equity_penalty": -0.05,
    },
    "low-cost": {
        "ridership_capture": 0.22,
        "accessibility_improvement": 0.18,
        "coverage": 0.10,
        "directness_vs_car": 0.10,
        "operating_cost": -0.35,
        "equity_penalty": -0.05,
    },
    "coverage-max": {
        "ridership_capture": 0.20,
        "accessibility_improvement": 0.15,
        "coverage": 0.45,
        "directness_vs_car": 0.10,
        "operating_cost": -0.05,
        "equity_penalty": -0.05,
    },
}

# Backward compatibility with earlier underscore profile labels.
PROFILE_ALIASES: dict[str, str] = {
    "ridership_max": "ridership-max",
    "accessibility_max": "accessibility-max",
    "low_cost": "low-cost",
    "coverage_max": "coverage-max",
}


NetworkMetrics = MetricComponents


@dataclass(frozen=True)
class ScoreSummary:
    profile: str
    score: float
    metrics: NetworkMetrics
    weighted_terms: dict[str, float]

    @property
    def components(self) -> dict[str, float]:
        """Explainable metric breakdown for GUI results tabs."""
        return self.metrics.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "score": self.score,
            "metrics": self.metrics.to_dict(),
            "components": self.components,
            "weighted_terms": dict(self.weighted_terms),
        }


def evaluate_metrics(
    *,
    plan: NetworkPlan,
    population: np.ndarray,
    jobs: np.ndarray,
    od_matrix: np.ndarray,
    access_radius_tiles: float = 2.0,
) -> NetworkMetrics:
    """Compute normalized metric components for objective scoring."""
    return compute_metric_components(
        plan=plan,
        population=population,
        jobs=jobs,
        od_matrix=od_matrix,
        access_radius_tiles=access_radius_tiles,
    )


def score_network(
    *,
    plan: NetworkPlan,
    population: np.ndarray,
    jobs: np.ndarray,
    od_matrix: np.ndarray,
    profile: str = "balanced",
) -> ScoreSummary:
    """Compute weighted objective score and explainable component breakdown."""
    canonical_profile = PROFILE_ALIASES.get(profile, profile)
    weights = PROFILE_WEIGHTS.get(canonical_profile)
    if weights is None:
        valid = ", ".join(REQUIRED_PROFILES)
        raise ValueError(f"Unknown objective profile '{profile}'. Valid: {valid}")

    metrics = evaluate_metrics(plan=plan, population=population, jobs=jobs, od_matrix=od_matrix)

    weighted_terms: dict[str, float] = {}
    for name, weight in weights.items():
        value = getattr(metrics, name)
        if name == "operating_cost":
            value = value / 500.0
        weighted_terms[name] = weight * value

    score = float(sum(weighted_terms.values()))
    return ScoreSummary(
        profile=canonical_profile,
        score=score,
        metrics=metrics,
        weighted_terms=weighted_terms,
    )
