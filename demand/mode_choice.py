"""Binary logit mode choice between car and transit generalized costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModeChoiceResult:
    transit_probability: np.ndarray
    expected_transit_riders: np.ndarray
    expected_car_users: np.ndarray


def run_binary_logit_mode_choice(
    *,
    od_matrix: np.ndarray,
    car_generalized_cost: np.ndarray,
    transit_generalized_cost: np.ndarray,
    logit_theta: float,
) -> ModeChoiceResult:
    """Compute OD transit probability and expected users via binary logit.

    P(transit) = 1 / (1 + exp(-theta * (U_t - U_c)))
    with utility approximated as U = -generalized_cost.
    """
    od = np.asarray(od_matrix, dtype=float)
    car_cost = np.asarray(car_generalized_cost, dtype=float)
    transit_cost = np.asarray(transit_generalized_cost, dtype=float)

    if od.ndim != 2:
        raise ValueError("od_matrix must be 2D.")
    if car_cost.shape != od.shape or transit_cost.shape != od.shape:
        raise ValueError("car_generalized_cost and transit_generalized_cost must match od_matrix shape.")
    if logit_theta <= 0.0:
        raise ValueError("logit_theta must be positive.")

    utility_diff = -transit_cost - (-car_cost)
    z = float(logit_theta) * utility_diff
    z = np.clip(z, -60.0, 60.0)

    transit_probability = 1.0 / (1.0 + np.exp(-z))
    expected_transit_riders = od * transit_probability
    expected_car_users = od - expected_transit_riders

    return ModeChoiceResult(
        transit_probability=transit_probability,
        expected_transit_riders=expected_transit_riders,
        expected_car_users=expected_car_users,
    )
