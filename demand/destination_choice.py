"""Gravity-style destination attractiveness and probabilities."""

from __future__ import annotations

import numpy as np


def gravity_attractiveness(
    *,
    jobs: np.ndarray,
    car_time_min: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Return OD attractiveness weights jobs^alpha * exp(-beta * car_time)."""
    jobs_grid = np.asarray(jobs, dtype=float)
    if jobs_grid.ndim != 2:
        raise ValueError("jobs must be a 2D array.")

    times = np.asarray(car_time_min, dtype=float)
    if times.ndim != 2:
        raise ValueError("car_time_min must be a 2D OD matrix.")

    n_tiles = jobs_grid.size
    if times.shape != (n_tiles, n_tiles):
        raise ValueError("car_time_min shape must be (n_tiles, n_tiles).")

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")

    jobs_flat = np.clip(jobs_grid.reshape(-1), 0.0, None)
    dest_term = np.power(jobs_flat, float(alpha), dtype=float)
    friction_term = np.exp(-float(beta) * np.clip(times, 0.0, None))
    return friction_term * dest_term[np.newaxis, :]


def destination_probabilities(
    *,
    jobs: np.ndarray,
    car_time_min: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Return row-normalized destination probabilities by origin."""
    attractiveness = gravity_attractiveness(jobs=jobs, car_time_min=car_time_min, alpha=alpha, beta=beta)
    row_sums = attractiveness.sum(axis=1, keepdims=True)

    n_destinations = attractiveness.shape[1]
    uniform = np.full_like(attractiveness, 1.0 / n_destinations)
    return np.divide(attractiveness, row_sums, out=uniform, where=row_sums > 0.0)
