"""OD matrix helpers for preserving origin worker totals."""

from __future__ import annotations

import numpy as np


def _normalize_probabilities_per_origin(destination_probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(destination_probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("destination_probabilities must be a 2D matrix.")

    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    n_dest = probs.shape[1]
    uniform = np.full_like(probs, 1.0 / n_dest)
    return np.divide(probs, row_sums, out=uniform, where=row_sums > 0.0)


def build_od_matrix(*, workers: np.ndarray, destination_probabilities: np.ndarray) -> np.ndarray:
    """Build OD matrix with exact origin-preservation after per-origin normalization."""
    worker_grid = np.asarray(workers, dtype=float)
    if worker_grid.ndim != 2:
        raise ValueError("workers must be a 2D array.")

    worker_vector = np.clip(worker_grid.reshape(-1), 0.0, None)
    probs = _normalize_probabilities_per_origin(destination_probabilities)

    n_origins = worker_vector.size
    if probs.shape[0] != n_origins:
        raise ValueError("destination_probabilities rows must match number of origins.")

    od = worker_vector[:, np.newaxis] * probs

    # Numerical guard: force each row sum to equal the worker total exactly.
    row_sums = od.sum(axis=1)
    residual = worker_vector - row_sums
    od[:, -1] += residual

    return od
