"""Worker generation from population surface."""

from __future__ import annotations

import numpy as np


def generate_workers(population: np.ndarray, worker_ratio: float) -> np.ndarray:
    """Return worker totals by tile using workers = worker_ratio * population."""
    pop = np.asarray(population, dtype=float)
    if pop.ndim != 2:
        raise ValueError("population must be a 2D array.")
    if not 0.0 <= float(worker_ratio) <= 1.0:
        raise ValueError("worker_ratio must be in [0, 1].")
    workers = pop * float(worker_ratio)
    return np.clip(workers, 0.0, None)
