"""Travel impedance and generalized cost helpers for demand modeling."""

from __future__ import annotations

import numpy as np


def compute_car_time(
    *,
    distance_km: np.ndarray,
    car_speed_kmh: float,
) -> np.ndarray:
    """Convert distance matrix in km to free-flow car time in minutes."""
    distances = np.asarray(distance_km, dtype=float)
    if distances.ndim != 2:
        raise ValueError("distance_km must be a 2D array.")
    if car_speed_kmh <= 0:
        raise ValueError("car_speed_kmh must be positive.")
    return np.clip(distances, 0.0, None) / float(car_speed_kmh) * 60.0


def compute_car_generalized_cost(
    *,
    car_time_min: np.ndarray,
    congestion_penalty: float,
    parking_penalty: float,
) -> np.ndarray:
    """Compute car generalized cost in minutes including congestion and parking.

    cost = car_time * (1 + congestion_penalty) + parking_penalty
    """
    times = np.asarray(car_time_min, dtype=float)
    if times.ndim != 2:
        raise ValueError("car_time_min must be a 2D array.")

    congestion = max(float(congestion_penalty), 0.0)
    parking = max(float(parking_penalty), 0.0)

    congested = np.clip(times, 0.0, None) * (1.0 + congestion)
    return congested + parking
