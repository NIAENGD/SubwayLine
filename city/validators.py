"""Validation and deterministic placement utilities for employment centers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

PlacementZone = Literal["center", "inner", "middle", "outer", "edge", "any"]


@dataclass(frozen=True)
class RingConstraint:
    """Optional annulus-style placement rule around downtown."""

    min_radius: float = 0.0
    max_radius: float = float("inf")


@dataclass(frozen=True)
class CenterSpec:
    """Placement specification for one center."""

    center_id: int
    size: int
    zone: PlacementZone
    ring: RingConstraint | None = None


@dataclass(frozen=True)
class CenterPlacement:
    """Top-left anchored placed center footprint."""

    center_id: int
    x0: int
    y0: int
    size: int

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        return self.y0 + self.size

    @property
    def centroid(self) -> tuple[float, float]:
        half = (self.size - 1) / 2.0
        return self.x0 + half, self.y0 + half


def overlaps(a: CenterPlacement, b: CenterPlacement) -> bool:
    """Return True when two square footprints overlap."""
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)


def separation_distance(a: CenterPlacement, b: CenterPlacement) -> float:
    """Euclidean distance between center centroids."""
    ax, ay = a.centroid
    bx, by = b.centroid
    return float(np.hypot(ax - bx, ay - by))


def zone_bounds(grid_size: int, zone: PlacementZone) -> tuple[int, int]:
    """Inclusive min/max radius from downtown for each placement zone."""
    max_r = (grid_size - 1) / 2.0
    if zone == "center":
        return 0, int(max_r * 0.25)
    if zone == "inner":
        return int(max_r * 0.15), int(max_r * 0.45)
    if zone == "middle":
        return int(max_r * 0.35), int(max_r * 0.70)
    if zone == "outer":
        return int(max_r * 0.60), int(max_r * 0.92)
    if zone == "edge":
        return int(max_r * 0.80), int(max_r)
    return 0, int(max_r)


def _valid_position(
    x0: int,
    y0: int,
    spec: CenterSpec,
    placed: list[CenterPlacement],
    grid_size: int,
    min_separation: float,
) -> bool:
    candidate = CenterPlacement(center_id=spec.center_id, x0=x0, y0=y0, size=spec.size)
    cx, cy = candidate.centroid
    downtown_c = (grid_size - 1) / 2.0
    r = float(np.hypot(cx - downtown_c, cy - downtown_c))

    zone_min, zone_max = zone_bounds(grid_size=grid_size, zone=spec.zone)
    if r < zone_min or r > zone_max:
        return False

    if spec.ring is not None and (r < spec.ring.min_radius or r > spec.ring.max_radius):
        return False

    for existing in placed:
        if overlaps(candidate, existing):
            return False
        if separation_distance(candidate, existing) < min_separation:
            return False
    return True


def deterministic_center_placement(
    *,
    grid_size: int,
    specs: list[CenterSpec],
    seed: int,
    min_separation: float,
    max_attempts_per_center: int = 50_000,
) -> list[CenterPlacement]:
    """Deterministically place center footprints with constraints and seed reproducibility."""
    rng = np.random.default_rng(seed)
    placed: list[CenterPlacement] = []

    # Downtown is always fixed to central 5x5.
    downtown_size = specs[0].size
    downtown_x0 = (grid_size // 2) - (downtown_size // 2)
    downtown = CenterPlacement(center_id=specs[0].center_id, x0=downtown_x0, y0=downtown_x0, size=downtown_size)
    placed.append(downtown)

    for spec in specs[1:]:
        max_start = grid_size - spec.size
        attempts = 0
        selected: CenterPlacement | None = None
        while attempts < max_attempts_per_center:
            attempts += 1
            x0 = int(rng.integers(0, max_start + 1))
            y0 = int(rng.integers(0, max_start + 1))
            if _valid_position(
                x0=x0,
                y0=y0,
                spec=spec,
                placed=placed,
                grid_size=grid_size,
                min_separation=min_separation,
            ):
                selected = CenterPlacement(center_id=spec.center_id, x0=x0, y0=y0, size=spec.size)
                break
        if selected is None:
            raise ValueError(
                f"Could not place center {spec.center_id} with provided constraints after {max_attempts_per_center} attempts."
            )
        placed.append(selected)

    return placed
