"""Line geometry validation utilities for turn-angle and backtracking constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from network.stations import Line, Station


@dataclass(frozen=True)
class GeometryViolation:
    line_id: str
    index: int
    reason: str


def _vec(a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    return np.asarray([b[0] - a[0], b[1] - a[1]], dtype=float)


def _angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def validate_line_geometry(
    line: Line,
    stations_by_id: dict[str, Station],
    *,
    max_turn_angle_deg: float = 90.0,
    allow_terminal_reversal: bool = True,
) -> list[GeometryViolation]:
    """Validate max turn angle and no U-turn/backtracking constraints."""
    violations: list[GeometryViolation] = []
    station_ids = line.stations
    if len(station_ids) < 2:
        return violations

    points: list[tuple[float, float]] = []
    for sid in station_ids:
        if sid not in stations_by_id:
            violations.append(GeometryViolation(line.id, -1, f"unknown station {sid}"))
            return violations
        st = stations_by_id[sid]
        points.append((st.x, st.y))

    for i in range(1, len(points) - 1):
        v1 = _vec(points[i - 1], points[i])
        v2 = _vec(points[i], points[i + 1])
        angle = _angle_degrees(v1, v2)

        if angle > max_turn_angle_deg + 1e-9:
            violations.append(GeometryViolation(line.id, i, f"turn angle {angle:.2f} > {max_turn_angle_deg:.2f}"))

        if float(np.dot(v1, v2)) < 0:
            terminal_reversal = allow_terminal_reversal and (i == len(points) - 2)
            if not terminal_reversal:
                violations.append(GeometryViolation(line.id, i, "backtracking/U-turn not allowed"))

    # Explicit exact A->B->A check for nonterminal interior points.
    for i in range(1, len(station_ids) - 1):
        if station_ids[i - 1] == station_ids[i + 1]:
            terminal_reversal = allow_terminal_reversal and (i == len(station_ids) - 2)
            if not terminal_reversal:
                violations.append(GeometryViolation(line.id, i, "exact U-turn A-B-A not allowed"))

    return violations


def validate_network_geometry(
    lines: list[Line],
    stations_by_id: dict[str, Station],
    *,
    max_turn_angle_deg: float = 90.0,
    allow_terminal_reversal: bool = True,
) -> dict[str, list[GeometryViolation]]:
    """Validate all lines and return non-empty violation lists by line id."""
    out: dict[str, list[GeometryViolation]] = {}
    for line in lines:
        issues = validate_line_geometry(
            line,
            stations_by_id,
            max_turn_angle_deg=max_turn_angle_deg,
            allow_terminal_reversal=allow_terminal_reversal,
        )
        if issues:
            out[line.id] = issues
    return out
