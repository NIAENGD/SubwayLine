"""Mutation operators for simulated annealing network refinement."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from optimize.initial_builder import NetworkLine, NetworkPlan


def _neighbors(point: tuple[int, int], grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = point
    rows, cols = grid_shape
    out: list[tuple[int, int]] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            out.append((nx, ny))
    return out


def _unique_points(lines: list[NetworkLine]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for line in lines:
        for p in line.points:
            if p not in seen:
                out.append(p)
                seen.add(p)
    return out


def add_station_at_line_end(plan: NetworkPlan, rng: np.random.Generator, grid_shape: tuple[int, int]) -> NetworkPlan:
    mutated = deepcopy(plan)
    line = rng.choice(mutated.lines)
    end_is_tail = bool(rng.integers(0, 2))
    anchor = line.points[-1] if end_is_tail else line.points[0]
    candidates = _neighbors(anchor, grid_shape)
    if not candidates:
        return plan
    new_pt = candidates[int(rng.integers(0, len(candidates)))]
    if end_is_tail:
        line.points.append(new_pt)
    else:
        line.points.insert(0, new_pt)
    return mutated


def remove_station_at_line_end(plan: NetworkPlan, rng: np.random.Generator, _grid_shape: tuple[int, int]) -> NetworkPlan:
    mutated = deepcopy(plan)
    eligible = [ln for ln in mutated.lines if len(ln.points) > 2]
    if not eligible:
        return plan
    line = rng.choice(eligible)
    if bool(rng.integers(0, 2)):
        line.points.pop()
    else:
        line.points.pop(0)
    return mutated


def insert_intermediate_station(plan: NetworkPlan, rng: np.random.Generator, grid_shape: tuple[int, int]) -> NetworkPlan:
    mutated = deepcopy(plan)
    eligible = [ln for ln in mutated.lines if len(ln.points) >= 2]
    if not eligible:
        return plan
    line = rng.choice(eligible)
    idx = int(rng.integers(1, len(line.points)))
    left = line.points[idx - 1]
    right = line.points[idx]
    mid = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
    neighbors = _neighbors(mid, grid_shape) + [mid]
    line.points.insert(idx, neighbors[int(rng.integers(0, len(neighbors)))])
    return mutated


def delete_intermediate_station(plan: NetworkPlan, rng: np.random.Generator, _grid_shape: tuple[int, int]) -> NetworkPlan:
    mutated = deepcopy(plan)
    eligible = [ln for ln in mutated.lines if len(ln.points) > 3]
    if not eligible:
        return plan
    line = rng.choice(eligible)
    idx = int(rng.integers(1, len(line.points) - 1))
    line.points.pop(idx)
    return mutated


def reroute_one_bend(plan: NetworkPlan, rng: np.random.Generator, grid_shape: tuple[int, int]) -> NetworkPlan:
    mutated = deepcopy(plan)
    eligible = [ln for ln in mutated.lines if len(ln.points) > 2]
    if not eligible:
        return plan
    line = rng.choice(eligible)
    idx = int(rng.integers(1, len(line.points) - 1))
    bend = line.points[idx]
    candidates = _neighbors(bend, grid_shape)
    if not candidates:
        return plan
    line.points[idx] = candidates[int(rng.integers(0, len(candidates)))]
    return mutated


def split_one_line_into_two(plan: NetworkPlan, rng: np.random.Generator, _grid_shape: tuple[int, int]) -> NetworkPlan:
    if not plan.lines:
        return plan
    mutated = deepcopy(plan)
    eligible = [ln for ln in mutated.lines if len(ln.points) >= 4]
    if not eligible:
        return plan
    line = rng.choice(eligible)
    cut = int(rng.integers(2, len(line.points) - 1))
    a, b = line.points[:cut], line.points[cut - 1 :]
    line.points = a
    mutated.lines.append(NetworkLine(id=f"L{len(mutated.lines)+1}", points=b))
    return mutated


def merge_two_lines(plan: NetworkPlan, rng: np.random.Generator, _grid_shape: tuple[int, int]) -> NetworkPlan:
    if len(plan.lines) < 2:
        return plan
    mutated = deepcopy(plan)
    i, j = rng.choice(len(mutated.lines), size=2, replace=False)
    i, j = int(i), int(j)
    a = mutated.lines[min(i, j)]
    b = mutated.lines[max(i, j)]
    merged = a.points + b.points
    a.points = merged
    mutated.lines.pop(max(i, j))
    return mutated


def shift_transfer_node(plan: NetworkPlan, rng: np.random.Generator, grid_shape: tuple[int, int]) -> NetworkPlan:
    mutated = deepcopy(plan)
    points = _unique_points(mutated.lines)
    if not points:
        return plan
    transfer = points[int(rng.integers(0, len(points)))]
    moved = _neighbors(transfer, grid_shape)
    if not moved:
        return plan
    replacement = moved[int(rng.integers(0, len(moved)))]
    for line in mutated.lines:
        line.points = [replacement if p == transfer else p for p in line.points]
    return mutated


def reduce_overlap(plan: NetworkPlan, rng: np.random.Generator, grid_shape: tuple[int, int]) -> NetworkPlan:
    mutated = deepcopy(plan)
    counts: dict[tuple[int, int], int] = {}
    for line in mutated.lines:
        for p in line.points:
            counts[p] = counts.get(p, 0) + 1

    hotspots = [p for p, c in counts.items() if c > 1]
    if not hotspots:
        return plan
    target = hotspots[int(rng.integers(0, len(hotspots)))]
    for line in mutated.lines:
        for idx, p in enumerate(line.points):
            if p == target:
                choices = _neighbors(p, grid_shape)
                if choices:
                    line.points[idx] = choices[int(rng.integers(0, len(choices)))]
                    return mutated
    return plan


def create_partial_circumferential_connector(
    plan: NetworkPlan,
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
) -> NetworkPlan:
    mutated = deepcopy(plan)
    rows, cols = grid_shape
    cx, cy = cols // 2, rows // 2
    r = max(4, int(min(rows, cols) * 0.20))
    theta0 = float(rng.uniform(0, np.pi))
    theta1 = theta0 + float(rng.uniform(np.pi / 3, np.pi * 0.8))

    steps = 8
    arc: list[tuple[int, int]] = []
    for i in range(steps + 1):
        t = theta0 + (theta1 - theta0) * (i / steps)
        x = int(np.clip(round(cx + r * np.cos(t)), 0, cols - 1))
        y = int(np.clip(round(cy + r * np.sin(t)), 0, rows - 1))
        if not arc or arc[-1] != (x, y):
            arc.append((x, y))

    if len(arc) >= 2:
        mutated.lines.append(NetworkLine(id=f"L{len(mutated.lines)+1}", points=arc))
    return mutated


MUTATION_OPERATORS = {
    "add_station_at_line_end": add_station_at_line_end,
    "remove_station_at_line_end": remove_station_at_line_end,
    "insert_intermediate_station": insert_intermediate_station,
    "delete_intermediate_station": delete_intermediate_station,
    "reroute_one_bend": reroute_one_bend,
    "split_one_line_into_two": split_one_line_into_two,
    "merge_two_lines": merge_two_lines,
    "shift_transfer_node": shift_transfer_node,
    "reduce_overlap": reduce_overlap,
    "create_partial_circumferential_connector": create_partial_circumferential_connector,
}


def apply_random_mutation(
    *,
    plan: NetworkPlan,
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    target_line_count: int,
) -> NetworkPlan:
    """Apply one random required mutation and normalize line count toward target."""
    op_name = list(MUTATION_OPERATORS.keys())[int(rng.integers(0, len(MUTATION_OPERATORS)))]
    mutated = MUTATION_OPERATORS[op_name](plan, rng, grid_shape)

    if len(mutated.lines) > target_line_count:
        mutated.lines = mutated.lines[:target_line_count]
    elif len(mutated.lines) < target_line_count and mutated.lines:
        # Duplicate and perturb the last line to preserve exact line-count constraints.
        base = deepcopy(mutated.lines[-1])
        if len(base.points) >= 2:
            base.points = base.points[::-1]
        base.id = f"L{len(mutated.lines)+1}"
        mutated.lines.append(base)

    for i, line in enumerate(mutated.lines, start=1):
        line.id = f"L{i}"
        if len(line.points) < 2:
            line.points = [(0, 0), (1, 0)]

    mutated.metadata = {**mutated.metadata, "last_mutation": op_name}
    return mutated
