"""Passenger assignment with max transfers and generalized-cost shortest paths."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq

import numpy as np

from network.graph_builder import AssignmentGraph, GraphEdge, build_assignment_graph
from network.stations import Line, Station


@dataclass(frozen=True)
class ODDemand:
    origin: tuple[int, int]
    destination: tuple[int, int]
    riders: float


@dataclass(frozen=True)
class PathSolution:
    cost: float
    transfers: int
    edges: list[tuple[str, GraphEdge]]


@dataclass(frozen=True)
class AssignmentOutputs:
    station_boardings: dict[str, float]
    segment_loads: dict[tuple[str, str, str], float]
    transfer_volumes: dict[str, float]


def _walk_time_minutes(tile_a: tuple[int, int], tile_b: tuple[int, int], walk_mph: float = 3.0) -> float:
    dist_tiles = float(np.hypot(tile_a[0] - tile_b[0], tile_a[1] - tile_b[1]))
    miles = 0.5 * dist_tiles
    if walk_mph <= 1e-9:
        return 1e9
    return 60.0 * miles / walk_mph


def _station_access_candidates(
    tile: tuple[int, int],
    stations: list[Station],
    *,
    max_access_tiles: float,
) -> list[tuple[Station, float]]:
    out: list[tuple[Station, float]] = []
    for st in stations:
        d = float(np.hypot(tile[0] - st.x, tile[1] - st.y))
        if d <= max_access_tiles + 1e-9:
            out.append((st, _walk_time_minutes(tile, (int(round(st.x)), int(round(st.y))))))
    out.sort(key=lambda t: t[1])
    return out


def _dijkstra_with_transfer_cap(
    graph: AssignmentGraph,
    source_station_ids: list[tuple[str, float]],
    target_station_ids: set[str],
    *,
    max_transfers: int,
) -> PathSolution | None:
    """Shortest path in line-context graph with explicit transfer-count state."""
    best_cost: dict[tuple[str, int], float] = {}
    prev: dict[tuple[str, int], tuple[tuple[str, int], GraphEdge]] = {}
    heap: list[tuple[float, int, str]] = []

    for sid, access_cost in source_station_ids:
        node = f"station:{sid}"
        state = (node, 0)
        best_cost[state] = access_cost
        heapq.heappush(heap, (access_cost, 0, node))

    best_target_state: tuple[str, int] | None = None
    best_target_cost = float("inf")

    while heap:
        cost, transfers, node = heapq.heappop(heap)
        state = (node, transfers)
        if cost > best_cost.get(state, float("inf")) + 1e-9:
            continue

        if node.startswith("station:"):
            sid = node.split(":", 1)[1]
            if sid in target_station_ids and cost < best_target_cost:
                best_target_state = state
                best_target_cost = cost

        for edge in graph.adjacency.get(node, []):
            next_transfers = transfers + (1 if edge.kind == "transfer" else 0)
            if next_transfers > max_transfers:
                continue
            next_state = (edge.to_node, next_transfers)
            next_cost = cost + edge.cost
            if next_cost + 1e-9 < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = next_cost
                prev[next_state] = (state, edge)
                heapq.heappush(heap, (next_cost, next_transfers, edge.to_node))

    if best_target_state is None:
        return None

    edges_rev: list[tuple[str, GraphEdge]] = []
    cur = best_target_state
    while cur in prev:
        pstate, edge = prev[cur]
        edges_rev.append((pstate[0], edge))
        cur = pstate
    edges_rev.reverse()
    return PathSolution(cost=best_target_cost, transfers=best_target_state[1], edges=edges_rev)


def assign_passengers(
    *,
    od_demands: list[ODDemand],
    stations: list[Station],
    lines: list[Line],
    max_transfers: int = 2,
    max_access_tiles: float = 1.0,
    transfer_penalty_min: float = 8.0,
) -> AssignmentOutputs:
    """Assign OD passengers to minimum generalized-cost paths with transfer cap."""
    stations_by_id = {s.id: s for s in stations}
    graph = build_assignment_graph(
        lines=lines,
        stations_by_id=stations_by_id,
        transfer_penalty_min=transfer_penalty_min,
    )

    station_boardings: defaultdict[str, float] = defaultdict(float)
    segment_loads: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    transfer_volumes: defaultdict[str, float] = defaultdict(float)

    for od in od_demands:
        riders = float(od.riders)
        if riders <= 0.0:
            continue

        origin_access = _station_access_candidates(od.origin, stations, max_access_tiles=max_access_tiles)
        dest_access = _station_access_candidates(od.destination, stations, max_access_tiles=max_access_tiles)
        if not origin_access or not dest_access:
            continue

        src = [(st.id, access_cost) for st, access_cost in origin_access[:8]]
        dst_ids = {st.id for st, _ in dest_access[:8]}

        path = _dijkstra_with_transfer_cap(
            graph,
            src,
            dst_ids,
            max_transfers=max_transfers,
        )
        if path is None:
            continue

        boarded = False
        for from_node, edge in path.edges:
            if edge.kind == "board" and not boarded and edge.station_id is not None:
                station_boardings[edge.station_id] += riders
                boarded = True
            elif edge.kind == "ride" and edge.line_id is not None and edge.station_id is not None:
                from_station = edge.station_id
                to_station = edge.to_node.split(":", 1)[1].split("|")[0] if edge.to_node.startswith("ctx:") else "?"
                segment_loads[(edge.line_id, from_station, to_station)] += riders
            elif edge.kind == "transfer" and edge.station_id is not None:
                transfer_volumes[edge.station_id] += riders

    return AssignmentOutputs(
        station_boardings=dict(station_boardings),
        segment_loads=dict(segment_loads),
        transfer_volumes=dict(transfer_volumes),
    )
