"""Assignment graph builder with line-context nodes and transfer penalties."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from network.stations import Line, Station


@dataclass(frozen=True)
class GraphEdge:
    to_node: str
    cost: float
    kind: str
    line_id: str | None = None
    station_id: str | None = None


@dataclass(frozen=True)
class AssignmentGraph:
    adjacency: dict[str, list[GraphEdge]]
    node_station_line: dict[str, tuple[str, str]]


def _ctx_node(station_id: str, line_id: str) -> str:
    return f"ctx:{station_id}|{line_id}"


def _segment_time_minutes(a: Station, b: Station, speed_mph: float) -> float:
    dist_tiles = float(np.hypot(a.x - b.x, a.y - b.y))
    dist_miles = dist_tiles * 0.5
    if speed_mph <= 1e-9:
        return 1e9
    return 60.0 * dist_miles / speed_mph


def build_assignment_graph(
    *,
    lines: list[Line],
    stations_by_id: dict[str, Station],
    transfer_penalty_min: float = 8.0,
    in_vehicle_speed_mph: float = 24.0,
) -> AssignmentGraph:
    """Build line-context graph with in-vehicle and transfer edges."""
    adjacency: defaultdict[str, list[GraphEdge]] = defaultdict(list)
    node_meta: dict[str, tuple[str, str]] = {}

    # In-vehicle edges along each line (bidirectional for now).
    for line in lines:
        wait_cost = max(float(line.headway_min), 0.0) * 0.5
        for sid in line.stations:
            node = _ctx_node(sid, line.id)
            node_meta[node] = (sid, line.id)

        for i in range(len(line.stations) - 1):
            s0 = line.stations[i]
            s1 = line.stations[i + 1]
            st0 = stations_by_id[s0]
            st1 = stations_by_id[s1]
            ride_cost = _segment_time_minutes(st0, st1, in_vehicle_speed_mph)

            n0 = _ctx_node(s0, line.id)
            n1 = _ctx_node(s1, line.id)
            adjacency[n0].append(GraphEdge(to_node=n1, cost=ride_cost, kind="ride", line_id=line.id, station_id=s0))
            adjacency[n1].append(GraphEdge(to_node=n0, cost=ride_cost, kind="ride", line_id=line.id, station_id=s1))

            # Boarding from station pseudo node into line context includes waiting.
            station_node0 = f"station:{s0}"
            station_node1 = f"station:{s1}"
            adjacency[station_node0].append(GraphEdge(to_node=n0, cost=wait_cost, kind="board", line_id=line.id, station_id=s0))
            adjacency[station_node1].append(GraphEdge(to_node=n1, cost=wait_cost, kind="board", line_id=line.id, station_id=s1))
            adjacency[n0].append(GraphEdge(to_node=station_node0, cost=0.0, kind="alight", line_id=line.id, station_id=s0))
            adjacency[n1].append(GraphEdge(to_node=station_node1, cost=0.0, kind="alight", line_id=line.id, station_id=s1))

    # Transfer edges between line contexts at shared station.
    station_to_lines: defaultdict[str, list[str]] = defaultdict(list)
    for line in lines:
        for sid in line.stations:
            station_to_lines[sid].append(line.id)

    for sid, line_ids in station_to_lines.items():
        unique = sorted(set(line_ids))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                n_a = _ctx_node(sid, unique[i])
                n_b = _ctx_node(sid, unique[j])
                adjacency[n_a].append(
                    GraphEdge(to_node=n_b, cost=transfer_penalty_min, kind="transfer", line_id=unique[j], station_id=sid)
                )
                adjacency[n_b].append(
                    GraphEdge(to_node=n_a, cost=transfer_penalty_min, kind="transfer", line_id=unique[i], station_id=sid)
                )

    return AssignmentGraph(adjacency=dict(adjacency), node_station_line=node_meta)
