"""Transit network modeling package."""

from network.assignment import AssignmentOutputs, ODDemand, assign_passengers
from network.graph_builder import AssignmentGraph, GraphEdge, build_assignment_graph
from network.line_geometry import GeometryViolation, validate_line_geometry, validate_network_geometry
from network.rasterization import OverlapSummary, compute_overlap_penalties, rasterize_segment
from network.stations import CandidateConfig, Line, Segment, Station, generate_station_candidates

__all__ = [
    "AssignmentGraph",
    "AssignmentOutputs",
    "CandidateConfig",
    "GeometryViolation",
    "GraphEdge",
    "Line",
    "ODDemand",
    "OverlapSummary",
    "Segment",
    "Station",
    "assign_passengers",
    "build_assignment_graph",
    "compute_overlap_penalties",
    "generate_station_candidates",
    "rasterize_segment",
    "validate_line_geometry",
    "validate_network_geometry",
]
