"""Employment presets for synthetic city job generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PlacementZone = Literal["center", "inner", "middle", "outer", "edge", "any"]


@dataclass(frozen=True)
class EmploymentPreset:
    """Preset knobs controlling the employment generation profile."""

    name: str
    center_shares: tuple[float, float, float, float, float]
    center_sizes: tuple[int, int, int, int, int]
    placement_zones: tuple[PlacementZone, PlacementZone, PlacementZone, PlacementZone, PlacementZone]
    halo_strength: tuple[float, float, float, float, float]
    background_share: float

    def __post_init__(self) -> None:
        if len(self.center_shares) != 5 or len(self.center_sizes) != 5:
            raise ValueError("Presets must define exactly 5 centers.")
        if len(self.placement_zones) != 5 or len(self.halo_strength) != 5:
            raise ValueError("Placement zones and halo strengths must include 5 entries.")
        if self.center_sizes[0] != 5:
            raise ValueError("Center 1 is always fixed as downtown 5x5.")
        center_total = sum(self.center_shares)
        if abs((center_total + self.background_share) - 1.0) > 1e-9:
            raise ValueError("Center shares plus background_share must sum to 1.0.")
        if any(size < 3 for size in self.center_sizes):
            raise ValueError("Center footprints must be at least 3x3.")
        if any(strength < 0.0 for strength in self.halo_strength):
            raise ValueError("Halo strengths must be non-negative.")


# Section 3.2 default: 30/15/10/8/5 center shares and 32% background.
DEFAULT_FIVE_CENTER = EmploymentPreset(
    name="moderate_polycentric",
    center_shares=(0.30, 0.15, 0.10, 0.08, 0.05),
    center_sizes=(5, 4, 3, 3, 3),
    placement_zones=("center", "middle", "outer", "middle", "outer"),
    halo_strength=(0.50, 0.35, 0.30, 0.30, 0.25),
    background_share=0.32,
)

EMPLOYMENT_PRESETS: dict[str, EmploymentPreset] = {
    "monocentric": EmploymentPreset(
        name="monocentric",
        center_shares=(0.56, 0.06, 0.03, 0.02, 0.01),
        center_sizes=(5, 3, 3, 3, 3),
        placement_zones=("center", "inner", "middle", "middle", "outer"),
        halo_strength=(0.55, 0.20, 0.15, 0.10, 0.10),
        background_share=0.32,
    ),
    "concentrated_polycentric": EmploymentPreset(
        name="concentrated_polycentric",
        center_shares=(0.40, 0.12, 0.08, 0.05, 0.03),
        center_sizes=(5, 4, 3, 3, 3),
        placement_zones=("center", "inner", "middle", "middle", "outer"),
        halo_strength=(0.50, 0.35, 0.28, 0.24, 0.20),
        background_share=0.32,
    ),
    "moderate_polycentric": DEFAULT_FIVE_CENTER,
    "distributed_polycentric": EmploymentPreset(
        name="distributed_polycentric",
        center_shares=(0.24, 0.13, 0.11, 0.10, 0.08),
        center_sizes=(5, 4, 4, 3, 3),
        placement_zones=("center", "middle", "outer", "outer", "edge"),
        halo_strength=(0.45, 0.36, 0.34, 0.30, 0.28),
        background_share=0.34,
    ),
    "edge_city_heavy": EmploymentPreset(
        name="edge_city_heavy",
        center_shares=(0.18, 0.12, 0.10, 0.10, 0.10),
        center_sizes=(5, 4, 4, 3, 3),
        placement_zones=("center", "edge", "edge", "outer", "edge"),
        halo_strength=(0.40, 0.38, 0.38, 0.35, 0.35),
        background_share=0.40,
    ),
}


def get_employment_preset(name: str) -> EmploymentPreset:
    """Return a preset by name, with section-3.2 default fallback."""
    return EMPLOYMENT_PRESETS.get(name, DEFAULT_FIVE_CENTER)
