"""Employment grid generation for synthetic 64x64 city jobs field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from city.fields import build_background_jobs_field, reconcile_total_jobs
from city.presets import EmploymentPreset, get_employment_preset
from city.validators import CenterPlacement, CenterSpec, RingConstraint, deterministic_center_placement

GRID_SIZE = 64


@dataclass(frozen=True)
class JobsGenerationResult:
    jobs: np.ndarray
    centers: tuple[CenterPlacement, ...]
    preset: EmploymentPreset


def _specs_from_preset(preset: EmploymentPreset) -> list[CenterSpec]:
    specs: list[CenterSpec] = []
    for i, (size, zone) in enumerate(zip(preset.center_sizes, preset.placement_zones, strict=True), start=1):
        ring: RingConstraint | None = None
        # Optional ring constraints tuned for plausible edge-city placement.
        if zone == "edge":
            ring = RingConstraint(min_radius=18.0, max_radius=31.5)
        elif zone == "outer":
            ring = RingConstraint(min_radius=12.0, max_radius=29.0)
        specs.append(CenterSpec(center_id=i, size=size, zone=zone, ring=ring))
    return specs


def _core_mask(center: CenterPlacement, grid_size: int) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[center.y0 : center.y1, center.x0 : center.x1] = True
    return mask


def _distance_from_core_edge(center: CenterPlacement, grid_size: int) -> np.ndarray:
    """Distance from each tile center to the nearest core-edge point (0 in core)."""
    yy, xx = np.indices((grid_size, grid_size), dtype=float)
    left = float(center.x0)
    right = float(center.x1 - 1)
    top = float(center.y0)
    bottom = float(center.y1 - 1)

    dx = np.maximum(np.maximum(left - xx, 0.0), xx - right)
    dy = np.maximum(np.maximum(top - yy, 0.0), yy - bottom)
    return np.hypot(dx, dy)


def _allocate_core_jobs(
    jobs: np.ndarray,
    center: CenterPlacement,
    core_jobs: float,
) -> None:
    """Allocate core jobs exactly and uniformly within the core footprint."""
    area = center.size * center.size
    jobs_per_tile = core_jobs / area
    jobs[center.y0 : center.y1, center.x0 : center.x1] += jobs_per_tile


def _allocate_halo_jobs(
    *,
    jobs: np.ndarray,
    center: CenterPlacement,
    core_jobs: float,
    halo_ratio: float,
    occupied_mask: np.ndarray,
) -> None:
    """Allocate radial halo with normalized weights and 150% cap (halo <= 50% core)."""
    capped_ratio = min(max(halo_ratio, 0.0), 0.5)
    halo_jobs_target = core_jobs * capped_ratio
    if halo_jobs_target <= 0.0:
        return

    dist = _distance_from_core_edge(center, jobs.shape[0])
    core = _core_mask(center, jobs.shape[0])

    if center.size == 3:
        max_outside_blocks = 4.0
    else:
        max_outside_blocks = 6.0

    halo_band = (~core) & (dist > 0.0) & (dist <= max_outside_blocks) & (~occupied_mask)
    if not np.any(halo_band):
        return

    decay_scale = max_outside_blocks / 2.2
    weights = np.exp(-dist / decay_scale)
    weights *= halo_band.astype(float)
    wsum = float(weights.sum())
    if wsum <= 0.0:
        return

    halo_alloc = weights * (halo_jobs_target / wsum)
    jobs += halo_alloc


def generate_jobs_grid(
    *,
    total_jobs: int,
    preset_name: str = "moderate_polycentric",
    seed: int = 101,
    grid_size: int = GRID_SIZE,
    min_separation: float = 8.0,
    attraction_strength: float = 0.12,
) -> JobsGenerationResult:
    """Generate full jobs field with fixed downtown core and default 5-center setup."""
    if grid_size != GRID_SIZE:
        raise ValueError("This generator currently supports exactly a 64x64 grid.")

    preset = get_employment_preset(preset_name)
    specs = _specs_from_preset(preset)
    centers = deterministic_center_placement(
        grid_size=grid_size,
        specs=specs,
        seed=seed,
        min_separation=min_separation,
    )

    jobs = np.zeros((grid_size, grid_size), dtype=float)

    # Core allocation first and exact by definition.
    core_jobs_by_center = [share * total_jobs for share in preset.center_shares]
    for center, core_jobs in zip(centers, core_jobs_by_center, strict=True):
        _allocate_core_jobs(jobs=jobs, center=center, core_jobs=core_jobs)

    # Protect all center footprints from halo/background overlap.
    occupied_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for center in centers:
        occupied_mask |= _core_mask(center, grid_size)

    # Radial halo allocation second: normalized sums and center-wise cap enforcement.
    for center, core_jobs, halo_ratio in zip(centers, core_jobs_by_center, preset.halo_strength, strict=True):
        _allocate_halo_jobs(
            jobs=jobs,
            center=center,
            core_jobs=core_jobs,
            halo_ratio=halo_ratio,
            occupied_mask=occupied_mask,
        )

    # Background jobs outside center footprints.
    background_jobs = total_jobs * preset.background_share
    background = build_background_jobs_field(
        grid_size=grid_size,
        background_jobs=background_jobs,
        centers=centers,
        seed=seed + 1,
        attraction_strength=attraction_strength,
    )
    background[occupied_mask] = 0.0
    residual = background_jobs / max(float(background.sum()), 1e-12)
    jobs += background * residual

    jobs = reconcile_total_jobs(jobs, float(total_jobs))
    return JobsGenerationResult(jobs=jobs, centers=tuple(centers), preset=preset)
