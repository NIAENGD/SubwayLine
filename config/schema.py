"""Typed configuration schema for SubwayLine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SeedContext:
    """Explicit random seeds used by each module/run pathway."""

    city_seed: int
    demand_seed: int
    optimization_seed: int
    batch_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.city_seed < 0 or self.demand_seed < 0 or self.optimization_seed < 0:
            raise ValueError("Seeds must be non-negative integers.")
        if not self.batch_seeds:
            raise ValueError("batch_seeds must include at least one seed.")
        if any(seed < 0 for seed in self.batch_seeds):
            raise ValueError("All batch seeds must be non-negative integers.")


@dataclass(frozen=True)
class CityConfig:
    total_population: int
    total_jobs: int
    tile_size: int
    employment_preset: str
    population_preset: str
    halo_strength: float
    center_shares: tuple[float, ...]
    center_sizes: tuple[int, ...]
    jobs_housing_interaction: float
    residential_cluster_count: int
    random_seed: int

    def __post_init__(self) -> None:
        if self.total_population <= 0 or self.total_jobs <= 0:
            raise ValueError("Population and jobs must be positive.")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if len(self.center_shares) != len(self.center_sizes):
            raise ValueError("center_shares and center_sizes must have equal length.")
        if not self.center_shares:
            raise ValueError("At least one center must be defined.")
        if abs(sum(self.center_shares) - 1.0) > 1e-6:
            raise ValueError("center_shares must sum to 1.0.")
        if any(size <= 0 for size in self.center_sizes):
            raise ValueError("center_sizes must all be positive.")
        if self.residential_cluster_count <= 0:
            raise ValueError("residential_cluster_count must be positive.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")


@dataclass(frozen=True)
class DemandConfig:
    worker_ratio: float
    destination_choice_alpha: float
    impedance_beta: float
    car_speed_kmh: float
    congestion_penalty: float
    parking_penalty: float
    logit_theta: float

    def __post_init__(self) -> None:
        if not 0.0 < self.worker_ratio <= 1.0:
            raise ValueError("worker_ratio must be in (0, 1].")
        if self.car_speed_kmh <= 0:
            raise ValueError("car_speed_kmh must be positive.")
        if self.logit_theta <= 0:
            raise ValueError("logit_theta must be positive.")


@dataclass(frozen=True)
class TransitRulesConfig:
    line_count: int
    station_spacing_m: int
    access_radius_m: int
    transfer_penalty_min: float
    max_turn_angle_deg: float
    shared_track_penalty: float
    maximum_transfers: int
    service_template: str
    headway_min: float

    def __post_init__(self) -> None:
        if self.line_count <= 0:
            raise ValueError("line_count must be positive.")
        if self.station_spacing_m <= 0 or self.access_radius_m <= 0:
            raise ValueError("station spacing and access radius must be positive.")
        if self.maximum_transfers < 0:
            raise ValueError("maximum_transfers must be >= 0.")
        if self.headway_min <= 0:
            raise ValueError("headway_min must be positive.")


@dataclass(frozen=True)
class OptimizationConfig:
    solver_type: str
    iterations: int
    temperature_schedule: tuple[float, float]
    beam_width: int
    candidate_pool_size: int
    random_seed: int
    parallel_workers: int

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")
        if len(self.temperature_schedule) != 2:
            raise ValueError("temperature_schedule must include (initial, final).")
        t0, t1 = self.temperature_schedule
        if t0 <= 0 or t1 <= 0:
            raise ValueError("Temperature values must be positive.")
        if self.beam_width <= 0 or self.candidate_pool_size <= 0:
            raise ValueError("beam_width and candidate_pool_size must be positive.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        if self.parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive.")


@dataclass(frozen=True)
class BatchConfig:
    seeds: tuple[int, ...]
    presets: tuple[str, ...]
    line_count_sweep: tuple[int, ...]
    export_aggregated_metrics: bool

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("Batch runs require at least one seed.")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Batch seeds must be non-negative integers.")
        if not self.presets:
            raise ValueError("At least one batch preset must be configured.")
        if not self.line_count_sweep:
            raise ValueError("line_count_sweep cannot be empty.")
        if any(line_count <= 0 for line_count in self.line_count_sweep):
            raise ValueError("line_count_sweep values must be positive.")


@dataclass(frozen=True)
class AppConfig:
    schema_version: str
    city: CityConfig
    demand: DemandConfig
    transit_rules: TransitRulesConfig
    optimization: OptimizationConfig
    batch: BatchConfig
    seed_context: SeedContext

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        seed_context_data = data.get("seed_context") or {
            "city_seed": data["city"]["random_seed"],
            "demand_seed": data["city"]["random_seed"],
            "optimization_seed": data["optimization"]["random_seed"],
            "batch_seeds": data["batch"]["seeds"],
        }
        return cls(
            schema_version=data["schema_version"],
            city=CityConfig(
                **{
                    **data["city"],
                    "center_shares": tuple(data["city"]["center_shares"]),
                    "center_sizes": tuple(data["city"]["center_sizes"]),
                }
            ),
            demand=DemandConfig(**data["demand"]),
            transit_rules=TransitRulesConfig(**data["transit_rules"]),
            optimization=OptimizationConfig(
                **{
                    **data["optimization"],
                    "temperature_schedule": tuple(data["optimization"]["temperature_schedule"]),
                }
            ),
            batch=BatchConfig(
                **{
                    **data["batch"],
                    "seeds": tuple(data["batch"]["seeds"]),
                    "presets": tuple(data["batch"]["presets"]),
                    "line_count_sweep": tuple(data["batch"]["line_count_sweep"]),
                }
            ),
            seed_context=SeedContext(
                city_seed=seed_context_data["city_seed"],
                demand_seed=seed_context_data["demand_seed"],
                optimization_seed=seed_context_data["optimization_seed"],
                batch_seeds=tuple(seed_context_data["batch_seeds"]),
            ),
        )
