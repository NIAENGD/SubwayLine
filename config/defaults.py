"""Default configuration values used by the GUI and CLI entrypoints."""

from __future__ import annotations

from config.schema import (
    AppConfig,
    BatchConfig,
    CityConfig,
    DemandConfig,
    OptimizationConfig,
    SeedContext,
    TransitRulesConfig,
)

DEFAULT_SCHEMA_VERSION = "1.0.0"

DEFAULT_CITY = CityConfig(
    total_population=1_000_000,
    total_jobs=600_000,
    tile_size=128,
    employment_preset="polycentric",
    population_preset="mixed_ringed_us_style",
    halo_strength=0.35,
    center_shares=(0.45, 0.30, 0.25),
    center_sizes=(12, 9, 7),
    jobs_housing_interaction=0.5,
    residential_cluster_count=10,
    random_seed=101,
)

DEFAULT_DEMAND = DemandConfig(
    worker_ratio=0.48,
    destination_choice_alpha=1.2,
    impedance_beta=0.08,
    car_speed_kmh=28.0,
    congestion_penalty=1.15,
    parking_penalty=0.20,
    logit_theta=0.65,
)

DEFAULT_TRANSIT_RULES = TransitRulesConfig(
    line_count=4,
    station_spacing_m=900,
    access_radius_m=600,
    transfer_penalty_min=6.0,
    max_turn_angle_deg=50.0,
    shared_track_penalty=0.35,
    maximum_transfers=2,
    service_template="urban_rapid",
    headway_min=6.0,
)

DEFAULT_OPTIMIZATION = OptimizationConfig(
    solver_type="simulated_annealing",
    iterations=600,
    temperature_schedule=(10.0, 0.1),
    beam_width=16,
    candidate_pool_size=64,
    random_seed=202,
    parallel_workers=4,
)

DEFAULT_BATCH = BatchConfig(
    seeds=(101, 202, 303),
    presets=("balanced", "ridership_max", "coverage_max"),
    line_count_sweep=(1, 2, 3, 4, 5, 6, 7, 8),
    export_aggregated_metrics=True,
)

DEFAULT_SEED_CONTEXT = SeedContext(
    city_seed=DEFAULT_CITY.random_seed,
    demand_seed=DEFAULT_CITY.random_seed,
    optimization_seed=DEFAULT_OPTIMIZATION.random_seed,
    batch_seeds=DEFAULT_BATCH.seeds,
)


def build_default_config() -> AppConfig:
    """Build a full typed config with explicit seeds for every run path."""
    return AppConfig(
        schema_version=DEFAULT_SCHEMA_VERSION,
        city=DEFAULT_CITY,
        demand=DEFAULT_DEMAND,
        transit_rules=DEFAULT_TRANSIT_RULES,
        optimization=DEFAULT_OPTIMIZATION,
        batch=DEFAULT_BATCH,
        seed_context=DEFAULT_SEED_CONTEXT,
    )
