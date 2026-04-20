"""Central application state shared across GUI modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from config.defaults import build_default_config
from config.schema import AppConfig, SeedContext


@dataclass
class RunContext:
    """Metadata for a single generation/optimization run."""

    run_id: str
    started_at_utc: str
    seeds: SeedContext


@dataclass
class AppState:
    """Holds the complete typed config plus generated artifacts."""

    config: AppConfig = field(default_factory=build_default_config)
    current_run: RunContext | None = None

    jobs: Any = None
    population: Any = None
    od: Any = None
    networks: list[Any] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Optional optimization artifacts
    candidate_jobs: list[Any] = field(default_factory=list)
    population_samples: list[Any] = field(default_factory=list)

    def start_run(self, run_id: str, seeds: SeedContext | None = None) -> RunContext:
        """Start a new run with explicit seeds."""
        run_seeds = seeds or self.config.seed_context
        self.current_run = RunContext(
            run_id=run_id,
            started_at_utc=datetime.now(UTC).isoformat(),
            seeds=run_seeds,
        )
        return self.current_run

    def set_config(self, config: AppConfig) -> None:
        """Replace config and clear old generated artifacts."""
        self.config = config
        self.reset_generated_artifacts()

    def reset_generated_artifacts(self) -> None:
        """Clear derived model outputs when inputs change."""
        self.current_run = None
        self.jobs = None
        self.population = None
        self.od = None
        self.networks = []
        self.metrics = {}
        self.candidate_jobs = []
        self.population_samples = []
