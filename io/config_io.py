"""JSON persistence helpers for app configuration."""

from __future__ import annotations

import json
from pathlib import Path

from config.defaults import build_default_config
from config.schema import AppConfig


def save_config(config: AppConfig, path: str | Path) -> Path:
    """Persist an AppConfig to JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def load_config(path: str | Path) -> AppConfig:
    """Load AppConfig from JSON and validate required seed context."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = AppConfig.from_dict(payload)
    validate_seed_context(config)
    return config


def validate_seed_context(config: AppConfig) -> None:
    """Guarantee every run carries explicit seeds."""
    seeds = config.seed_context
    if seeds.city_seed < 0 or seeds.demand_seed < 0 or seeds.optimization_seed < 0:
        raise ValueError("Explicit city/demand/optimization seeds are required.")
    if not seeds.batch_seeds:
        raise ValueError("At least one batch seed is required.")


def load_or_default(path: str | Path | None) -> AppConfig:
    """Load config if path exists, otherwise return defaults."""
    if path is None:
        return build_default_config()

    cfg_path = Path(path)
    if not cfg_path.exists():
        return build_default_config()

    return load_config(cfg_path)
