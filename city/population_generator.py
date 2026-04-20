"""Population grid generation for synthetic 64x64 city population field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from city.fields import smooth_field

GRID_SIZE = 64


@dataclass(frozen=True)
class PopulationPreset:
    """Preset knobs controlling population surface morphology."""

    name: str
    ring_radii: tuple[float, ...]
    ring_strengths: tuple[float, ...]
    angular_amplitude: float
    angular_modes: tuple[int, ...]
    bump_count_range: tuple[int, int]
    bump_amplitude_range: tuple[float, float]
    bump_sigma_range: tuple[float, float]
    bump_outer_bias: float
    noise_strength: float
    interaction_multiplier: float

    def __post_init__(self) -> None:
        if len(self.ring_radii) != len(self.ring_strengths):
            raise ValueError("ring_radii and ring_strengths must be same length.")
        if not self.ring_radii:
            raise ValueError("At least one radial ring must be provided.")
        if any(m <= 0 for m in self.angular_modes):
            raise ValueError("angular_modes must be positive integers.")
        lo, hi = self.bump_count_range
        if lo < 8 or hi > 12 or lo > hi:
            raise ValueError("bump_count_range must remain inside [8, 12].")


POPULATION_PRESETS: dict[str, PopulationPreset] = {
    "inner_ring_dense": PopulationPreset(
        name="inner_ring_dense",
        ring_radii=(0.14, 0.24, 0.36, 0.52, 0.72, 0.92),
        ring_strengths=(1.35, 1.60, 1.15, 0.82, 0.92, 0.72),
        angular_amplitude=0.26,
        angular_modes=(2, 3, 5),
        bump_count_range=(8, 10),
        bump_amplitude_range=(0.12, 0.34),
        bump_sigma_range=(2.0, 4.2),
        bump_outer_bias=0.62,
        noise_strength=0.30,
        interaction_multiplier=0.70,
    ),
    "mixed_ringed_us_style": PopulationPreset(
        name="mixed_ringed_us_style",
        ring_radii=(0.10, 0.20, 0.34, 0.48, 0.66, 0.86),
        ring_strengths=(0.95, 1.20, 1.00, 1.14, 0.92, 0.78),
        angular_amplitude=0.30,
        angular_modes=(2, 4, 6),
        bump_count_range=(9, 11),
        bump_amplitude_range=(0.18, 0.40),
        bump_sigma_range=(2.2, 4.8),
        bump_outer_bias=0.70,
        noise_strength=0.34,
        interaction_multiplier=1.00,
    ),
    "suburban_clustered": PopulationPreset(
        name="suburban_clustered",
        ring_radii=(0.08, 0.20, 0.36, 0.54, 0.74, 0.94),
        ring_strengths=(0.78, 0.96, 1.05, 1.22, 1.18, 0.92),
        angular_amplitude=0.34,
        angular_modes=(3, 5, 7),
        bump_count_range=(10, 12),
        bump_amplitude_range=(0.28, 0.58),
        bump_sigma_range=(2.2, 5.4),
        bump_outer_bias=0.82,
        noise_strength=0.32,
        interaction_multiplier=1.20,
    ),
    "outer_clustered_sprawl": PopulationPreset(
        name="outer_clustered_sprawl",
        ring_radii=(0.10, 0.26, 0.42, 0.58, 0.78, 0.98),
        ring_strengths=(0.62, 0.72, 0.86, 1.08, 1.22, 1.10),
        angular_amplitude=0.38,
        angular_modes=(2, 5, 8),
        bump_count_range=(10, 12),
        bump_amplitude_range=(0.34, 0.66),
        bump_sigma_range=(2.8, 5.8),
        bump_outer_bias=0.88,
        noise_strength=0.36,
        interaction_multiplier=1.35,
    ),
    "randomized_uneven": PopulationPreset(
        name="randomized_uneven",
        ring_radii=(0.08, 0.22, 0.38, 0.54, 0.72, 0.92),
        ring_strengths=(1.00, 0.84, 1.18, 0.94, 1.12, 0.82),
        angular_amplitude=0.42,
        angular_modes=(2, 3, 7),
        bump_count_range=(8, 12),
        bump_amplitude_range=(0.16, 0.62),
        bump_sigma_range=(2.0, 6.0),
        bump_outer_bias=0.74,
        noise_strength=0.40,
        interaction_multiplier=1.08,
    ),
}

DEFAULT_POPULATION_PRESET = POPULATION_PRESETS["mixed_ringed_us_style"]


@dataclass(frozen=True)
class PopulationGenerationResult:
    population: np.ndarray
    preset: PopulationPreset
    bump_count: int


def get_population_preset(name: str) -> PopulationPreset:
    """Return a preset by name, with section-4.4 default fallback."""
    return POPULATION_PRESETS.get(name, DEFAULT_POPULATION_PRESET)


def _build_geometry(grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((grid_size, grid_size), dtype=float)
    center = (grid_size - 1) / 2.0
    x = xx - center
    y = yy - center
    radius = np.hypot(x, y)
    radius_norm = radius / max(radius.max(), 1e-12)
    theta = np.arctan2(y, x)
    return radius_norm, theta, xx, yy


def _build_radial_ring_layer(radius_norm: np.ndarray, preset: PopulationPreset) -> np.ndarray:
    radii = np.asarray(preset.ring_radii, dtype=float)
    strengths = np.asarray(preset.ring_strengths, dtype=float)
    interp = np.interp(radius_norm, radii, strengths, left=strengths[0], right=strengths[-1])
    layer = smooth_field(interp, sigma=1.4, radius=3)
    return np.clip(layer, 0.05, None)


def _build_angular_distortion(theta: np.ndarray, radius_norm: np.ndarray, preset: PopulationPreset, rng: np.random.Generator) -> np.ndarray:
    signal = np.ones_like(theta, dtype=float)
    amp = preset.angular_amplitude
    # Slightly strengthen asymmetry outside the core.
    radial_gain = 0.65 + 0.70 * radius_norm
    for mode in preset.angular_modes:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        local_amp = amp * rng.uniform(0.60, 1.00) / np.sqrt(mode)
        signal += local_amp * np.sin(mode * theta + phase) * radial_gain
    return np.clip(signal, 0.18, None)


def _sample_bump_radius(rng: np.random.Generator, outer_bias: float) -> float:
    # Bias towards suburban/outer rings while still allowing occasional inner peaks.
    return 0.10 + 0.90 * ((rng.random() ** (1.0 + 2.8 * outer_bias)))


def _build_suburban_bumps(
    *,
    xx: np.ndarray,
    yy: np.ndarray,
    radius_norm: np.ndarray,
    preset: PopulationPreset,
    rng: np.random.Generator,
    residential_cluster_count: int | None,
) -> tuple[np.ndarray, int]:
    lo, hi = preset.bump_count_range
    bump_count = int(residential_cluster_count) if residential_cluster_count is not None else int(rng.integers(lo, hi + 1))
    bump_count = int(np.clip(bump_count, 8, 12))

    bumps = np.zeros_like(xx, dtype=float)
    amp_lo, amp_hi = preset.bump_amplitude_range
    sigma_lo, sigma_hi = preset.bump_sigma_range

    for _ in range(bump_count):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        r = _sample_bump_radius(rng, preset.bump_outer_bias)
        cx = ((xx.shape[1] - 1) / 2.0) + np.cos(angle) * r * ((xx.shape[1] - 1) / 2.0)
        cy = ((yy.shape[0] - 1) / 2.0) + np.sin(angle) * r * ((yy.shape[0] - 1) / 2.0)
        sigma = rng.uniform(sigma_lo, sigma_hi)
        amp = rng.uniform(amp_lo, amp_hi)
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        bumps += amp * np.exp(-0.5 * dist2 / (sigma * sigma))

    # Discourage inner-core bump concentration by default.
    bumps *= 0.85 + 0.55 * np.clip(radius_norm, 0.0, 1.0)
    return 1.0 + bumps, bump_count


def _build_correlated_noise(grid_size: int, preset: PopulationPreset, rng: np.random.Generator) -> np.ndarray:
    low = rng.normal(loc=0.0, scale=1.0, size=(grid_size, grid_size))
    corr = smooth_field(low, sigma=4.2, radius=7)
    corr = corr - float(corr.mean())
    corr /= float(np.std(corr) + 1e-12)
    layer = 1.0 + preset.noise_strength * corr
    return np.clip(layer, 0.15, None)


def _build_jobs_repulsion(
    *,
    jobs: np.ndarray | None,
    jobs_housing_interaction: float,
    interaction_multiplier: float,
) -> np.ndarray | None:
    if jobs is None:
        return None

    interaction = float(np.clip(jobs_housing_interaction, 0.0, 1.0))
    if interaction <= 0.0:
        return np.ones_like(jobs, dtype=float)

    jobs = np.asarray(jobs, dtype=float)
    p90 = float(np.percentile(jobs, 90))
    p99 = float(np.percentile(jobs, 99))
    denom = max(p99 - p90, 1e-9)
    cores = np.clip((jobs - p90) / denom, 0.0, None)

    # Nonlinear repulsion: high-density job cores receive stronger penalty.
    strength = interaction * interaction_multiplier
    repulsion = 1.0 / (1.0 + (1.8 + 3.2 * strength) * np.power(cores, 1.35 + 1.20 * strength))

    # Smooth to avoid pixel-level artifacts and preserve low-frequency structure.
    repulsion = smooth_field(repulsion, sigma=1.8, radius=3)
    return np.clip(repulsion, 0.05, 1.5)


def reconcile_total_population(field: np.ndarray, target_total_population: float) -> np.ndarray:
    """Rescale and reconcile to exact population total with stable correction."""
    adjusted = np.array(field, dtype=float, copy=True)
    adjusted[adjusted < 0.0] = 0.0

    total = float(adjusted.sum())
    if total <= 0.0:
        return adjusted

    adjusted *= target_total_population / total
    drift = target_total_population - float(adjusted.sum())
    if abs(drift) > 1e-8:
        idx = np.unravel_index(np.argmax(adjusted), adjusted.shape)
        adjusted[idx] += drift
    return adjusted


def generate_population_grid(
    *,
    total_population: int,
    preset_name: str = "mixed_ringed_us_style",
    seed: int = 101,
    grid_size: int = GRID_SIZE,
    jobs: np.ndarray | None = None,
    jobs_housing_interaction: float = 0.50,
    residential_cluster_count: int | None = None,
) -> PopulationGenerationResult:
    """Generate an uneven, non-monotonic residential population surface.

    The surface is composed from section-4.2 layers: radial rings, angular distortion,
    suburban Gaussian bumps (8-12), and correlated low-frequency noise. Optionally,
    an interaction term repels housing from major job cores.
    """
    if grid_size != GRID_SIZE:
        raise ValueError("This generator currently supports exactly a 64x64 grid.")
    if total_population <= 0:
        raise ValueError("total_population must be positive.")

    preset = get_population_preset(preset_name)
    rng = np.random.default_rng(seed)

    radius_norm, theta, xx, yy = _build_geometry(grid_size)

    radial_layer = _build_radial_ring_layer(radius_norm, preset)
    angular_layer = _build_angular_distortion(theta, radius_norm, preset, rng)
    bumps_layer, bump_count = _build_suburban_bumps(
        xx=xx,
        yy=yy,
        radius_norm=radius_norm,
        preset=preset,
        rng=rng,
        residential_cluster_count=residential_cluster_count,
    )
    noise_layer = _build_correlated_noise(grid_size, preset, rng)

    base = radial_layer * angular_layer * bumps_layer * noise_layer

    repulsion = _build_jobs_repulsion(
        jobs=jobs,
        jobs_housing_interaction=jobs_housing_interaction,
        interaction_multiplier=preset.interaction_multiplier,
    )
    if repulsion is not None:
        base *= repulsion

    # Keep strictly non-negative then enforce exact total through reconciliation.
    base = np.clip(base, 0.0, None)
    population = reconcile_total_population(base, float(total_population))
    return PopulationGenerationResult(population=population, preset=preset, bump_count=bump_count)
