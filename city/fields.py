"""Grid field construction utilities for jobs generation."""

from __future__ import annotations

import numpy as np

from city.validators import CenterPlacement


def gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    """Construct a normalized 1D Gaussian kernel."""
    x = np.arange(-radius, radius + 1, dtype=float)
    w = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return w / w.sum()


def smooth_field(field: np.ndarray, sigma: float = 2.2, radius: int = 4) -> np.ndarray:
    """Apply separable Gaussian smoothing to a 2D field without SciPy dependency."""
    kernel = gaussian_kernel1d(sigma=sigma, radius=radius)
    padded = np.pad(field, ((0, 0), (radius, radius)), mode="reflect")
    out_x = np.empty_like(field, dtype=float)
    for r in range(field.shape[0]):
        out_x[r, :] = np.convolve(padded[r, :], kernel, mode="valid")

    padded_y = np.pad(out_x, ((radius, radius), (0, 0)), mode="reflect")
    out = np.empty_like(field, dtype=float)
    for c in range(field.shape[1]):
        out[:, c] = np.convolve(padded_y[:, c], kernel, mode="valid")
    return out


def build_background_jobs_field(
    *,
    grid_size: int,
    background_jobs: float,
    centers: list[CenterPlacement],
    seed: int,
    attraction_strength: float = 0.12,
) -> np.ndarray:
    """Generate dispersed background jobs from smoothed noise + weak center attraction."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(shape=1.5, scale=1.0, size=(grid_size, grid_size))
    smoothed = smooth_field(base, sigma=2.4, radius=4)

    if attraction_strength > 0.0 and centers:
        yy, xx = np.indices((grid_size, grid_size), dtype=float)
        attract = np.zeros_like(smoothed)
        max_size = max(c.size for c in centers)
        for c in centers:
            cx, cy = c.centroid
            # Larger centers attract slightly more dispersed jobs.
            scale = (c.size / max_size) ** 1.2
            dist = np.hypot(xx - cx, yy - cy)
            attract += scale / (1.0 + dist)
        attract = attract / (attract.max() + 1e-12)
        smoothed = smoothed * (1.0 + attraction_strength * attract)

    smoothed = np.clip(smoothed, 0.0, None)
    total = float(smoothed.sum())
    if total <= 0:
        return np.zeros((grid_size, grid_size), dtype=float)
    return smoothed * (background_jobs / total)


def reconcile_total_jobs(field: np.ndarray, target_total_jobs: float) -> np.ndarray:
    """Rescale and reconcile to exact job total with numerically stable correction."""
    adjusted = np.array(field, dtype=float, copy=True)
    total = float(adjusted.sum())
    if total <= 0.0:
        return adjusted

    adjusted *= target_total_jobs / total
    drift = target_total_jobs - float(adjusted.sum())
    if abs(drift) > 1e-8:
        # Deterministic correction on max tile to force exact total.
        idx = np.unravel_index(np.argmax(adjusted), adjusted.shape)
        adjusted[idx] += drift

    adjusted[adjusted < 0.0] = 0.0
    return adjusted
