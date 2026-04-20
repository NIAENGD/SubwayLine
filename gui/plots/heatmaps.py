"""Heatmap plot helpers for GUI result views."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def plot_jobs_heatmap(ax: Axes, jobs: np.ndarray) -> None:
    arr = np.asarray(jobs, dtype=float)
    ax.clear()
    im = ax.imshow(arr, cmap="YlOrRd", origin="lower")
    ax.set_title("Jobs heatmap")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_population_heatmap(ax: Axes, population: np.ndarray) -> None:
    arr = np.asarray(population, dtype=float)
    ax.clear()
    im = ax.imshow(arr, cmap="Blues", origin="lower")
    ax.set_title("Population heatmap")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_coverage_map(ax: Axes, coverage_mask: np.ndarray) -> None:
    mask = np.asarray(coverage_mask, dtype=float)
    ax.clear()
    im = ax.imshow(mask, cmap="Greens", origin="lower", vmin=0.0, vmax=1.0)
    ax.set_title("Coverage map")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_accessibility_map(ax: Axes, jobs: np.ndarray, coverage_mask: np.ndarray) -> None:
    arr = np.asarray(jobs, dtype=float) * np.asarray(coverage_mask, dtype=float)
    ax.clear()
    im = ax.imshow(arr, cmap="viridis", origin="lower")
    ax.set_title("Accessibility map")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
