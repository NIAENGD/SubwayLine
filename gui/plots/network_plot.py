"""Network and OD plotting helpers."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from optimize.initial_builder import NetworkPlan


def plot_od_desire_lines(ax: Axes, od_matrix: np.ndarray, grid_shape: tuple[int, int], max_lines: int = 80) -> None:
    rows, cols = grid_shape
    od = np.asarray(od_matrix, dtype=float)
    ax.clear()
    if od.size == 0:
        ax.set_title("OD desire lines (no data)")
        return

    flat_idx = np.argsort(od.reshape(-1))[::-1][:max_lines]
    for idx in flat_idx:
        val = float(od.reshape(-1)[idx])
        if val <= 0:
            continue
        o, d = divmod(int(idx), od.shape[1])
        oy, ox = divmod(o, cols)
        dy, dx = divmod(d, cols)
        ax.plot([ox, dx], [oy, dy], color="purple", alpha=min(0.8, 0.15 + val / (od.max() + 1e-9)), linewidth=0.8)

    ax.set_xlim(0, cols - 1)
    ax.set_ylim(0, rows - 1)
    ax.set_title("OD desire lines")


def plot_network_map(ax: Axes, plan: NetworkPlan, grid_shape: tuple[int, int]) -> None:
    rows, cols = grid_shape
    ax.clear()
    palette = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a4c93", "#118ab2", "#ef476f", "#3a86ff"]
    for i, line in enumerate(plan.lines):
        if not line.points:
            continue
        xs = [p[0] for p in line.points]
        ys = [p[1] for p in line.points]
        color = palette[i % len(palette)]
        ax.plot(xs, ys, "-", color=color, linewidth=2, label=line.id)
        ax.scatter(xs, ys, color=color, s=10)
    ax.set_xlim(0, cols - 1)
    ax.set_ylim(0, rows - 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Generated line map")
