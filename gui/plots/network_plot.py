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


def plot_interactive_density_map(
    ax: Axes,
    *,
    base_layer: np.ndarray,
    layer_title: str,
    plan: NetworkPlan,
    grid_shape: tuple[int, int],
    show_lines: bool,
    show_stations: bool,
    show_transfers: bool,
) -> None:
    rows, cols = grid_shape
    ax.clear()
    img = np.asarray(base_layer, dtype=float)
    ax.imshow(img, origin="lower", cmap="viridis", alpha=0.95)

    station_to_lines: dict[tuple[float, float], set[str]] = {}
    for line in plan.lines:
        for pt in line.points:
            station_to_lines.setdefault((float(pt[0]), float(pt[1])), set()).add(line.id)

    palette = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a4c93", "#118ab2", "#ef476f", "#3a86ff"]
    if show_lines:
        for i, line in enumerate(plan.lines):
            if not line.points:
                continue
            xs = [p[0] for p in line.points]
            ys = [p[1] for p in line.points]
            color = palette[i % len(palette)]
            ax.plot(xs, ys, "-", color=color, linewidth=2.0, alpha=0.9)

    if show_stations and station_to_lines:
        pts = np.array(list(station_to_lines), dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], s=16, c="white", edgecolors="black", linewidths=0.7, label="Stations")

    if show_transfers:
        transfer_pts = np.array([pt for pt, lines in station_to_lines.items() if len(lines) > 1], dtype=float)
        if transfer_pts.size > 0:
            ax.scatter(
                transfer_pts[:, 0],
                transfer_pts[:, 1],
                s=60,
                marker="*",
                c="gold",
                edgecolors="black",
                linewidths=0.8,
                label="Transfer stations",
                zorder=6,
            )

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_title(layer_title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if show_stations or show_transfers:
        ax.legend(loc="upper right", fontsize=8)
