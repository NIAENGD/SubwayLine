"""Chart helpers for load and score diagnostics."""

from __future__ import annotations

from matplotlib.axes import Axes


def plot_load_chart(ax: Axes, per_line_metrics: dict[str, float]) -> None:
    ax.clear()
    labels = list(per_line_metrics.keys())
    values = [per_line_metrics[k] for k in labels]
    ax.bar(labels, values, color="#457b9d")
    ax.set_title("Boardings / loads")
    ax.tick_params(axis="x", rotation=30)


def plot_score_breakdown(ax: Axes, weighted_terms: dict[str, float]) -> None:
    ax.clear()
    labels = list(weighted_terms.keys())
    values = [weighted_terms[k] for k in labels]
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in values]
    ax.barh(labels, values, color=colors)
    ax.set_title("Score breakdown")
