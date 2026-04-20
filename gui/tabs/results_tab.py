"""Results display tab with required visualizations."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QGridLayout, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from gui.plots.charts import plot_load_chart, plot_score_breakdown
from gui.plots.heatmaps import (
    plot_accessibility_map,
    plot_coverage_map,
    plot_jobs_heatmap,
    plot_population_heatmap,
)
from gui.plots.network_plot import plot_network_map, plot_od_desire_lines
from network.metrics import compute_metric_components
from optimize.initial_builder import NetworkPlan


class ResultsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        self.figures: dict[str, tuple[Figure, FigureCanvasQTAgg]] = {}

        keys = [
            "jobs",
            "population",
            "od",
            "network",
            "coverage",
            "accessibility",
            "loads",
            "score",
        ]
        for i, key in enumerate(keys):
            fig = Figure(figsize=(4, 3))
            canvas = FigureCanvasQTAgg(fig)
            self.figures[key] = (fig, canvas)
            grid.addWidget(canvas, i // 2, i % 2)

        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        layout.addLayout(grid)
        layout.addWidget(self.metrics_table)

    def render(
        self,
        *,
        jobs: np.ndarray,
        population: np.ndarray,
        od: np.ndarray,
        plan: NetworkPlan,
        weighted_terms: dict[str, float],
    ) -> None:
        rows, cols = population.shape
        station_pts = np.array([pt for line in plan.lines for pt in line.points], dtype=float)
        yy, xx = np.indices((rows, cols), dtype=float)
        if station_pts.size == 0:
            coverage = np.zeros((rows, cols), dtype=float)
        else:
            d = np.sqrt((xx[..., None] - station_pts[:, 0]) ** 2 + (yy[..., None] - station_pts[:, 1]) ** 2)
            coverage = (d.min(axis=2) <= 2.0).astype(float)

        plot_jobs_heatmap(self.figures["jobs"][0].subplots(), jobs)
        plot_population_heatmap(self.figures["population"][0].subplots(), population)
        plot_od_desire_lines(self.figures["od"][0].subplots(), od, (rows, cols))
        plot_network_map(self.figures["network"][0].subplots(), plan, (rows, cols))
        plot_coverage_map(self.figures["coverage"][0].subplots(), coverage)
        plot_accessibility_map(self.figures["accessibility"][0].subplots(), jobs, coverage)

        line_loads = {line.id: float(len(line.points)) for line in plan.lines}
        plot_load_chart(self.figures["loads"][0].subplots(), line_loads)
        plot_score_breakdown(self.figures["score"][0].subplots(), weighted_terms)

        for _, canvas in self.figures.values():
            canvas.draw_idle()

        metrics = compute_metric_components(plan=plan, population=population, jobs=jobs, od_matrix=od)
        metrics_dict = metrics.to_dict()
        self.metrics_table.setRowCount(len(metrics_dict))
        for i, (name, val) in enumerate(metrics_dict.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{val:.4f}"))
