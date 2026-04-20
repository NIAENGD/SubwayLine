"""Results display tab with required visualizations."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.plots.charts import plot_load_chart, plot_score_breakdown
from gui.plots.heatmaps import (
    plot_accessibility_map,
    plot_coverage_map,
    plot_jobs_heatmap,
    plot_population_heatmap,
)
from gui.plots.network_plot import plot_interactive_density_map, plot_network_map, plot_od_desire_lines
from network.metrics import compute_metric_components
from optimize.initial_builder import NetworkPlan


class ResultsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Interactive map layer"))
        self.layer_selector = QComboBox()
        self.layer_selector.addItems(["Population density", "Job density", "Transit coverage", "Accessibility score"])
        self.show_lines = QCheckBox("Show lines")
        self.show_stations = QCheckBox("Show stations")
        self.show_transfers = QCheckBox("Highlight transfers")
        self.show_lines.setChecked(True)
        self.show_stations.setChecked(True)
        self.show_transfers.setChecked(True)
        controls.addWidget(self.layer_selector)
        controls.addWidget(self.show_lines)
        controls.addWidget(self.show_stations)
        controls.addWidget(self.show_transfers)
        controls.addStretch(1)

        grid = QGridLayout()
        self.figures: dict[str, tuple[Figure, FigureCanvasQTAgg]] = {}
        self._latest_payload: dict[str, object] = {}

        keys = [
            "interactive_map",
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
        layout.addLayout(controls)
        layout.addLayout(grid)
        layout.addWidget(self.metrics_table)
        self.layer_selector.currentIndexChanged.connect(self._redraw_interactive_map)
        self.show_lines.toggled.connect(self._redraw_interactive_map)
        self.show_stations.toggled.connect(self._redraw_interactive_map)
        self.show_transfers.toggled.connect(self._redraw_interactive_map)

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
        self._latest_payload = {
            "jobs": jobs,
            "population": population,
            "coverage": coverage,
            "plan": plan,
            "grid_shape": (rows, cols),
        }
        self._redraw_interactive_map()

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

    def _redraw_interactive_map(self) -> None:
        if not self._latest_payload:
            return
        jobs = np.asarray(self._latest_payload["jobs"], dtype=float)
        population = np.asarray(self._latest_payload["population"], dtype=float)
        coverage = np.asarray(self._latest_payload["coverage"], dtype=float)
        plan = self._latest_payload["plan"]
        grid_shape = self._latest_payload["grid_shape"]

        layer_name = self.layer_selector.currentText()
        if layer_name == "Population density":
            base_layer = population
        elif layer_name == "Job density":
            base_layer = jobs
        elif layer_name == "Transit coverage":
            base_layer = coverage
        else:
            base_layer = jobs * (0.25 + 0.75 * coverage)

        fig, canvas = self.figures["interactive_map"]
        plot_interactive_density_map(
            fig.subplots(),
            base_layer=base_layer,
            layer_title=f"Interactive map: {layer_name}",
            plan=plan,
            grid_shape=grid_shape,
            show_lines=self.show_lines.isChecked(),
            show_stations=self.show_stations.isChecked(),
            show_transfers=self.show_transfers.isChecked(),
        )
        canvas.draw_idle()
