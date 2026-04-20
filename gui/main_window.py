"""Main application window wiring tabs and one-click workflows."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from city.jobs_generator import generate_jobs_grid
from city.population_generator import generate_population_grid
from config.defaults import build_default_config
from config.schema import AppConfig, BatchConfig, CityConfig, DemandConfig, OptimizationConfig, TransitRulesConfig
from demand.destination_choice import destination_probabilities
from demand.impedance import compute_car_time
from demand.od_matrix import build_od_matrix
from demand.worker_generation import generate_workers
from gui.state import AppState
from gui.tabs.batch_tab import BatchTab
from gui.tabs.city_tab import CityTab
from gui.tabs.demand_tab import DemandTab
from gui.tabs.optimization_tab import OptimizationTab
from gui.tabs.transit_tab import TransitTab
from gui.plots.network_plot import plot_interactive_density_map
from optimize.initial_builder import NetworkPlan
from optimize.sweep_runner import run_staged_line_sweep


class GenerateCityWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            cfg = self.config
            self.progress.emit(5, "Generating city: jobs")
            jobs = generate_jobs_grid(total_jobs=cfg.city.total_jobs, preset_name=cfg.city.employment_preset, seed=cfg.city.random_seed).jobs
            self.progress.emit(25, "Generating city: population")
            population = generate_population_grid(
                total_population=cfg.city.total_population,
                preset_name=cfg.city.population_preset,
                seed=cfg.city.random_seed,
                jobs=jobs,
                jobs_housing_interaction=cfg.city.jobs_housing_interaction,
                residential_cluster_count=cfg.city.residential_cluster_count,
            ).population
            self.progress.emit(45, "Generating city: demand prep")

            rows, cols = jobs.shape
            yy, xx = np.indices((rows, cols), dtype=float)
            coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
            dif = coords[:, None, :] - coords[None, :, :]
            distance_km = np.sqrt((dif**2).sum(axis=2))
            car_time = compute_car_time(distance_km=distance_km, car_speed_kmh=cfg.demand.car_speed_kmh)
            dest_probs = destination_probabilities(
                jobs=jobs,
                car_time_min=car_time,
                alpha=cfg.demand.destination_choice_alpha,
                beta=cfg.demand.impedance_beta,
            )
            self.progress.emit(75, "Generating city: OD matrix")
            workers = generate_workers(population, cfg.demand.worker_ratio)
            od = build_od_matrix(workers=workers, destination_probabilities=dest_probs)
            self.progress.emit(100, "City generated")
            self.finished.emit({"jobs": jobs, "population": population, "od": od})
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit(str(exc))


class SolveWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, *, config: AppConfig, jobs: np.ndarray, population: np.ndarray, od: np.ndarray, max_lines: int) -> None:
        super().__init__()
        self.config = config
        self.jobs = jobs
        self.population = population
        self.od = od
        self.max_lines = max_lines

    def run(self) -> None:
        try:
            cfg = self.config

            def _on_progress(line_count: int, max_count: int, stage: str) -> None:
                base = int(((line_count - 1) / max_count) * 100)
                stage_boost = {"build_initial_network": 8, "annealing": 16, "saved": 28}.get(stage, 0)
                self.progress.emit(min(99, base + stage_boost), f"Solving {line_count}/{max_count}: {stage.replace('_', ' ')}")

            result_map = run_staged_line_sweep(
                population=self.population,
                jobs=self.jobs,
                od_matrix=self.od,
                max_lines=self.max_lines,
                objective_profile=cfg.batch.presets[0] if cfg.batch.presets else "balanced",
                iterations=cfg.optimization.iterations,
                temperature_schedule=cfg.optimization.temperature_schedule,
                output_dir=Path("outputs/gui") / datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
                random_seed=cfg.optimization.random_seed,
                progress_callback=_on_progress,
            )
            selected_count = min(cfg.transit_rules.line_count, self.max_lines)
            selected = result_map[selected_count]
            self.finished.emit(
                {
                    "networks": [result_map[k].best_plan for k in sorted(result_map)],
                    "selected_plan": selected.best_plan,
                    "metrics": selected.score_summary,
                }
            )
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit(str(exc))



def _load_config_io_module():
    import importlib.util
    module_path = Path(__file__).resolve().parents[1] / "io" / "config_io.py"
    spec = importlib.util.spec_from_file_location("subwayline_config_io", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config I/O module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MainWindow(QMainWindow):
    def __init__(self, initial_config: AppConfig | None = None) -> None:
        super().__init__()
        self.setWindowTitle("SubwayLine")
        self.state = AppState()
        self.current_plan: NetworkPlan | None = None
        self._active_thread: QThread | None = None
        self._active_worker: QObject | None = None
        if initial_config is not None:
            self.state.set_config(initial_config)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        self.city_tab = CityTab(self.state.config.city)
        self.demand_tab = DemandTab(self.state.config.demand)
        self.transit_tab = TransitTab(self.state.config.transit_rules)
        self.optimization_tab = OptimizationTab(self.state.config.optimization)
        self.batch_tab = BatchTab(self.state.config.batch)

        buttons = QHBoxLayout()
        self.btn_generate = QPushButton("Generate City")
        self.btn_solve_selected = QPushButton("Solve Selected Line Count")
        self.btn_solve_1_8 = QPushButton("Solve 1–8")
        self.btn_save_cfg = QPushButton("Save Config")
        self.btn_load_cfg = QPushButton("Load Config")
        self.btn_export_images = QPushButton("Export images")
        self.btn_export_csv = QPushButton("Export metrics CSV")
        self.btn_reset = QPushButton("Reset defaults")
        for b in [
            self.btn_generate,
            self.btn_solve_selected,
            self.btn_solve_1_8,
            self.btn_save_cfg,
            self.btn_load_cfg,
            self.btn_export_images,
            self.btn_export_csv,
            self.btn_reset,
        ]:
            buttons.addWidget(b)

        body_split = QSplitter()
        body_split.setChildrenCollapsible(False)
        body_split.addWidget(self._build_left_sidebar())
        body_split.addWidget(self._build_center_map())
        body_split.addWidget(self._build_right_sidebar())
        body_split.setStretchFactor(0, 0)
        body_split.setStretchFactor(1, 1)
        body_split.setStretchFactor(2, 0)

        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        outer.addLayout(buttons)
        outer.addWidget(body_split)
        outer.addWidget(self.progress_bar)
        outer.addWidget(self.status_label)

        self.btn_generate.clicked.connect(self.on_generate_city)
        self.btn_solve_selected.clicked.connect(self.on_solve_selected)
        self.btn_solve_1_8.clicked.connect(self.on_solve_1_8)
        self.btn_save_cfg.clicked.connect(self.on_save_config)
        self.btn_load_cfg.clicked.connect(self.on_load_config)
        self.btn_export_images.clicked.connect(self.on_export_images)
        self.btn_export_csv.clicked.connect(self.on_export_csv)
        self.btn_reset.clicked.connect(self.on_reset_defaults)

        if initial_config is not None:
            self.city_tab.load_from_config(initial_config.city)
            self.demand_tab.load_from_config(initial_config.demand)
            self.transit_tab.load_from_config(initial_config.transit_rules)
            self.optimization_tab.load_from_config(initial_config.optimization)
            self.batch_tab.load_from_config(initial_config.batch)
        self._redraw_map()

    def _build_left_sidebar(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._wrap_section("City", self.city_tab))
        layout.addWidget(self._wrap_section("Demand", self.demand_tab))
        layout.addWidget(self._wrap_section("Transit Rules", self.transit_tab))
        layout.addWidget(self._wrap_section("Optimization", self.optimization_tab))
        layout.addWidget(self._wrap_section("Batch", self.batch_tab))
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        return scroll

    def _build_center_map(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.map_figure = Figure(figsize=(6, 6))
        self.map_canvas = FigureCanvasQTAgg(self.map_figure)
        layout.addWidget(self.map_canvas)
        return panel

    def _build_right_sidebar(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        map_controls = QGroupBox("Map controls")
        map_form = QFormLayout(map_controls)
        self.layer_selector = QComboBox()
        self.layer_selector.addItems(["Population", "Jobs"])
        self.show_lines = QCheckBox("Show generated lines overlay")
        self.show_stations = QCheckBox("Show stations")
        self.show_transfers = QCheckBox("Highlight transfers")
        self.show_lines.setChecked(True)
        self.show_stations.setChecked(True)
        self.show_transfers.setChecked(True)
        map_form.addRow("Base layer", self.layer_selector)
        map_form.addRow(self.show_lines)
        map_form.addRow(self.show_stations)
        map_form.addRow(self.show_transfers)
        layout.addWidget(map_controls)

        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        layout.addWidget(self.metrics_table)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Run logs will appear here.")
        layout.addWidget(self.log_output)
        layout.addStretch(1)

        self.layer_selector.currentIndexChanged.connect(self._redraw_map)
        self.show_lines.toggled.connect(self._redraw_map)
        self.show_stations.toggled.connect(self._redraw_map)
        self.show_transfers.toggled.connect(self._redraw_map)
        return panel

    def _wrap_section(self, title: str, widget: QWidget) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _parse_csv_floats(self, text: str) -> tuple[float, ...]:
        return tuple(float(x.strip()) for x in text.split(",") if x.strip())

    def _parse_csv_ints(self, text: str) -> tuple[int, ...]:
        return tuple(int(x.strip()) for x in text.split(",") if x.strip())

    def _collect_config(self) -> AppConfig:
        c = CityConfig(
            total_population=self.city_tab.total_population.value(),
            total_jobs=self.city_tab.total_jobs.value(),
            tile_size=self.city_tab.tile_size.value(),
            employment_preset=self.city_tab.employment_preset.text().strip() or "moderate_polycentric",
            population_preset=self.city_tab.population_preset.text().strip() or "mixed_ringed_us_style",
            halo_strength=self.city_tab.halo_strength.value(),
            center_shares=self._parse_csv_floats(self.city_tab.center_shares.text()) or (1.0,),
            center_sizes=self._parse_csv_ints(self.city_tab.center_sizes.text()) or (1,),
            jobs_housing_interaction=self.city_tab.jobs_housing_interaction.value(),
            residential_cluster_count=self.city_tab.residential_cluster_count.value(),
            random_seed=self.city_tab.random_seed.value(),
        )
        d = DemandConfig(
            worker_ratio=self.demand_tab.worker_ratio.value(),
            destination_choice_alpha=self.demand_tab.destination_choice_alpha.value(),
            impedance_beta=self.demand_tab.impedance_beta.value(),
            car_speed_kmh=self.demand_tab.car_speed_kmh.value(),
            congestion_penalty=self.demand_tab.congestion_penalty.value(),
            parking_penalty=self.demand_tab.parking_penalty.value(),
            logit_theta=self.demand_tab.logit_theta.value(),
        )
        t = TransitRulesConfig(
            line_count=self.transit_tab.line_count.value(),
            station_spacing_m=self.transit_tab.station_spacing_m.value(),
            access_radius_m=self.transit_tab.access_radius_m.value(),
            transfer_penalty_min=self.transit_tab.transfer_penalty_min.value(),
            max_turn_angle_deg=self.transit_tab.max_turn_angle_deg.value(),
            shared_track_penalty=self.transit_tab.shared_track_penalty.value(),
            maximum_transfers=self.transit_tab.maximum_transfers.value(),
            service_template=self.transit_tab.service_template.text().strip() or "urban_rapid",
            headway_min=self.transit_tab.headway_min.value(),
        )
        temp_vals = self._parse_csv_floats(self.optimization_tab.temperature_schedule.text())
        if len(temp_vals) != 2:
            temp_vals = (10.0, 0.1)
        o = OptimizationConfig(
            solver_type=self.optimization_tab.solver_type.text().strip() or "simulated_annealing",
            iterations=self.optimization_tab.iterations.value(),
            temperature_schedule=(temp_vals[0], temp_vals[1]),
            beam_width=self.optimization_tab.beam_width.value(),
            candidate_pool_size=self.optimization_tab.candidate_pool_size.value(),
            random_seed=self.optimization_tab.random_seed.value(),
            parallel_workers=self.optimization_tab.parallel_workers.value(),
        )
        b = BatchConfig(
            seeds=self._parse_csv_ints(self.batch_tab.seeds.text()) or (101,),
            presets=tuple(x.strip() for x in self.batch_tab.presets.text().split(",") if x.strip()) or ("balanced",),
            line_count_sweep=self._parse_csv_ints(self.batch_tab.line_count_sweep.text()) or (1, 2, 3, 4, 5, 6, 7, 8),
            export_aggregated_metrics=self.batch_tab.export_aggregated_metrics.isChecked(),
        )
        return replace(self.state.config, city=c, demand=d, transit_rules=t, optimization=o, batch=b)

    def on_generate_city(self) -> None:
        cfg = self._collect_config()
        randomized_seed = int(np.random.default_rng().integers(0, 1_000_000_001))
        cfg = replace(cfg, city=replace(cfg.city, random_seed=randomized_seed))
        self.city_tab.random_seed.setValue(randomized_seed)
        self.state.set_config(cfg)
        self.current_plan = None
        self._set_busy(True)
        self.progress_bar.setValue(0)
        self._append_log(f"Starting city generation (seed={randomized_seed}).")
        self._start_worker(GenerateCityWorker(self.state.config), self._on_generation_finished, "Generate City failed")

    def _run_solver(self, max_lines: int) -> None:
        if self.state.jobs is None or self.state.population is None or self.state.od is None:
            QMessageBox.information(self, "No city", "Generate city data before solving.")
            return
        self._set_busy(True)
        self.progress_bar.setValue(0)
        self._append_log(f"Starting solver up to {max_lines} lines.")
        self._start_worker(
            SolveWorker(
                config=self.state.config,
                jobs=self.state.jobs,
                population=self.state.population,
                od=self.state.od,
                max_lines=max_lines,
            ),
            self._on_solver_finished,
            "Solve failed",
        )

    def on_solve_selected(self) -> None:
        try:
            self.state.config = self._collect_config()
            self._run_solver(self.state.config.transit_rules.line_count)
        except Exception as exc:
            QMessageBox.critical(self, "Solve failed", str(exc))

    def on_solve_1_8(self) -> None:
        try:
            self.state.config = self._collect_config()
            self._run_solver(8)
        except Exception as exc:
            QMessageBox.critical(self, "Solve 1-8 failed", str(exc))

    def on_save_config(self) -> None:
        save_config = _load_config_io_module().save_config

        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "config.json", "JSON files (*.json)")
        if not path:
            return
        self.state.set_config(self._collect_config())
        save_config(self.state.config, path)
        self.status_label.setText(f"Saved config: {path}")

    def on_load_config(self) -> None:
        load_config = _load_config_io_module().load_config

        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON files (*.json)")
        if not path:
            return
        cfg = load_config(path)
        self.state.set_config(cfg)
        self.city_tab.load_from_config(cfg.city)
        self.demand_tab.load_from_config(cfg.demand)
        self.transit_tab.load_from_config(cfg.transit_rules)
        self.optimization_tab.load_from_config(cfg.optimization)
        self.batch_tab.load_from_config(cfg.batch)
        self.status_label.setText(f"Loaded config: {path}")

    def on_export_images(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export images")
        if not folder:
            return
        base = Path(folder)
        self.map_figure.savefig(base / "city_map.png", dpi=150, bbox_inches="tight")
        self.status_label.setText(f"Exported images to {folder}")

    def on_export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export metrics CSV", "metrics.csv", "CSV files (*.csv)")
        if not path:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["metric", "value"])
            for k, v in self.state.metrics.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        writer.writerow([f"{k}.{sk}", sv])
                else:
                    writer.writerow([k, v])
        self.status_label.setText(f"Exported metrics CSV: {path}")

    def on_reset_defaults(self) -> None:
        cfg = build_default_config()
        self.state.set_config(cfg)
        self.city_tab.load_from_config(cfg.city)
        self.demand_tab.load_from_config(cfg.demand)
        self.transit_tab.load_from_config(cfg.transit_rules)
        self.optimization_tab.load_from_config(cfg.optimization)
        self.batch_tab.load_from_config(cfg.batch)
        self._redraw_map()
        self.status_label.setText("Reset to defaults")

    def _start_worker(self, worker: QObject, on_finished, error_title: str) -> None:
        thread = QThread(self)
        self._active_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # type: ignore[attr-defined]
        worker.progress.connect(self._on_worker_progress)  # type: ignore[attr-defined]
        worker.finished.connect(on_finished)  # type: ignore[attr-defined]
        worker.finished.connect(thread.quit)  # type: ignore[attr-defined]
        worker.failed.connect(lambda message: self._on_worker_failed(error_title, message))  # type: ignore[attr-defined]
        worker.failed.connect(thread.quit)  # type: ignore[attr-defined]
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._active_thread = thread
        thread.start()

    def _on_worker_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self._append_log(message)

    def _on_generation_finished(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        self.state.jobs = data.get("jobs")
        self.state.population = data.get("population")
        self.state.od = data.get("od")
        self.progress_bar.setValue(100)
        self.status_label.setText("City generated")
        self._append_log("City generation complete.")
        self._set_busy(False)
        self._redraw_map()

    def _on_solver_finished(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        self.state.networks = data.get("networks", [])
        self.current_plan = data.get("selected_plan")
        self.state.metrics = data.get("metrics", {})
        self._refresh_metrics_table()
        self.progress_bar.setValue(100)
        self.status_label.setText("Solver finished")
        self._append_log("Solver complete.")
        self._set_busy(False)
        self._redraw_map()

    def _on_worker_failed(self, title: str, message: str) -> None:
        self._set_busy(False)
        self._append_log(f"{title}: {message}")
        QMessageBox.critical(self, title, message)

    def _on_thread_finished(self) -> None:
        self._active_thread = None
        self._active_worker = None

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")

    def _set_busy(self, busy: bool) -> None:
        self.btn_generate.setEnabled(not busy)
        self.btn_solve_selected.setEnabled(not busy)
        self.btn_solve_1_8.setEnabled(not busy)
        self.btn_reset.setEnabled(not busy)

    def _refresh_metrics_table(self) -> None:
        metrics = self.state.metrics if isinstance(self.state.metrics, dict) else {}
        flat_rows: list[tuple[str, object]] = []
        for key, value in metrics.items():
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    flat_rows.append((f"{key}.{child_key}", child_value))
            else:
                flat_rows.append((key, value))
        self.metrics_table.setRowCount(len(flat_rows))
        for row, (name, value) in enumerate(flat_rows):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(name)))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{value}"))

    def _redraw_map(self) -> None:
        self.map_figure.clear()
        ax = self.map_figure.add_subplot(111)
        if self.state.jobs is None or self.state.population is None:
            ax.clear()
            ax.set_title("Generate a city to view the map")
            self.map_canvas.draw_idle()
            return

        base_layer = self.state.population if self.layer_selector.currentText() == "Population" else self.state.jobs
        rows, cols = base_layer.shape
        plot_interactive_density_map(
            ax,
            base_layer=base_layer,
            layer_title=f"City map: {self.layer_selector.currentText()}",
            plan=self.current_plan if self.current_plan is not None else NetworkPlan(lines=[]),
            grid_shape=(rows, cols),
            show_lines=self.show_lines.isChecked() and self.current_plan is not None,
            show_stations=self.show_stations.isChecked() and self.current_plan is not None,
            show_transfers=self.show_transfers.isChecked() and self.current_plan is not None,
        )
        self.map_canvas.draw_idle()
