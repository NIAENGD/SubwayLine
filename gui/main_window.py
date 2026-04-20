"""Main application window wiring tabs and one-click workflows."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
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
from gui.tabs.results_tab import ResultsTab
from gui.tabs.transit_tab import TransitTab
from optimize.sweep_runner import run_staged_line_sweep




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
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SubwayLine")
        self.state = AppState()

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        self.tabs = QTabWidget()
        self.city_tab = CityTab(self.state.config.city)
        self.demand_tab = DemandTab(self.state.config.demand)
        self.transit_tab = TransitTab(self.state.config.transit_rules)
        self.optimization_tab = OptimizationTab(self.state.config.optimization)
        self.results_tab = ResultsTab()
        self.batch_tab = BatchTab(self.state.config.batch)

        self.tabs.addTab(self.city_tab, "City")
        self.tabs.addTab(self.demand_tab, "Demand")
        self.tabs.addTab(self.transit_tab, "Transit rules")
        self.tabs.addTab(self.optimization_tab, "Optimization")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.batch_tab, "Batch runs")

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

        self.status_label = QLabel("Ready")
        outer.addLayout(buttons)
        outer.addWidget(self.tabs)
        outer.addWidget(self.status_label)

        self.btn_generate.clicked.connect(self.on_generate_city)
        self.btn_solve_selected.clicked.connect(self.on_solve_selected)
        self.btn_solve_1_8.clicked.connect(self.on_solve_1_8)
        self.btn_save_cfg.clicked.connect(self.on_save_config)
        self.btn_load_cfg.clicked.connect(self.on_load_config)
        self.btn_export_images.clicked.connect(self.on_export_images)
        self.btn_export_csv.clicked.connect(self.on_export_csv)
        self.btn_reset.clicked.connect(self.on_reset_defaults)

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
        try:
            self.state.set_config(self._collect_config())
            cfg = self.state.config
            jobs = generate_jobs_grid(total_jobs=cfg.city.total_jobs, preset_name=cfg.city.employment_preset, seed=cfg.city.random_seed).jobs
            population = generate_population_grid(
                total_population=cfg.city.total_population,
                preset_name=cfg.city.population_preset,
                seed=cfg.city.random_seed,
                jobs=jobs,
                jobs_housing_interaction=cfg.city.jobs_housing_interaction,
                residential_cluster_count=cfg.city.residential_cluster_count,
            ).population

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
            workers = generate_workers(population, cfg.demand.worker_ratio)
            od = build_od_matrix(workers=workers, destination_probabilities=dest_probs)

            self.state.jobs = jobs
            self.state.population = population
            self.state.od = od
            self.status_label.setText("City generated")
        except Exception as exc:
            QMessageBox.critical(self, "Generate City failed", str(exc))

    def _run_solver(self, max_lines: int) -> None:
        if self.state.jobs is None or self.state.population is None or self.state.od is None:
            self.on_generate_city()
        cfg = self.state.config
        result_map = run_staged_line_sweep(
            population=self.state.population,
            jobs=self.state.jobs,
            od_matrix=self.state.od,
            max_lines=max_lines,
            objective_profile=cfg.batch.presets[0] if cfg.batch.presets else "balanced",
            iterations=cfg.optimization.iterations,
            temperature_schedule=cfg.optimization.temperature_schedule,
            output_dir=Path("outputs/gui") / datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
            random_seed=cfg.optimization.random_seed,
        )
        self.state.networks = [result_map[k].best_plan for k in sorted(result_map)]
        selected_count = min(cfg.transit_rules.line_count, max_lines)
        selected = result_map[selected_count]
        self.state.metrics = selected.score_summary
        self.results_tab.render(
            jobs=self.state.jobs,
            population=self.state.population,
            od=self.state.od,
            plan=selected.best_plan,
            weighted_terms=selected.score_summary.get("weighted_terms", {}),
        )

    def on_solve_selected(self) -> None:
        try:
            self.state.set_config(self._collect_config())
            self._run_solver(self.state.config.transit_rules.line_count)
            self.status_label.setText("Solved selected line count")
            self.tabs.setCurrentWidget(self.results_tab)
        except Exception as exc:
            QMessageBox.critical(self, "Solve failed", str(exc))

    def on_solve_1_8(self) -> None:
        try:
            self.state.set_config(self._collect_config())
            self._run_solver(8)
            self.status_label.setText("Solved 1–8 sweep")
            self.tabs.setCurrentWidget(self.results_tab)
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
        for key, (fig, _) in self.results_tab.figures.items():
            fig.savefig(base / f"{key}.png", dpi=150, bbox_inches="tight")
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
        self.status_label.setText("Reset to defaults")
