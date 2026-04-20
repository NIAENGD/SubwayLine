"""Optimization controls tab."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLineEdit, QSpinBox, QWidget

from config.schema import OptimizationConfig


class OptimizationTab(QWidget):
    def __init__(self, config: OptimizationConfig) -> None:
        super().__init__()
        self.solver_type = QLineEdit()
        self.iterations = QSpinBox(); self.iterations.setRange(1, 100_000)
        self.temperature_schedule = QLineEdit()
        self.beam_width = QSpinBox(); self.beam_width.setRange(1, 1024)
        self.candidate_pool_size = QSpinBox(); self.candidate_pool_size.setRange(1, 4096)
        self.random_seed = QSpinBox(); self.random_seed.setRange(0, 1_000_000_000)
        self.parallel_workers = QSpinBox(); self.parallel_workers.setRange(1, 128)

        form = QFormLayout(self)
        form.addRow("solver type", self.solver_type)
        form.addRow("iterations", self.iterations)
        form.addRow("temperature schedule", self.temperature_schedule)
        form.addRow("beam width", self.beam_width)
        form.addRow("candidate pool size", self.candidate_pool_size)
        form.addRow("random seed", self.random_seed)
        form.addRow("parallel workers", self.parallel_workers)
        self.load_from_config(config)

    def load_from_config(self, cfg: OptimizationConfig) -> None:
        self.solver_type.setText(cfg.solver_type)
        self.iterations.setValue(cfg.iterations)
        self.temperature_schedule.setText(f"{cfg.temperature_schedule[0]},{cfg.temperature_schedule[1]}")
        self.beam_width.setValue(cfg.beam_width)
        self.candidate_pool_size.setValue(cfg.candidate_pool_size)
        self.random_seed.setValue(cfg.random_seed)
        self.parallel_workers.setValue(cfg.parallel_workers)
