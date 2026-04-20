"""Demand controls tab."""

from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QWidget

from config.schema import DemandConfig


class DemandTab(QWidget):
    def __init__(self, config: DemandConfig) -> None:
        super().__init__()
        self.worker_ratio = QDoubleSpinBox(); self.worker_ratio.setRange(0.01, 1.0); self.worker_ratio.setSingleStep(0.01)
        self.destination_choice_alpha = QDoubleSpinBox(); self.destination_choice_alpha.setRange(0.0, 10.0)
        self.impedance_beta = QDoubleSpinBox(); self.impedance_beta.setRange(0.0, 10.0); self.impedance_beta.setDecimals(4)
        self.car_speed_kmh = QDoubleSpinBox(); self.car_speed_kmh.setRange(1.0, 160.0)
        self.congestion_penalty = QDoubleSpinBox(); self.congestion_penalty.setRange(0.0, 10.0)
        self.parking_penalty = QDoubleSpinBox(); self.parking_penalty.setRange(0.0, 100.0)
        self.logit_theta = QDoubleSpinBox(); self.logit_theta.setRange(0.01, 10.0)

        form = QFormLayout(self)
        form.addRow("worker ratio", self.worker_ratio)
        form.addRow("destination choice alpha", self.destination_choice_alpha)
        form.addRow("impedance beta", self.impedance_beta)
        form.addRow("car speed parameters", self.car_speed_kmh)
        form.addRow("congestion penalties", self.congestion_penalty)
        form.addRow("parking penalty", self.parking_penalty)
        form.addRow("logit parameters", self.logit_theta)
        self.load_from_config(config)

    def load_from_config(self, cfg: DemandConfig) -> None:
        self.worker_ratio.setValue(cfg.worker_ratio)
        self.destination_choice_alpha.setValue(cfg.destination_choice_alpha)
        self.impedance_beta.setValue(cfg.impedance_beta)
        self.car_speed_kmh.setValue(cfg.car_speed_kmh)
        self.congestion_penalty.setValue(cfg.congestion_penalty)
        self.parking_penalty.setValue(cfg.parking_penalty)
        self.logit_theta.setValue(cfg.logit_theta)
