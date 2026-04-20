"""Transit rules controls tab."""

from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QLineEdit, QSpinBox, QWidget

from config.schema import TransitRulesConfig


class TransitTab(QWidget):
    def __init__(self, config: TransitRulesConfig) -> None:
        super().__init__()
        self.line_count = QSpinBox(); self.line_count.setRange(1, 32)
        self.access_radius_m = QSpinBox(); self.access_radius_m.setRange(100, 5000)
        self.transfer_penalty_min = QDoubleSpinBox(); self.transfer_penalty_min.setRange(0.0, 60.0)
        self.max_turn_angle_deg = QDoubleSpinBox(); self.max_turn_angle_deg.setRange(0.0, 180.0)
        self.shared_track_penalty = QDoubleSpinBox(); self.shared_track_penalty.setRange(0.0, 10.0)
        self.maximum_transfers = QSpinBox(); self.maximum_transfers.setRange(0, 8)
        self.service_template = QLineEdit()
        self.headway_min = QDoubleSpinBox(); self.headway_min.setRange(1.0, 30.0)

        form = QFormLayout(self)
        form.addRow("line count", self.line_count)
        form.addRow("access radius", self.access_radius_m)
        form.addRow("transfer penalty", self.transfer_penalty_min)
        form.addRow("max turn angle", self.max_turn_angle_deg)
        form.addRow("shared track penalty", self.shared_track_penalty)
        form.addRow("maximum transfers", self.maximum_transfers)
        form.addRow("service template", self.service_template)
        form.addRow("headway", self.headway_min)
        self.load_from_config(config)

    def load_from_config(self, cfg: TransitRulesConfig) -> None:
        self.line_count.setValue(cfg.line_count)
        self.access_radius_m.setValue(cfg.access_radius_m)
        self.transfer_penalty_min.setValue(cfg.transfer_penalty_min)
        self.max_turn_angle_deg.setValue(cfg.max_turn_angle_deg)
        self.shared_track_penalty.setValue(cfg.shared_track_penalty)
        self.maximum_transfers.setValue(cfg.maximum_transfers)
        self.service_template.setText(cfg.service_template)
        self.headway_min.setValue(cfg.headway_min)
