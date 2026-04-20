"""Batch run controls tab."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QFormLayout, QLineEdit, QWidget

from config.schema import BatchConfig


class BatchTab(QWidget):
    def __init__(self, config: BatchConfig) -> None:
        super().__init__()
        self.seeds = QLineEdit()
        self.presets = QLineEdit()
        self.line_count_sweep = QLineEdit()
        self.export_aggregated_metrics = QCheckBox("export aggregated metrics")

        form = QFormLayout(self)
        form.addRow("multiple seeds", self.seeds)
        form.addRow("multiple presets", self.presets)
        form.addRow("line-count sweeps", self.line_count_sweep)
        form.addRow("export aggregated metrics", self.export_aggregated_metrics)
        self.load_from_config(config)

    def load_from_config(self, cfg: BatchConfig) -> None:
        self.seeds.setText(",".join(str(x) for x in cfg.seeds))
        self.presets.setText(",".join(cfg.presets))
        self.line_count_sweep.setText(",".join(str(x) for x in cfg.line_count_sweep))
        self.export_aggregated_metrics.setChecked(cfg.export_aggregated_metrics)
