"""City generation controls tab."""

from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QLineEdit, QSpinBox, QWidget

from config.schema import CityConfig


class CityTab(QWidget):
    def __init__(self, config: CityConfig) -> None:
        super().__init__()
        self.total_population = QSpinBox(); self.total_population.setRange(1, 50_000_000)
        self.total_jobs = QSpinBox(); self.total_jobs.setRange(1, 50_000_000)
        self.tile_size = QSpinBox(); self.tile_size.setRange(16, 1024)
        self.employment_preset = QLineEdit()
        self.population_preset = QLineEdit()
        self.halo_strength = QDoubleSpinBox(); self.halo_strength.setRange(0.0, 2.0); self.halo_strength.setSingleStep(0.05)
        self.center_shares = QLineEdit()
        self.center_sizes = QLineEdit()
        self.jobs_housing_interaction = QDoubleSpinBox(); self.jobs_housing_interaction.setRange(0.0, 1.0); self.jobs_housing_interaction.setSingleStep(0.05)
        self.residential_cluster_count = QSpinBox(); self.residential_cluster_count.setRange(8, 12)
        self.random_seed = QSpinBox(); self.random_seed.setRange(0, 1_000_000_000)

        form = QFormLayout(self)
        form.addRow("total population", self.total_population)
        form.addRow("total jobs", self.total_jobs)
        form.addRow("tile size", self.tile_size)
        form.addRow("employment preset", self.employment_preset)
        form.addRow("population preset", self.population_preset)
        form.addRow("halo strength", self.halo_strength)
        form.addRow("center shares (csv)", self.center_shares)
        form.addRow("center sizes (csv)", self.center_sizes)
        form.addRow("jobs-housing interaction", self.jobs_housing_interaction)
        form.addRow("residential cluster count", self.residential_cluster_count)
        form.addRow("random seed", self.random_seed)
        self.load_from_config(config)

    def load_from_config(self, cfg: CityConfig) -> None:
        self.total_population.setValue(cfg.total_population)
        self.total_jobs.setValue(cfg.total_jobs)
        self.tile_size.setValue(cfg.tile_size)
        self.employment_preset.setText(cfg.employment_preset)
        self.population_preset.setText(cfg.population_preset)
        self.halo_strength.setValue(cfg.halo_strength)
        self.center_shares.setText(",".join(f"{x:.4f}" for x in cfg.center_shares))
        self.center_sizes.setText(",".join(str(x) for x in cfg.center_sizes))
        self.jobs_housing_interaction.setValue(cfg.jobs_housing_interaction)
        self.residential_cluster_count.setValue(cfg.residential_cluster_count)
        self.random_seed.setValue(cfg.random_seed)
