"""
Phase 12 — Tests: Visualization Plots
Smoke tests verifying that each plot function runs without error
when the expected input files exist.
"""

import pytest
import numpy as np
import pandas as pd
import pathlib
import pickle
import tempfile

import src.config as config


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_test_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_anom = 20
    n_norm = n - n_anom
    types = ["normal"] * n_norm + ["sudden_drop"] * 10 + ["false_data_injection"] * 10
    is_anom = [False] * n_norm + [True] * n_anom
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-05-15", periods=n, freq="30min"),
        "device_id": ["P1"] * n,
        "protocol": ["lora"] * n,
        "mppt_w": rng.uniform(0, 4000, n),
        "volt_v": rng.uniform(23, 26, n),
        "curr_a": rng.uniform(0, 160, n),
        "batt_pct": rng.uniform(30, 95, n),
        "irradiance": rng.uniform(100, 900, n),
        "temp_c": rng.uniform(30, 60, n),
        "physics_residual": rng.uniform(0, 0.5, n),
        "is_anomaly": is_anom,
        "anomaly_type": types,
        "is_weather_event": [False] * n,
        "is_low_irradiance_period": [0] * n,
    })


# ─── Smoke Tests ─────────────────────────────────────────────────────────────

class TestPlotsSmoke:
    """
    Verify each plot function executes without crashing.
    Uses temporary directories patched into config paths.
    """

    @pytest.fixture(autouse=True)
    def _setup_dirs(self, tmp_path, monkeypatch):
        self.splits_dir = tmp_path / "splits"
        self.results_dir = tmp_path / "results"
        self.fig_dir = tmp_path / "figures"
        self.splits_dir.mkdir()
        self.results_dir.mkdir()
        self.fig_dir.mkdir()

        monkeypatch.setattr(config, "SPLITS_DIR", self.splits_dir)
        monkeypatch.setattr(config, "RESULTS_DIR", self.results_dir)
        monkeypatch.setattr(config, "FIGURES_DIR", self.fig_dir)

    def _write_test_split(self):
        df = _make_test_df(200)
        df.to_csv(self.splits_dir / "test.csv", index=False)

    def test_plot_figure_3_f1(self, monkeypatch):
        """F1 bar chart reads Table_1 CSV — verify it runs."""
        from src.visualization.plots import plot_figure_3_f1

        t1 = pd.DataFrame({
            "Method": ["Rule-Based", "LSTM-AE"],
            "sudden_drop": [0.8, 0.9],
            "battery_fault": [0.7, 0.85],
            "normal": [0.95, 0.99],
            "low_irradiance": [0.9, 0.95],
        })
        t1.to_csv(self.results_dir / "Table_1_F1_Per_Class.csv", index=False)
        monkeypatch.setattr(config, "FIGURES_DIR", self.fig_dir)

        # Re-import to pick up monkeypatched paths
        import src.visualization.plots as plots_mod
        monkeypatch.setattr(plots_mod, "FIG_DIR", self.fig_dir)
        monkeypatch.setattr(plots_mod, "RESULTS_DIR", self.results_dir)

        plot_figure_3_f1()
        assert (self.fig_dir / "Figure_3_F1_BarChart.pdf").exists()

    def test_plot_figure_5_heatmap(self, monkeypatch):
        """Reconstruction heatmap reads errors CSV."""
        from src.visualization.plots import plot_figure_5_heatmap

        n = 500
        err_data = {"timestamp": pd.date_range("2020-05-15", periods=n, freq="30min"),
                     "is_anomaly": [False] * n,
                     "anomaly_type": ["normal"] * n}
        for feat in ["mppt_w", "volt_v", "curr_a"]:
            err_data[feat] = np.random.rand(n)
        pd.DataFrame(err_data).to_csv(
            self.results_dir / "lstmae_reconstruction_errors_test.csv", index=False
        )

        import src.visualization.plots as plots_mod
        monkeypatch.setattr(plots_mod, "FIG_DIR", self.fig_dir)
        monkeypatch.setattr(plots_mod, "RESULTS_DIR", self.results_dir)

        plot_figure_5_heatmap()
        assert (self.fig_dir / "Figure_5_Recon_Heatmap.pdf").exists()
