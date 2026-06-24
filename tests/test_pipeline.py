"""
Integration Test — End-to-End Pipeline Smoke Test
Runs a lightweight version of the full SCIO-Bench pipeline on synthetic data
to verify module interfaces and prevent regression.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import _handle_nan_inf, _rename_and_derive
from src.data.anomaly_injection import inject_all_anomalies
from src.evaluation.metrics import compute_detection_metrics


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_plant_df() -> pd.DataFrame:
    """Create a minimal 200-row synthetic plant DataFrame (post-augmentation)."""
    n = 200
    rng = np.random.default_rng(42)
    mppt = rng.uniform(0, 500, n)
    # Derive volt/curr/batt as augmentation.py would
    soc = np.clip(
        70.0 + np.cumsum(rng.uniform(-2, 3, n)),
        5.0, 100.0
    )
    volt = 24.0 + (soc - 50) * 0.1  # rough LiFePO4 curve
    curr = np.where(volt > 0, mppt / volt, 0.0)
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-05-15", periods=n, freq="30min"),
        "device_id": ["SIM_PLANT1_INV01"] * n,
        "mppt_w": mppt,
        "prod_wh": mppt * 0.5,
        "temp_c": rng.uniform(25, 55, n),
        "irradiance": rng.uniform(0, 1200, n),
        "ambient_temp_c": rng.uniform(20, 40, n),
        "daily_yield_kwh": rng.uniform(0, 50, n),
        "ac_power_kw": rng.uniform(0, 0.5, n),
        # Post-augmentation columns required by injectors
        "batt_pct": soc,
        "volt_v": volt,
        "curr_a": curr,
        "rssi": rng.normal(-70, 15, n),
        "protocol": ["lora"] * n,
    })


# ─── Pipeline Tests ───────────────────────────────────────────────────────────

class TestPreprocessIntegration:
    def test_handle_nan_inf_no_crash(self, synthetic_plant_df):
        df = synthetic_plant_df.copy()
        # Introduce some NaN
        df.loc[5:7, "mppt_w"] = np.nan
        result = _handle_nan_inf(df)
        assert not result[[c for c in result.columns if c != "timestamp"]].isnull().values.any()

    def test_rename_and_derive(self, synthetic_plant_df):
        df = synthetic_plant_df.copy()
        # Add a fake mppt_kw column to test rename
        df["mppt_kw"] = df["mppt_w"] / 1000.0
        df = df.drop(columns=["mppt_w"])
        result = _rename_and_derive(df, plant_id=1)
        assert "mppt_w" in result.columns
        assert "mppt_kw" not in result.columns
        assert "prod_wh" in result.columns


class TestAnomalyInjectionIntegration:
    def test_inject_all_anomalies(self, synthetic_plant_df):
        df1 = synthetic_plant_df.copy()
        df2 = synthetic_plant_df.copy()
        df2["device_id"] = "SIM_PLANT2_INV01"

        combined = inject_all_anomalies(df1, df2, random_seed=42)
        assert "is_anomaly" in combined.columns
        assert "anomaly_type" in combined.columns
        assert "is_weather_event" in combined.columns
        # Check proportions roughly match targets
        # Note: on small synthetic data (~200 rows), segment-based injection
        # may overshoot target proportions; this is expected.
        anomaly_rate = combined["is_anomaly"].mean()
        assert 0.01 < anomaly_rate < 0.30, (
            f"Anomaly rate {anomaly_rate:.3f} way off expected range on small data"
        )
        # Check injection order prevented overlap
        assert combined["anomaly_type"].notna().all()

    def test_a6_is_not_anomaly(self, synthetic_plant_df):
        df1 = synthetic_plant_df.copy()
        df2 = synthetic_plant_df.copy()
        df2["device_id"] = "SIM_PLANT2_INV01"
        combined = inject_all_anomalies(df1, df2, random_seed=42)
        a6_mask = combined["anomaly_type"] == "low_irradiance"
        if a6_mask.any():
            assert not combined.loc[a6_mask, "is_anomaly"].any()


class TestMetricsIntegration:
    def test_compute_metrics_basic(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        metrics = compute_detection_metrics(
            y_true, y_pred, method="test", split_name="test"
        )
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert metrics["n_predicted"] == 3
        assert metrics["n_true"] == 3

    def test_compute_metrics_with_scores(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_scores = np.array([0.1, 0.6, 0.8, 0.3])
        metrics = compute_detection_metrics(
            y_true, y_pred, y_scores=y_scores, method="test", split_name="test"
        )
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == pytest.approx(0.75, abs=0.01)

    def test_compute_metrics_with_weather(self):
        n = 100
        y_true = np.zeros(n, dtype=int)
        y_pred = np.zeros(n, dtype=int)
        y_pred[::10] = 1  # 10% flagged
        df = pd.DataFrame({
            "anomaly_type": ["normal"] * 50 + ["low_irradiance"] * 30 + ["sudden_drop"] * 20,
            "is_anomaly": [False] * 50 + [False] * 30 + [True] * 20,
        })
        metrics = compute_detection_metrics(
            y_true, y_pred, df=df, method="test", split_name="test"
        )
        assert "fpr_a6" in metrics
        assert "fpr_global" in metrics
        assert "f1_sudden_drop" in metrics
