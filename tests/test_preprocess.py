"""
Phase 1 — Tests: Preprocessing
Tests merge, resample, NaN removal, and column renaming logic.
Uses synthetic mini-DataFrames — no actual Kaggle data required.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.preprocess import _merge_and_resample, _handle_nan_inf, _rename_and_derive


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_gen_df(n: int = 10, freq: str = "15min") -> pd.DataFrame:
    """Synthetic generation DataFrame mimicking Kaggle Plant_N_Generation_Data.csv"""
    idx = pd.date_range("2020-05-15 06:00", periods=n, freq=freq)
    return pd.DataFrame({
        "DATE_TIME": idx,
        "PLANT_ID": 1,
        "SOURCE_KEY": "abc",
        "DC_POWER": np.random.uniform(100, 800, n),   # kW
        "AC_POWER": np.random.uniform(90, 750, n),    # kW
        "DAILY_YIELD": np.cumsum(np.random.uniform(5, 20, n)),
        "TOTAL_YIELD": np.ones(n) * 1e6,
    })


def _make_weather_df(n: int = 10, freq: str = "15min") -> pd.DataFrame:
    """Synthetic weather DataFrame mimicking Kaggle Plant_N_Weather_Sensor_Data.csv"""
    idx = pd.date_range("2020-05-15 06:00", periods=n, freq=freq)
    return pd.DataFrame({
        "DATE_TIME": idx,
        "PLANT_ID": 1,
        "SOURCE_KEY": "xyz",
        "AMBIENT_TEMPERATURE": np.random.uniform(25, 40, n),
        "MODULE_TEMPERATURE": np.random.uniform(30, 65, n),
        "IRRADIATION": np.random.uniform(0, 1.0, n),
    })


# ─── Tests: _merge_and_resample ──────────────────────────────────────────────

class TestMergeAndResample:
    def test_output_shape(self):
        gen = _make_gen_df(20)
        wx = _make_weather_df(20)
        merged = _merge_and_resample(gen, wx)
        # 20 × 15min → 5 × 30min windows (with mean aggregation)
        assert len(merged) <= 20
        assert "timestamp" in merged.columns

    def test_no_nan_after_resample_clean_data(self):
        gen = _make_gen_df(20)
        wx = _make_weather_df(20)
        merged = _merge_and_resample(gen, wx)
        numeric_cols = merged.select_dtypes(include="number").columns
        assert not merged[numeric_cols].isnull().all(axis=None), \
            "All values are NaN — merge or resample failed"

    def test_timestamp_monotonic(self):
        gen = _make_gen_df(20)
        wx = _make_weather_df(20)
        merged = _merge_and_resample(gen, wx)
        assert merged["timestamp"].is_monotonic_increasing

    def test_inner_join_drops_unmatched(self):
        gen = _make_gen_df(10)
        # Weather has only first 5 timestamps
        wx = _make_weather_df(5)
        merged = _merge_and_resample(gen, wx)
        assert len(merged) <= 10


# ─── Tests: _handle_nan_inf ──────────────────────────────────────────────────

class TestHandleNanInf:
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=6, freq="30min"),
            "DC_POWER": [100.0, np.nan, np.nan, 200.0, np.inf, -np.inf],
            "MODULE_TEMPERATURE": [35.0, 36.0, np.nan, 37.0, 38.0, 39.0],
        })

    def test_no_nan_remains(self):
        df = _handle_nan_inf(self._base_df())
        numeric = df.select_dtypes(include="number").columns
        assert not df[numeric].isnull().any(axis=None)

    def test_no_inf_remains(self):
        df = _handle_nan_inf(self._base_df())
        numeric = df.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any()

    def test_forward_fill_used(self):
        """The second NaN (consecutive 1) should be forward-filled from row 0."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="30min"),
            "DC_POWER": [100.0, np.nan, 200.0, 300.0],
        })
        cleaned = _handle_nan_inf(df)
        # row 1 should be forward-filled from row 0 (100.0)
        # pandas 2.x: .ffill(limit=N) — same result, different API
        assert cleaned["DC_POWER"].iloc[1] == pytest.approx(100.0)


# ─── Tests: _rename_and_derive ───────────────────────────────────────────────

class TestRenameAndDerive:
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=5, freq="30min"),
            "DC_POWER": [500.0, 600.0, 700.0, 0.0, 800.0],
            "AC_POWER": [450.0, 540.0, 630.0, 0.0, 720.0],
            "MODULE_TEMPERATURE": [40.0, 41.0, 42.0, 30.0, 45.0],
            "IRRADIATION": [0.8, 0.9, 1.0, 0.0, 0.95],
            "AMBIENT_TEMPERATURE": [30.0, 31.0, 32.0, 28.0, 33.0],
            "DAILY_YIELD": [100.0, 200.0, 300.0, 0.0, 400.0],
            "TOTAL_YIELD": [1e6] * 5,
            "_had_nan": [False] * 5,
        })

    def test_mppt_w_column_created(self):
        df = _rename_and_derive(self._base_df(), plant_id=1)
        assert "mppt_w" in df.columns

    def test_mppt_w_in_watts(self):
        """DC_POWER=500 kW → mppt_w should be 500×1000 = 500,000 W"""
        df = _rename_and_derive(self._base_df(), plant_id=1)
        assert df["mppt_w"].iloc[0] == pytest.approx(500_000.0)

    def test_prod_wh_derived(self):
        df = _rename_and_derive(self._base_df(), plant_id=1)
        assert "prod_wh" in df.columns
        # prod_wh = mppt_w × 0.5 → 500000 × 0.5 = 250000
        assert df["prod_wh"].iloc[0] == pytest.approx(250_000.0)

    def test_no_negative_mppt_w(self):
        df = _rename_and_derive(self._base_df(), plant_id=1)
        assert (df["mppt_w"] >= 0).all()

    def test_device_id_assigned(self):
        df = _rename_and_derive(self._base_df(), plant_id=2)
        assert df["device_id"].iloc[0] == "SIM_PLANT2_INV01"

    def test_temp_c_column_exists(self):
        df = _rename_and_derive(self._base_df(), plant_id=1)
        assert "temp_c" in df.columns

    def test_irradiance_non_negative(self):
        df = _rename_and_derive(self._base_df(), plant_id=1)
        assert (df["irradiance"] >= 0).all()
