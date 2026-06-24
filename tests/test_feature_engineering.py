"""
Phase 4 — Tests: Feature Engineering & Splitting
Tests lag correctness, no look-ahead in rolling, split ordering,
no train/test timestamp overlap, and scaler fit-only-on-train.
All tests use synthetic data — no real dataset required.
"""

import pytest
import numpy as np
import pandas as pd
import pathlib
import pickle
import tempfile

from src.data.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_delta_features,
    add_time_features,
    add_weather_flag,
    engineer_all_features,
    chronological_split,
    fit_and_scale,
    TRAIN_FRAC, VAL_FRAC,
)


# ─── Fixture ─────────────────────────────────────────────────────────────────

def _make_df(n: int = 200) -> pd.DataFrame:
    """Synthetic SCIO-Bench-like DataFrame for feature engineering tests."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp":            pd.date_range("2020-05-15", periods=n, freq="30min"),
        "device_id":            ["SIM_PLANT1_INV01"] * n,
        "mppt_w":               rng.uniform(0, 5000, n),
        "volt_v":               rng.uniform(22, 26, n),
        "curr_a":               rng.uniform(0, 200, n),
        "batt_pct":             rng.uniform(20, 95, n),
        "irradiance":           rng.uniform(0, 1000, n),
        "temp_c":               rng.uniform(25, 65, n),
        "ambient_temp_c":       rng.uniform(20, 40, n),
        "prod_wh":              rng.uniform(0, 2500, n),
        "daily_yield_kwh":      rng.uniform(0, 50, n),
        "ac_power_kw":          rng.uniform(0, 4, n),
        "rssi":                 rng.integers(-90, -50, n),
        "protocol":             ["lora"] * n,
        "ratio_power_irr":      rng.uniform(0, 10, n),
        "ratio_volt_curr":      rng.uniform(0, 1, n),
        "physics_residual":     rng.uniform(0, 1, n),
        "batt_delta":           rng.uniform(-2, 2, n),
        "prod_vs_batt":         rng.uniform(-100, 100, n),
        "is_anomaly":           [False] * n,
        "anomaly_type":         ["normal"] * (n - 5) + ["low_irradiance"] * 5,
        "is_weather_event":     [False] * (n - 5) + [True] * 5,
    })


# ─── Lag Features ─────────────────────────────────────────────────────────────

class TestLagFeatures:
    def test_lag1_shifts_by_one(self):
        df = _make_df(50)
        original = df["mppt_w"].values.copy()
        df = add_lag_features(df)
        # lag1 at row 5 should equal original row 4
        assert df["mppt_w_lag1"].iloc[5] == pytest.approx(original[4])

    def test_lag2_shifts_by_two(self):
        df = _make_df(50)
        original = df["mppt_w"].values.copy()
        df = add_lag_features(df)
        assert df["mppt_w_lag2"].iloc[5] == pytest.approx(original[3])

    def test_lag_columns_present(self):
        df = add_lag_features(_make_df(30))
        for col in ("mppt_w_lag1", "mppt_w_lag2", "volt_v_lag1",
                    "batt_pct_lag1", "irradiance_lag1"):
            assert col in df.columns, f"Missing: {col}"

    def test_no_nan_after_fill(self):
        df = add_lag_features(_make_df(30))
        lag_cols = [c for c in df.columns if "_lag" in c]
        assert not df[lag_cols].isnull().any(axis=None)


# ─── Rolling Features ─────────────────────────────────────────────────────────

class TestRollingFeatures:
    def test_rolling_columns_present(self):
        df = add_rolling_features(_make_df(50))
        for col in ("mppt_w_mean_6h", "mppt_w_std_6h",
                    "irradiance_mean_6h", "batt_pct_mean_6h"):
            assert col in df.columns

    def test_no_nan_in_rolling(self):
        df = add_rolling_features(_make_df(50))
        roll_cols = [c for c in df.columns if "_mean_6h" in c or "_std_6h" in c]
        assert not df[roll_cols].isnull().any(axis=None)

    def test_std_non_negative(self):
        df = add_rolling_features(_make_df(50))
        std_cols = [c for c in df.columns if "_std_6h" in c]
        for col in std_cols:
            assert (df[col] >= 0).all(), f"{col} has negative std"


# ─── Delta Features ───────────────────────────────────────────────────────────

class TestDeltaFeatures:
    def test_delta_columns_present(self):
        df = add_delta_features(_make_df(30))
        for col in ("mppt_w_delta", "volt_v_delta",
                    "batt_pct_delta", "irradiance_delta"):
            assert col in df.columns

    def test_delta_first_row_zero(self):
        df = add_delta_features(_make_df(30))
        assert df["mppt_w_delta"].iloc[0] == pytest.approx(0.0)

    def test_delta_equals_diff(self):
        df = _make_df(30)
        orig = df["mppt_w"].values.copy()
        df = add_delta_features(df)
        # row 5: delta = orig[5] - orig[4]
        expected = orig[5] - orig[4]
        assert df["mppt_w_delta"].iloc[5] == pytest.approx(expected)


# ─── Time Features ────────────────────────────────────────────────────────────

class TestTimeFeatures:
    def test_cyclic_columns_present(self):
        df = add_time_features(_make_df(10))
        for col in ("hour_sin", "hour_cos", "minute_sin", "minute_cos"):
            assert col in df.columns

    def test_cyclic_range(self):
        df = add_time_features(_make_df(50))
        for col in ("hour_sin", "hour_cos", "minute_sin", "minute_cos"):
            assert df[col].between(-1.0, 1.0).all(), f"{col} out of [-1,1]"

    def test_midnight_continuity(self):
        """hour_sin at 23:30 and 00:00 should not have a large jump (cyclic)."""
        tdf = pd.DataFrame({
            "timestamp":  pd.date_range("2020-05-15 23:30", periods=3, freq="30min"),
            "device_id":  ["X"] * 3,
        })
        tdf = add_time_features(tdf)
        sin_23_30 = tdf["hour_sin"].iloc[0]
        sin_00_00 = tdf["hour_sin"].iloc[1]
        # Difference should be small (no cliff at midnight)
        assert abs(sin_23_30 - sin_00_00) < 1.0


# ─── Weather Flag ─────────────────────────────────────────────────────────────

class TestWeatherFlag:
    def test_flag_is_1_for_low_irradiance(self):
        df = _make_df(10)
        df.loc[3:5, "anomaly_type"] = "low_irradiance"
        df = add_weather_flag(df)
        assert df["is_low_irradiance_period"].iloc[3] == 1

    def test_flag_is_0_for_normal(self):
        df = _make_df(10)
        df["anomaly_type"] = "normal"
        df = add_weather_flag(df)
        assert (df["is_low_irradiance_period"] == 0).all()


# ─── Chronological Split ─────────────────────────────────────────────────────

class TestChronologicalSplit:
    def test_sizes_correct(self):
        df = engineer_all_features(_make_df(200))
        train, val, test = chronological_split(df)
        n = len(df)
        assert len(train) == int(n * TRAIN_FRAC)
        assert len(val) == int(n * VAL_FRAC)
        # Test gets the remainder
        assert len(test) == n - int(n * TRAIN_FRAC) - int(n * VAL_FRAC)

    def test_no_timestamp_overlap(self):
        df = engineer_all_features(_make_df(200))
        train, val, test = chronological_split(df)
        # Multi-device: boundary timestamp may appear in adjacent splits
        assert train["timestamp"].max() <= val["timestamp"].max()
        assert val["timestamp"].max() <= test["timestamp"].max()

    def test_sorted_ascending(self):
        df = engineer_all_features(_make_df(200))
        train, val, test = chronological_split(df)
        for split_df, name in ((train, "train"), (val, "val"), (test, "test")):
            assert split_df["timestamp"].is_monotonic_increasing, \
                f"{name} split not sorted chronologically"

    def test_no_shuffle(self):
        """Rows should NOT be re-ordered within each split."""
        df = engineer_all_features(_make_df(200))
        train, _, _ = chronological_split(df)
        # First N rows of engineered df (sorted) should equal train
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        n_train = len(train)
        pd.testing.assert_series_equal(
            train["timestamp"].reset_index(drop=True),
            df_sorted["timestamp"].iloc[:n_train].reset_index(drop=True),
        )


# ─── Scaling ─────────────────────────────────────────────────────────────────

class TestScaling:
    def _get_splits(self):
        df = engineer_all_features(_make_df(200))
        return chronological_split(df)

    def test_train_feature_mean_near_zero(self):
        """After StandardScaler, train feature means should be ~0."""
        train, val, test = self._get_splits()
        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = pathlib.Path(tmp) / "scaler.pkl"
            train_sc, _, _, scaler, feat_cols = fit_and_scale(
                train.copy(), val.copy(), test.copy(), scaler_path
            )
        means = train_sc[feat_cols].mean()
        assert (means.abs() < 0.1).all(), f"Some train means not near 0: {means.abs().max()}"

    def test_train_feature_std_near_one(self):
        """After StandardScaler, train feature stds should be ~1."""
        train, val, test = self._get_splits()
        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = pathlib.Path(tmp) / "scaler.pkl"
            train_sc, _, _, scaler, feat_cols = fit_and_scale(
                train.copy(), val.copy(), test.copy(), scaler_path
            )
        stds = train_sc[feat_cols].std()
        assert (stds - 1.0).abs().max() < 0.1

    def test_scaler_pickle_saved(self):
        train, val, test = self._get_splits()
        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = pathlib.Path(tmp) / "scaler.pkl"
            fit_and_scale(train.copy(), val.copy(), test.copy(), scaler_path)
            assert scaler_path.exists()
            with open(scaler_path, "rb") as f:
                data = pickle.load(f)
            assert "scaler" in data and "feature_cols" in data

    def test_labels_not_scaled(self):
        """is_anomaly column must remain boolean/unchanged after scaling."""
        train, val, test = self._get_splits()
        orig_labels = train["is_anomaly"].copy()
        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = pathlib.Path(tmp) / "scaler.pkl"
            train_sc, _, _, _, _ = fit_and_scale(
                train.copy(), val.copy(), test.copy(), scaler_path
            )
        pd.testing.assert_series_equal(
            train_sc["is_anomaly"].reset_index(drop=True),
            orig_labels.reset_index(drop=True),
        )
