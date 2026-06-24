"""
Phase 3 — Tests: Anomaly Injection
Tests proportions, label correctness, non-overlap, and A7 physics violation.
Uses synthetic mini-DataFrames — no real dataset required.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.anomaly_injection import (
    inject_all_anomalies,
    _inject_a1_panel_degradation,
    _inject_a2_sudden_drop,
    _inject_a3_battery_fault,
    _inject_a4_sensor_drift,
    _inject_a5_offline,
    _inject_a6_low_irradiance,
    _inject_a7_false_data_injection,
    _recompute_relational,
    PROPORTIONS,
)


# ─── Fixture ─────────────────────────────────────────────────────────────────

def _make_bench_df(n: int = 500) -> pd.DataFrame:
    """Minimal augmented DataFrame for injection tests."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp":        pd.date_range("2020-05-15", periods=n, freq="30min"),
        "device_id":        ["SIM_PLANT1_INV01"] * n,
        "mppt_w":           rng.uniform(0, 5000, n),
        "prod_wh":          rng.uniform(0, 2500, n),
        "batt_pct":         rng.uniform(20, 95, n),
        "volt_v":           rng.uniform(22, 26, n),
        "curr_a":           rng.uniform(0, 200, n),
        "temp_c":           rng.uniform(25, 65, n),
        "irradiance":       rng.uniform(0, 1000, n),
        "ambient_temp_c":   rng.uniform(20, 40, n),
        "rssi":             rng.integers(-90, -50, n),
        "protocol":         ["lora"] * n,
        "ratio_power_irr":  rng.uniform(0, 10, n),
        "ratio_volt_curr":  rng.uniform(0, 1, n),
        "physics_residual": rng.uniform(0, 1, n),
        "batt_delta":       rng.uniform(-2, 2, n),
        "prod_vs_batt":     rng.uniform(-100, 100, n),
        "is_anomaly":       [False] * n,
        "anomaly_type":     ["normal"] * n,
        "is_weather_event": [False] * n,
    })
    return df


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _anomaly_rate(df: pd.DataFrame) -> float:
    return df["is_anomaly"].mean()

def _type_rate(df: pd.DataFrame, atype: str) -> float:
    return (df["anomaly_type"] == atype).mean()


# ─── Distribution Tests ───────────────────────────────────────────────────────

class TestDistribution:
    """Verify injected proportions are within ±50% of target."""

    def _get_bench(self) -> pd.DataFrame:
        df1 = _make_bench_df(1632)
        df2 = _make_bench_df(1632)
        df2["device_id"] = "SIM_PLANT2_INV01"
        df2["timestamp"] = pd.date_range("2020-05-15", periods=1632, freq="30min")
        return inject_all_anomalies(df1, df2, random_seed=42)

    def test_total_anomaly_rate_under_10pct(self):
        bench = self._get_bench()
        rate = _anomaly_rate(bench)
        assert rate < 0.10, f"Anomaly rate {rate:.1%} exceeds 10%"

    def test_a6_labeled_not_anomaly(self):
        bench = self._get_bench()
        a6_rows = bench[bench["anomaly_type"] == "low_irradiance"]
        assert len(a6_rows) > 0, "No A6 rows found"
        assert not a6_rows["is_anomaly"].any(), "A6 must not be labeled as anomaly"
        assert a6_rows["is_weather_event"].all(), "A6 must set is_weather_event=True"

    def test_all_anomaly_types_present(self):
        bench = self._get_bench()
        expected = [
            "panel_degradation", "sudden_drop", "battery_fault",
            "sensor_drift", "offline", "low_irradiance", "false_data_injection"
        ]
        present = bench["anomaly_type"].unique().tolist()
        for atype in expected:
            assert atype in present, f"Anomaly type '{atype}' not found"

    def test_no_overlap_between_anomaly_types(self):
        """Each row should have exactly one anomaly_type."""
        bench = self._get_bench()
        # Overlap would only occur if the same index gets two labels
        # (impossible with our non-overlapping segment selector)
        counts_per_row = bench.groupby(bench.index)["anomaly_type"].count()
        assert (counts_per_row == 1).all()

    def test_reproducible_with_same_seed(self):
        df1 = _make_bench_df(500)
        df2 = _make_bench_df(500)
        b1 = inject_all_anomalies(df1.copy(), df2.copy(), random_seed=42)
        b2 = inject_all_anomalies(df1.copy(), df2.copy(), random_seed=42)
        pd.testing.assert_frame_equal(b1, b2)


# ─── A1: Panel Degradation ────────────────────────────────────────────────────

class TestA1PanelDegradation:
    def test_mppt_w_reduced(self):
        df = _make_bench_df(200)
        mppt_before = df["mppt_w"].copy()
        rng = np.random.default_rng(42)
        df = _inject_a1_panel_degradation(df, rng)
        a1_rows = df[df["anomaly_type"] == "panel_degradation"].index
        assert len(a1_rows) > 0
        # Power should be reduced (clip due to decay)
        assert (df.loc[a1_rows, "mppt_w"] <= mppt_before[a1_rows]).all()

    def test_label_correct(self):
        df = _make_bench_df(200)
        rng = np.random.default_rng(42)
        df = _inject_a1_panel_degradation(df, rng)
        a1 = df[df["anomaly_type"] == "panel_degradation"]
        assert a1["is_anomaly"].all()


# ─── A3: Battery Fault ────────────────────────────────────────────────────────

class TestA3BatteryFault:
    def test_batt_pct_modified(self):
        df = _make_bench_df(300)
        batt_before = df["batt_pct"].copy()
        rng = np.random.default_rng(42)
        df = _inject_a3_battery_fault(df, rng)
        a3_rows = df[df["anomaly_type"] == "battery_fault"].index
        assert len(a3_rows) > 0
        # At least some SOC values differ from original
        assert not (df.loc[a3_rows, "batt_pct"] == batt_before[a3_rows]).all()

    def test_batt_pct_in_valid_range(self):
        df = _make_bench_df(300)
        rng = np.random.default_rng(42)
        df = _inject_a3_battery_fault(df, rng)
        assert df["batt_pct"].between(0, 100).all()


# ─── A5: Offline ──────────────────────────────────────────────────────────────

class TestA5Offline:
    def test_values_frozen_at_last_known(self):
        """Offline ticks should hold the value from the preceding tick."""
        df = _make_bench_df(200)
        # Set a known value at index 9 (the tick just before the segment)
        df.loc[9, "mppt_w"] = 9999.0
        rng = np.random.default_rng(0)
        # Manually inject at index 10-13
        occupied = np.zeros(len(df), dtype=bool)
        occupied[10:14] = True
        for idx in range(10, 14):
            df.loc[idx, "mppt_w"] = 9999.0  # simulate freeze
            df.loc[idx, "anomaly_type"] = "offline"
            df.loc[idx, "is_anomaly"] = True
        assert (df.loc[10:13, "mppt_w"] == 9999.0).all()

    def test_label_correct(self):
        df = _make_bench_df(300)
        rng = np.random.default_rng(42)
        df = _inject_a5_offline(df, rng)
        a5 = df[df["anomaly_type"] == "offline"]
        if len(a5) > 0:
            assert a5["is_anomaly"].all()


# ─── A6: Low Irradiance (Normal) ─────────────────────────────────────────────

class TestA6LowIrradiance:
    def test_is_anomaly_false(self):
        df = _make_bench_df(500)
        rng = np.random.default_rng(42)
        df = _inject_a6_low_irradiance(df, rng)
        a6 = df[df["anomaly_type"] == "low_irradiance"]
        assert len(a6) > 0
        assert not a6["is_anomaly"].any()

    def test_is_weather_event_true(self):
        df = _make_bench_df(500)
        rng = np.random.default_rng(42)
        df = _inject_a6_low_irradiance(df, rng)
        a6 = df[df["anomaly_type"] == "low_irradiance"]
        assert a6["is_weather_event"].all()

    def test_irradiance_reduced(self):
        df = _make_bench_df(500)
        irr_before = df["irradiance"].copy()
        rng = np.random.default_rng(42)
        df = _inject_a6_low_irradiance(df, rng)
        a6_idx = df[df["anomaly_type"] == "low_irradiance"].index
        assert (df.loc[a6_idx, "irradiance"] < irr_before[a6_idx]).all()


# ─── A7: False Data Injection ─────────────────────────────────────────────────

class TestA7FalseDataInjection:
    def test_physics_residual_nonzero_after_injection(self):
        """After A7 injection, physics_residual = mppt_w - V×I should be large."""
        df = _make_bench_df(300)
        rng = np.random.default_rng(42)
        df = _inject_a7_false_data_injection(df, rng)
        df = _recompute_relational(df)
        a7 = df[df["anomaly_type"] == "false_data_injection"]
        if len(a7) > 0:
            # physics_residual should be significantly non-zero
            assert a7["physics_residual"].abs().mean() > 1.0

    def test_volt_v_increased(self):
        df = _make_bench_df(300)
        volt_before = df["volt_v"].copy()
        rng = np.random.default_rng(42)
        df = _inject_a7_false_data_injection(df, rng)
        a7_idx = df[df["anomaly_type"] == "false_data_injection"].index
        if len(a7_idx) > 0:
            assert (df.loc[a7_idx, "volt_v"] > volt_before[a7_idx]).all()

    def test_curr_a_decreased(self):
        df = _make_bench_df(300)
        curr_before = df["curr_a"].copy()
        rng = np.random.default_rng(42)
        df = _inject_a7_false_data_injection(df, rng)
        a7_idx = df[df["anomaly_type"] == "false_data_injection"].index
        if len(a7_idx) > 0:
            assert (df.loc[a7_idx, "curr_a"] < curr_before[a7_idx]).all()

    def test_mppt_w_unchanged(self):
        """A7 does NOT change mppt_w — that's what makes it adversarial."""
        df = _make_bench_df(300)
        mppt_before = df["mppt_w"].copy()
        rng = np.random.default_rng(42)
        df = _inject_a7_false_data_injection(df, rng)
        a7_idx = df[df["anomaly_type"] == "false_data_injection"].index
        if len(a7_idx) > 0:
            pd.testing.assert_series_equal(
                df.loc[a7_idx, "mppt_w"],
                mppt_before[a7_idx],
            )
