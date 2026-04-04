"""
Phase 5 — Tests: Rule-Based Detector
Tests threshold calibration, per-rule trigger logic, and metrics correctness.
Uses synthetic DataFrames — no real dataset required.
"""

import pytest
import numpy as np
import pandas as pd
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.models.rule_based import (
    RuleBasedDetector,
    RuleThresholds,
    fit_thresholds,
    _mad_threshold,
    _apply_rules,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_normal_df(n: int = 300) -> pd.DataFrame:
    """Clean normal data for fitting thresholds."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp":        pd.date_range("2020-05-15", periods=n, freq="30min"),
        "device_id":        ["P1"] * n,
        "mppt_w":           rng.uniform(0, 4000, n),
        "volt_v":           rng.uniform(23, 26, n),
        "curr_a":           rng.uniform(0, 160, n),
        "batt_pct":         rng.uniform(30, 95, n),
        "irradiance":       rng.uniform(100, 900, n),
        "temp_c":           rng.uniform(30, 60, n),
        "mppt_w_delta":     rng.uniform(-200, 200, n),
        "volt_v_delta":     rng.uniform(-0.3, 0.3, n),
        "batt_delta":       rng.uniform(-1.0, 1.5, n),
        "physics_residual": rng.uniform(0, 0.5, n),
        "is_anomaly":       [False] * n,
        "anomaly_type":     ["normal"] * n,
        "is_weather_event": [False] * n,
    })


def _make_test_df(normal_df: pd.DataFrame) -> pd.DataFrame:
    """Add known anomaly rows to test detection."""
    df = normal_df.copy()
    # R1: negative power
    df.loc[10, "mppt_w"] = -50.0
    df.loc[10, "is_anomaly"] = True
    df.loc[10, "anomaly_type"] = "sudden_drop"

    # R3: rapid battery drain
    df.loc[20, "batt_delta"] = -15.0
    df.loc[20, "is_anomaly"] = True
    df.loc[20, "anomaly_type"] = "battery_fault"

    # R6: physics residual spike (FDI)
    df.loc[30, "physics_residual"] = 5000.0
    df.loc[30, "is_anomaly"] = True
    df.loc[30, "anomaly_type"] = "false_data_injection"

    return df


# ─── MAD Threshold ────────────────────────────────────────────────────────────

class TestMADThreshold:
    def test_returns_float(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(_mad_threshold(arr), float)

    def test_higher_k_gives_higher_threshold(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _mad_threshold(arr, k=5.0) > _mad_threshold(arr, k=2.0)

    def test_constant_series_gives_median(self):
        """If all values are equal, MAD=0 → threshold = median."""
        arr = np.full(20, 42.0)
        assert _mad_threshold(arr, k=3.5) == pytest.approx(42.0)

    def test_robust_to_outlier(self):
        """One massive outlier should not blow up the threshold."""
        arr = np.concatenate([np.ones(99), [1e9]])
        threshold = _mad_threshold(arr[:-1], k=3.5)  # without outlier
        threshold_with = _mad_threshold(arr, k=3.5)  # with outlier
        # The threshold itself shouldn't change much when outlier is excluded
        assert threshold_with < 1e6   # should not be astronomically large


# ─── Threshold Calibration ────────────────────────────────────────────────────

class TestFitThresholds:
    def test_returns_rulethresholds(self):
        df = _make_normal_df()
        t = fit_thresholds(df)
        assert isinstance(t, RuleThresholds)

    def test_r2_power_delta_positive(self):
        df = _make_normal_df()
        t = fit_thresholds(df)
        assert t.r2_power_delta > 0

    def test_r6_physics_residual_positive(self):
        df = _make_normal_df()
        t = fit_thresholds(df)
        assert t.r6_physics_residual > 0

    def test_uses_only_normal_rows(self):
        """Thresholds fitted on mixed data should differ from normal-only."""
        df_normal = _make_normal_df(200)
        df_mixed = df_normal.copy()
        # Add extreme anomaly rows
        df_mixed.loc[0:10, "mppt_w_delta"] = 5000.0
        df_mixed.loc[0:10, "anomaly_type"] = "sudden_drop"
        df_mixed.loc[0:10, "is_anomaly"]   = True

        t_normal = fit_thresholds(df_normal)
        t_mixed  = fit_thresholds(df_mixed)
        # Both should give similar thresholds because anomaly rows are excluded
        assert abs(t_normal.r2_power_delta - t_mixed.r2_power_delta) < 50.0


# ─── Rule Detection ──────────────────────────────────────────────────────────

class TestRuleDetection:
    def _fitted_det(self) -> tuple[RuleBasedDetector, pd.DataFrame]:
        train = _make_normal_df(300)
        det = RuleBasedDetector(k=3.5)
        det.fit(train)
        return det, train

    def test_r1_negative_power(self):
        det, train = self._fitted_det()
        df = train.copy()
        df.loc[5, "mppt_w"] = -100.0
        preds = det.predict(df)
        assert preds[5] == 1, "Negative power not detected by R1"

    def test_r3_rapid_drain_detected(self):
        det, train = self._fitted_det()
        df = train.copy()
        df.loc[5, "batt_delta"] = -20.0   # extreme drain
        preds = det.predict(df)
        assert preds[5] == 1, "Rapid battery drain not detected by R3"

    def test_r6_fdi_detected(self):
        det, train = self._fitted_det()
        df = train.copy()
        df.loc[5, "physics_residual"] = 99999.0  # extreme FDI
        preds = det.predict(df)
        assert preds[5] == 1, "FDI physics residual not detected by R6"

    def test_normal_data_low_flag_rate(self):
        """Clean normal data should have a low false-positive rate (<20%)."""
        det, train = self._fitted_det()
        preds = det.predict(train)
        fpr = preds.mean()
        assert fpr < 0.20, f"False positive rate too high on train: {fpr:.2%}"

    def test_output_is_binary(self):
        det, train = self._fitted_det()
        preds = det.predict(train)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_output_length_matches_input(self):
        det, train = self._fitted_det()
        preds = det.predict(train)
        assert len(preds) == len(train)


# ─── Metrics ─────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_evaluate_returns_required_keys(self):
        train = _make_normal_df(300)
        test  = _make_test_df(_make_normal_df(200))
        det = RuleBasedDetector(k=3.5)
        det.fit(train)
        m = det.evaluate(test, split_name="test")
        for key in ("f1", "precision", "recall", "fpr_a6", "fpr_global",
                    "n_predicted", "n_true", "method"):
            assert key in m, f"Missing key: {key}"

    def test_f1_between_0_and_1(self):
        train = _make_normal_df(300)
        test  = _make_test_df(_make_normal_df(200))
        det = RuleBasedDetector(k=3.5)
        det.fit(train)
        m = det.evaluate(test)
        assert 0.0 <= m["f1"] <= 1.0

    def test_fpr_a6_zero_if_no_a6_rows(self):
        """If no A6 rows exist, FPR@A6 should be 0."""
        train = _make_normal_df(200)
        test  = _make_normal_df(100)
        det = RuleBasedDetector(k=3.5)
        det.fit(train)
        m = det.evaluate(test)
        assert m["fpr_a6"] == pytest.approx(0.0)

    def test_known_anomaly_detected(self):
        """Manually injected R3 anomaly should raise recall above 0."""
        train = _make_normal_df(300)
        test  = _make_test_df(_make_normal_df(200))
        det = RuleBasedDetector(k=3.5)
        det.fit(train)
        m = det.evaluate(test)
        assert m["recall"] > 0.0, "Known anomalies not detected"


# ─── End-to-end: fit + predict ────────────────────────────────────────────────

class TestEndToEnd:
    def test_fit_predict_works(self):
        train = _make_normal_df(300)
        test  = _make_test_df(_make_normal_df(200))
        det = RuleBasedDetector(k=3.5)
        det.fit(train)
        preds = det.predict(test)
        assert len(preds) == len(test)

    def test_not_fitted_raises(self):
        det = RuleBasedDetector()
        with pytest.raises(RuntimeError):
            det.predict(_make_normal_df(10))

    def test_proba_proxy_range(self):
        train = _make_normal_df(300)
        det = RuleBasedDetector(k=3.5)
        det.fit(train)
        scores = det.predict_proba_proxy(train)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0
