"""
Phase 8 — Tests: L2 Hierarchical Classifier
Tests MIR@k correctness, classifier fit/predict, and evaluate() schema.
Uses synthetic DataFrames — no real dataset or TF model required.
"""

import pytest
import numpy as np
import pandas as pd
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.models.l2_classifier import (
    mir_at_k,
    L2Classifier,
    L2_TARGET_CLASSES,
    _get_feature_cols,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic DataFrame with all 6 anomaly types + normal rows."""
    rng = np.random.default_rng(seed)
    n_classes   = 6
    per_class   = max(1, n // (n_classes + 4))   # scale to n
    n_anom      = per_class * n_classes
    n_norm      = max(0, n - n_anom)
    types = (
        ["normal"] * n_norm +
        ["panel_degradation"]   * per_class +
        ["sudden_drop"]         * per_class +
        ["battery_fault"]       * per_class +
        ["sensor_drift"]        * per_class +
        ["offline"]             * per_class +
        ["false_data_injection"]* per_class
    )
    n_actual = len(types)
    is_anom = [t != "normal" for t in types]
    return pd.DataFrame({
        "timestamp":   pd.date_range("2020-05-15", periods=n_actual, freq="30min"),
        "device_id":   ["P1"] * n_actual,
        "protocol":    ["lora"] * n_actual,
        "mppt_w":      rng.normal(0, 1, n_actual),
        "volt_v":      rng.normal(0, 1, n_actual),
        "curr_a":      rng.normal(0, 1, n_actual),
        "batt_pct":    rng.normal(0, 1, n_actual),
        "irradiance":  rng.normal(0, 1, n_actual),
        "batt_delta":  rng.normal(0, 0.1, n_actual),
        "mppt_w_delta":rng.normal(0, 0.5, n_actual),
        "hour_sin":    np.sin(np.linspace(0, 2 * np.pi, n_actual)),
        "physics_residual": rng.uniform(0, 0.5, n_actual),
        "is_anomaly":  is_anom,
        "anomaly_type":types,
        "is_weather_event": [False] * n_actual,
        "is_low_irradiance_period": [0] * n_actual,
    })



# ─── MIR@k ────────────────────────────────────────────────────────────────────

class TestMIRAtK:
    def test_perfect_ranking_and_typing(self):
        """If all anomalies are ranked first and typed correctly, MIR@k=1."""
        y_true  = np.array(["normal", "normal", "sudden_drop", "battery_fault"])
        y_pred  = np.array(["normal", "normal", "sudden_drop", "battery_fault"])
        scores  = np.array([0.1, 0.2, 0.9, 0.8])
        result  = mir_at_k(y_true, y_pred, scores, k=2)
        assert result == pytest.approx(1.0)

    def test_zero_budget_captures_nothing(self):
        """k=0 means no alarms investigated → MIR@0 = 0 (unless no anomalies)."""
        y_true = np.array(["normal", "sudden_drop", "battery_fault"])
        y_pred = np.array(["normal", "sudden_drop", "battery_fault"])
        scores = np.array([0.1, 0.9, 0.8])
        result = mir_at_k(y_true, y_pred, scores, k=0)
        assert result == pytest.approx(0.0)

    def test_no_anomalies_returns_one(self):
        """Edge case: all normal → trivially satisfied."""
        y_true = np.array(["normal", "normal"])
        y_pred = np.array(["normal", "normal"])
        scores = np.array([0.1, 0.2])
        assert mir_at_k(y_true, y_pred, scores, k=1) == pytest.approx(1.0)

    def test_wrong_type_reduces_mir(self):
        """Wrong typing for ranked anomaly reduces MIR@k below 1."""
        y_true = np.array(["sudden_drop", "battery_fault"])
        y_pred = np.array(["battery_fault", "battery_fault"])  # 1 correct
        scores = np.array([0.9, 0.8])
        result = mir_at_k(y_true, y_pred, scores, k=2)
        assert 0.0 < result < 1.0

    def test_low_ranked_anomaly_not_counted(self):
        """Anomaly ranked below budget k is NOT counted even if typed correctly."""
        y_true = np.array(["normal", "normal", "sudden_drop"])
        y_pred = np.array(["normal", "normal", "sudden_drop"])
        scores = np.array([0.9, 0.8, 0.1])  # anomaly ranked last
        result = mir_at_k(y_true, y_pred, scores, k=2)
        assert result == pytest.approx(0.0)


# ─── Feature Column Helper ───────────────────────────────────────────────────

class TestGetFeatureCols:
    def test_excludes_label_cols(self):
        df = _make_df(50)
        cols = _get_feature_cols(df)
        for c in ("is_anomaly", "anomaly_type", "timestamp", "device_id"):
            assert c not in cols

    def test_nonempty(self):
        assert len(_get_feature_cols(_make_df(50))) > 0


# ─── L2Classifier ────────────────────────────────────────────────────────────

class TestL2Classifier:
    def _fitted_clf(self, seed: int = 0) -> tuple[L2Classifier, pd.DataFrame]:
        df = _make_df(300, seed=seed)
        clf = L2Classifier(n_estimators=20, use_smote=False, random_state=42)
        clf.fit(df)
        return clf, df

    def test_fit_sets_is_fitted(self):
        clf, _ = self._fitted_clf()
        assert clf.is_fitted

    def test_fit_sets_classes(self):
        clf, _ = self._fitted_clf()
        assert clf.classes_ is not None
        assert len(clf.classes_) > 0

    def test_predict_returns_strings(self):
        clf, df = self._fitted_clf()
        anom_df = df[df["is_anomaly"]]
        preds = clf.predict(anom_df)
        assert all(isinstance(p, str) for p in preds)

    def test_predict_only_known_classes(self):
        clf, df = self._fitted_clf()
        anom_df = df[df["is_anomaly"]]
        preds = clf.predict(anom_df)
        for p in preds:
            assert p in L2_TARGET_CLASSES, f"Unknown class: {p}"

    def test_not_fitted_raises(self):
        clf = L2Classifier()
        with pytest.raises(RuntimeError):
            clf.predict(_make_df(10))

    def test_no_anomaly_rows_raises(self):
        clf = L2Classifier(use_smote=False)
        df = _make_df(50)
        df["is_anomaly"] = False
        df["anomaly_type"] = "normal"
        with pytest.raises(ValueError):
            clf.fit(df)

    def test_feature_importances_length(self):
        clf, _ = self._fitted_clf()
        fi = clf.feature_importances()
        assert len(fi) == len(clf.feat_cols)

    def test_feature_importances_sum_to_one(self):
        clf, _ = self._fitted_clf()
        fi = clf.feature_importances()
        assert abs(fi.sum() - 1.0) < 0.01


# ─── Evaluate ─────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_required_keys(self):
        df = _make_df(300, seed=5)
        clf = L2Classifier(n_estimators=20, use_smote=False)
        clf.fit(df)

        l1_pred   = df["is_anomaly"].astype(int).values
        l1_scores = np.random.rand(len(df))

        m = clf.evaluate(df, l1_pred, l1_scores, split_name="test", k_budget=20)
        for key in ("mir_at_k", "macro_f1_typing", "typing_accuracy"):
            assert key in m, f"Missing: {key}"

    def test_mir_in_0_1(self):
        df = _make_df(300, seed=5)
        clf = L2Classifier(n_estimators=20, use_smote=False)
        clf.fit(df)
        l1_pred   = df["is_anomaly"].astype(int).values
        l1_scores = np.random.rand(len(df))
        m = clf.evaluate(df, l1_pred, l1_scores, k_budget=20)
        assert 0.0 <= m["mir_at_k"] <= 1.0

    def test_typing_accuracy_in_0_1(self):
        df = _make_df(300, seed=5)
        clf = L2Classifier(n_estimators=20, use_smote=False)
        clf.fit(df)
        l1_pred   = df["is_anomaly"].astype(int).values
        l1_scores = np.random.rand(len(df))
        m = clf.evaluate(df, l1_pred, l1_scores, k_budget=20)
        assert 0.0 <= m["typing_accuracy"] <= 1.0
