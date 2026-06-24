"""
Phase 6 — Tests: Classical Unsupervised ML
Tests Isolation Forest and LOF: fit → predict → metrics.
No real dataset required — uses synthetic DataFrames.
"""

import pytest
import numpy as np
import pandas as pd

from src.models.classical_ml import (
    _compute_metrics,
    _fit_isolation_forest,
    tune_isolation_forest,
    tune_lof,
    evaluate_sklearn_model,
    score_sklearn_model,
)
from src.config import get_feature_cols


# ─── Fixture ─────────────────────────────────────────────────────────────────

def _make_df(n: int = 300, anomaly_frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_anom = int(n * anomaly_frac)
    n_norm = n - n_anom
    types  = ["normal"] * n_norm + ["sudden_drop"] * n_anom
    is_anom = [False] * n_norm + [True] * n_anom
    df = pd.DataFrame({
        "timestamp":             pd.date_range("2020-05-15", periods=n, freq="30min"),
        "device_id":             ["P1"] * n,
        "protocol":              ["lora"] * n,
        "mppt_w":                np.concatenate([rng.uniform(0, 4000, n_norm),
                                                 rng.uniform(-500, 5500, n_anom)]),
        "volt_v":                rng.uniform(23, 26, n),
        "curr_a":                rng.uniform(0, 160, n),
        "batt_pct":              rng.uniform(30, 95, n),
        "irradiance":            rng.uniform(100, 900, n),
        "temp_c":                rng.uniform(30, 60, n),
        "mppt_w_delta":          rng.normal(0, 1, n),
        "volt_v_delta":          rng.normal(0, 0.1, n),
        "batt_delta":            rng.normal(0, 0.5, n),
        "physics_residual":      rng.uniform(0, 0.5, n),
        "hour_sin":              np.sin(2 * np.pi * rng.uniform(0, 24, n) / 24),
        "hour_cos":              np.cos(2 * np.pi * rng.uniform(0, 24, n) / 24),
        "is_anomaly":            is_anom,
        "anomaly_type":          types,
        "is_weather_event":      [False] * n,
        "is_low_irradiance_period": [0] * n,
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ─── Feature Column Selection ─────────────────────────────────────────────────

class TestFeatureCols:
    def test_excludes_label_cols(self):
        df = _make_df(50)
        cols = get_feature_cols(df)
        for c in ("is_anomaly", "anomaly_type", "is_weather_event",
                  "timestamp", "device_id", "protocol"):
            assert c not in cols, f"Label col {c} should be excluded"

    def test_includes_numeric_features(self):
        df = _make_df(50)
        cols = get_feature_cols(df)
        for c in ("mppt_w", "volt_v", "batt_pct", "physics_residual"):
            assert c in cols, f"Feature {c} should be included"

    def test_returns_nonempty(self):
        df = _make_df(50)
        assert len(get_feature_cols(df)) > 0


# ─── Isolation Forest ────────────────────────────────────────────────────────

class TestIsolationForest:
    def _train_val(self):
        train = _make_df(300, anomaly_frac=0.09, seed=0)
        val   = _make_df(100, anomaly_frac=0.09, seed=1)
        return train, val

    def test_fit_returns_model(self):
        train, _ = self._train_val()
        feat_cols = get_feature_cols(train)
        X = train[feat_cols].values
        clf = _fit_isolation_forest(X)
        assert hasattr(clf, "predict")

    def test_predict_binary(self):
        train, val = self._train_val()
        feat_cols  = get_feature_cols(train)
        X_train = train[train["anomaly_type"] == "normal"][feat_cols].values
        clf = _fit_isolation_forest(X_train, contamination=0.09)
        raw  = clf.predict(val[feat_cols].values)
        assert set(np.unique(raw)).issubset({-1, 1})

    def test_tune_returns_model_and_params(self):
        train, val = self._train_val()
        clf, params, df_results = tune_isolation_forest(train, val)
        assert clf is not None
        assert "n_estimators"  in params
        assert "contamination" in params
        assert "val_f1" in df_results.columns

    def test_test_metrics_keys(self):
        train, val = self._train_val()
        clf, params, _ = tune_isolation_forest(train, val)
        test = _make_df(100, anomaly_frac=0.09, seed=2)
        m = evaluate_sklearn_model(clf, test, "isolation_forest", params)
        for key in ("f1", "precision", "recall", "fpr_a6", "n_predicted"):
            assert key in m

    def test_f1_in_range(self):
        train, val = self._train_val()
        clf, params, _ = tune_isolation_forest(train, val)
        test = _make_df(100, anomaly_frac=0.09, seed=2)
        m = evaluate_sklearn_model(clf, test, "isolation_forest", params)
        assert 0.0 <= m["f1"] <= 1.0


# ─── Local Outlier Factor ─────────────────────────────────────────────────────

class TestLOF:
    def _train_val(self):
        train = _make_df(300, anomaly_frac=0.09, seed=10)
        val   = _make_df(100, anomaly_frac=0.09, seed=11)
        return train, val

    def test_tune_returns_model_and_params(self):
        train, val = self._train_val()
        clf, params, df_results = tune_lof(train, val)
        assert clf is not None
        assert "n_neighbors"  in params
        assert "contamination" in params
        assert "val_f1" in df_results.columns

    def test_novelty_true(self):
        """LOF must be fitted with novelty=True to support predict() on new data."""
        train, val = self._train_val()
        clf, _, _ = tune_lof(train, val)
        assert clf.novelty is True

    def test_test_metrics_keys(self):
        train, val = self._train_val()
        clf, params, _ = tune_lof(train, val)
        test = _make_df(100, anomaly_frac=0.09, seed=12)
        m = evaluate_sklearn_model(clf, test, "lof", params)
        for key in ("f1", "precision", "recall", "fpr_a6", "n_predicted"):
            assert key in m

    def test_f1_in_range(self):
        train, val = self._train_val()
        clf, params, _ = tune_lof(train, val)
        test = _make_df(100, anomaly_frac=0.09, seed=12)
        m = evaluate_sklearn_model(clf, test, "lof", params)
        assert 0.0 <= m["f1"] <= 1.0


# ─── Anomaly Scoring ─────────────────────────────────────────────────────────

class TestScoring:
    def _get_if_model(self):
        train = _make_df(200, seed=20)
        feat_cols = get_feature_cols(train)
        X = train[train["anomaly_type"] == "normal"][feat_cols].values
        return _fit_isolation_forest(X, contamination=0.09)

    def test_scores_in_0_1(self):
        clf = self._get_if_model()
        test = _make_df(100, seed=21)
        scores = score_sklearn_model(clf, test)
        assert scores.min() >= -0.01   # allow tiny float error
        assert scores.max() <= 1.01

    def test_scores_length_matches_input(self):
        clf = self._get_if_model()
        test = _make_df(100, seed=21)
        scores = score_sklearn_model(clf, test)
        assert len(scores) == len(test)


# ─── Compute Metrics ─────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_all_keys(self):
        df = _make_df(100)
        y_true = df["is_anomaly"].astype(int).values
        y_pred = np.zeros(len(df), dtype=int)
        y_pred[:5] = 1
        m = _compute_metrics(y_true, y_pred, df, "test_method", "test")
        for key in ("f1", "precision", "recall", "fpr_a6", "fpr_global",
                    "n_predicted", "n_true", "method", "split"):
            assert key in m

    def test_per_type_f1_added(self):
        df = _make_df(100)
        y_true = df["is_anomaly"].astype(int).values
        y_pred = np.ones(len(df), dtype=int)
        m = _compute_metrics(y_true, y_pred, df, "test_method", "test")
        # should contain f1_normal and f1_sudden_drop
        assert any(k.startswith("f1_") for k in m)
