"""
Phase 9 — Tests: XAI Analysis
Tests SHAP logic and LSTM reconstruction error parsing.
"""

import pytest
import numpy as np
import pandas as pd

from src.xai.shap_analysis import analyze_shap
from src.config import get_feature_cols
from src.xai.reconstruction_analysis import build_sequences, sequences_to_row_scores


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_anom = 20
    n_norm = n - n_anom
    types  = ["normal"] * n_norm + ["sudden_drop"] * 10 + ["false_data_injection"] * 10
    is_anom = [False] * n_norm + [True] * n_anom
    
    return pd.DataFrame({
        "timestamp":             pd.date_range("2020-05-15", periods=n, freq="30min"),
        "device_id":             ["P1"] * n,
        "protocol":              ["lora"] * n,
        "mppt_w":                rng.normal(0, 1, n),
        "volt_v":                rng.normal(0, 1, n),
        "curr_a":                rng.normal(0, 1, n),
        "physics_residual":      rng.normal(0, 0.1, n),
        "is_anomaly":            is_anom,
        "anomaly_type":          types,
        "is_weather_event":      [False] * n,
        "is_low_irradiance_period": [0] * n,
    })


from sklearn.ensemble import IsolationForest

class TestSHAPAnalysis:
    def test_feature_cols_exclusion(self):
        df = _make_df(50)
        cols = get_feature_cols(df)
        assert "is_anomaly" not in cols
        assert "mppt_w" in cols

    def test_analyze_shap_returns_dict(self):
        df = _make_df(50)
        feat_cols = get_feature_cols(df)
        
        # Use a real IsolationForest so SHAP accepts it
        clf = IsolationForest(random_state=42)
        clf.fit(df[feat_cols].values)
        
        res = analyze_shap(clf, df, feat_cols, bg_size=10)
        assert isinstance(res, dict)
        if "global_importance" in res:
            assert isinstance(res["global_importance"], pd.Series)
            assert "type_importances" in res
            assert isinstance(res["type_importances"], dict)

    def test_analyze_shap_no_anomalies(self):
        df = _make_df(50)
        df["anomaly_type"] = "normal"
        df["is_anomaly"] = False
        feat_cols = get_feature_cols(df)
        
        clf = IsolationForest(random_state=42)
        clf.fit(df[feat_cols].values)
        
        res = analyze_shap(clf, df, feat_cols, bg_size=10)
        assert res == {}


# ─── Reconstruction Analysis Tests ───────────────────────────────────────────

class TestReconstructionAnalysis:
    def test_build_sequences_shape(self):
        data = np.random.randn(100, 5).astype(np.float32)
        seqs = build_sequences(data, seq_len=10)
        assert seqs.shape == (91, 10, 5)

    def test_sequences_to_row_scores_shape(self):
        n_rows, seq_len, n_features = 100, 10, 4
        n_seqs = n_rows - seq_len + 1
        seq_errors = np.ones((n_seqs, n_features), dtype=np.float64)
        scores = sequences_to_row_scores(seq_errors, n_rows, seq_len)
        assert scores.shape == (n_rows, n_features)

    def test_sequences_to_row_scores_values(self):
        n_rows, seq_len, n_features = 50, 10, 3
        n_seqs = n_rows - seq_len + 1
        seq_errors = np.full((n_seqs, n_features), 2.5)
        scores = sequences_to_row_scores(seq_errors, n_rows, seq_len)
        np.testing.assert_allclose(scores, 2.5, atol=1e-6)
