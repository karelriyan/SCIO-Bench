"""
Phase 7 — Tests: LSTM Autoencoder
Tests sequence building, row-score mapping, threshold calibration, and
prediction logic using a lightweight mock model (avoids full TF training).
"""

import pytest
import numpy as np
import pandas as pd
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.models.lstm_autoencoder import (
    build_sequences,
    sequences_to_row_scores,
    _mad_threshold,
    _get_feature_cols,
    LSTMAutoencoder,
    SEQ_LEN,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, anomaly_frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_anom = int(n * anomaly_frac)
    n_norm = n - n_anom
    return pd.DataFrame({
        "timestamp":             pd.date_range("2020-05-15", periods=n, freq="30min"),
        "device_id":             ["P1"] * n,
        "protocol":              ["lora"] * n,
        "mppt_w":                rng.normal(0, 1, n),
        "volt_v":                rng.normal(0, 1, n),
        "curr_a":                rng.normal(0, 1, n),
        "batt_pct":              rng.normal(0, 1, n),
        "irradiance":            rng.normal(0, 1, n),
        "physics_residual":      rng.normal(0, 0.1, n),
        "is_anomaly":            [False] * n_norm + [True] * n_anom,
        "anomaly_type":          ["normal"] * n_norm + ["sudden_drop"] * n_anom,
        "is_weather_event":      [False] * n,
        "is_low_irradiance_period": [0] * n,
    })


# ─── Sequence Building ────────────────────────────────────────────────────────

class TestBuildSequences:
    def test_output_shape(self):
        data = np.random.randn(100, 5).astype(np.float32)
        seqs = build_sequences(data, seq_len=10, step=1)
        assert seqs.shape == (91, 10, 5)   # 100 - 10 + 1 = 91

    def test_step_2_reduces_count(self):
        data = np.random.randn(100, 5).astype(np.float32)
        seqs = build_sequences(data, seq_len=10, step=2)
        expected = len(range(0, 100 - 10 + 1, 2))
        assert seqs.shape[0] == expected

    def test_first_seq_matches_data(self):
        data = np.arange(50 * 3).reshape(50, 3).astype(np.float32)
        seqs = build_sequences(data, seq_len=5, step=1)
        np.testing.assert_array_equal(seqs[0], data[:5])

    def test_last_seq_matches_data(self):
        data = np.arange(50 * 3).reshape(50, 3).astype(np.float32)
        seqs = build_sequences(data, seq_len=5, step=1)
        np.testing.assert_array_equal(seqs[-1], data[45:50])

    def test_dtype_float32(self):
        data = np.random.randn(50, 4)
        seqs = build_sequences(data, seq_len=5)
        assert seqs.dtype == np.float32


# ─── Row Score Mapping ────────────────────────────────────────────────────────

class TestSequencesToRowScores:
    def test_output_length_matches_input(self):
        n_rows, seq_len = 100, 10
        n_seqs = n_rows - seq_len + 1
        seq_errors = np.ones(n_seqs, dtype=np.float64)
        scores = sequences_to_row_scores(seq_errors, n_rows, seq_len)
        assert len(scores) == n_rows

    def test_constant_errors_give_constant_scores(self):
        """If all sequence errors = 1, every row should score 1."""
        n_rows, seq_len = 50, 10
        n_seqs = n_rows - seq_len + 1
        seq_errors = np.ones(n_seqs)
        scores = sequences_to_row_scores(seq_errors, n_rows, seq_len)
        np.testing.assert_allclose(scores, 1.0, atol=1e-6)

    def test_high_error_propagates_to_row(self):
        """A spike at sequence index 10 should raise scores around row 10-20."""
        n_rows, seq_len = 100, 10
        n_seqs = n_rows - seq_len + 1
        seq_errors = np.zeros(n_seqs)
        seq_errors[10] = 999.0     # spike at sequence 10 covers rows 10-19
        scores = sequences_to_row_scores(seq_errors, n_rows, seq_len)
        assert scores[15] > 0.0    # row 15 is covered by this sequence


# ─── MAD Threshold ────────────────────────────────────────────────────────────

class TestMADThreshold:
    def test_returns_float(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(_mad_threshold(arr, k=3.0), float)

    def test_higher_k_gives_higher_threshold(self):
        arr = np.random.randn(100)
        assert _mad_threshold(arr, k=5.0) > _mad_threshold(arr, k=2.0)

    def test_zero_variance_gives_median(self):
        arr = np.full(20, 7.0)
        assert _mad_threshold(arr, k=3.5) == pytest.approx(7.0)


# ─── Feature Columns ─────────────────────────────────────────────────────────

class TestGetFeatureCols:
    def test_excludes_label_cols(self):
        df = _make_df(50)
        cols = _get_feature_cols(df)
        for c in ("is_anomaly", "anomaly_type", "timestamp", "device_id", "protocol"):
            assert c not in cols

    def test_nonempty(self):
        assert len(_get_feature_cols(_make_df(50))) > 0


# ─── LSTMAutoencoder (light weight — no TF training) ─────────────────────────

class TestLSTMAutoencoderLogic:
    """
    Tests that do NOT invoke model.fit() (too slow for unit tests).
    Instead, we mock the model and only test surrounding logic.
    """

    def _make_ae_with_mock(self, df: pd.DataFrame) -> LSTMAutoencoder:
        """Create AE with a fake model that returns zeros (perfect reconstruction)."""
        ae = LSTMAutoencoder(seq_len=10, batch_size=32, epochs=1)
        ae.feat_cols  = _get_feature_cols(df)
        ae.n_features = len(ae.feat_cols)
        ae.threshold  = 0.05        # arbitrary fixed threshold for testing
        ae.k          = 3.0
        ae.is_fitted  = True

        # Mock model: always returns input unchanged (zero reconstruction error)
        class MockModel:
            def predict(self, X, batch_size=32, verbose=0):
                return X   # perfect reconstruction → error = 0

        ae.model = MockModel()
        return ae

    def test_predict_all_zeros_for_perfect_reconstruction(self):
        """With zero reconstruction error, all predictions should be 0 (normal)."""
        df = _make_df(100)
        ae = self._make_ae_with_mock(df)
        ae.threshold = 0.5   # high threshold so error=0 never fires
        preds = ae.predict(df)
        assert (preds == 0).all()

    def test_predict_output_binary(self):
        df = _make_df(100)
        ae = self._make_ae_with_mock(df)
        preds = ae.predict(df)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_length_matches_input(self):
        df = _make_df(100)
        ae = self._make_ae_with_mock(df)
        assert len(ae.predict(df)) == len(df)

    def test_anomaly_scores_length_matches_input(self):
        df = _make_df(100)
        ae = self._make_ae_with_mock(df)
        scores = ae.anomaly_scores(df)
        assert len(scores) == len(df)

    def test_evaluate_returns_required_keys(self):
        df = _make_df(100)
        ae = self._make_ae_with_mock(df)
        m = ae.evaluate(df, split_name="test")
        for key in ("f1", "precision", "recall", "fpr_a6", "fpr_global",
                    "n_predicted", "n_true", "method", "threshold"):
            assert key in m, f"Missing: {key}"

    def test_not_fitted_raises(self):
        ae = LSTMAutoencoder()
        with pytest.raises(RuntimeError):
            ae.predict(_make_df(10))
