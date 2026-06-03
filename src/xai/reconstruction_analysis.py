"""
Phase 9 — XAI: Reconstruction Analysis
Explainability for deep learning models.

Uses per-feature reconstruction error from the LSTM Autoencoder to explain
which sensor signals deviated most from the learned "normal" physics.

Reference: SCIO Research Framework §8.2
"""

import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd

from src import config

warnings.filterwarnings("ignore")

SPLITS_DIR  = config.SPLITS_DIR
RESULTS_DIR = config.RESULTS_DIR

def build_sequences(
    data:    np.ndarray,
    seq_len: int,
    step:    int = 1,
) -> np.ndarray:
    n = len(data)
    starts = range(0, n - seq_len + 1, step)
    seqs   = np.stack([data[i : i + seq_len] for i in starts], axis=0)
    return seqs.astype(np.float32)

def sequences_to_row_scores(
    seq_errors: np.ndarray,
    n_rows:     int,
    seq_len:    int,
    step:       int = 1,
) -> np.ndarray:
    # seq_errors shape: (n_seq, n_features)
    n_features = seq_errors.shape[1]
    scores  = np.zeros((n_rows, n_features), dtype=np.float64)
    counts  = np.zeros(n_rows, dtype=np.int32)
    starts  = list(range(0, n_rows - seq_len + 1, step))

    for idx, start in enumerate(starts):
        scores[start : start + seq_len] += seq_errors[idx]
        counts[start : start + seq_len] += 1

    counts = np.maximum(counts, 1)
    scores /= counts[:, None]

    if len(starts) > 0:
        last_start = starts[-1]
        if last_start + seq_len < n_rows:
            scores[last_start + seq_len :] = seq_errors[-1]

    return scores

def load_lstm_ae():
    """Load trained LSTM AE model and metadata."""
    lstm_meta_path  = RESULTS_DIR / "lstm_ae_model_meta.pkl"
    lstm_model_path = RESULTS_DIR / "lstm_ae_model.keras"

    if not (lstm_model_path.exists() and lstm_meta_path.exists()):
        raise FileNotFoundError("[recon_xai] LSTM-AE model missing. Run Phase 7.")

    # Only load TF locally when needed to save import time and warnings if not used
    import tensorflow as tf
    with open(lstm_meta_path, "rb") as f:
        meta = pickle.load(f)

    lstm_model = tf.keras.models.load_model(str(lstm_model_path))
    print("[recon_xai] Loaded LSTM Autoencoder.")
    return lstm_model, meta

def get_per_feature_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get per-row, per-feature MAE from LSTM-AE.
    Returns DataFrame matching df length, with feature columns containing reconstruction error.
    """
    model, meta = load_lstm_ae()
    feat_cols   = meta["feat_cols"]
    seq_len     = meta["seq_len"]

    X_raw  = df[feat_cols].values.astype(np.float32)
    X_seqs = build_sequences(X_raw, seq_len)
    
    # Predict and calculate per-sequence, per-feature error
    print("[recon_xai] Running LSTM AE inference for per-feature errors...")
    X_hat  = model.predict(X_seqs, batch_size=64, verbose=0)
    
    # Calculate MAE over time dimension (axis=1) leaving (n_seq, n_features)
    seq_errs = np.mean(np.abs(X_seqs - X_hat), axis=1)
    
    # Map back to rows (n_rows, n_features)
    row_errs = sequences_to_row_scores(seq_errs, len(df), seq_len)
    
    err_df = pd.DataFrame(row_errs, columns=feat_cols)
    err_df["timestamp"] = df["timestamp"].values
    err_df["is_anomaly"] = df["is_anomaly"].values
    err_df["anomaly_type"] = df["anomaly_type"].values
    
    return err_df

# ─── Entry Point ─────────────────────────────────────────────────────────────

def run_phase9_reconstruction(
    splits_dir:  str | pathlib.Path = SPLITS_DIR,
    results_dir: str | pathlib.Path = RESULTS_DIR,
):
    """Run Phase 9 Reconstruction Error Analysis pipeline."""
    splits_dir  = pathlib.Path(splits_dir)
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[recon_xai] Loading splits …")
    test_df = pd.read_csv(splits_dir / "test.csv", parse_dates=["timestamp"])
    
    try:
        err_df = get_per_feature_error(test_df)
    except FileNotFoundError as e:
        print(f"[recon_xai] {e}")
        return

    # Save for visualization (Phase 12 Phase 5 Heatmap)
    err_df.to_csv(RESULTS_DIR / "lstmae_reconstruction_errors_test.csv", index=False)
    print(f"[recon_xai] Per-feature errors saved to {RESULTS_DIR / 'lstmae_reconstruction_errors_test.csv'}")

    # Analyze top features per anomaly type
    print("\n=== Phase 9 Reconstruction Error Analysis (LSTM-AE) ===")
    
    # We want features where error during anomaly is much higher than error during normal
    anom_errs = err_df[err_df["is_anomaly"]].drop(columns=["timestamp", "is_anomaly"], errors="ignore")
    norm_errs = err_df[~err_df["is_anomaly"]].drop(columns=["timestamp", "is_anomaly", "anomaly_type"], errors="ignore")
    
    if len(anom_errs) == 0:
        print("  No anomalies in dataset.")
        return

    # Baseline normal error
    baseline_err = norm_errs.mean()
    
    print("  Dominant Features by Anomaly Type (Error ratio wrt Baseline):")
    for atype in anom_errs["anomaly_type"].unique():
        type_errs = anom_errs[anom_errs["anomaly_type"] == atype].drop(columns=["anomaly_type"])
        mean_type_err = type_errs.mean()
        
        # Ratio of anomaly error to normal error
        ratio = (mean_type_err / (baseline_err + 1e-6)).sort_values(ascending=False)
        top_cols = ratio.head(3).index.tolist()
        ratios   = ratio.head(3).values.round(1).tolist()
        
        col_ratio_str = ", ".join(f"{c} ({r}x)" for c, r in zip(top_cols, ratios))
        print(f"    {atype:<25}: {col_ratio_str}")

if __name__ == "__main__":
    run_phase9_reconstruction()
