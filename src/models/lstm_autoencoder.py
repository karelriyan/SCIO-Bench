"""
Phase 7 — Method C: LSTM Autoencoder (L1 Anomaly Detector)
Sequence-to-sequence reconstruction model trained on NORMAL data only.

Architecture (L1 — binary anomaly detection):
  Encoder: LSTM(64) → LSTM(32) → latent(16)
  Decoder: RepeatVector → LSTM(32) → LSTM(64) → TimeDistributed Dense

Anomaly scoring:
  score(x) = MAE(x, x_hat)   (mean absolute reconstruction error across features)

Threshold calibration:
  On validation set (normal rows only):
    threshold = median(score_normal) + k × MAD(score_normal)
  k is tuned to maximise val F1.

Training:
  - Fit ONLY on normal training rows (pure unsupervised)
  - Input: sliding window sequences of length SEQ_LEN=24 (12h)
  - Batch size 64, EarlyStopping patience=5 on val_loss
  - Adam lr=1e-3, reduce-on-plateau lr scheduler

Reference: SCIO Research Framework §7.3
"""

import os
import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src import config

# Suppress TF logs before importing
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

SPLITS_DIR  = config.SPLITS_DIR
RESULTS_DIR = config.RESULTS_DIR
MODEL_PATH  = config.RESULTS_DIR / "lstm_ae_model.keras"

# ─── Hyper-parameters (from config) ───────────────────────────────────────────

_AE_CFG = config.LSTMAEConfig()

SEQ_LEN    = _AE_CFG.seq_len
BATCH_SIZE = _AE_CFG.batch_size
EPOCHS     = _AE_CFG.epochs
LATENT_DIM = _AE_CFG.latent_dim
ENC_UNITS  = list(_AE_CFG.enc_units)
DEC_UNITS  = list(_AE_CFG.dec_units)
PATIENCE   = _AE_CFG.patience

LABEL_COLS = config.LABEL_COLS


# ─── Feature helpers ─────────────────────────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in LABEL_COLS
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
            and c != "is_low_irradiance_period"]


# ─── Sequence Builder ─────────────────────────────────────────────────────────

def build_sequences(
    data:    np.ndarray,
    seq_len: int = SEQ_LEN,
    step:    int = 1,
) -> np.ndarray:
    """
    Build overlapping sliding-window sequences.

    Args:
        data:    (n_rows, n_features) array
        seq_len: window length (ticks)
        step:    stride (1 = fully overlapping)

    Returns:
        (n_sequences, seq_len, n_features)
    """
    n = len(data)
    starts = range(0, n - seq_len + 1, step)
    seqs   = np.stack([data[i : i + seq_len] for i in starts], axis=0)
    return seqs.astype(np.float32)


def sequences_to_row_scores(
    seq_errors: np.ndarray,
    n_rows:     int,
    seq_len:    int = SEQ_LEN,
    step:       int = 1,
) -> np.ndarray:
    """
    Map per-sequence reconstruction errors back to per-row scores.
    Each row's score = mean of all sequences it participates in.

    Args:
        seq_errors: (n_sequences,) array of per-sequence MAE
        n_rows:     total number of original rows
        seq_len:    window length used
        step:       stride used

    Returns:
        (n_rows,) array of per-row anomaly scores
    """
    scores  = np.zeros(n_rows,  dtype=np.float64)
    counts  = np.zeros(n_rows,  dtype=np.int32)
    starts  = list(range(0, n_rows - seq_len + 1, step))

    for idx, start in enumerate(starts):
        scores[start : start + seq_len] += seq_errors[idx]
        counts[start : start + seq_len] += 1

    # Rows not covered by any sequence (tail) get score of last seq
    counts = np.maximum(counts, 1)
    scores /= counts

    # Fill tail rows (if any) that were never covered
    if len(starts) > 0:
        last_start = starts[-1]
        if last_start + seq_len < n_rows:
            scores[last_start + seq_len :] = seq_errors[-1]

    return scores


# ─── Model Builder ────────────────────────────────────────────────────────────

def build_lstm_ae(n_features: int, seq_len: int = SEQ_LEN) -> "tf.keras.Model":
    """
    Build LSTM Autoencoder:
      Encoder: LSTM(64) → LSTM(32) → Dense(latent_dim)
      Decoder: RepeatVector(seq_len) → LSTM(32) → LSTM(64) → TimeDistributed Dense

    Returns a compiled Keras model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(seq_len, n_features), name="input")

    # Encoder
    x = layers.LSTM(ENC_UNITS[0], return_sequences=True,
                    name="enc_lstm1")(inputs)
    x = layers.LSTM(ENC_UNITS[1], return_sequences=False,
                    name="enc_lstm2")(x)
    latent = layers.Dense(LATENT_DIM, activation="relu",
                          name="latent")(x)

    # Decoder
    x = layers.RepeatVector(seq_len, name="repeat")(latent)
    x = layers.LSTM(DEC_UNITS[0], return_sequences=True,
                    name="dec_lstm1")(x)
    x = layers.LSTM(DEC_UNITS[1], return_sequences=True,
                    name="dec_lstm2")(x)
    outputs = layers.TimeDistributed(
        layers.Dense(n_features), name="reconstruction"
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTM_AE")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae",
    )
    return model


# ─── MAD Threshold ─────────────────────────────────────────────────────────────

def _mad_threshold(scores: np.ndarray, k: float) -> float:
    med = np.nanmedian(scores)
    mad = np.nanmedian(np.abs(scores - med))
    return float(med + k * mad)


# ─── Trainer ──────────────────────────────────────────────────────────────────

class LSTMAutoencoder:
    """
    LSTM Autoencoder wrapper for the SCIO-Bench L1 anomaly detector.
    """

    def __init__(
        self,
        seq_len:    int   = SEQ_LEN,
        batch_size: int   = BATCH_SIZE,
        epochs:     int   = EPOCHS,
        patience:   int   = PATIENCE,
        k:          float = 3.0,
    ):
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.epochs     = epochs
        self.patience   = patience
        self.k          = k

        self.model       = None
        self.threshold   = None
        self.feat_cols   = None
        self.n_features  = None
        self.history     = None
        self.is_fitted   = False

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
    ) -> "LSTMAutoencoder":
        """
        Train on NORMAL-only rows from train_df.
        Val_df is used for EarlyStopping (val_loss) and threshold calibration.
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau,
        )

        self.feat_cols  = _get_feature_cols(train_df)
        self.n_features = len(self.feat_cols)

        # Normal-only training rows
        X_train_raw = train_df[train_df["anomaly_type"] == "normal"][
            self.feat_cols
        ].values.astype(np.float32)

        # Val: all rows for early stopping; normal-only for threshold calibration
        X_val_raw = val_df[self.feat_cols].values.astype(np.float32)

        print(f"[lstm_ae] Building model (seq_len={self.seq_len}, "
              f"n_features={self.n_features}) …")
        self.model = build_lstm_ae(self.n_features, self.seq_len)
        self.model.summary(print_fn=lambda x: None)   # suppress summary spam

        # Sequences
        X_train = build_sequences(X_train_raw, self.seq_len)
        X_val   = build_sequences(X_val_raw,   self.seq_len)
        print(f"[lstm_ae] Train seqs: {X_train.shape} | Val seqs: {X_val.shape}")

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        print(f"[lstm_ae] Training (max {self.epochs} epochs, patience={self.patience}) …")
        self.history = self.model.fit(
            X_train, X_train,           # autoencoder: input == target
            validation_data=(X_val, X_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )

        # Calibrate threshold on val NORMAL rows only
        self._calibrate_threshold(val_df)
        self.is_fitted = True
        return self

    def _calibrate_threshold(self, val_df: pd.DataFrame) -> None:
        """Compute MAD-based threshold on val normal scores."""
        normal_val = val_df[val_df["anomaly_type"] == "normal"]
        scores     = self._row_scores(normal_val)
        self.threshold = _mad_threshold(scores, self.k)
        print(f"[lstm_ae] Threshold={self.threshold:.6f} (k={self.k}, "
              f"on {len(normal_val)} val-normal rows)")

    def _seq_mae(self, X_seqs: np.ndarray) -> np.ndarray:
        """Per-sequence MAE between input and reconstruction."""
        X_hat = self.model.predict(X_seqs, batch_size=self.batch_size, verbose=0)
        # (n_seq, seq_len, n_feat) → mean over seq_len and n_feat
        return np.mean(np.abs(X_seqs - X_hat), axis=(1, 2))

    def _row_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row anomaly score for any DataFrame."""
        X_raw  = df[self.feat_cols].values.astype(np.float32)
        X_seqs = build_sequences(X_raw, self.seq_len)
        seq_errs = self._seq_mae(X_seqs)
        return sequences_to_row_scores(seq_errs, len(df), self.seq_len)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Binary predictions (1=anomaly) based on calibrated threshold."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        scores = self._row_scores(df)
        return (scores > self.threshold).astype(int)

    def anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Raw reconstruction error scores for ROC curve generation."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self._row_scores(df)

    def tune_k(
        self,
        val_df:   pd.DataFrame,
        k_values: list[float] = [2.0, 2.5, 3.0, 3.5, 4.0],
    ) -> float:
        """
        Find the k that maximises F1 on the validation set.
        Updates self.threshold to use best k.
        """
        if not self.model:
            raise RuntimeError("Call fit() first.")

        normal_val = val_df[val_df["anomaly_type"] == "normal"]
        normal_scores = self._row_scores(normal_val)
        med = np.nanmedian(normal_scores)
        mad = np.nanmedian(np.abs(normal_scores - med))

        # Score ALL val rows for evaluation
        all_scores = self._row_scores(val_df)
        y_val      = val_df["is_anomaly"].astype(int).values

        best_f1, best_k = -1.0, self.k
        print("\n[lstm_ae] k-sweep on val set:")
        for k in k_values:
            thr  = float(med + k * mad)
            pred = (all_scores > thr).astype(int)
            f1   = f1_score(y_val, pred, zero_division=0)
            p    = precision_score(y_val, pred, zero_division=0)
            r    = recall_score(y_val, pred, zero_division=0)
            print(f"  k={k:.1f} thr={thr:.5f} | F1={f1:.4f} P={p:.4f} R={r:.4f}")
            if f1 > best_f1:
                best_f1, best_k = f1, k

        self.k         = best_k
        self.threshold = float(med + best_k * mad)
        print(f"[lstm_ae] Best k={best_k} → threshold={self.threshold:.6f}, "
              f"val F1={best_f1:.4f}")
        return best_k

    def evaluate(
        self,
        df:         pd.DataFrame,
        split_name: str = "test",
    ) -> dict:
        """Full metrics dict (same schema as Phase 5/6)."""
        scores = self.anomaly_scores(df)
        pred   = (scores > self.threshold).astype(int)
        y_true = df["is_anomaly"].astype(int).values

        a6_mask     = (df["anomaly_type"] == "low_irradiance").values
        normal_mask = (~df["is_anomaly"].values)

        fpr_a6     = pred[a6_mask].mean()     if a6_mask.sum()     > 0 else 0.0
        fpr_global = pred[normal_mask].mean() if normal_mask.sum() > 0 else 0.0

        metrics = {
            "method":      "lstm_ae",
            "split":       split_name,
            "f1":          f1_score(y_true, pred, zero_division=0),
            "precision":   precision_score(y_true, pred, zero_division=0),
            "recall":      recall_score(y_true, pred, zero_division=0),
            "fpr_global":  float(fpr_global),
            "fpr_a6":      float(fpr_a6),
            "n_predicted": int(pred.sum()),
            "n_true":      int(y_true.sum()),
            "threshold":   self.threshold,
            "k":           self.k,
        }

        # Per-type F1
        for atype in df["anomaly_type"].unique():
            mask = (df["anomaly_type"] == atype).values
            if mask.sum() == 0:
                continue
            y_t = df.loc[mask, "is_anomaly"].astype(int).values
            y_p = pred[mask]
            metrics[f"f1_{atype}"] = f1_score(y_t, y_p, zero_division=0)

        return metrics

    def save(self, path: pathlib.Path = MODEL_PATH) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        # Save wrapper (without model — loaded separately)
        meta = {
            "threshold":  self.threshold,
            "k":          self.k,
            "feat_cols":  self.feat_cols,
            "n_features": self.n_features,
            "seq_len":    self.seq_len,
        }
        meta_path = path.parent / (path.stem + "_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        print(f"[lstm_ae] Model saved → {path}")
        print(f"[lstm_ae] Meta  saved → {meta_path}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run_phase7(
    splits_dir:  str | pathlib.Path = SPLITS_DIR,
    results_dir: str | pathlib.Path = RESULTS_DIR,
) -> dict:
    """
    Full Phase 7 pipeline:
      1. Load splits
      2. Build + train LSTM AE on normal train rows
      3. Tune k on val
      4. Evaluate on test
      5. Save model + results
    """
    splits_dir  = pathlib.Path(splits_dir)
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for f in ("train.csv", "val.csv", "test.csv"):
        if not (splits_dir / f).exists():
            raise FileNotFoundError(
                f"[lstm_ae] {f} not found. Run Phase 4 first:\n"
                "  python -m src.data.feature_engineering"
            )

    print("[lstm_ae] Loading splits …")
    train = pd.read_csv(splits_dir / "train.csv", parse_dates=["timestamp"])
    val   = pd.read_csv(splits_dir / "val.csv",   parse_dates=["timestamp"])
    test  = pd.read_csv(splits_dir / "test.csv",  parse_dates=["timestamp"])
    print(f"[lstm_ae] Train={len(train):,} Val={len(val):,} Test={len(test):,}")

    ae = LSTMAutoencoder(seq_len=SEQ_LEN, epochs=EPOCHS, patience=PATIENCE)
    ae.fit(train, val)
    ae.tune_k(val)
    ae.save(results_dir / "lstm_ae_model.keras")

    test_metrics = ae.evaluate(test, split_name="test")

    # Save metrics
    pd.DataFrame([test_metrics]).to_csv(
        results_dir / "lstm_ae_results.csv", index=False
    )

    print("\n=== Phase 7 Results (Test Set) ===")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  FPR@A6:    {test_metrics['fpr_a6']:.4f}")
    print(f"  Predicted: {test_metrics['n_predicted']} anomalies "
          f"(true: {test_metrics['n_true']})")

    return test_metrics


if __name__ == "__main__":
    run_phase7()
