"""
Phase 11 — Centralized Evaluation Metrics
All model-agnostic metric computation lives here to avoid duplication
across rule_based.py, lstm_autoencoder.py, and classical_ml.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
    method: str = "unknown",
    split_name: str = "test",
    threshold: float | None = None,
) -> dict:
    """
    Compute the full metrics dict used across all SCIO-Bench methods.

    Args:
        y_true:      Ground-truth binary labels (1=anomaly).
        y_pred:      Binary predictions (1=anomaly).
        y_scores:    Continuous anomaly scores for AUC-ROC (optional).
        df:          Full DataFrame with 'anomaly_type' and 'is_anomaly' columns.
        method:      Method name (e.g. 'rule_based', 'lstm_ae').
        split_name:  Data split evaluated ('train', 'val', 'test').
        threshold:   Decision threshold (if applicable).

    Returns:
        Dictionary with f1, precision, recall, fpr_global, fpr_a6,
        per-type F1s, and optional roc_auc.
    """
    metrics: dict = {
        "method": method,
        "split": split_name,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_predicted": int(y_pred.sum()),
        "n_true": int(y_true.sum()),
    }

    if threshold is not None:
        metrics["threshold"] = float(threshold)

    # AUC-ROC (requires continuous scores)
    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            metrics["roc_auc"] = float("nan")

    # FPR on normal rows and A6 weather events
    if df is not None and len(df) == len(y_pred):
        a6_mask = (df["anomaly_type"] == "low_irradiance").values
        normal_mask = (~df["is_anomaly"].values)

        metrics["fpr_a6"] = float(
            y_pred[a6_mask].mean() if a6_mask.sum() > 0 else 0.0
        )
        metrics["fpr_global"] = float(
            y_pred[normal_mask].mean() if normal_mask.sum() > 0 else 0.0
        )

        # Per-anomaly-type F1
        for atype in df["anomaly_type"].unique():
            mask = (df["anomaly_type"] == atype).values
            if mask.sum() == 0:
                continue
            y_t = df.loc[mask, "is_anomaly"].astype(int).values
            y_p = y_pred[mask]
            metrics[f"f1_{atype}"] = float(f1_score(y_t, y_p, zero_division=0))

    return metrics


def confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a human-readable confusion matrix dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
