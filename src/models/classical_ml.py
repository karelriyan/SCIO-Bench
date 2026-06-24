"""
Phase 6 — Method B: Classical Unsupervised ML
Implements Isolation Forest and Local Outlier Factor with
grid-search hyperparameter tuning on the validation set.

Models:
  B1: Isolation Forest (scikit-learn)
      Hyperparams: n_estimators, contamination, max_features
      Contamination is tuned to match expected anomaly rate (~9%)

  B2: Local Outlier Factor (scikit-learn)
      Hyperparams: n_neighbors, contamination
      Note: LOF in sklearn is transductive only — we use novelty=True for test

Both models:
  - Fit on train NORMAL rows only (unsupervised anomaly detection paradigm)
  - Threshold tuned by maximising F1 on val set
  - Save predictions, per-type F1, and FPR@A6

Reference: SCIO Research Framework §7.2
"""

import pathlib
import pickle
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, precision_score, recall_score

from src import config
from src.config import get_feature_cols
from src.evaluation.metrics import compute_detection_metrics

warnings.filterwarnings("ignore")

SPLITS_DIR  = config.SPLITS_DIR
RESULTS_DIR = config.RESULTS_DIR

LABEL_COLS = config.LABEL_COLS


# ─── Shared Evaluation ───────────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    method: str,
    split: str,
    **extra,
) -> dict:
    """Compute standard metrics dict using centralized metrics module."""
    metrics = compute_detection_metrics(
        y_true=y_true,
        y_pred=y_pred,
        df=df,
        method=method,
        split_name=split
    )
    metrics.update(extra)
    return metrics


# ─── Isolation Forest ────────────────────────────────────────────────────────

IF_PARAM_GRID = {
    "n_estimators":  [100, 200],
    "contamination": [0.05, 0.09, 0.12],
    "max_features":  [0.7, 1.0],
}


def _fit_isolation_forest(
    X_train: np.ndarray,
    n_estimators: int   = 100,
    contamination: float = 0.09,
    max_features: float  = 1.0,
    random_state: int    = 42,
) -> IsolationForest:
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train)
    return clf


def tune_isolation_forest(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
) -> tuple[IsolationForest, dict, pd.DataFrame]:
    """
    Grid search over IF hyperparams, maximising val F1.

    Fits ONLY on normal training rows (unsupervised paradigm).
    Returns: (best_model, best_params, all_results_df)
    """
    feat_cols = get_feature_cols(train_df)
    # Fit on normal-only rows
    X_train_normal = train_df[train_df["anomaly_type"] == "normal"][feat_cols].values
    X_val  = val_df[feat_cols].values
    y_val  = val_df["is_anomaly"].astype(int).values

    best_f1, best_clf, best_params = -1.0, None, {}
    rows = []

    keys   = list(IF_PARAM_GRID.keys())
    combos = list(product(*[IF_PARAM_GRID[k] for k in keys]))
    total  = len(combos)

    print(f"[iso_forest] Grid search: {total} combinations …")
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        clf    = _fit_isolation_forest(X_train_normal, **params)
        # IF: -1 = outlier, 1 = inlier → convert to 0/1
        raw  = clf.predict(X_val)
        pred = (raw == -1).astype(int)
        f1   = f1_score(y_val, pred, zero_division=0)
        rows.append({**params, "val_f1": f1})
        if f1 > best_f1:
            best_f1, best_clf, best_params = f1, clf, params

    print(f"[iso_forest] Best params: {best_params} → val F1={best_f1:.4f}")
    return best_clf, best_params, pd.DataFrame(rows)


# ─── Local Outlier Factor ─────────────────────────────────────────────────────

LOF_PARAM_GRID = {
    "n_neighbors":   [10, 20, 35],
    "contamination": [0.05, 0.09, 0.12],
}


def tune_lof(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
) -> tuple[LocalOutlierFactor, dict, pd.DataFrame]:
    """
    Grid search over LOF hyperparams, maximising val F1.

    LOF uses novelty=True so it can predict on unseen data.
    Fits ONLY on normal training rows.
    Returns: (best_model, best_params, all_results_df)
    """
    feat_cols = get_feature_cols(train_df)
    X_train_normal = train_df[train_df["anomaly_type"] == "normal"][feat_cols].values
    X_val  = val_df[feat_cols].values
    y_val  = val_df["is_anomaly"].astype(int).values

    best_f1, best_clf, best_params = -1.0, None, {}
    rows = []

    keys   = list(LOF_PARAM_GRID.keys())
    combos = list(product(*[LOF_PARAM_GRID[k] for k in keys]))
    total  = len(combos)

    print(f"[lof] Grid search: {total} combinations …")
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        clf = LocalOutlierFactor(
            n_neighbors=params["n_neighbors"],
            contamination=params["contamination"],
            novelty=True,   # required to call predict() on unseen data
            n_jobs=-1,
        )
        clf.fit(X_train_normal)
        raw  = clf.predict(X_val)
        pred = (raw == -1).astype(int)
        f1   = f1_score(y_val, pred, zero_division=0)
        rows.append({**params, "val_f1": f1})
        if f1 > best_f1:
            best_f1, best_clf, best_params = f1, clf, params

    print(f"[lof] Best params: {best_params} → val F1={best_f1:.4f}")
    return best_clf, best_params, pd.DataFrame(rows)


# ─── Shared Test Evaluation ───────────────────────────────────────────────────

def evaluate_sklearn_model(
    clf,
    test_df:  pd.DataFrame,
    method:   str,
    params:   dict,
) -> dict:
    """
    Evaluate a fitted sklearn IF/LOF on test set.
    Maps sklearn -1/1 convention to 1/0 (anomaly/normal).
    """
    feat_cols = get_feature_cols(test_df)
    X_test = test_df[feat_cols].values
    y_true = test_df["is_anomaly"].astype(int).values

    raw  = clf.predict(X_test)
    pred = (raw == -1).astype(int)

    return _compute_metrics(y_true, pred, test_df, method, "test", **params)


def score_sklearn_model(clf, df: pd.DataFrame) -> np.ndarray:
    """
    Return anomaly scores in [0, 1] (higher = more anomalous).
    Used for ROC curve generation.
    For IF: negate decision function (lower = more anomalous internally).
    For LOF: negate score_samples.
    """
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].values
    if hasattr(clf, "decision_function"):
        raw = -clf.decision_function(X)   # IF: negate so higher = more anomalous
    elif hasattr(clf, "score_samples"):
        raw = -clf.score_samples(X)       # LOF: negate
    else:
        raw = clf.predict(X).astype(float)
    # Normalise to [0, 1]
    mn, mx = raw.min(), raw.max()
    if mx > mn:
        return (raw - mn) / (mx - mn)
    return raw


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_phase6(
    splits_dir:  str | pathlib.Path = SPLITS_DIR,
    results_dir: str | pathlib.Path = RESULTS_DIR,
) -> tuple[dict, dict]:
    """
    Full Phase 6 pipeline:
      1. Load splits
      2. Grid search IF on val  → best model → test metrics
      3. Grid search LOF on val → best model → test metrics
      4. Save both models + results

    Returns: (if_metrics, lof_metrics)
    """
    splits_dir  = pathlib.Path(splits_dir)
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for f in ("train.csv", "val.csv", "test.csv"):
        if not (splits_dir / f).exists():
            raise FileNotFoundError(
                f"[classical_ml] {f} not found. Run Phase 4 first:\n"
                "  python -m src.data.feature_engineering"
            )

    print("[classical_ml] Loading splits …")
    train = pd.read_csv(splits_dir / "train.csv", parse_dates=["timestamp"])
    val   = pd.read_csv(splits_dir / "val.csv",   parse_dates=["timestamp"])
    test  = pd.read_csv(splits_dir / "test.csv",  parse_dates=["timestamp"])
    print(f"[classical_ml] Train={len(train):,} Val={len(val):,} Test={len(test):,}")

    all_results = []

    # ── Isolation Forest ──────────────────────────────────────────────────
    print("\n=== Isolation Forest ===")
    if_clf, if_params, if_sweep = tune_isolation_forest(train, val)
    if_sweep.to_csv(results_dir / "if_sweep.csv", index=False)

    if_metrics = evaluate_sklearn_model(if_clf, test, "isolation_forest", if_params)
    _print_metrics(if_metrics, "Isolation Forest")
    all_results.append(if_metrics)

    with open(results_dir / "isolation_forest_model.pkl", "wb") as f:
        pickle.dump(if_clf, f)
    print(f"[iso_forest] Model saved → {results_dir}/isolation_forest_model.pkl")

    # ── Local Outlier Factor ──────────────────────────────────────────────
    print("\n=== Local Outlier Factor ===")
    lof_clf, lof_params, lof_sweep = tune_lof(train, val)
    lof_sweep.to_csv(results_dir / "lof_sweep.csv", index=False)

    lof_metrics = evaluate_sklearn_model(lof_clf, test, "lof", lof_params)
    _print_metrics(lof_metrics, "LOF")
    all_results.append(lof_metrics)

    with open(results_dir / "lof_model.pkl", "wb") as f:
        pickle.dump(lof_clf, f)
    print(f"[lof] Model saved → {results_dir}/lof_model.pkl")

    # ── Save combined results ─────────────────────────────────────────────
    pd.DataFrame(all_results).to_csv(
        results_dir / "classical_ml_results.csv", index=False
    )

    return if_metrics, lof_metrics


def _print_metrics(m: dict, label: str) -> None:
    print(f"\n=== {label} (Test Set) ===")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  FPR@A6:    {m['fpr_a6']:.4f}")
    print(f"  Predicted: {m['n_predicted']} anomalies (true: {m['n_true']})")


if __name__ == "__main__":
    run_phase6()
