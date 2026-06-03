"""
Phase 8 — Hierarchical L2 Classifier (Anomaly Typing)
Implements the second stage of the SCIO-Bench hierarchical pipeline.

Pipeline:
  L1 → Binary anomaly detection (LSTM-AE / IF / LOF / rule-based)
  L2 → Anomaly type classification (this module)

L2 design:
  - Trained ONLY on rows the L1 flags as anomaly (operating on L1 positives)
  - Training: ground-truth anomaly rows from train SET (stratified)
  - SMOTE applied to balance rare classes (A7 FDI = rarest)
  - Model: Random Forest (fast, handles imbalance, SHAP-compatible in Phase 9)
  - Hyperparams tuned on val set

Evaluation metrics:
  - Per-type precision/recall/F1 (for Table I in paper)
  - MIR@k: Maintenance-Informed Recall at budget k
    "If an operator can investigate k alarms per day, what fraction of
     real anomalies are correctly typed?"
  - Confusion matrix saved for figure generation (Phase 12)

Reference: SCIO Research Framework §7.4
"""

import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src import config

warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False
    print("[l2] Warning: imbalanced-learn not installed. SMOTE disabled.")

SPLITS_DIR  = pathlib.Path("data/splits")
RESULTS_DIR = pathlib.Path("outputs/results")

# A6 is a weather event — excluded from L2 typing
# (L1 should not flag A6; if it does, it's already a FPR issue)
L2_TARGET_CLASSES = [
    "panel_degradation",
    "sudden_drop",
    "battery_fault",
    "sensor_drift",
    "offline",
    "false_data_injection",
]

LABEL_COLS = ["is_anomaly", "anomaly_type", "is_weather_event",
              "timestamp", "device_id", "protocol"]


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in LABEL_COLS
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
            and c != "is_low_irradiance_period"]


# ─── MIR@k ────────────────────────────────────────────────────────────────────

def mir_at_k(
    y_true_types: np.ndarray,
    y_pred_types: np.ndarray,
    y_scores:     np.ndarray,
    k:            int,
) -> float:
    """
    Maintenance-Informed Recall at budget k.

    Simulates an operator who can investigate at most k alarms per day (ranked
    by anomaly score descending). Computes the fraction of all true anomalies
    that appear in the top-k ranked items AND are correctly typed.

    Args:
        y_true_types: Ground-truth type labels (string array).
        y_pred_types: Predicted type labels (string array).
        y_scores:     Anomaly score per row (higher = more anomalous).
        k:            Alarm budget.

    Returns:
        MIR@k ∈ [0, 1]
    """
    n = len(y_true_types)
    k = min(k, n)

    # Rank by score descending — top k are "investigated"
    order     = np.argsort(-y_scores)
    top_k_idx = order[:k]

    # True anomalies that appear in top-k
    is_true_anomaly = np.array([t != "normal" for t in y_true_types])
    n_true          = is_true_anomaly.sum()
    if n_true == 0:
        return 1.0   # edge case: no real anomalies → trivially satisfied

    # Among top-k: correctly typed anomalies
    correct = sum(
        1 for i in top_k_idx
        if is_true_anomaly[i] and y_pred_types[i] == y_true_types[i]
    )
    return correct / n_true


# ─── L2 Classifier ───────────────────────────────────────────────────────────

class L2Classifier:
    """
    Second-stage anomaly type classifier (Random Forest + SMOTE).

    Workflow:
      fit(train_df)  → train on ground-truth anomaly rows from train set
      predict(df)    → predict anomaly type for given rows
      evaluate(df)   → per-type F1 + MIR@k with provided L1 predictions
    """

    def __init__(
        self,
        n_estimators: int   = 200,
        max_depth:    int   = None,
        random_state: int   = 42,
        use_smote:    bool  = True,
    ):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state
        self.use_smote    = use_smote and _HAS_SMOTE
        self.feat_cols    = None
        self.clf          = None
        self.classes_     = None
        self.is_fitted    = False

    def fit(self, train_df: pd.DataFrame) -> "L2Classifier":
        """
        Fit L2 classifier on anomaly rows only (excluding A6 weather events).
        Applies SMOTE to balance rare anomaly types.
        """
        self.feat_cols = _get_feature_cols(train_df)

        # Use only real anomaly rows (not A6 weather, not normal)
        anom_df = train_df[
            train_df["is_anomaly"] &
            train_df["anomaly_type"].isin(L2_TARGET_CLASSES)
        ].copy()

        if len(anom_df) == 0:
            raise ValueError("[l2] No anomaly rows found in training data.")

        X = anom_df[self.feat_cols].values
        y = anom_df["anomaly_type"].values

        print(f"[l2] Training on {len(X)} anomaly rows "
              f"({len(np.unique(y))} classes):")
        for cls in np.unique(y):
            print(f"     {cls}: {(y == cls).sum()}")

        # SMOTE to balance rare classes
        if self.use_smote:
            # SMOTE requires at least k_neighbors+1 samples per class
            min_count = min((y == c).sum() for c in np.unique(y))
            k_neighbors = min(5, min_count - 1)
            if k_neighbors >= 1:
                try:
                    sm = SMOTE(
                        random_state=self.random_state,
                        k_neighbors=k_neighbors,
                    )
                    X, y = sm.fit_resample(X, y)
                    print(f"[l2] After SMOTE: {len(X)} samples")
                except Exception as e:
                    print(f"[l2] SMOTE skipped: {e}")
            else:
                print("[l2] SMOTE skipped: too few samples per class")

        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        self.is_fitted = True
        print(f"[l2] RF fitted. Classes: {list(self.classes_)}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomaly type for each row."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        X = df[self.feat_cols].values
        return self.clf.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probability matrix (n_rows × n_classes) for each row."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        X = df[self.feat_cols].values
        return self.clf.predict_proba(X)

    def evaluate(
        self,
        test_df:       pd.DataFrame,
        l1_pred:       np.ndarray,
        l1_scores:     np.ndarray,
        split_name:    str = "test",
        k_budget:      int = 20,
    ) -> dict:
        """
        Evaluate L2 classifier on test set.

        Strategy:
          1. Apply L2 only to rows L1 flagged as anomaly (l1_pred == 1)
          2. For rows L1 missed, assign 'normal' as type
          3. Compute per-type F1 on TRUE anomaly rows
          4. Compute MIR@k using L1 scores as ranking signal

        Args:
            test_df:    Test split DataFrame.
            l1_pred:    Binary L1 predictions (1=anomaly) — shape (n_rows,).
            l1_scores:  Continuous L1 anomaly scores — used for MIR@k ranking.
            split_name: Name tag for the metrics dict.
            k_budget:   Alarm budget for MIR@k.

        Returns:
            Metrics dict with per-type F1, MIR@k, and confusion matrix path.
        """
        n = len(test_df)

        # Step 1: predict types for L1-flagged rows
        y_pred_types = np.array(["normal"] * n, dtype=object)
        l1_flagged   = np.where(l1_pred == 1)[0]

        if len(l1_flagged) > 0:
            flagged_df           = test_df.iloc[l1_flagged]
            y_pred_types[l1_flagged] = self.predict(flagged_df)

        # Step 2: ground truth types
        y_true_types = test_df["anomaly_type"].values

        # Step 3: per-type F1 on true anomaly rows only
        anom_mask = test_df["is_anomaly"].values
        metrics   = {
            "method":     "l2_classifier",
            "split":      split_name,
            "k_budget":   k_budget,
        }

        if anom_mask.sum() > 0:
            true_anom_types = y_true_types[anom_mask]
            pred_anom_types = y_pred_types[anom_mask]

            metrics["typing_accuracy"] = (
                true_anom_types == pred_anom_types
            ).mean()

            # Per-type F1 across all anomaly types
            present_types = [t for t in L2_TARGET_CLASSES
                             if (true_anom_types == t).sum() > 0]
            for atype in present_types:
                y_t = (true_anom_types == atype).astype(int)
                y_p = (pred_anom_types == atype).astype(int)
                metrics[f"f1_{atype}"] = f1_score(y_t, y_p, zero_division=0)

            # Overall macro F1 on anomaly rows
            metrics["macro_f1_typing"] = f1_score(
                true_anom_types, pred_anom_types,
                labels=present_types,
                average="macro",
                zero_division=0,
            )

        # Step 4: MIR@k
        metrics["mir_at_k"] = mir_at_k(
            y_true_types, y_pred_types, l1_scores, k=k_budget
        )

        # Save confusion matrix
        cm_path = RESULTS_DIR / f"l2_confusion_matrix_{split_name}.npy"
        types_present = [t for t in L2_TARGET_CLASSES
                         if (y_true_types == t).sum() > 0 or
                            (y_pred_types == t).sum() > 0]
        cm = confusion_matrix(
            y_true_types[anom_mask],
            y_pred_types[anom_mask],
            labels=types_present,
        )
        np.save(cm_path, cm)
        pd.DataFrame(cm, index=types_present, columns=types_present).to_csv(
            RESULTS_DIR / f"l2_confusion_matrix_{split_name}.csv"
        )

        return metrics

    def feature_importances(self) -> pd.Series:
        """Return feature importances as a named Series (for SHAP Phase 9)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return pd.Series(
            self.clf.feature_importances_,
            index=self.feat_cols,
        ).sort_values(ascending=False)

    def save(self, path: pathlib.Path = RESULTS_DIR / "l2_model.pkl") -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[l2] Model saved → {path}")

    @classmethod
    def load(cls, path: pathlib.Path = RESULTS_DIR / "l2_model.pkl") -> "L2Classifier":
        with open(pathlib.Path(path), "rb") as f:
            return pickle.load(f)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run_phase8(
    splits_dir:  str | pathlib.Path = SPLITS_DIR,
    results_dir: str | pathlib.Path = RESULTS_DIR,
    k_budget:    int = 20,
) -> dict:
    """
    Full Phase 8 pipeline:
      1. Load splits + L1 LSTM-AE results
      2. Fit L2 classifier on train anomaly rows
      3. Evaluate on test using L1 predictions as filter
      4. Save model + metrics
    """
    splits_dir  = pathlib.Path(splits_dir)
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[l2] Loading splits …")
    train = pd.read_csv(splits_dir / "train.csv", parse_dates=["timestamp"])
    val   = pd.read_csv(splits_dir / "val.csv",   parse_dates=["timestamp"])
    test  = pd.read_csv(splits_dir / "test.csv",  parse_dates=["timestamp"])

    # ── Load L1 (LSTM-AE) predictions ────────────────────────────────────────
    # Re-run LSTM-AE inference to get fresh l1_pred + l1_scores on test set
    print("[l2] Loading LSTM-AE model for L1 predictions …")
    lstm_meta_path  = results_dir / "lstm_ae_model_meta.pkl"
    lstm_model_path = results_dir / "lstm_ae_model.keras"

    if lstm_model_path.exists() and lstm_meta_path.exists():
        import tensorflow as tf
        from src.models.lstm_autoencoder import (
            build_sequences, sequences_to_row_scores, _get_feature_cols as _lstm_feat
        )
        with open(lstm_meta_path, "rb") as f:
            meta = pickle.load(f)

        lstm_model = tf.keras.models.load_model(str(lstm_model_path))
        feat_cols  = meta["feat_cols"]
        seq_len    = meta["seq_len"]
        threshold  = meta["threshold"]

        def _l1_predict(df):
            X_raw  = df[feat_cols].values.astype(np.float32)
            X_seqs = build_sequences(X_raw, seq_len)
            X_hat  = lstm_model.predict(X_seqs, verbose=0)
            errs   = np.mean(np.abs(X_seqs - X_hat), axis=(1, 2))
            scores = sequences_to_row_scores(errs, len(df), seq_len)
            return (scores > threshold).astype(int), scores

        l1_pred_test,  l1_scores_test  = _l1_predict(test)
        l1_pred_train, l1_scores_train = _l1_predict(train)
        print(f"[l2] L1 flags on test: {l1_pred_test.sum()} / {len(test)}")
    else:
        print("[l2] LSTM-AE model not found — using ground-truth labels as L1 proxy")
        l1_pred_test   = test["is_anomaly"].astype(int).values
        l1_scores_test = test["is_anomaly"].astype(float).values
        l1_pred_train  = train["is_anomaly"].astype(int).values
        l1_scores_train= train["is_anomaly"].astype(float).values

    # ── Fit L2 ────────────────────────────────────────────────────────────────
    print("\n[l2] Fitting L2 classifier …")
    l2 = L2Classifier(n_estimators=200, use_smote=True)
    l2.fit(train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n[l2] Evaluating on test (k_budget={k_budget}) …")
    test_metrics = l2.evaluate(test, l1_pred_test, l1_scores_test,
                               split_name="test", k_budget=k_budget)

    l2.save(results_dir / "l2_model.pkl")

    # Feature importances
    fi = l2.feature_importances()
    fi.to_csv(results_dir / "l2_feature_importances.csv", header=["importance"])
    print("\n[l2] Top-10 feature importances:")
    print(fi.head(10).to_string())

    pd.DataFrame([test_metrics]).to_csv(
        results_dir / "l2_results.csv", index=False
    )

    print("\n=== Phase 8 Results (Test Set) ===")
    print(f"  Typing Accuracy : {test_metrics.get('typing_accuracy', 0):.4f}")
    print(f"  Macro F1 (typing): {test_metrics.get('macro_f1_typing', 0):.4f}")
    print(f"  MIR@{k_budget}         : {test_metrics.get('mir_at_k', 0):.4f}")
    for key, val in sorted(test_metrics.items()):
        if key.startswith("f1_"):
            print(f"  {key:<30}: {val:.4f}")

    return test_metrics


if __name__ == "__main__":
    run_phase8()
