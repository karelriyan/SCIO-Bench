"""
Phase 5 — Method A: Rule-Based Threshold Detector
Implements 7 interpretable detection rules with MAD-calibrated thresholds.

Rules (R1–R7):
  R1: mppt_w < 0 + epsilon               → A2 sudden drop (negative power)
  R2: |mppt_w_delta| > MAD_threshold     → A1/A2 sudden change in power
  R3: batt_delta < -threshold (rapid)    → A3 battery fault (fast drain)
  R4: batt_pct stuck (std=0 over window) → A3 battery fault (frozen BMS)
  R5: |volt_v - volt_v_lag1| > threshold → A4 sensor drift
  R6: physics_residual > threshold       → A7 FDI (P ≠ V×I)
  R7: mppt_w > 0 and irradiance < eps    → A1/A2 generation without sunlight

Threshold calibration:
  All thresholds are computed from the TRAINING split using:
    threshold = median(|x - median(x)|) × k  (MAD × k)
  The multiplier k is the primary hyperparameter tuned on val set.

Reference: SCIO Research Framework §7.1
"""

import pathlib
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

SPLITS_DIR   = pathlib.Path("data/splits")
RESULTS_DIR  = pathlib.Path("outputs/results")
MODEL_DIR    = pathlib.Path("outputs/results")

# ─── Threshold Dataclass ─────────────────────────────────────────────────────

@dataclass
class RuleThresholds:
    """Calibrated thresholds for each rule (fitted on train)."""
    r1_mppt_w_min:       float = 0.0        # R1: negative power (always 0)
    r2_power_delta:      float = 500.0      # R2: power change per tick (W)
    r3_batt_rapid_drain: float = 5.0        # R3: SOC drop % per tick
    r4_batt_stuck_std:   float = 0.01       # R4: batt std over 4 ticks
    r5_volt_jump:        float = 1.5        # R5: voltage jump (V)
    r6_physics_residual: float = 200.0      # R6: |mppt_w - V×I| threshold (W)
    r7_irradiance_eps:   float = 10.0       # R7: irradiance below which power=0 expected
    r7_mppt_w_min:       float = 100.0      # R7: min power to flag as suspicious
    k_multiplier:        float = 3.5        # MAD × k for R2/R3/R5/R6

    # Runtime — how many rules triggered each prediction
    rule_votes: dict = field(default_factory=dict)


# ─── MAD Helper ──────────────────────────────────────────────────────────────

def _mad_threshold(series: np.ndarray, k: float = 3.5, eps: float = 1e-5) -> float:
    """
    Median Absolute Deviation threshold.
    threshold = median(|x - median(x)|) × k

    Robust to outliers and distribution-free (no Gaussian assumption).
    """
    med = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - med))
    mad = max(mad, eps)  # Prevent zero-variance collapse (e.g. constant signals)
    return float(med + k * mad)


# ─── Threshold Calibration ───────────────────────────────────────────────────

def fit_thresholds(
    train_df: pd.DataFrame,
    k: float = 3.5,
) -> RuleThresholds:
    """
    Fit MAD-based thresholds from TRAINING data only.
    Only uses rows labeled 'normal' to avoid contamination from known anomalies.

    Args:
        train_df: Training split DataFrame (output of feature_engineering).
        k:        MAD multiplier. Higher k → fewer false positives.

    Returns:
        RuleThresholds with calibrated values.
    """
    # Use only normal rows for threshold calibration (pure distribution)
    normal = train_df[train_df["anomaly_type"] == "normal"].copy()

    t = RuleThresholds(k_multiplier=k)

    # R1: always 0 (negative power is definitively wrong)
    t.r1_mppt_w_min = 0.0

    # R2: power change per tick — MAD on absolute delta
    if "mppt_w_delta" in normal.columns:
        t.r2_power_delta = _mad_threshold(normal["mppt_w_delta"].abs().values, k)
    else:
        # Compute from raw values if delta not present
        t.r2_power_delta = _mad_threshold(
            normal["mppt_w"].diff().abs().dropna().values, k
        )

    # R3: rapid battery drain — MAD on negative batt_delta
    if "batt_delta" in normal.columns:
        drain = -normal["batt_delta"].values   # positive = draining
        t.r3_batt_rapid_drain = _mad_threshold(drain[drain > 0], k) if (drain > 0).any() else 5.0
    else:
        t.r3_batt_rapid_drain = 5.0

    # R4: stuck battery — threshold stays fixed (near-zero std = stuck)
    t.r4_batt_stuck_std = 0.01

    # R5: voltage jump — MAD on absolute voltage delta
    if "volt_v_delta" in normal.columns:
        t.r5_volt_jump = _mad_threshold(normal["volt_v_delta"].abs().values, k)
    else:
        t.r5_volt_jump = _mad_threshold(
            normal["volt_v"].diff().abs().dropna().values, k
        )

    # R6: physics residual — MAD on |mppt_w - V×I|
    if "physics_residual" in normal.columns:
        t.r6_physics_residual = _mad_threshold(
            normal["physics_residual"].abs().values, k
        )
    else:
        residual = (normal["mppt_w"] - normal["volt_v"] * normal["curr_a"]).abs()
        t.r6_physics_residual = _mad_threshold(residual.values, k)

    # R7: fixed irradiance epsilon (sensor noise floor)
    t.r7_irradiance_eps = 10.0
    t.r7_mppt_w_min     = 100.0   # 100W minimum to flag

    return t


# ─── Rule Evaluations ────────────────────────────────────────────────────────

def _apply_rules(df: pd.DataFrame, t: RuleThresholds) -> pd.Series:
    """
    Apply all 7 rules to a DataFrame.

    Returns:
        Boolean Series — True = anomaly flagged.
    """
    n = len(df)
    flags = np.zeros(n, dtype=bool)

    is_night_mask = df["is_night"].values == 1 if "is_night" in df.columns else np.zeros(n, dtype=bool)

    # R1: Negative power output
    if "mppt_w" in df.columns:
        r1_flags = df["mppt_w"].values < -1.0   # allow -1W sensor noise
        r1_flags &= ~is_night_mask
        flags |= r1_flags

    # R2: Sudden power change (>MAD×k)
    if "mppt_w_delta" in df.columns:
        r2_flags = df["mppt_w_delta"].abs().values > t.r2_power_delta
        r2_flags &= ~is_night_mask
        flags |= r2_flags

    # R3: Rapid battery drain (SOC drops >threshold per tick)
    if "batt_delta" in df.columns:
        flags |= df["batt_delta"].values < -t.r3_batt_rapid_drain

    # R4: Battery stuck (batt_delta ≈ 0 continuously) — proxy: |batt_delta| very small
    #     AND batt_pct not at typical overnight level (distinguish from stable charging)
    if "batt_delta" in df.columns and "batt_pct" in df.columns:
        stuck_mask = (
            (df["batt_delta"].abs().values < t.r4_batt_stuck_std) &
            (df["batt_pct"].values > 5.0) &
            (df["batt_pct"].values < 99.0)
        )
        flags |= stuck_mask

    # R5: Voltage jump/drift
    if "volt_v_delta" in df.columns:
        r5_flags = df["volt_v_delta"].abs().values > t.r5_volt_jump
        r5_flags &= ~is_night_mask
        flags |= r5_flags

    # R6: Physics residual spike → FDI
    if "physics_residual" in df.columns:
        flags |= df["physics_residual"].abs().values > t.r6_physics_residual

    # R7: Generation during darkness (irradiance ≈ 0 but power > threshold)
    if "irradiance" in df.columns and "mppt_w" in df.columns:
        flags |= (
            (df["irradiance"].values < t.r7_irradiance_eps) &
            (df["mppt_w"].values > t.r7_mppt_w_min)
        )

    return pd.Series(flags, index=df.index)


# ─── Detector Class ──────────────────────────────────────────────────────────

class RuleBasedDetector:
    """
    Deterministic rule-based anomaly detector.
    Fits thresholds on training data; predicts on any split.
    """

    def __init__(self, k: float = 3.5):
        self.k          = k
        self.thresholds: Optional[RuleThresholds] = None
        self.is_fitted   = False

    def fit(self, train_df: pd.DataFrame) -> "RuleBasedDetector":
        """Calibrate thresholds from training data."""
        self.thresholds = fit_thresholds(train_df, k=self.k)
        self.is_fitted = True
        print(f"[rule_based] Thresholds calibrated (k={self.k:.1f}):")
        print(f"  R2 power_delta : {self.thresholds.r2_power_delta:8.2f} W")
        print(f"  R3 batt_drain  : {self.thresholds.r3_batt_rapid_drain:8.2f} %/tick")
        print(f"  R5 volt_jump   : {self.thresholds.r5_volt_jump:8.4f} V")
        print(f"  R6 phys_resid  : {self.thresholds.r6_physics_residual:8.2f} W")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary anomaly predictions (1=anomaly, 0=normal)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        flags = _apply_rules(df, self.thresholds)
        return flags.astype(int).values

    def predict_proba_proxy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Rule-count proxy for 'anomaly probability'.
        Counts how many rules fire per row (max 7), normalises to [0, 1].
        Used for ROC curve generation.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        t    = self.thresholds
        cols = {
            "r1": ("mppt_w",          lambda x: (x < -1.0).astype(int)),
            "r2": ("mppt_w_delta",    lambda x: (x.abs() > t.r2_power_delta).astype(int)),
            "r3": ("batt_delta",      lambda x: (x < -t.r3_batt_rapid_drain).astype(int)),
            "r5": ("volt_v_delta",    lambda x: (x.abs() > t.r5_volt_jump).astype(int)),
            "r6": ("physics_residual",lambda x: (x.abs() > t.r6_physics_residual).astype(int)),
        }
        scores = np.zeros(len(df))
        is_night_mask = df["is_night"].values == 1 if "is_night" in df.columns else np.zeros(len(df), dtype=bool)

        for name, (col, fn) in cols.items():
            if col in df.columns:
                res = fn(df[col]).values.copy()
                if name in ["r1", "r2", "r5"]:
                    res[is_night_mask] = 0
                scores += res
        return scores / 7.0   # normalise to [0,1]

    def evaluate(
        self,
        df: pd.DataFrame,
        split_name: str = "test",
    ) -> dict:
        """
        Run prediction and compute full metrics.

        Returns dict with: f1, precision, recall, fpr, n_anomalies_predicted,
                           fpr_a6 (false positive rate on A6 rows only).
        """
        y_pred = self.predict(df)
        y_true = df["is_anomaly"].astype(int).values

        # FPR@A6 — FPR computed ONLY on A6 (normal weather) rows
        a6_mask = (df["anomaly_type"] == "low_irradiance").values
        normal_mask = (~df["is_anomaly"].values)

        fpr_a6 = 0.0
        if a6_mask.sum() > 0:
            fpr_a6 = y_pred[a6_mask].mean()    # fraction of A6 flagged as anomaly

        # Global FPR on all normal rows
        fpr_global = 0.0
        if normal_mask.sum() > 0:
            fpr_global = y_pred[normal_mask].mean()

        metrics = {
            "method":        "rule_based",
            "split":         split_name,
            "f1":            f1_score(y_true, y_pred, zero_division=0),
            "precision":     precision_score(y_true, y_pred, zero_division=0),
            "recall":        recall_score(y_true, y_pred, zero_division=0),
            "fpr_global":    fpr_global,
            "fpr_a6":        fpr_a6,
            "n_predicted":   int(y_pred.sum()),
            "n_true":        int(y_true.sum()),
            "k":             self.k,
        }

        # Per-type F1 (for Table I in paper)
        for atype in df["anomaly_type"].unique():
            mask = (df["anomaly_type"] == atype).values
            if mask.sum() == 0:
                continue
            y_t = (df.loc[mask, "is_anomaly"]).astype(int).values
            y_p = y_pred[mask]
            metrics[f"f1_{atype}"] = f1_score(y_t, y_p, zero_division=0)

        return metrics

    def save(self, path: pathlib.Path = MODEL_DIR / "rule_based_model.pkl") -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[rule_based] Model saved → {path}")

    @classmethod
    def load(cls, path: pathlib.Path = MODEL_DIR / "rule_based_model.pkl") -> "RuleBasedDetector":
        with open(pathlib.Path(path), "rb") as f:
            return pickle.load(f)


# ─── k-sweep Val Tuning ──────────────────────────────────────────────────────

def tune_k_on_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    k_values: list[float] = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
) -> tuple[float, pd.DataFrame]:
    """
    Grid search over k values using val F1 score.

    Returns:
        (best_k, results_df)
    """
    rows = []
    for k in k_values:
        det = RuleBasedDetector(k=k)
        det.fit(train_df)
        m = det.evaluate(val_df, split_name="val")
        rows.append(m)
        print(f"  k={k:.1f} | F1={m['f1']:.4f} | P={m['precision']:.4f} "
              f"| R={m['recall']:.4f} | FPR@A6={m['fpr_a6']:.4f}")

    results_df = pd.DataFrame(rows)
    best_k = results_df.loc[results_df["f1"].idxmax(), "k"]
    print(f"\n[rule_based] Best k={best_k} (val F1={results_df['f1'].max():.4f})")
    return float(best_k), results_df


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run_phase5(
    splits_dir:  str | pathlib.Path = SPLITS_DIR,
    results_dir: str | pathlib.Path = RESULTS_DIR,
) -> dict:
    """
    Full Phase 5 pipeline:
      1. Load train/val/test splits
      2. Tune k on val
      3. Refit on train with best k
      4. Evaluate on test
      5. Save model + results

    Returns: test metrics dict
    """
    splits_dir  = pathlib.Path(splits_dir)
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for f in ("train.csv", "val.csv", "test.csv"):
        if not (splits_dir / f).exists():
            raise FileNotFoundError(
                f"[rule_based] {f} not found. Run Phase 4 first:\n"
                "  python -m src.data.feature_engineering"
            )

    print("[rule_based] Loading splits …")
    train = pd.read_csv(splits_dir / "train.csv", parse_dates=["timestamp"])
    val   = pd.read_csv(splits_dir / "val.csv",   parse_dates=["timestamp"])
    test  = pd.read_csv(splits_dir / "test.csv",  parse_dates=["timestamp"])

    print(f"[rule_based] Train={len(train):,} Val={len(val):,} Test={len(test):,}")

    # k tuning
    print("\n[rule_based] Tuning k on validation set …")
    best_k, tune_df = tune_k_on_val(train, val)

    # Save tuning results
    tune_df.to_csv(results_dir / "rule_based_k_sweep.csv", index=False)

    # Refit with best k and evaluate on test
    print(f"\n[rule_based] Fitting with best k={best_k} …")
    detector = RuleBasedDetector(k=best_k)
    detector.fit(train)

    test_metrics = detector.evaluate(test, split_name="test")
    detector.save(results_dir / "rule_based_model.pkl")

    # Save test metrics
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(results_dir / "rule_based_results.csv", index=False)

    print("\n=== Phase 5 Results (Test Set) ===")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  FPR@A6:    {test_metrics['fpr_a6']:.4f}")
    print(f"  Predicted: {test_metrics['n_predicted']} anomalies "
          f"(true: {test_metrics['n_true']})")

    return test_metrics


if __name__ == "__main__":
    run_phase5()
