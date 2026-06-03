"""
Phase 9 — XAI: SHAP Analysis
Explainability for classical machine learning models.

Uses SHAP (SHapley Additive exPlanations) to explain predictions.
Specifically, uses TreeExplainer for the Isolation Forest model.
Saves local explanations (anomaly instances) and global feature importance.

Reference: SCIO Research Framework §8.1
"""

import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd

from src import config

warnings.filterwarnings("ignore")

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    print("[shap_xai] Warning: shap library not installed. Running dummy mode.")

SPLITS_DIR  = config.SPLITS_DIR
RESULTS_DIR = config.RESULTS_DIR

LABEL_COLS = ["is_anomaly", "anomaly_type", "is_weather_event",
              "timestamp", "device_id", "protocol"]


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in LABEL_COLS
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
            and c != "is_low_irradiance_period"]


def load_isolation_forest():
    """Load trained Isolation Forest from Phase 6."""
    model_path = RESULTS_DIR / "isolation_forest_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"[shap_xai] Model not found: {model_path}")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    print(f"[shap_xai] Loaded Isolation Forest: {clf}")
    return clf


def analyze_shap(
    clf,
    df: pd.DataFrame,
    feat_cols: list[str],
    bg_size: int = 100,
) -> dict:
    """
    Compute SHAP values for anomalies using a subset of normal data as background.

    Args:
        clf:        Trained Isolation Forest
        df:         DataFrame containing anomalies and normal rows
        feat_cols:  List of feature columns
        bg_size:    Number of normal samples to use as background/baseline

    Returns:
        dict with top features globally and locally.
    """
    if not _HAS_SHAP:
        print("[shap_xai] SHAP not available. Returning dummy importance.")
        return {"global_top_features": feat_cols[:3]}

    # 1. Background dataset (normal rows only)
    normal_df = df[df["anomaly_type"] == "normal"]
    if len(normal_df) > bg_size:
        bg_df = normal_df.sample(n=bg_size, random_state=42)
    else:
        bg_df = normal_df
    X_bg = bg_df[feat_cols].values

    # 2. Target dataset (true anomalies)
    anom_df = df[df["anomaly_type"] != "normal"]
    X_anom  = anom_df[feat_cols].values
    y_anom_types = anom_df["anomaly_type"].values

    if len(X_anom) == 0:
        print("[shap_xai] No anomalies found in given dataframe.")
        return {}

    print(f"[shap_xai] Computing SHAP values for {len(X_anom)} anomalies "
          f"(background size = {len(X_bg)}) ...")

    # TreeExplainer for Isolation Forest
    explainer = shap.TreeExplainer(clf)
    
    # explainer.shap_values for IF returns values for anomaly score
    # Lower output = more anomalous (negative values typically).
    shap_vals = explainer.shap_values(X_anom)

    # Scikit-learn IF returns [n_samples] output. shap_vals is usually [n_samples, n_features].
    # Sometimes it wraps in a list for multiclass, but IF is single output.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    # Global feature importance (mean absolute SHAP across all anomalies)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    feat_importance = pd.Series(mean_abs_shap, index=feat_cols).sort_values(ascending=False)
    
    # Save SHAP values for visualization (Phase 12)
    np.save(RESULTS_DIR / "if_shap_values.npy", shap_vals)
    np.save(RESULTS_DIR / "if_shap_features.npy", X_anom)
    with open(RESULTS_DIR / "if_shap_feat_cols.pkl", "wb") as f:
        pickle.dump(feat_cols, f)

    # Local explanations: Top 3 features driving the score per anomaly type
    type_importances = {}
    for atype in np.unique(y_anom_types):
        idx = (y_anom_types == atype)
        type_shap = shap_vals[idx]
        mean_type_abs_shap = np.abs(type_shap).mean(axis=0)
        type_feat_imp = pd.Series(mean_type_abs_shap, index=feat_cols).sort_values(ascending=False)
        type_importances[atype] = type_feat_imp.head(3).index.tolist()

    return {
        "global_importance": feat_importance,
        "type_importances":  type_importances,
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run_phase9_shap(
    splits_dir:  str | pathlib.Path = SPLITS_DIR,
    results_dir: str | pathlib.Path = RESULTS_DIR,
) -> dict:
    """Run Phase 9 SHAP Analysis pipeline."""
    splits_dir  = pathlib.Path(splits_dir)
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[shap_xai] Loading splits …")
    test_df = pd.read_csv(splits_dir / "test.csv", parse_dates=["timestamp"])
    
    feat_cols = _get_feature_cols(test_df)
    
    try:
        clf = load_isolation_forest()
    except FileNotFoundError as e:
        print(f"[shap_xai] {e}. Run Phase 6 first.")
        return {}

    # Run analysis
    res = analyze_shap(clf, test_df, feat_cols)
    
    print("\n=== Phase 9 SHAP Feature Importance (Isolation Forest) ===")
    if "global_importance" in res:
        print("  Global Top 5:")
        print(res["global_importance"].head(5).to_string())
    
    if "type_importances" in res:
        print("\n  Top 3 Features by True Anomaly Type:")
        for atype, cols in res["type_importances"].items():
            print(f"    {atype:<25}: {cols}")

    return res

if __name__ == "__main__":
    run_phase9_shap()
