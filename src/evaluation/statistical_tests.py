"""
Phase 11 — Statistical Tests & Results Compilation
Creates publication-ready tables and performs McNemar's test.

Produces:
- Table I: F1 per anomaly type per method
- Table II: AUC, ADL, Inference Time, FPR@A6, Model Size
- Table III: Best hyperparameters from sweeps

Reference: SCIO Research Framework §9.2
"""

import pathlib
import pickle
import warnings
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src import config
from src.config import get_feature_cols

warnings.filterwarnings("ignore")

SPLITS_DIR = config.SPLITS_DIR
RESULTS_DIR = config.RESULTS_DIR

# ─── McNemar's Test ───────────────────────────────────────────────────────────

def mcnemar_test(y_true, y_pred1, y_pred2) -> tuple[float, float]:
    """
    Computes McNemar's test with continuity correction.
    Returns (chi2_stat, p_value)
    """
    correct1 = (y_true == y_pred1)
    correct2 = (y_true == y_pred2)
    
    # b: Model 1 correct, Model 2 incorrect
    b = np.sum(correct1 & ~correct2)
    # c: Model 1 incorrect, Model 2 correct
    c = np.sum(~correct1 & correct2)
    
    if b + c == 0:
        return 0.0, 1.0 # Identical predictions
        
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_value = 1.0 - chi2.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_value)

def _get_l1_test_predictions(test_df: pd.DataFrame) -> dict:
    import tensorflow as tf
    results = {}
    
    # 1. Truth
    y_true = test_df["is_anomaly"].values

    feat_cols = get_feature_cols(test_df)
    X = test_df[feat_cols].values
    
    # 2. Rule based
    rb_path = RESULTS_DIR / "rule_based_model.json"
    if rb_path.exists():
        from src.models.rule_based import RuleBasedDetector
        rb_rules = RuleBasedDetector.load(rb_path)
        results["Rule-Based"] = rb_rules.predict(test_df)
    
    # 3. Isolation Forest
    if_path = RESULTS_DIR / "isolation_forest_model.pkl"
    if if_path.exists():
        with open(if_path, "rb") as f:
            if_model = pickle.load(f)
            pred = if_model.predict(X)
            # IF output: 1 normal, -1 anomaly
            results["Isolation Forest"] = (pred == -1)

    # 4. LSTM AE
    lstm_path = RESULTS_DIR / "lstm_ae_model.keras"
    meta_path = RESULTS_DIR / "lstm_ae_model_meta.json"
    if lstm_path.exists() and meta_path.exists():
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        model = tf.keras.models.load_model(str(lstm_path))
        with open(meta_path, "r") as f:
            meta = json.load(f)

        seq_len = meta["seq_len"]
        threshold = meta["threshold"]
        feat_cols_lstm = meta["feat_cols"]
        X_lstm = test_df[feat_cols_lstm].values.astype(np.float32)

        from src.models.lstm_autoencoder import build_sequences, sequences_to_row_scores

        seqs = build_sequences(X_lstm, seq_len)
        X_hat = model.predict(seqs, verbose=0, batch_size=128)
        seq_err = np.mean(np.abs(seqs - X_hat), axis=(1, 2))
        row_scores = sequences_to_row_scores(seq_err, len(test_df), seq_len)
        results["LSTM-AE"] = (row_scores > threshold)

    return y_true, results

# ─── Table Compilation ────────────────────────────────────────────────────────

def compile_table_I() -> pd.DataFrame:
    """Table I: F1 per Anomaly Type"""
    files = {
        "Rule-Based": "rule_based_results.csv",
        "Isolation Forest": "classical_ml_results.csv",
        "LSTM-AE": "lstm_ae_results.csv",
        "L2 (RF)": "l2_results.csv" # L2 gives f1_... for types directly!
    }
    
    rows = []
    
    for method, fn in files.items():
        p = RESULTS_DIR / fn
        if not p.exists(): continue
        
        df = pd.read_csv(p)
        col_method = "method" if "method" in df.columns else "Method"
        
        if method == "Isolation Forest" and col_method in df.columns:
            df = df[df[col_method] == "Isolation Forest"]
            
        if len(df) == 0: continue
        row = df.iloc[0].to_dict()
        
        # Extract f1 columns
        rec = {"Method": method}
        for k, v in row.items():
            if k.startswith("f1_"):
                anom = k.replace("f1_", "")
                rec[anom] = v
        rows.append(rec)
        
    if not rows:
        return pd.DataFrame()
        
    return pd.DataFrame(rows).fillna("-")

def compile_table_II() -> pd.DataFrame:
    """Table II: Overall System performance and edge deployment footprint"""
    
    methods = ["Rule-Based", "Isolation Forest", "LSTM-AE"]
    files = {
        "Rule-Based": "rule_based_results.csv",
        "Isolation Forest": "classical_ml_results.csv",
        "LSTM-AE": "lstm_ae_results.csv"
    }
    
    edge_p = RESULTS_DIR / "edge_profiling_results.csv"
    edge_df = pd.read_csv(edge_p) if edge_p.exists() else pd.DataFrame()
    
    rows = []
    for method, fn in files.items():
        p = RESULTS_DIR / fn
        if not p.exists(): continue
        
        df = pd.read_csv(p)
        if df.empty: continue
        
        col_method = "method" if "method" in df.columns else "Method"
        if col_method in df.columns:
            r = df[df[col_method].str.contains(method, na=False)]
            if len(r) == 0: r = df.iloc[[0]]
        else:
            r = df.iloc[[0]]
            
        r = r.iloc[0]
        
        rec = {
            "Method": method,
            "Macro_F1": r.get("Macro_F1", r.get("f1", r.get("F1", np.nan))),
            "Precision": r.get("global_precision", r.get("precision", r.get("Precision", np.nan))),
            "Recall": r.get("global_recall", r.get("recall", r.get("Recall", np.nan))),
            "FPR@A6": r.get("fpr_a6", r.get("FPR_A6", np.nan)),
        }
        
        # Try to join edge results
        if not edge_df.empty and "Method" in edge_df.columns:
            e_r = edge_df[edge_df["Method"].str.startswith(method)]
            if len(e_r) > 0:
                rec["Latency_ms"] = e_r.iloc[0].get("Latency_ms", np.nan)
                rec["Peak_RAM_MB"] = e_r.iloc[0].get("Peak_RAM_MB", np.nan)
                rec["Size_KB"] = e_r.iloc[0].get("Size_KB", np.nan)
                
        rows.append(rec)
        
    return pd.DataFrame(rows)

def compile_table_III() -> pd.DataFrame:
    """Table III: Best Model Hyperparameters"""
    rows = []
    
    if (RESULTS_DIR / "rule_based_k_sweep.csv").exists():
        df = pd.read_csv(RESULTS_DIR / "rule_based_k_sweep.csv")
        target_col = "Macro_F1" if "Macro_F1" in df.columns else "f1"
        try:
            best = df.iloc[df[target_col].idxmax()]
            rows.append({"Method": "Rule-Based", "Hyperparameters": f"MAD k={best.get('k', '?')}"})
        except (KeyError, ValueError, IndexError) as e:
            print(f"[stats] Warning: could not parse rule_based_k_sweep.csv: {e}")

    if (RESULTS_DIR / "if_sweep.csv").exists():
        df = pd.read_csv(RESULTS_DIR / "if_sweep.csv")
        target_col = "valid_f1" if "valid_f1" in df.columns else "f1"
        try:
            best = df.iloc[df[target_col].idxmax()]
            rows.append({"Method": "Isolation Forest", "Hyperparameters": f"contamination={best.get('contamination', '?')}"})
        except (KeyError, ValueError, IndexError) as e:
            print(f"[stats] Warning: could not parse if_sweep.csv: {e}")

    if (RESULTS_DIR / "lof_sweep.csv").exists():
        df = pd.read_csv(RESULTS_DIR / "lof_sweep.csv")
        target_col = "valid_f1" if "valid_f1" in df.columns else "f1"
        try:
            best = df.iloc[df[target_col].idxmax()]
            rows.append({"Method": "Local Outlier Factor", "Hyperparameters": f"neighbors={int(best.get('n_neighbors', 0))}, cont={best.get('contamination', '?')}"})
        except (KeyError, ValueError, IndexError) as e:
            print(f"[stats] Warning: could not parse lof_sweep.csv: {e}")

    if (RESULTS_DIR / "lstm_ae_model_meta.json").exists():
        with open(RESULTS_DIR / "lstm_ae_model_meta.json", "r") as f:
            meta = json.load(f)
        rows.append({"Method": "LSTM-AE", "Hyperparameters": f"seq_len={meta.get('seq_len', '')}, threshold={meta.get('threshold', 0):.4f}"})

    return pd.DataFrame(rows)


def run_phase11_stats():
    """Run all Phase 11 stats logic."""
    print("[stats] Compiling Phase 11 Tables...")
    
    t1 = compile_table_I()
    t2 = compile_table_II()
    t3 = compile_table_III()
    
    print("\n" + "="*80)
    print(" TABLE I: F1 Score Breakdown per Anomaly Type")
    print("="*80)
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
    
    print("\n" + "="*80)
    print(" TABLE II: Overall Metrics & Deployment Footprint")
    print("="*80)
    print(t2.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
    
    print("\n" + "="*80)
    print(" TABLE III: Best Hyperparameters")
    print("="*80)
    print(t3.to_string(index=False))
    print("="*80 + "\n")
    
    # Save Tables
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t1.to_csv(RESULTS_DIR / "Table_1_F1_Per_Class.csv", index=False)
    t2.to_csv(RESULTS_DIR / "Table_2_Overall_Metrics.csv", index=False)
    t3.to_csv(RESULTS_DIR / "Table_3_Hyperparams.csv", index=False)
    
    print("[stats] Tables saved to outputs/results/")
    
    # McNemar tests
    print("\n[stats] Running McNemar Statistical Tests...")
    test_path = SPLITS_DIR / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path, parse_dates=["timestamp"])
        y_true, preds = _get_l1_test_predictions(test_df)
        
        if "Rule-Based" in preds and "Isolation Forest" in preds:
            s, p = mcnemar_test(y_true, preds["Rule-Based"], preds["Isolation Forest"])
            print(f"  Rule-Based vs Isolation Forest: chi2={s:.3f}, p_value={p:.3e}")
            if p < 0.05: print("    -> Statistically Significant difference (p < 0.05)")
            
        if "Isolation Forest" in preds and "LSTM-AE" in preds:
            # Need to align lengths due to LSTM seq drop
            y_lstmae = preds["LSTM-AE"]
            n = len(y_lstmae)
            y_true_cut = y_true[-n:]
            y_if_cut = preds["Isolation Forest"][-n:]
            s, p = mcnemar_test(y_true_cut, y_if_cut, y_lstmae)
            print(f"  Isolation Forest vs LSTM-AE : chi2={s:.3f}, p_value={p:.3e}")
            if p < 0.05: print("    -> Statistically Significant difference (p < 0.05)")
    else:
        print("[stats] Error: Cannot find test.csv to run McNemar tests.")


if __name__ == "__main__":
    run_phase11_stats()
