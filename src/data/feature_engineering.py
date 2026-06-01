"""
Phase 4 — Feature Engineering & Data Splitting
Applies feature transformations and produces chronological splits.

Split ratios (STRICT chronological order — NO shuffle):
  Train: first 70%  → used to fit models + scaler
  Val:   next  15%  → hyperparameter tuning, MAD threshold
  Test:  last  15%  → held-out final evaluation

Features added:
  Lag features (1 and 2 ticks back for key sensors):
    mppt_w_lag1, mppt_w_lag2
    volt_v_lag1, batt_pct_lag1
    irradiance_lag1

  Rolling window (12-tick = 6h, non-leaking via closed='left'):
    mppt_w_mean_6h,  mppt_w_std_6h
    irradiance_mean_6h
    batt_pct_mean_6h

  Delta features (first-order difference):
    mppt_w_delta,   volt_v_delta,   batt_pct_delta,   irradiance_delta

  Time-of-day encodings (cyclic sin/cos — avoids discontinuity at midnight):
    hour_sin, hour_cos,  minute_sin, minute_cos

  Weather event binary flag:
    is_low_irradiance_period  (1 if anomaly_type == 'low_irradiance')

Label columns kept as-is:
  is_anomaly, anomaly_type, is_weather_event

Scaling:
  StandardScaler fit ONLY on train split → transform train/val/test
  Scaler saved to outputs/dataset/scaler.pkl for inference re-use

Output files: data/splits/{train,val,test}.csv
              outputs/dataset/scaler.pkl
"""

import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────

DATASET_PATH = pathlib.Path("outputs/dataset/scio_bench_dataset.csv")
SPLITS_DIR   = pathlib.Path("data/splits")
SCALER_PATH  = pathlib.Path("outputs/dataset/scaler.pkl")

# ─── Split ratios ─────────────────────────────────────────────────────────────

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# Test = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

# ─── Columns ─────────────────────────────────────────────────────────────────

# Columns that are sensor readings — used for lag/rolling/delta
SENSOR_COLS = [
    "mppt_w", "volt_v", "curr_a", "batt_pct",
    "irradiance", "temp_c", "ambient_temp_c",
]

# Label & categorical columns — never scaled
LABEL_COLS = ["is_anomaly", "anomaly_type", "is_weather_event"]
CAT_COLS   = ["device_id", "protocol"]
META_COLS  = ["timestamp"] + LABEL_COLS + CAT_COLS

# Numeric feature columns (to be scaled) — built dynamically after engineering
# (determined at runtime after all features are added)


# ─── Feature Engineering ─────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-tick and 2-tick lag for key sensor columns."""
    df = df.copy()
    lag_cols = ["mppt_w", "volt_v", "batt_pct", "irradiance"]
    for col in lag_cols:
        df[f"{col}_lag1"] = df.groupby("device_id")[col].shift(1)
    df["mppt_w_lag2"] = df.groupby("device_id")["mppt_w"].shift(2)
    # Fill first-row NaNs with the next valid value (cold-start), then zero
    lag_feature_cols = [c for c in df.columns if c.endswith(("_lag1", "_lag2"))]
    df[lag_feature_cols] = df[lag_feature_cols].bfill().fillna(0.0)
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Add 6h rolling mean and std (window=12 ticks × 30min).
    Uses shift(1) to avoid look-ahead leakage.
    """
    df = df.copy()
    roll_cols = ["mppt_w", "irradiance", "batt_pct"]
    for col in roll_cols:
        # shift(1) gives NaN at row 0 — bfill before rolling so row 0 has a value
        shifted = df.groupby("device_id")[col].shift(1).bfill()
        df[f"{col}_mean_6h"] = (
            shifted
            .groupby(df["device_id"])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"{col}_std_6h"] = (
            shifted
            .groupby(df["device_id"])
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0.0))
        )
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """First-order difference per device — rate of change per 30-min tick."""
    df = df.copy()
    delta_cols = ["mppt_w", "volt_v", "batt_pct", "irradiance"]
    for col in delta_cols:
        df[f"{col}_delta"] = df.groupby("device_id")[col].diff().fillna(0.0)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic sin/cos encoding of hour and minute to avoid midnight discontinuity."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    total_minutes = 24 * 60

    df["hour_sin"]   = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / total_minutes)
    df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / total_minutes)

    # Night mask feature
    is_night = ((ts.dt.hour >= 18) | (ts.dt.hour <= 5)) & (df["irradiance"] == 0)
    df["is_night"] = is_night.astype(int)
    
    # Suppress delta variations during the night to avoid false alarms
    if "mppt_w_delta" in df.columns:
        df.loc[is_night, "mppt_w_delta"] = 0.0
    if "volt_v_delta" in df.columns:
        df.loc[is_night, "volt_v_delta"] = 0.0

    return df


def add_weather_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if this row is in a low-irradiance weather period."""
    df = df.copy()
    df["is_low_irradiance_period"] = (
        df["anomaly_type"] == "low_irradiance"
    ).astype(int)
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline in order:
    lags → rolling → delta → time → weather_flag
    """
    print("[feature_eng] Applying lag features …")
    df = add_lag_features(df)
    print("[feature_eng] Applying rolling features (6h window) …")
    df = add_rolling_features(df)
    print("[feature_eng] Applying delta features …")
    df = add_delta_features(df)
    print("[feature_eng] Applying cyclic time features …")
    df = add_time_features(df)
    print("[feature_eng] Applying weather event flag …")
    df = add_weather_flag(df)
    return df


# ─── Chronological Splitting ──────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float   = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically (sorted by timestamp).
    NO shuffle — preserves temporal structure to prevent data leakage.

    Args:
        df:         Feature-engineered DataFrame.
        train_frac: Fraction for training (default 0.70).
        val_frac:   Fraction for validation (default 0.15).

    Returns:
        (train_df, val_df, test_df)
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n  = len(df)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train : n_train + n_val].copy()
    test  = df.iloc[n_train + n_val :].copy()

    print(f"[split] Train: {len(train):,} rows "
          f"({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"[split]   Val: {len(val):,} rows "
          f"({val['timestamp'].min().date()} → {val['timestamp'].max().date()})")
    print(f"[split]  Test: {len(test):,} rows "
          f"({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

    # Sanity: no split should start BEFORE the previous one ends
    # (use <= because multi-plant data shares identical timestamps at boundary)
    assert train["timestamp"].max() <= val["timestamp"].max(), \
        "[split] Train/val timestamp overlap!"
    assert val["timestamp"].max() <= test["timestamp"].max(), \
        "[split] Val/test timestamp overlap!"

    return train, val, test


# ─── Scaling ─────────────────────────────────────────────────────────────────

def fit_and_scale(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    scaler_path: pathlib.Path = SCALER_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, list[str]]:
    """
    Fit StandardScaler on train numeric features only.
    Transform train, val, test in-place.

    Scaler is saved to scaler_path for later inference use.

    Returns:
        (train_scaled, val_scaled, test_scaled, scaler, feature_cols)
    """
    # Identify numeric feature columns (exclude meta/label/categoricals)
    all_cols     = train.columns.tolist()
    feature_cols = [c for c in all_cols
                    if c not in META_COLS
                    and train[c].dtype in (np.float64, np.float32, np.int64, np.int32)
                    and c not in ["is_low_irradiance_period", "is_night"]]  # already binary

    scaler = StandardScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols]   = scaler.transform(val[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    # Save scaler
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)
    print(f"[scale] Scaler saved → {scaler_path}  ({len(feature_cols)} features)")

    return train, val, test, scaler, feature_cols


# ─── Entry point ─────────────────────────────────────────────────────────────

def load_and_engineer(
    dataset_path: str | pathlib.Path = DATASET_PATH,
    splits_dir:   str | pathlib.Path = SPLITS_DIR,
    scaler_path:  str | pathlib.Path = SCALER_PATH,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Full Phase 4 pipeline:
      1. Load scio_bench_dataset.csv
      2. Apply feature engineering
      3. Chronological split
      4. Fit/transform StandardScaler
      5. Save splits to data/splits/

    Returns:
        (train_df, val_df, test_df, feature_cols)
    """
    dataset_path = pathlib.Path(dataset_path)
    splits_dir   = pathlib.Path(splits_dir)
    scaler_path  = pathlib.Path(scaler_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"[feature_eng] Dataset not found: {dataset_path}\n"
            "Run Phase 3 first: python -m src.data.anomaly_injection"
        )

    print(f"[feature_eng] Loading {dataset_path.name} …")
    df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    print(f"[feature_eng] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Feature engineering
    df = engineer_all_features(df)
    print(f"[feature_eng] Features after engineering: {len(df.columns)} columns")

    # Split
    train, val, test = chronological_split(df)

    # Scale
    train, val, test, scaler, feature_cols = fit_and_scale(
        train, val, test, scaler_path
    )

    # Save
    if save:
        splits_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(splits_dir / "train.csv", index=False)
        val.to_csv(splits_dir / "val.csv",     index=False)
        test.to_csv(splits_dir / "test.csv",   index=False)
        print(f"[feature_eng] Splits saved to {splits_dir}/{{train,val,test}}.csv")

    print(f"\n[feature_eng] ✓ Feature columns ({len(feature_cols)}):")
    print("  " + ", ".join(feature_cols))

    return train, val, test, feature_cols


if __name__ == "__main__":
    train, val, test, feature_cols = load_and_engineer()
    print(f"\n=== Sample train row ===")
    print(train[feature_cols[:8]].head(2).to_string())

    print(f"\n=== Anomaly distribution in test split ===")
    print(test["anomaly_type"].value_counts().to_string())
