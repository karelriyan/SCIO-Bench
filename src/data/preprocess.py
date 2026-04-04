"""
Phase 1 — Preprocessing
Loads, merges, resamples, cleans, and renames the Kaggle dataset to SCIO conventions.

Input:  data/raw/  (Plant_1/2 × Generation + Weather Sensor CSVs)
Output: data/processed/plant1_clean.parquet
        data/processed/plant2_clean.parquet

Column mapping (Kaggle → SCIO):
  DC_POWER         → mppt_w        (W — multiply ×1000 from kW)
  MODULE_TEMPERATURE → temp_c      (°C)
  IRRADIATION      → irradiance    (W/m²)
  AMBIENT_TEMPERATURE → ambient_temp_c
  AC_POWER         → ac_power_kw   (kW, kept for reference)
  DAILY_YIELD      → daily_yield_kwh

Timestamps: dataset is in IST (UTC+5:30), kept as-is (relative pattern matters, not timezone).
"""

import pathlib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

RAW_DIR = pathlib.Path("data/raw")
OUT_DIR = pathlib.Path("data/processed")

# Expected raw files per plant
PLANT_FILES = {
    1: {
        "gen": "Plant_1_Generation_Data.csv",
        "weather": "Plant_1_Weather_Sensor_Data.csv",
    },
    2: {
        "gen": "Plant_2_Generation_Data.csv",
        "weather": "Plant_2_Weather_Sensor_Data.csv",
    },
}

# Kaggle → SCIO column rename mapping
RENAME_MAP = {
    "DC_POWER": "mppt_kw",             # kW — converted to W below
    "AC_POWER": "ac_power_kw",         # kW — kept for reference
    "DAILY_YIELD": "daily_yield_kwh",
    "TOTAL_YIELD": "total_yield_kwh",
    "MODULE_TEMPERATURE": "temp_c",
    "IRRADIATION": "irradiance",       # W/m²
    "AMBIENT_TEMPERATURE": "ambient_temp_c",
}

# Columns to keep as SCIO features (after rename)
SCIO_COLUMNS = [
    "timestamp",
    "device_id",
    "mppt_w",           # DC Power in Watts (= mppt_kw × 1000)
    "ac_power_kw",
    "daily_yield_kwh",
    "temp_c",
    "irradiance",
    "ambient_temp_c",
]

# Max consecutive NaN ticks to forward-fill (per SRS FR-034)
FFILL_LIMIT = 2
# Resample interval (30 minutes, matching SCIO IoT gateway polling)
RESAMPLE_INTERVAL = "30min"


def _load_plant_csvs(plant_id: int, raw_dir: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load generation and weather CSVs for a given plant."""
    files = PLANT_FILES[plant_id]
    gen_path = raw_dir / files["gen"]
    weather_path = raw_dir / files["weather"]

    for p in (gen_path, weather_path):
        if not p.exists():
            raise FileNotFoundError(
                f"[preprocess] File not found: {p}\n"
                "Run download first: python -m src.data.download"
            )

    gen_df = pd.read_csv(gen_path, parse_dates=["DATE_TIME"])
    weather_df = pd.read_csv(weather_path, parse_dates=["DATE_TIME"])
    return gen_df, weather_df


def _merge_and_resample(gen_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join generation + weather on DATE_TIME, then resample to 30-minute intervals.
    """
    # Normalize column names (strip whitespace)
    gen_df.columns = gen_df.columns.str.strip()
    weather_df.columns = weather_df.columns.str.strip()

    # Inner join on DATE_TIME
    merged = pd.merge(gen_df, weather_df, on="DATE_TIME", how="inner", suffixes=("_gen", "_wx"))

    # Set datetime index for resampling
    merged = merged.set_index("DATE_TIME").sort_index()

    # Resample: mean of all numeric columns per 30-minute window
    numeric_cols = merged.select_dtypes(include="number").columns
    resampled = merged[numeric_cols].resample(RESAMPLE_INTERVAL).mean()

    # Reset index — timestamp becomes a column
    resampled = resampled.reset_index().rename(columns={"DATE_TIME": "timestamp"})
    return resampled


def _handle_nan_inf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust NaN/Inf cleanup per SRS FR-034:
    1. Replace ±Inf with NaN.
    2. Forward-fill up to FFILL_LIMIT consecutive NaN.
    3. Flag remaining NaN as potential offline (A5 candidate).
    4. Fill remaining NaN with column median.
    5. Assert no NaN or Inf remain.
    """
    # Step 1: Inf → NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Mark rows with ANY NaN before filling (offline candidates)
    df["_had_nan"] = df.drop(columns=["timestamp"], errors="ignore").isnull().any(axis=1)

    # Step 2: Forward-fill up to limit (pandas 2.x: ffill() replaces fillna(method=))
    non_ts_cols = [c for c in df.columns if c not in ("timestamp", "_had_nan")]
    df[non_ts_cols] = df[non_ts_cols].ffill(limit=FFILL_LIMIT)

    # Step 3: Remaining NaN (gap > 2) → median fill
    medians = df[non_ts_cols].median()
    df[non_ts_cols] = df[non_ts_cols].fillna(medians)

    # Step 4: Validate
    assert not df[non_ts_cols].isnull().values.any(), \
        "[preprocess] NaN still present after cleanup!"
    assert not np.isinf(df[non_ts_cols].values).any(), \
        "[preprocess] Inf still present after cleanup!"

    return df


def _rename_and_derive(df: pd.DataFrame, plant_id: int) -> pd.DataFrame:
    """Rename columns to SCIO convention and derive key variables."""
    # Rename
    df = df.rename(columns=RENAME_MAP)

    # Convert DC Power kW → W
    if "mppt_kw" in df.columns:
        df["mppt_w"] = df["mppt_kw"] * 1000.0
        df = df.drop(columns=["mppt_kw"])
    elif "mppt_w" not in df.columns:
        raise ValueError("[preprocess] DC_POWER / mppt_w column not found!")

    # Derive prod_wh: approximate via 30-min power integral
    # prod_wh ≈ mppt_w × 0.5 hour (energy in Wh per tick)
    df["prod_wh"] = df["mppt_w"] * 0.5

    # Clip negative values (sensor noise during darkness → 0)
    df["mppt_w"] = df["mppt_w"].clip(lower=0.0)
    df["prod_wh"] = df["prod_wh"].clip(lower=0.0)
    if "irradiance" in df.columns:
        df["irradiance"] = df["irradiance"].clip(lower=0.0)

    # Add device_id
    df["device_id"] = f"SIM_PLANT{plant_id}_INV01"

    # Drop internal helper column
    df = df.drop(columns=["_had_nan"], errors="ignore")

    # Drop columns we don't need downstream
    drop_cols = [c for c in df.columns
                 if c not in SCIO_COLUMNS + ["prod_wh", "ambient_temp_c", "daily_yield_kwh"]]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Reorder
    ordered = ["timestamp", "device_id", "mppt_w", "prod_wh",
               "temp_c", "irradiance", "ambient_temp_c", "daily_yield_kwh", "ac_power_kw"]
    existing = [c for c in ordered if c in df.columns]
    rest = [c for c in df.columns if c not in existing]
    df = df[existing + rest]

    return df


def preprocess_plant(
    plant_id: int,
    raw_dir: pathlib.Path = RAW_DIR,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for one plant.

    Returns:
        Cleaned DataFrame with SCIO column convention.
    """
    print(f"[preprocess] Processing Plant {plant_id} …")

    gen_df, weather_df = _load_plant_csvs(plant_id, raw_dir)
    print(f"  Gen rows: {len(gen_df)} | Weather rows: {len(weather_df)}")

    df = _merge_and_resample(gen_df, weather_df)
    print(f"  After merge+resample: {len(df)} rows "
          f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")

    df = _handle_nan_inf(df)
    df = _rename_and_derive(df, plant_id)

    print(f"  ✓ Plant {plant_id} ready: {df.shape} | Columns: {df.columns.tolist()}")
    return df


def load_and_preprocess(
    raw_dir: str | pathlib.Path = RAW_DIR,
    out_dir: str | pathlib.Path = OUT_DIR,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess both plants and optionally save to data/processed/.

    Args:
        raw_dir: Directory containing raw Kaggle CSVs.
        out_dir: Directory to save cleaned parquet files.
        save:    Save outputs as parquet (fast loading for subsequent phases).

    Returns:
        (df_plant1, df_plant2)
    """
    raw_dir = pathlib.Path(raw_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df1 = preprocess_plant(1, raw_dir)
    df2 = preprocess_plant(2, raw_dir)

    if save:
        df1.to_csv(out_dir / "plant1_clean.csv", index=False)
        df2.to_csv(out_dir / "plant2_clean.csv", index=False)
        print(f"[preprocess] Saved to {out_dir}/plant{{1,2}}_clean.csv")

    return df1, df2


if __name__ == "__main__":
    df1, df2 = load_and_preprocess()
    print("\n=== Plant 1 sample ===")
    print(df1.head(3).to_string())
    print("\n=== Plant 2 sample ===")
    print(df2.head(3).to_string())
