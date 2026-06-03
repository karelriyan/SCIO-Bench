"""
SCIO-Bench — Central Configuration
All tunable constants, paths, and hyperparameters live here.
"""

import pathlib
from dataclasses import dataclass

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_DIR = OUTPUTS_DIR / "dataset"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# ─── Preprocessing ────────────────────────────────────────────────────────────
RESAMPLE_INTERVAL: str = "30min"          # Kaggle data → SCIO IoT gateway polling
FFILL_LIMIT: int = 2                       # Max consecutive NaN ticks to forward-fill

PLANT_FILES = {
    1: {
        "gen": "Plant_1_Generation_Data.csv",
        "weather": "Plant_1_Weather_Sensor_Data.csv",
        "gen_date_fmt": "%d-%m-%Y %H:%M",
        "weather_date_fmt": "%Y-%m-%d %H:%M:%S",
    },
    2: {
        "gen": "Plant_2_Generation_Data.csv",
        "weather": "Plant_2_Weather_Sensor_Data.csv",
        "gen_date_fmt": "%Y-%m-%d %H:%M:%S",
        "weather_date_fmt": "%Y-%m-%d %H:%M:%S",
    },
}

RENAME_MAP = {
    "DC_POWER": "mppt_kw",
    "AC_POWER": "ac_power_kw",
    "DAILY_YIELD": "daily_yield_kwh",
    "TOTAL_YIELD": "total_yield_kwh",
    "MODULE_TEMPERATURE": "temp_c",
    "IRRADIATION": "irradiance",
    "AMBIENT_TEMPERATURE": "ambient_temp_c",
}

SCIO_BASE_COLUMNS = [
    "timestamp",
    "device_id",
    "mppt_w",
    "ac_power_kw",
    "daily_yield_kwh",
    "temp_c",
    "irradiance",
    "ambient_temp_c",
]

# ─── Anomaly Injection ────────────────────────────────────────────────────────

ANOMALY_PROPORTIONS = {
    "A1": 0.020,   # panel_degradation
    "A2": 0.015,   # sudden_drop
    "A3": 0.020,   # battery_fault
    "A4": 0.015,   # sensor_drift
    "A5": 0.020,   # offline
    "A6": 0.150,   # low_irradiance  (normal weather — NOT anomaly)
    "A7": 0.010,   # false_data_injection
}

# Segment lengths (ticks, each tick = 30 min)
ANOMALY_SEGMENT_LEN = {
    "A1": 12,   # 6 hours
    "A2": 2,    # 1 hour
    "A3": 8,    # 4 hours
    "A4": 6,    # 3 hours
    "A5": 4,    # 2 hours
    "A6": 48,   # 24 hours
    "A7": 3,    # 1.5 hours
}

ANOMALY_CODE_MAP = {
    "A1": "panel_degradation",
    "A2": "sudden_drop",
    "A3": "battery_fault",
    "A4": "sensor_drift",
    "A5": "offline",
    "A6": "low_irradiance",
    "A7": "false_data_injection",
}

# ─── Models ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LSTMAEConfig:
    seq_len: int = 24          # 12 hours at 30-min resolution
    batch_size: int = 64
    epochs: int = 50
    latent_dim: int = 16
    enc_units: tuple[int, ...] = (64, 32)
    dec_units: tuple[int, ...] = (32, 64)
    patience: int = 5
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class RuleBasedConfig:
    k: float = 3.5             # MAD multiplier
    eps: float = 1e-5          # Prevent zero-variance collapse
    
    # Per-feature thresholds (updated dynamically, defaults here for docs)
    mppt_w_max: float = 500.0
    mppt_w_min: float = 0.0
    irradiance_max: float = 1500.0
    temp_c_max: float = 85.0
    batt_pct_max: float = 100.0
    batt_pct_min: float = 0.0


# ─── Evaluation ───────────────────────────────────────────────────────────────

LABEL_COLS = [
    "is_anomaly",
    "anomaly_type",
    "is_weather_event",
    "timestamp",
    "device_id",
    "protocol",
]

# ─── Feature Engineering ──────────────────────────────────────────────────────

ROLLING_WINDOW: int = 12       # 6 hours for rolling mean/std features

