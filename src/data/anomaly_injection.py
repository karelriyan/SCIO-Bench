"""
Phase 3 — Anomaly Injection (SCIO-Bench Dataset)
Injects 7 anomaly types into the augmented dataset to produce the final
labeled SCIO-Bench benchmark.

Anomaly types:
  A1: Panel Degradation       — 2%  — gradual 30-50% mppt_w decay over 6h
  A2: Sudden Panel Drop       — 1.5%— instant 60-80% drop for 1-3 ticks
  A3: Battery Fault           — 2%  — rapid SOC drop (>5%/tick) or stuck
  A4: Sensor Drift            — 1.5%— ±15% persistent volt_v offset
  A5: Device Offline          — 2%  — NaN / last-value-held ≥3 ticks
  A6: Extended Low Irradiance — ~15%— (NORMAL!) prolonged cloud cover
  A7: False Data Injection    — 1%  — physics-inconsistent: volt↑ curr↓ P same

Design rules:
  - np.random.seed(42) before ALL injections for full reproducibility
  - Injections do NOT overlap (each tick gets at most one anomaly type)
  - A6 is labeled is_anomaly=False (it is a normal weather event)
  - A7 stays within normal individual ranges (makes rule-based detection hard)
  - Output columns: is_anomaly (bool), anomaly_type (str), is_weather_event (bool)
"""

import pathlib
import numpy as np
import pandas as pd

# ─── Proportions ──────────────────────────────────────────────────────────────

from src import config

PROPORTIONS = config.ANOMALY_PROPORTIONS
ANOMALY_SEGMENT_LEN = config.ANOMALY_SEGMENT_LEN
DATA_DIR = config.PROCESSED_DIR


# ─── Segment selector ─────────────────────────────────────────────────────────

def _select_segments(
    n: int,
    proportion: float,
    segment_len: int,
    rng: np.random.Generator,
    occupied: np.ndarray,
) -> list[np.ndarray]:
    """
    Select non-overlapping, contiguous segments of fixed length.

    Args:
        n:           Total number of rows.
        proportion:  Fraction of rows to cover.
        segment_len: Length of each contiguous segment.
        rng:         Seeded RNG for reproducibility.
        occupied:    Boolean mask of already-used indices (mutated in-place).

    Returns:
        List of index arrays.
    """
    target_count = int(n * proportion)
    n_segments = max(1, target_count // segment_len)
    segments = []
    candidates = np.where(~occupied)[0]

    for _ in range(n_segments * 10):   # retry budget
        if len(segments) >= n_segments or len(candidates) < segment_len:
            break
        start_pool = candidates[candidates <= n - segment_len]
        if len(start_pool) == 0:
            break
        start = rng.choice(start_pool)
        seg = np.arange(start, min(start + segment_len, n))
        if occupied[seg].any():
            candidates = np.where(~occupied)[0]
            continue
        occupied[seg] = True
        segments.append(seg)
        candidates = np.where(~occupied)[0]

    return segments


# ─── Individual Injectors ─────────────────────────────────────────────────────

def _get_segments(df: pd.DataFrame, code: str, rng: np.random.Generator) -> list[np.ndarray]:
    """Helper: select non-overlapping segments for anomaly type *code*."""
    n = len(df)
    occupied = df["anomaly_type"].ne("normal").to_numpy().copy()
    return _select_segments(
        n, PROPORTIONS[code], ANOMALY_SEGMENT_LEN[code], rng, occupied
    )


def _inject_a1_panel_degradation(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A1: Gradual 30-50% mppt_w decay over 6h (12 ticks × 30min)."""
    for seg in _get_segments(df, "A1", rng):
        decay = np.linspace(1.0, rng.uniform(0.50, 0.70), len(seg))
        df.loc[seg, "mppt_w"]   *= decay
        df.loc[seg, "prod_wh"]  *= decay
        df.loc[seg, "anomaly_type"] = "panel_degradation"
        df.loc[seg, "is_anomaly"]   = True
    return df


def _inject_a2_sudden_drop(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A2: Instant 60-80% power drop for 1-3 ticks."""
    for seg in _get_segments(df, "A2", rng):
        drop = rng.uniform(0.20, 0.40)   # keep 20-40% of power
        df.loc[seg, "mppt_w"]  *= drop
        df.loc[seg, "prod_wh"] *= drop
        df.loc[seg, "volt_v"]  *= rng.uniform(0.85, 0.95)
        df.loc[seg, "anomaly_type"] = "sudden_drop"
        df.loc[seg, "is_anomaly"]   = True
    return df


def _inject_a3_battery_fault(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A3: Rapid SOC drop (>5%/tick) OR SOC stuck at fixed value."""
    for seg in _get_segments(df, "A3", rng):
        fault_type = rng.choice(["rapid_drop", "stuck"])
        if fault_type == "rapid_drop":
            start_soc = df.loc[seg[0], "batt_pct"]
            drop_per_tick = rng.uniform(5.5, 8.0)
            new_soc = [max(5.0, start_soc - drop_per_tick * i) for i in range(len(seg))]
            df.loc[seg, "batt_pct"] = new_soc
        else:
            df.loc[seg, "batt_pct"] = rng.uniform(20.0, 60.0)
        df.loc[seg, "anomaly_type"] = "battery_fault"
        df.loc[seg, "is_anomaly"]   = True
    return df


def _inject_a4_sensor_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A4: ±15% persistent volt_v offset (calibration error / loose connection)."""
    for seg in _get_segments(df, "A4", rng):
        drift = rng.uniform(0.85, 1.15)   # ±15%
        if rng.random() < 0.5:
            drift = 1.0 / drift            # also allow downward drift
        df.loc[seg, "volt_v"] *= drift
        df.loc[seg, "curr_a"] /= drift   # P stays same, P=VI → I adjusts
        df.loc[seg, "anomaly_type"] = "sensor_drift"
        df.loc[seg, "is_anomaly"]   = True
    return df


def _inject_a5_offline(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A5: Device offline — last-value-held for 3-5 ticks (mimics NaN filled by sensor)."""
    sensor_cols = ["mppt_w", "prod_wh", "volt_v", "curr_a",
                   "batt_pct", "temp_c", "irradiance"]
    for seg in _get_segments(df, "A5", rng):
        if seg[0] > 0:
            last_vals = df.loc[seg[0] - 1, sensor_cols]
            for col in sensor_cols:
                df.loc[seg, col] = last_vals[col]
        else:
            for col in sensor_cols:
                df.loc[seg, col] = 0.0     # edge case: start of dataset
        df.loc[seg, "anomaly_type"] = "offline"
        df.loc[seg, "is_anomaly"]   = True
    return df


def _inject_a6_low_irradiance(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    A6: Extended Low Irradiance — NORMAL weather event (is_anomaly=False!).
    Simulates prolonged tropical cloud cover (12–48h = 24–96 ticks).
    Model must NOT flag these as anomalies (used for FPR evaluation).
    """
    for seg in _get_segments(df, "A6", rng):
        reduction = rng.uniform(0.20, 0.50)   # keep 20-50% of irradiance/power
        df.loc[seg, "irradiance"] *= reduction
        df.loc[seg, "mppt_w"]     *= reduction
        df.loc[seg, "prod_wh"]    *= reduction
        df.loc[seg, "anomaly_type"]    = "low_irradiance"
        df.loc[seg, "is_anomaly"]      = False
        df.loc[seg, "is_weather_event"] = True
    return df


def _inject_a7_false_data_injection(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    A7: False Data Injection (adversarial) — physics-inconsistent sensor manipulation.
    volt_v ↑10-20%, curr_a ↓30-50%, mppt_w unchanged → P ≠ V×I
    Both volt_v and curr_a individually appear within normal ranges.
    Rule-based methods will likely miss this; physics_residual should catch it.
    """
    for seg in _get_segments(df, "A7", rng):
        volt_boost  = rng.uniform(1.10, 1.20)   # volt_v +10-20%
        curr_reduce = rng.uniform(0.50, 0.70)   # curr_a -30-50%
        df.loc[seg, "volt_v"] *= volt_boost
        df.loc[seg, "curr_a"] *= curr_reduce
        # mppt_w intentionally NOT changed → creates physics_residual spike
        df.loc[seg, "anomaly_type"] = "false_data_injection"
        df.loc[seg, "is_anomaly"]   = True
    return df


# ─── Recompute relational features after injection ────────────────────────────

def _recompute_relational(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute 5 relational features after anomaly injection, since volt_v/curr_a
    may have changed. This ensures physics_residual reflects the injected state.
    """
    eps = 1e-6
    df["ratio_power_irr"] = df["mppt_w"] / (df["irradiance"] + eps)
    df["ratio_volt_curr"] = df["volt_v"] / (df["curr_a"] + eps)
    df["physics_residual"] = df["mppt_w"] - df["volt_v"] * df["curr_a"]
    df["batt_delta"]       = df["batt_pct"].diff().fillna(0.0)
    df["prod_vs_batt"]     = (
        df["prod_wh"]
        - df["batt_delta"] * 2400.0 / 100.0
    )
    return df


# ─── Main injection pipeline ──────────────────────────────────────────────────

def inject_all_anomalies(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Merge both plants, inject all 7 anomaly types, return labeled SCIO-Bench dataset.

    Injection order: A6 first (to claim large segments), then A1-A5, finally A7.
    This ensures realistic distribution without segment overlap.

    Args:
        df1, df2:    Augmented plant DataFrames.
        random_seed: Master seed (42 for full reproducibility).

    Returns:
        Combined labeled DataFrame (~3264 rows).
    """
    # RNG is localised per-plant via default_rng — no global state pollution

    def _inject_one_plant(df: pd.DataFrame, plant_seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(plant_seed)
        df = df.copy()

        # Initialise label columns
        df["is_anomaly"]      = False
        df["anomaly_type"]    = "normal"
        df["is_weather_event"] = False

        # Inject in order: A6 first (claims large blocks), then real anomalies
        df = _inject_a6_low_irradiance(df, rng)
        df = _inject_a1_panel_degradation(df, rng)
        df = _inject_a2_sudden_drop(df, rng)
        df = _inject_a3_battery_fault(df, rng)
        df = _inject_a4_sensor_drift(df, rng)
        df = _inject_a5_offline(df, rng)
        df = _inject_a7_false_data_injection(df, rng)

        # Recompute relational features with injected values
        df = _recompute_relational(df)

        return df

    df1 = _inject_one_plant(df1, plant_seed=random_seed)
    df2 = _inject_one_plant(df2, plant_seed=random_seed + 1)

    combined = pd.concat([df1, df2], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    return combined


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_distribution_report(df: pd.DataFrame) -> None:
    """Print anomaly type distribution for paper reporting."""
    n = len(df)
    counts = df["anomaly_type"].value_counts()
    print("\n=== SCIO-Bench Distribution Report ===")
    print(f"Total rows: {n:,}")
    print(f"{'Type':<25} {'Count':>6} {'%':>7}")
    print("-" * 42)
    for atype, count in counts.items():
        is_anom = df.loc[df["anomaly_type"] == atype, "is_anomaly"].any()
        tag = "[ANOMALY]" if is_anom else "[NORMAL ]"
        print(f"{atype:<25} {count:>6} {count/n*100:>6.1f}%  {tag}")
    print("-" * 42)
    real_anomalies = df[df["is_anomaly"]].shape[0]
    print(f"Real anomaly rate: {real_anomalies/n*100:.1f}%  "
          f"(target: ~9%, paper says <10%)")
    weather_count = df[df["is_weather_event"]].shape[0]
    print(f"A6 weather events: {weather_count/n*100:.1f}%  (target: ~15%)")


# ─── Entry point ──────────────────────────────────────────────────────────────

def load_and_inject(
    processed_dir: str | pathlib.Path = DATA_DIR,
    out_dir: str | pathlib.Path = config.DATASET_DIR,
    random_seed: int = 42,
    save: bool = True,
) -> pd.DataFrame:
    """Load augmented CSVs, inject anomalies, save SCIO-Bench dataset."""
    processed_dir = pathlib.Path(processed_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in ("plant1_augmented.csv", "plant2_augmented.csv"):
        if not (processed_dir / f).exists():
            raise FileNotFoundError(
                f"[inject] {f} not found. Run Phase 2 first:\n"
                "  python -m src.data.augmentation"
            )

    df1 = pd.read_csv(processed_dir / "plant1_augmented.csv", parse_dates=["timestamp"])
    df2 = pd.read_csv(processed_dir / "plant2_augmented.csv", parse_dates=["timestamp"])

    print(f"[inject] Injecting anomalies (seed={random_seed}) …")
    bench = inject_all_anomalies(df1, df2, random_seed=random_seed)

    if save:
        out_path = out_dir / "scio_bench_dataset.csv"
        bench.to_csv(out_path, index=False)
        print(f"[inject] Saved → {out_path}  ({len(bench):,} rows)")

    print_distribution_report(bench)
    return bench


if __name__ == "__main__":
    bench = load_and_inject()
