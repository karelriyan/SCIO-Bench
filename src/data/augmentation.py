"""
Phase 2 — Synthetic Variable Augmentation
Adds battery SOC, voltage, current, RSSI, protocol, and relational physics features
to the cleaned SCIO dataset.

Outputs per plant (added columns):
  batt_pct        — Battery State of Charge (%) — non-linear SOC model
  volt_v          — Bus voltage (V) — derived from LiFePO4 discharge curve
  curr_a          — Current (A) — derived as mppt_w / volt_v
  rssi            — Signal strength (dBm) — synthetic normal distribution
  protocol        — Communication protocol (lora / 4g) — based on rssi

Relational features (physics-based, scale-invariant):
  ratio_power_irr — DC Power / Irradiance (efficiency proxy)
  ratio_volt_curr — V / I (impedance proxy; sensitive to A7 FDI)
  physics_residual— mppt_w - volt_v × curr_a (should ≈ 0 for honest data)
  batt_delta      — SOC rate of change per tick
  prod_vs_batt    — Energy produced vs SOC-implied energy change

References:
  Battery SOC model: NASA Battery Dataset (B0005-B0018)
  Voltage curve: LiFePO4 24V discharge polynomial fit
  Relational features: SCIO Research Framework §6.1
"""

import pathlib
import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

# Battery parameters (100Ah × 24V nominal = 2400 Wh)
BATTERY_CAPACITY_WH = 2400.0
NOMINAL_VOLTAGE     = 24.0      # V (24V LiFePO4 pack)
INITIAL_SOC         = 0.70      # 70% initial charge
CHARGE_EFF          = 0.92      # base charge efficiency
DISCHARGE_EFF       = 0.95      # base discharge efficiency
DEGRADATION_RATE    = 0.0002    # capacity loss per cycle (0.02 %)
CYCLE_COUNT         = 50        # assumed cycles already completed
CONSTANT_LOAD_W     = 50.0      # background load (W) — always-on devices
DT_HOURS            = 0.5       # 30-minute ticks

# RSSI parameters (dBm)
RSSI_MEAN  = -70
RSSI_STD   = 15
RSSI_4G_THRESHOLD = -80         # below → switch to 4G (worse LoRa)

DATA_DIR = pathlib.Path("data/processed")


# ─── Battery SOC Simulation (non-linear, from framework §5.2) ────────────────

def simulate_soc_nonlinear(
    mppt_w_series: np.ndarray,
    initial_soc: float = INITIAL_SOC,
    cycle_count: int   = CYCLE_COUNT,
    random_seed: int   = 42,
) -> np.ndarray:
    """
    Non-linear battery SOC simulation with:
    - Tapering charge efficiency as SOC → 1.0
    - Capacity degradation per cycle
    - Gaussian sensor noise (organic fluctuation)

    Args:
        mppt_w_series: DC power output in Watts per tick.
        initial_soc:   Starting SOC [0, 1].
        cycle_count:   Assumed prior charge/discharge cycles.
        random_seed:   For reproducibility.

    Returns:
        SOC percentage array in range [5.0, 100.0].
    """
    rng = np.random.default_rng(random_seed)
    capacity = BATTERY_CAPACITY_WH * (1.0 - DEGRADATION_RATE * cycle_count)

    soc = initial_soc
    soc_series = []

    for power_w in mppt_w_series:
        # Tapering: charge efficiency drops when SOC > 80% (CV phase)
        charge_eff = CHARGE_EFF * (1.0 - 0.3 * max(0.0, soc - 0.8))

        # Net energy this tick (Wh)
        charge_wh    = power_w * charge_eff * DT_HOURS
        discharge_wh = (CONSTANT_LOAD_W / DISCHARGE_EFF) * DT_HOURS
        net_wh       = charge_wh - discharge_wh

        # Update SOC
        soc = soc + net_wh / capacity
        soc = float(np.clip(soc, 0.05, 1.0))

        # Organic sensor noise (BMS rounding + measurement error, std ≈ 0.3 %)
        noise = rng.normal(0.0, 0.003)
        soc_series.append(np.clip(soc + noise, 0.05, 1.0) * 100.0)

    return np.array(soc_series)


# ─── Voltage Derivation (LiFePO4 discharge curve) ─────────────────────────────

def derive_voltage(
    soc_pct: np.ndarray,
    nominal_v: float = NOMINAL_VOLTAGE,
    random_seed: int  = 42,
) -> np.ndarray:
    """
    Non-linear voltage derived from SOC using a polynomial fit of a
    24V LiFePO4 discharge curve.

    V = (a·SOC³ + b·SOC² + c·SOC + d) × nominal_v  + sensor_noise

    Polynomial coefficients tuned to exhibit:
      SOC=100% → ~25.6V, SOC=50% → ~24.0V, SOC=10% → ~22.8V

    Args:
        soc_pct:   SOC percentage array (5–100).
        nominal_v: Nominal pack voltage [V].
        random_seed: For reproducible sensor noise.

    Returns:
        Voltage array in Volts.
    """
    rng = np.random.default_rng(random_seed)
    soc_norm = np.array(soc_pct) / 100.0  # normalise to [0, 1]

    # Polynomial: characterised from LiFePO4 OCV curve
    v = (2.1 * soc_norm**3
         - 3.8 * soc_norm**2
         + 2.9 * soc_norm
         + 0.8) * nominal_v

    # Sensor noise ±0.15 V
    noise = rng.normal(0.0, 0.15, size=len(v))
    return np.clip(v + noise, nominal_v * 0.85, nominal_v * 1.10)


# ─── Current Derivation ────────────────────────────────────────────────────────

def derive_current(mppt_w: np.ndarray, volt_v: np.ndarray) -> np.ndarray:
    """
    Current = Power / Voltage  (P = V × I).
    A small epsilon prevents division by zero during night-time (mppt_w ≈ 0).

    Returns: curr_a in Amperes (non-negative).
    """
    epsilon = 1e-6
    return np.clip(mppt_w / (volt_v + epsilon), 0.0, None)


# ─── RSSI & Protocol ──────────────────────────────────────────────────────────

def generate_rssi_protocol(
    n: int,
    random_seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    Synthetic RSSI values (dBm) and protocol labels.

    Protocol: 'lora'  if rssi ≥ RSSI_4G_THRESHOLD  (good signal)
              '4g'    if rssi <  RSSI_4G_THRESHOLD  (weak LoRa → fallback)

    Returns: (rssi_array, protocol_list)
    """
    rng = np.random.default_rng(random_seed)
    rssi = rng.normal(RSSI_MEAN, RSSI_STD, size=n).astype(int)
    protocol = ["lora" if r >= RSSI_4G_THRESHOLD else "4g" for r in rssi]
    return rssi, protocol


# ─── Relational Features ──────────────────────────────────────────────────────

def compute_relational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5 physics-based relational features.
    These are scale-invariant and critical for False Data Injection (A7) detection.

    Features added:
      ratio_power_irr — mppt_w / (irradiance + ε)  [W per W/m², efficiency proxy]
      ratio_volt_curr — volt_v / (curr_a + ε)       [Ohm, impedance proxy]
      physics_residual— mppt_w - volt_v × curr_a    [W, should ≈ 0 for honest data]
      batt_delta      — SOC rate of change per 30-min tick [% / tick]
      prod_vs_batt    — prod_wh - batt_delta × BATTERY_CAPACITY_WH / 100
                        (energy balance: generated vs stored)
    """
    df = df.copy()
    eps = 1e-6

    df["ratio_power_irr"] = df["mppt_w"] / (df["irradiance"] + eps)
    df["ratio_volt_curr"] = df["volt_v"] / (df["curr_a"] + eps)
    df["physics_residual"] = df["mppt_w"] - df["volt_v"] * df["curr_a"]
    df["batt_delta"]       = df["batt_pct"].diff().fillna(0.0)
    df["prod_vs_batt"]     = (
        df["prod_wh"]
        - df["batt_delta"] * BATTERY_CAPACITY_WH / 100.0
    )

    return df


# ─── Main Augmentation Pipeline ───────────────────────────────────────────────

def augment_dataset(
    df: pd.DataFrame,
    plant_id: str = "PLANT1",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Full augmentation pipeline for one plant DataFrame.

    Adds: batt_pct, volt_v, curr_a, rssi, protocol,
          ratio_power_irr, ratio_volt_curr, physics_residual,
          batt_delta, prod_vs_batt

    Args:
        df:          Cleaned plant DataFrame (output of preprocess.py).
        plant_id:    Identifier string for seed offset (ensures plants differ).
        random_seed: Base random seed (42 for reproducibility).

    Returns:
        Augmented DataFrame.
    """
    df = df.copy()
    n = len(df)

    # Seed offset per plant so Plant 1 and Plant 2 have different noise
    seed_offset = hash(plant_id) % 1000
    seed = random_seed + seed_offset

    print(f"  [augment] Adding synthetic variables for {plant_id} (n={n}) …")

    # Battery SOC
    df["batt_pct"] = simulate_soc_nonlinear(
        df["mppt_w"].to_numpy(),
        random_seed=seed,
    )

    # Voltage (LiFePO4 curve)
    df["volt_v"] = derive_voltage(
        df["batt_pct"].to_numpy(),
        random_seed=seed + 1,
    )

    # Current
    df["curr_a"] = derive_current(
        df["mppt_w"].to_numpy(),
        df["volt_v"].to_numpy(),
    )

    # RSSI & Protocol
    rssi, protocol = generate_rssi_protocol(n, random_seed=seed + 2)
    df["rssi"]     = rssi
    df["protocol"] = protocol

    # Relational physics features
    df = compute_relational_features(df)

    print(f"  [augment] ✓ Columns added: batt_pct, volt_v, curr_a, rssi, protocol, "
          f"ratio_power_irr, ratio_volt_curr, physics_residual, batt_delta, prod_vs_batt")

    return df


# ─── Entry point ──────────────────────────────────────────────────────────────

def load_and_augment(
    processed_dir: str | pathlib.Path = DATA_DIR,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cleaned CSVs from preprocess phase, augment both plants, optionally save.

    Returns: (df_plant1_aug, df_plant2_aug)
    """
    processed_dir = pathlib.Path(processed_dir)

    for f in ("plant1_clean.csv", "plant2_clean.csv"):
        if not (processed_dir / f).exists():
            raise FileNotFoundError(
                f"[augment] {f} not found. Run Phase 1 first:\n"
                "  python -m src.data.preprocess"
            )

    df1 = pd.read_csv(processed_dir / "plant1_clean.csv", parse_dates=["timestamp"])
    df2 = pd.read_csv(processed_dir / "plant2_clean.csv", parse_dates=["timestamp"])

    df1 = augment_dataset(df1, plant_id="PLANT1")
    df2 = augment_dataset(df2, plant_id="PLANT2")

    if save:
        df1.to_csv(processed_dir / "plant1_augmented.csv", index=False)
        df2.to_csv(processed_dir / "plant2_augmented.csv", index=False)
        print(f"[augment] Saved to {processed_dir}/plant{{1,2}}_augmented.csv")

    return df1, df2


if __name__ == "__main__":
    df1, df2 = load_and_augment()
    print("\n=== Plant 1 columns ===")
    print(df1.columns.tolist())
    print("\n=== Plant 1 sample (augmented) ===")
    aug_cols = ["timestamp", "mppt_w", "batt_pct", "volt_v", "curr_a",
                "rssi", "protocol", "physics_residual"]
    print(df1[aug_cols].head(5).to_string())
    print(f"\nphysics_residual stats:\n{df1['physics_residual'].describe()}")
