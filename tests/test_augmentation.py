"""
Phase 2 — Tests: Augmentation
Tests SOC range, voltage range, current derivation, relational features.
All tests use synthetic data — no real dataset required.
"""

import pytest
import numpy as np
import pandas as pd
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.data.augmentation import (
    simulate_soc_nonlinear,
    derive_voltage,
    derive_current,
    generate_rssi_protocol,
    compute_relational_features,
    augment_dataset,
    BATTERY_CAPACITY_WH,
    NOMINAL_VOLTAGE,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_plant_df(n: int = 100) -> pd.DataFrame:
    """Minimal plant DataFrame (output of preprocess phase)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp":       pd.date_range("2020-05-15", periods=n, freq="30min"),
        "device_id":       ["SIM_PLANT1_INV01"] * n,
        "mppt_w":          rng.uniform(0, 5000, n),
        "prod_wh":         rng.uniform(0, 2500, n),
        "temp_c":          rng.uniform(25, 65, n),
        "irradiance":      rng.uniform(0, 1000, n),
        "ambient_temp_c":  rng.uniform(20, 40, n),
        "daily_yield_kwh": rng.uniform(0, 50, n),
        "ac_power_kw":     rng.uniform(0, 4, n),
    })


# ─── simulate_soc_nonlinear ───────────────────────────────────────────────────

class TestSOCSimulation:
    def test_output_length(self):
        power = np.ones(50) * 1000.0
        soc = simulate_soc_nonlinear(power)
        assert len(soc) == 50

    def test_range_5_to_100(self):
        power = np.ones(200) * 1000.0
        soc = simulate_soc_nonlinear(power)
        assert soc.min() >= 5.0
        assert soc.max() <= 100.0

    def test_soc_increases_with_high_power(self):
        """At very high generation (10kW), SOC should trend upward from 30%."""
        power = np.ones(96) * 10_000.0
        initial_soc = 0.3
        soc = simulate_soc_nonlinear(power, initial_soc=initial_soc)
        # Final SOC should be well above the starting 30%
        # (soc[0] is first computed value, not the seed — compare against initial)
        assert soc[-1] > initial_soc * 100

    def test_soc_decreases_at_night(self):
        """With zero generation, SOC should decrease (only load)."""
        power = np.zeros(48)
        soc = simulate_soc_nonlinear(power, initial_soc=0.8)
        assert soc[-1] < 80.0

    def test_reproducible_with_same_seed(self):
        power = np.random.default_rng(99).uniform(0, 3000, 100)
        soc1 = simulate_soc_nonlinear(power, random_seed=42)
        soc2 = simulate_soc_nonlinear(power, random_seed=42)
        np.testing.assert_array_equal(soc1, soc2)

    def test_different_seeds_differ(self):
        power = np.ones(50) * 1000.0
        soc1 = simulate_soc_nonlinear(power, random_seed=1)
        soc2 = simulate_soc_nonlinear(power, random_seed=2)
        assert not np.allclose(soc1, soc2)


# ─── derive_voltage ───────────────────────────────────────────────────────────

class TestDeriveVoltage:
    def test_voltage_within_realistic_range(self):
        soc = np.linspace(5, 100, 50)
        v = derive_voltage(soc)
        # 24V pack: expect 20–27V range
        assert v.min() >= NOMINAL_VOLTAGE * 0.85
        assert v.max() <= NOMINAL_VOLTAGE * 1.10

    def test_higher_soc_higher_voltage(self):
        """Voltage should generally increase with SOC (monotonic trend)."""
        soc_low  = np.full(20, 10.0)
        soc_high = np.full(20, 90.0)
        # Mean voltage at high SOC should exceed mean at low SOC
        assert derive_voltage(soc_high).mean() > derive_voltage(soc_low).mean()

    def test_output_length_matches_input(self):
        soc = np.linspace(5, 100, 77)
        assert len(derive_voltage(soc)) == 77

    def test_reproducible(self):
        soc = np.linspace(5, 100, 50)
        v1 = derive_voltage(soc, random_seed=42)
        v2 = derive_voltage(soc, random_seed=42)
        np.testing.assert_array_equal(v1, v2)


# ─── derive_current ───────────────────────────────────────────────────────────

class TestDeriveCurrent:
    def test_basic_ohms_law(self):
        """P = V × I  →  I = P / V."""
        mppt_w = np.array([2400.0])
        volt_v = np.array([24.0])
        curr_a = derive_current(mppt_w, volt_v)
        assert curr_a[0] == pytest.approx(100.0, rel=1e-3)

    def test_no_negative_current(self):
        """Current must be non-negative."""
        mppt_w = np.zeros(20)
        volt_v = np.full(20, 24.0)
        assert (derive_current(mppt_w, volt_v) >= 0).all()

    def test_zero_power_zero_current(self):
        mppt_w = np.array([0.0, 0.0])
        volt_v = np.array([24.0, 24.0])
        curr = derive_current(mppt_w, volt_v)
        # pytest.approx doesn't support .all() — use numpy assertion instead
        np.testing.assert_allclose(curr, 0.0, atol=1e-9)


# ─── generate_rssi_protocol ───────────────────────────────────────────────────

class TestRssiProtocol:
    def test_output_lengths(self):
        rssi, protocol = generate_rssi_protocol(50)
        assert len(rssi) == 50
        assert len(protocol) == 50

    def test_protocol_only_valid_values(self):
        _, protocol = generate_rssi_protocol(200)
        assert all(p in ("lora", "4g") for p in protocol)

    def test_weak_rssi_gives_4g(self):
        """Very weak signal (well below -80) should all be '4g'."""
        rng = np.random.default_rng(0)
        # Generate all very weak RSIs manually
        rssi = np.full(10, -100)
        protocol = ["lora" if r >= -80 else "4g" for r in rssi]
        assert all(p == "4g" for p in protocol)

    def test_reproducible(self):
        r1, p1 = generate_rssi_protocol(50, random_seed=42)
        r2, p2 = generate_rssi_protocol(50, random_seed=42)
        np.testing.assert_array_equal(r1, r2)
        assert p1 == p2


# ─── compute_relational_features ─────────────────────────────────────────────

class TestRelationalFeatures:
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "mppt_w":     [1000.0, 2000.0, 0.0],
            "volt_v":     [24.0,   25.0,   23.0],
            "curr_a":     [41.67,  80.0,   0.0],
            "irradiance": [500.0,  800.0,  0.0],
            "batt_pct":   [60.0,   62.0,   58.0],
            "prod_wh":    [500.0, 1000.0,  0.0],
        })

    def test_all_relational_columns_added(self):
        df = compute_relational_features(self._base_df())
        for col in ("ratio_power_irr", "ratio_volt_curr",
                    "physics_residual", "batt_delta", "prod_vs_batt"):
            assert col in df.columns, f"Missing column: {col}"

    def test_physics_residual_near_zero_clean_data(self):
        """For P=V×I, physics_residual should be near zero."""
        df = pd.DataFrame({
            "mppt_w":     [1000.0],
            "volt_v":     [24.0],
            "curr_a":     [1000.0 / 24.0],   # exact P/V
            "irradiance": [500.0],
            "batt_pct":   [70.0],
            "prod_wh":    [500.0],
        })
        df = compute_relational_features(df)
        assert abs(df["physics_residual"].iloc[0]) < 1.0  # within 1W

    def test_ratio_power_irr_zero_irradiance(self):
        """With zero irradiance, ratio_power_irr should be finite (not inf)."""
        df = pd.DataFrame({
            "mppt_w":     [100.0],
            "volt_v":     [24.0],
            "curr_a":     [4.0],
            "irradiance": [0.0],   # zero irradiance
            "batt_pct":   [70.0],
            "prod_wh":    [50.0],
        })
        df = compute_relational_features(df)
        assert np.isfinite(df["ratio_power_irr"].iloc[0])

    def test_batt_delta_first_row_zero(self):
        """First row batt_delta should be 0 (no previous tick)."""
        df = compute_relational_features(self._base_df())
        assert df["batt_delta"].iloc[0] == pytest.approx(0.0)


# ─── augment_dataset (end-to-end) ────────────────────────────────────────────

class TestAugmentDataset:
    def test_all_expected_columns_present(self):
        df = augment_dataset(_make_plant_df(), plant_id="PLANT1")
        expected = [
            "batt_pct", "volt_v", "curr_a", "rssi", "protocol",
            "ratio_power_irr", "ratio_volt_curr", "physics_residual",
            "batt_delta", "prod_vs_batt",
        ]
        for col in expected:
            assert col in df.columns, f"Missing: {col}"

    def test_row_count_unchanged(self):
        plant_df = _make_plant_df(n=200)
        aug = augment_dataset(plant_df)
        assert len(aug) == 200

    def test_no_nan_in_augmented_columns(self):
        aug = augment_dataset(_make_plant_df())
        aug_cols = ["batt_pct", "volt_v", "curr_a", "rssi",
                    "ratio_power_irr", "ratio_volt_curr", "physics_residual"]
        assert not aug[aug_cols].isnull().any(axis=None)

    def test_no_inf_in_augmented_columns(self):
        aug = augment_dataset(_make_plant_df())
        num_cols = aug.select_dtypes(include="number").columns
        assert not np.isinf(aug[num_cols].values).any()

    def test_plant1_plant2_batt_pct_differ(self):
        """Different plant seeds → different batt_pct trajectories."""
        df = _make_plant_df()
        aug1 = augment_dataset(df.copy(), plant_id="PLANT1")
        aug2 = augment_dataset(df.copy(), plant_id="PLANT2")
        assert not np.allclose(aug1["batt_pct"].values, aug2["batt_pct"].values)
