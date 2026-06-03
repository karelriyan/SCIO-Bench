# SCIO-Bench

> **A Labeled Benchmark Dataset and Comparative Study for Anomaly Detection in Off-Grid Renewable Energy IoT Monitoring**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20406391-darkgreen)](https://doi.org/10.5281/zenodo.20406391)
[![Paper: IEEE IoT Journal (Under Review)](https://img.shields.io/badge/Paper-IEEE%20IoT%20Journal-red.svg)]()
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-darkblue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**Affiliation:** Universitas Jenderal Soedirman, Purwokerto, Indonesia  
**Status:** Research Completed — April 2026

---

## Overview

SCIO-Bench is a **publicly available labeled synthetic dataset** and comparative benchmark for anomaly detection in off-grid solar+battery IoT telemetry. This research exposes the critical limitations of using purely unsupervised machine learning on raw electrical telemetry, demonstrating how fundamental physics features can stop cyber-attacks while meteorological variances naturally break detection models. It evaluates four approaches:

1. **Rule-Based Threshold** (SCIO M1 baseline) — MAD-based rules
2. **Isolation Forest** — classical unsupervised ML
3. **LOF (Local Outlier Factor)** — classical unsupervised ML
4. **LSTM Autoencoder** (L1) + **Random Forest** (L2 hierarchical) — lightweight deep learning

The dataset (`scio_bench_dataset.csv`) contains ~3,200 rows with **5 real anomaly types** injected at realistic rates (<10%), plus a **Tropical Weather Stress Test (A6)** and **False Data Injection (A7)**.

---

## Quick Setup

The SCIO-Bench pipeline has been engineered into fully modular Python packages for reproducibility. You can execute the entire pipeline via the master notebook or run each phase independently.

```bash
# 1. Clone the repository
git clone https://github.com/karelriyan/SCIO-Bench.git
cd SCIO-Bench

# 2. Install dependencies
pip install -e .

# 3. Add your Kaggle API key (or run: python setup_kaggle.py)
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 4. Open the master notebook
jupyter notebook notebooks/scio_anomaly_benchmark.ipynb

# 5. Or run each phase independently via terminal:
# python -m src.data.preprocess
# python -m src.data.augmentation
# python -m src.data.anomaly_injection
# python -m src.data.feature_engineering
# python -m src.models.rule_based
# python -m src.models.classical_ml
# python -m src.models.lstm_autoencoder
# python -m src.visualization.plots
```

All outputs are exported to `outputs/{dataset,results,figures}/`.

---

## Project Structure

```
SCIO-Bench/
├── src/
│   ├── data/               # Dataset download, preprocessing, augmentation, injection
│   ├── models/             # Rule-based, IF, LOF, LSTM AE, Hierarchical L2
│   ├── evaluation/         # Metrics, statistical tests, edge profiling
│   ├── xai/                # SHAP analysis, reconstruction error analysis
│   └── visualization/      # Plot generation (5 publication figures)
├── notebooks/
│   └── scio_anomaly_benchmark.ipynb   # Master orchestration notebook
├── outputs/
│   ├── dataset/            # scio_bench_dataset.csv (also on Zenodo)
│   ├── results/            # Table I, II, III as CSV + LaTeX
│   └── figures/            # Figure 1–5 as PDF (300 DPI)
├── data/
│   ├── raw/                # Kaggle download (not committed — see .gitignore)
│   ├── processed/          # Merged, resampled, cleaned CSVs
│   └── splits/             # Train/val/test splits (chronological)
├── tests/                  # Unit tests per component
├── requirements.txt
└── README.md
```

---

## Dataset: SCIO-Bench

| Anomaly Type | Code | Proportion | Description |
|---|---|---|---|
| Panel Degradation | A1 | 2% | Gradual 30–50% power decay over 6h |
| Sudden Panel Drop | A2 | 1.5% | Instant 60–80% drop for 1–3 ticks |
| Battery Fault | A3 | 2% | Rapid SOC drop or stuck value |
| Sensor Drift | A4 | 1.5% | ±15% persistent voltage/current offset |
| Device Offline | A5 | 2% | NaN / last-value-held >3 consecutive ticks |
| Extended Low Irradiance (**Normal!**) | A6 | ~15% | Tropical rainy season — NOT anomaly |
| False Data Injection (Adversarial) | A7 | 1% | Physics-inconsistent sensor manipulation |
| Normal | — | ~76% | Baseline operation |

**Base data:** Kaggle Solar Power Generation Data (Ani Kannal, 2020) — 2 plants, India, 34 days, resampled to 30-minute intervals.

---

## Reproducibility

- All random seeds: `random_state=42`, `np.random.default_rng(42)`
- Chronological train/val/test split (no data leakage)
- Paper available on Zenodo: [DOI: 10.5281/zenodo.20406391]

---

## Citation

```bibtex
@article{scio_bench_2026,
  title   = {Rule-Based vs. Machine Learning Anomaly Detection for Off-Grid Renewable Energy IoT Telemetry: A Comparative Benchmark with Adaptive Thresholding, Cyber-Physical Discrimination, and Hardware-Algorithmic Co-Design},
  author  = {Riyan, Karel Tsalasatir and Setyoko, Muhammad Azka Mauzaky and Ulumudin, Ihya},
  journal = {Zenodo (Preprint)},
  year    = {2026},
  doi     = {10.5281/zenodo.20406391}
}
```

---

## License

- Code: MIT License
- Dataset (SCIO-Bench): CC BY 4.0
