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

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/karelriyan/SCIO-Bench.git
cd SCIO-Bench

# Install dependencies (development mode)
pip install -e .
```

### 2. Secure Credentials Handling

Using Kaggle API credentials securely is crucial. Instead of writing raw credentials to files inside the workspace (which risk accidental exposure), you should configure them using standard environment variables.

**On Linux / macOS / WSL:**
```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
```

**On Windows PowerShell:**
```powershell
$env:KAGGLE_USERNAME="your_kaggle_username"
$env:KAGGLE_KEY="your_kaggle_api_key"
```

When you execute the download script, `src/data/download.py` automatically checks for these environment variables. If present, it writes them securely to `~/.kaggle/kaggle.json` and sets restricted read/write permissions (`chmod 600` under Unix/WSL) automatically.

### 3. Pipeline Execution

You can run the entire workflow in one go by opening the Jupyter Notebook:
```bash
jupyter notebook notebooks/scio_anomaly_benchmark.ipynb
```

Alternatively, you can run individual pipeline phases from the terminal.

---

## Directory Structures & Output Artifacts

Running the pipeline changes the directory structure as files are generated at each phase. Below is the sequence of state transitions:

### 1. Initial State (Source Code Only)
```
SCIO-Bench/
├── src/                 # Codebase modules
├── notebooks/           # Orchestration notebook
├── tests/               # Test suites
├── pyproject.toml
├── requirements.txt
└── README.md
```

### 2. After Phase 1 (Download & Preprocessing)
*Downloads raw Kaggle dataset and exports cleaned CSVs for each plant.*
*   **Run command:** `python -m src.data.preprocess` (which runs `download` automatically if missing)
*   **Outputs:**
```
data/
├── raw/
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   └── Plant_2_Weather_Sensor_Data.csv
└── processed/
    ├── plant1_clean.csv
    └── plant2_clean.csv
```

### 3. After Phase 2 (Synthetic Variable Augmentation)
*Simulates battery telemetry (SOC, voltage, current) and communication stats (RSSI, protocol).*
*   **Run command:** `python -m src.data.augmentation`
*   **Outputs:**
```
data/
└── processed/
    ├── plant1_augmented.csv
    └── plant2_augmented.csv
```

### 4. After Phase 3 (Anomaly Injection)
*Injects anomalies (A1-A5, A7) and weather events (A6) to form the combined benchmark dataset.*
*   **Run command:** `python -m src.data.anomaly_injection`
*   **Outputs:**
```
outputs/
└── dataset/
    └── scio_bench_dataset.csv
```

### 5. After Phase 4 (Feature Engineering & Splitting)
*Computes lags, cyclic time features, rolling statistics, and partitions splits chronologically.*
*   **Run command:** `python -m src.data.feature_engineering`
*   **Outputs:**
```
data/
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
outputs/
└── dataset/
    └── scaler.pkl           # StandardScaler fitted on train split
```

### 6. After Model Training & Evaluation (Phases 5–8)
*Fits rule-based, classical ML, deep autoencoders, and hierarchical classifiers.*
*   **Run commands:**
    ```bash
    python -m src.models.rule_based
    python -m src.models.classical_ml
    python -m src.models.lstm_autoencoder
    python -m src.models.l2_classifier
    ```
*   **Outputs:**
```
outputs/
└── results/
    ├── rule_based_model.json
    ├── rule_based_k_sweep.csv
    ├── rule_based_results.csv
    ├── isolation_forest_model.pkl
    ├── lof_model.pkl
    ├── if_sweep.csv
    ├── lof_sweep.csv
    ├── classical_ml_results.csv
    ├── lstm_ae_model.keras
    ├── lstm_ae_model_meta.json
    ├── lstm_ae_results.csv
    ├── l2_model.pkl
    ├── l2_feature_importances.csv
    ├── l2_confusion_matrix_test.csv
    └── l2_results.csv
```

### 7. After Explainability, Profiling, & Plots (Phases 9–12)
*Runs SHAP, reconstruction heatmaps, latency profiling, and generates publication figures.*
*   **Run commands:**
    ```bash
    python -m src.xai.shap_analysis
    python -m src.xai.reconstruction_analysis
    python -m src.evaluation.statistical_tests
    python -m src.evaluation.edge_profiling
    python -m src.visualization.plots
    ```
*   **Outputs:**
```
outputs/
├── results/
│   ├── if_shap_values.npy
│   ├── if_shap_features.npy
│   ├── if_shap_feat_cols.pkl
│   ├── lstmae_reconstruction_errors_test.csv
│   ├── Table_1_F1_Per_Class.csv
│   ├── Table_2_Overall_Metrics.csv
│   ├── Table_3_Hyperparams.csv
│   ├── edge_profiling_results.csv
│   └── lstm_ae_quantized.tflite     # INT8 model for microcontrollers
└── figures/
    ├── Figure_1_ROC_Curves.pdf
    ├── Figure_2_Time_Series.pdf
    ├── Figure_3_F1_BarChart.pdf
    ├── Figure_4_SHAP_Summary.pdf
    └── Figure_5_Recon_Heatmap.pdf
```

---

## Executing Partial Pipeline Steps

The SCIO-Bench codebase is fully modular. You can re-run parts of the pipeline without starting from scratch, provided the required upstream files exist:

*   **Tuning/Re-fitting Models:** If the splits are already generated in `data/splits/`, you can train or tune individual models directly:
    ```bash
    python -m src.models.rule_based
    python -m src.models.classical_ml
    python -m src.models.lstm_autoencoder
    ```
*   **Re-running the Hierarchical L2 Classifier:** Phase 8 requires `train.csv`/`test.csv` in `data/splits/` and Keras model files in `outputs/results/`. You can execute:
    ```bash
    python -m src.models.l2_classifier
    ```
*   **Regenerating Plots & Tables:** If you want to modify chart parameters or format tables, you can run the visualization/stats scripts directly without retraining the models:
    ```bash
    python -m src.evaluation.statistical_tests
    python -m src.visualization.plots
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
