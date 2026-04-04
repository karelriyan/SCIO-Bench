# SCIO Research Framework
## Anomaly Detection for Off-Grid Renewable Energy IoT Monitoring
### Kerangka Riset Lengkap — Acuan Eksekusi untuk Claude Code

---

## METADATA

```
Tanggal Dibuat  : April 2026
Penulis Kerangka: Karel Tsalasatir Riyan (CEO, SCIO)
Afiliasi        : Universitas Jenderal Soedirman, Purwokerto, Indonesia
Status          : READY FOR EXECUTION (v2.0 — post peer review revision)
Revisi v2.0     : Proporsi anomali realistis (<10%), SOC non-linear, XAI (SHAP),
                  Tropical Stress Test (A6), Fair Benchmark (Grid Search semua metode)
Revisi v3.0     : Adversarial Resilience (A7 False Data Injection), Edge Hardware
                  Profiling (RAM/CPU/ESP32), Imbalanced Data Strategy (SMOTE mention +
                  threshold tuning justification), 4 novelty contributions, RQ6
Revisi v4.0     : Relational features (physics_residual, V/I ratio, Power/Irradiance),
                  MAD-based threshold (robust vs 3-sigma), Two-Layer Hierarchical Model
                  (L1 semi-supervised + L2 supervised+SMOTE), Alarm-Budgeted Event-Level
                  Evaluation (MIR@k), Local Physical Alarm for 3T (LED/buzzer onsite),
                  TFLite INT8 quantization, Robust NaN/Inf preprocessing
Target Platform : Google Colab (Free Tier) — zero cost
Target Publish  : TechRxiv (IEEE Preprint) + Zenodo (DOI backup)
Deadline        : 3 hari dari tanggal pembuatan
```

---

## BAGIAN 1 — HASIL DEEP RESEARCH

### 1.1 Kriteria Paper Berkualitas Tinggi (IEEE/MDPI Standard)

Berdasarkan analisis standar MDPI Sensors, MDPI Energies, dan IEEE IoT Journal 2024–2025:

**Struktur yang Wajib Ada (rejection jika tidak ada):**
- [ ] Clear problem statement dengan justifikasi mengapa masalah ini penting
- [ ] Explicit research gap — mengutip paper sebelumnya dan menunjukkan apa yang belum dijawab
- [ ] Novelty claim yang spesifik (bukan "kami mengusulkan metode baru" — harus konkret)
- [ ] Dataset yang jelas sumbernya, reproducible, dan diakses publik
- [ ] Comparison baseline — minimal 2 metode pembanding
- [ ] Statistical evaluation: F1-Score, Precision, Recall adalah minimum. AUC-ROC dan Average Detection Latency menambah nilai signifikan
- [ ] Reproducibility: kode tersedia (GitHub link), random seed ditetapkan, environment dicatat
- [ ] Limitation section — paper tanpa limitation dianggap tidak jujur secara akademik

**Metrik Evaluasi Wajib untuk Anomaly Detection (IEEE Standard):**
| Metrik | Wajib? | Keterangan |
|--------|--------|-----------|
| Precision | ✅ | TP / (TP + FP) |
| Recall (Sensitivity) | ✅ | TP / (TP + FN) |
| F1-Score | ✅ | Harmonic mean Precision & Recall |
| AUC-ROC | ✅ Sangat disarankan | Area under ROC curve, threshold-independent |
| Average Detection Latency (ADL) | ✅ untuk IoT paper | Berapa "tick" setelah anomali terjadi, baru terdeteksi |
| False Positive Rate (FPR) | Disarankan | Critical untuk sistem real-time |
| Inference Time (ms) | Disarankan untuk edge IoT | Penting untuk klaim "lightweight" |

**Pola Paper yang Diterima di MDPI Energies/Sensors (2023–2025):**
- Fokus narrow dan spesifik lebih baik daripada broad claim
- Tabel hasil harus ada standard deviation atau confidence interval
- Visualisasi: confusion matrix + ROC curve + time-series plot dengan anomali ter-highlight
- Related work mengutip minimal 25–30 referensi, dominan 2021–2025

**Pelajaran dari Paper Unsoed (Alkaf et al., 2026, AITI Journal):**
> Paper dari Universitas Jenderal Soedirman tentang anomaly detection solar PV menggunakan
> K-means + Isolation Forest berhasil dipublikasikan — memvalidasi bahwa topik ini acceptable
> untuk afiliasi Unsoed. Paper ini menjadi referensi wajib dan bukti prior work dari institusi.

---

### 1.2 State of the Art & Research Gap

**Paper Kunci yang Harus Dikutip:**

| Paper | Tahun | Metode | Dataset | Gap yang Ditinggalkan |
|-------|-------|--------|---------|----------------------|
| Alkaf et al., AITI Journal | 2026 | K-means, Isolation Forest | Solar PV dataset | Tidak ada battery monitoring; tidak ada multi-metric comparison |
| MDPI Energies (Machine Learning Schemes for PV) | 2022 | AE-LSTM, Prophet, Isolation Forest | Grid-connected PV | Grid-connected only, bukan off-grid; tidak ada context developing country |
| PMC/ETRI (Ambat et al.) | 2025 | Multiple ML + PyOD | Smart home | Rumah tangga perkotaan; tidak ada battery+solar combined |
| IEEE Sensors (Nizam et al.) | 2022 | Deep anomaly detection | Industrial IoT multivariate | Industrial context; tidak ada EBT off-grid |
| Computer Science Review (Survey) | 2024 | Survey | Multiple | Identifies gap: kurangnya dataset untuk off-grid renewable energy IoT |

**Research Gap yang Diidentifikasi (Novelty Claim):**

1. **Gap Utama (paling kuat):** Tidak ada studi komparatif sistematis antara rule-based threshold, classical unsupervised ML, dan lightweight deep learning untuk konteks *off-grid* solar+battery monitoring di *developing countries* (specifically: sistem kecil 50–500W, battery 12–48V, monitoring interval 30 detik).

2. **Gap Dataset:** Dataset yang ada (Kaggle Solar Power Generation, dll.) adalah grid-connected utility-scale. Tidak ada public benchmark untuk off-grid residential/small-commercial dengan combined solar+battery telemetry.

3. **Gap Praktis:** Paper sebelumnya tidak mempertimbangkan *deployment constraint* nyata: latency requirement, model size, dan inference time untuk resource-constrained IoT gateway.

4. **Gap Konteks:** Pola anomali di Indonesia (musim hujan panjang, suhu tinggi, dusty environment, intermittent grid) berbeda dari dataset Eropa/India yang mendominasi literatur.

**Novelty Claim yang Diusulkan (Updated — 3 kontribusi eksplisit):**
> "Kami mempresentasikan studi komparatif pertama yang secara sistematis mengevaluasi tiga
> pendekatan anomaly detection (rule-based, classical ML unsupervised, LSTM Autoencoder) pada
> dataset solar+battery IoT off-grid dengan konteks operasional Indonesia. Tiga kontribusi utama:
> (1) SCIO-Bench — dataset sintetis berlabel publik dengan 5 tipe anomali pada proporsi realistis
> (<10%), derived dari data solar nyata; (2) Tropical Weather Stress Test (Extended Low Irradiance)
> — skenario baru untuk mengukur FPR saat musim hujan panjang, belum ada di benchmark sebelumnya;
> (3) SHAP-based Explainability — memvalidasi bahwa model belajar representasi fisik yang benar,
> bukan spurious correlation."

---

### 1.3 Dataset Selection

**Dataset Kaggle yang Dipilih (PRIMARY):**

```
Nama    : Solar Power Generation Data
Author  : Ani Kannal
URL     : https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
Ukuran  : ~3.000 baris per plant × 2 plant × 2 file (generation + weather sensor)
Resolusi: 15 menit
Durasi  : 34 hari (Mei–Juni 2020)
Lokasi  : 2 solar plant di India (iklim mirip Indonesia)
```

**Kolom Tersedia:**
| Kolom Dataset | Variabel SCIO | Mapping |
|---------------|---------------|---------|
| DC_POWER (kW) | `mppt_w` (W) | Direct × 1000 |
| AC_POWER (kW) | `prod_wh` proxy | Derived |
| IRRADIATION (W/m²) | Feature tambahan | Weather input |
| MODULE_TEMPERATURE (°C) | `temp_c` | Direct |
| AMBIENT_TEMPERATURE (°C) | Feature tambahan | Environmental context |
| DAILY_YIELD (kWh) | Cumulative `prod_wh` | Derived |

**Kolom yang Perlu Ditambahkan via Synthetic Augmentation:**
| Kolom SCIO | Strategi |
|-----------|----------|
| `batt_pct` | Simulasi fisika: SOC model sederhana (kapasitas 100Ah, charge rate dari DC_POWER) |
| `volt_v` | Derived dari DC_POWER dan estimated current (P = V×I, asumsikan V nominal 12/24V + ripple) |
| `curr_a` | Derived: `curr_a = DC_POWER × 1000 / volt_v` |

**Dataset Sekunder (untuk validasi silang):**
```
Nama    : Hourly Energy Consumption (dari berbagai smart home dataset di Kaggle)
Tujuan  : Verifikasi bahwa metode generalisasi ke pola konsumsi berbeda
```

---

### 1.4 Platform Publikasi — Keputusan Final

**Urutan Strategi Publikasi:**

| Platform | Kecepatan | Endorsement? | DOI? | Gratis? | Keputusan |
|----------|-----------|-------------|------|---------|-----------|
| **Zenodo** | Instan (< 1 jam) | ❌ Tidak ada | ✅ Otomatis (10.5281/zenodo.xxx) | ✅ | **PUBLISH PERTAMA — untuk timestamp & DOI** |
| **TechRxiv (IEEE)** | 1–3 hari (screening only) | ❌ Tidak ada | ✅ DOI TechRxiv | ✅ | **SUBMIT BERSAMAAN — untuk credibility IEEE** |
| arXiv | Butuh endorsement | ✅ WAJIB | ✅ | ✅ | Setelah P2MW, cari endorser |

**Instruksi Zenodo:**
1. Buat akun di zenodo.org (pakai email Unsoed)
2. Klik "+ New Upload"
3. Resource type: **Publication → Preprint**
4. Afiliasi: Universitas Jenderal Soedirman
5. Klik "Get a DOI now!" sebelum upload — DOI bisa dimasukkan ke dalam PDF
6. License: CC BY 4.0
7. Klik Publish → DOI aktif dalam < 1 menit

**Instruksi TechRxiv:**
1. Daftar di techrxiv.org
2. Submit → Category: **Signal Processing** atau **Computing and Processing**
3. Upload PDF, isi metadata lengkap
4. Screening 1–3 hari (bukan peer review, hanya cek basic quality)

---

## BAGIAN 2 — JUDUL PAPER (3 OPSI)

**Opsi A (paling kuat untuk IEEE):**
> "Rule-Based vs. Machine Learning Anomaly Detection for Off-Grid Solar-Battery IoT Telemetry: A Comparative Benchmark with Synthetic Dataset"

**Opsi B (fokus pada novelty dataset):**
> "SCIO-Bench: A Labeled Benchmark Dataset and Comparative Study for Anomaly Detection in Distributed Off-Grid Renewable Energy Monitoring Systems"

**Opsi C (fokus pada deployment):**
> "Lightweight Anomaly Detection for Resource-Constrained IoT in Off-Grid Renewable Energy Systems: A Three-Method Comparative Evaluation"

**Rekomendasi:** Opsi A atau B. Opsi B lebih kuat karena menyertakan nama dataset — meningkatkan citability.

---

## BAGIAN 3 — PROBLEM STATEMENT & NOVELTY

**Problem Statement:**
Off-grid renewable energy systems (solar PV + battery storage) deployed in rural and
remote areas of developing countries — particularly Indonesia's 3T regions (Terdepan,
Terluar, Tertinggal) — suffer from undetected performance degradation and failures.
Unlike grid-connected utility-scale systems, these small off-grid installations (50–500W
solar, 12–48V battery, 30-second telemetry interval) present unique challenges:
(1) limited labeled fault data, (2) deployment in resource-constrained IoT environments,
and (3) operational patterns that differ significantly from temperate-climate benchmarks.

**Novelty Statement (untuk abstract):**
"To the best of our knowledge, this is the first systematic comparative benchmark of
rule-based thresholding, classical unsupervised ML (Isolation Forest, Local Outlier
Factor), and lightweight deep learning (LSTM Autoencoder) specifically for off-grid
solar-battery IoT monitoring in tropical developing-country context. We make three
contributions: (1) SCIO-Bench — a publicly available labeled synthetic dataset with
five anomaly types injected at realistic rates (<10%); (2) a novel Tropical Weather
Stress Test (Extended Low Irradiance) evaluating model robustness against false positives
during prolonged cloud cover — a scenario absent from prior benchmarks; and (3) SHAP-based
explainability analysis validating that detected anomalies correspond to physically
meaningful sensor deviations rather than spurious correlations; and (4) adversarial
resilience evaluation against False Data Injection attacks — sensor manipulation
designed to stay within normal individual ranges yet violate physical consistency."

---

## BAGIAN 4 — RESEARCH QUESTIONS

| ID | Research Question | Measurable Answer |
|----|------------------|-------------------|
| RQ1 | Apakah rule-based threshold (baseline M1 SCIO) mampu mendeteksi kelima tipe anomali dengan F1 > 0.7 pada seluruh kondisi? | F1-Score per anomaly type per method |
| RQ2 | Apakah unsupervised ML (Isolation Forest, LOF) meningkatkan F1 secara signifikan dibanding rule-based, tanpa label training? | Delta F1 + statistical significance test |
| RQ3 | Apakah LSTM Autoencoder memberikan Average Detection Latency (ADL) yang lebih rendah dibanding metode non-sequence-aware? | ADL dalam satuan "number of 30-second ticks" |
| RQ4 | Apakah ada trade-off yang signifikan antara F1-Score dan inference time (ms) di antara ketiga metode, yang relevan untuk edge deployment? | Pareto frontier: F1 vs. inference time |
| RQ5 | Apakah fitur yang diidentifikasi paling penting oleh SHAP (Isolation Forest) dan reconstruction error (LSTM AE) konsisten dengan ground-truth anomaly type yang diinjeksikan? | Qualitative match antara top-SHAP features dan variabel yang dimodifikasi per anomaly type |
| RQ6 | Apakah ketiga metode mampu mendeteksi serangan False Data Injection (A7) yang dirancang berada dalam rentang nilai "normal" secara individual, namun tidak konsisten secara fisik? | F1-Score khusus A7 per method — rule-based diprediksi gagal; hipotesis: physics_residual feature + LSTM AE unggul |
| RQ7 | Apakah arsitektur dua lapis (L1 LSTM AE + L2 Random Forest) menghasilkan Missed Incident Rate (MIR) yang lebih rendah dalam skenario budget alarm terbatas (10 alarms/hari) dibanding metode single-layer? | MIR@10 per metode: L1+L2 diharapkan unggul karena pemisahan tugas deteksi vs klasifikasi |

---

## BAGIAN 5 — DATASET STRATEGY

### 5.1 Dataset Pipeline

```
[STEP 1] Download Kaggle Dataset
         └── kaggle datasets download -d anikannal/solar-power-generation-data
         └── Files: Plant_1_Generation_Data.csv, Plant_1_Weather_Sensor_Data.csv
                    Plant_2_Generation_Data.csv, Plant_2_Weather_Sensor_Data.csv

[STEP 2] Preprocessing & Alignment
         └── Merge generation + weather pada DATE_TIME
         └── Resample ke interval 30 menit (dari 15 menit) → match SCIO interval
         └── Handle missing values: forward-fill max 2 consecutive NaN
         └── Rename columns ke SCIO naming convention

[STEP 3] Synthetic Variable Augmentation
         └── Tambah batt_pct: SOC simulation model
         └── Tambah volt_v: derived dari DC_POWER nominal voltage model
         └── Tambah curr_a: derived P = V×I
         └── Tambah rssi: random normal distribution (-70 ± 15 dBm)
         └── Tambah protocol: categorical (lora/4g), based on signal quality

[STEP 4] Anomaly Injection (5 tipe)
         └── Injeksi ke proporsi tertentu dari dataset
         └── Label ground truth disimpan di kolom 'anomaly_type' & 'is_anomaly'

[STEP 5] Train/Val/Test Split
         └── Chronological split (JANGAN random — time series data)
         └── Train: hari 1–20, Val: hari 21–25, Test: hari 26–34
```

### 5.2 Battery SOC Simulation Model (Improved — Non-Linear)

> ⚠️ **Peer review note:** Model SOC linier sederhana akan dikritik reviewer kelistrikan.
> Gunakan pendekatan berikut untuk membuat fluktuasi batt_pct lebih organik.

**Pendekatan: Hybrid Physics + Noise dari NASA Battery Dataset**

```python
# Model SOC non-linier dengan degradation noise
# Referensi: NASA Battery Dataset (B0005, B0006, B0007, B0018)
# https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

BATTERY_CAPACITY_WH = 1200  # 100Ah × 12V
INITIAL_SOC = 0.7
CHARGE_EFF = 0.92            # non-linear: menurun saat SOC tinggi
DISCHARGE_EFF = 0.95
DEGRADATION_RATE = 0.0002    # kapasitas turun 0.02% per siklus

def simulate_soc_nonlinear(dc_power_series, timestamps, cycle_count=0):
    """
    Model SOC non-linier dengan:
    - Efficiency yang menurun saat SOC mendekati 1.0 (tapering effect)
    - Degradation ringan per siklus
    - Gaussian noise untuk organic fluctuation (std=0.3%)
    """
    soc = INITIAL_SOC
    soc_series = []
    capacity = BATTERY_CAPACITY_WH * (1 - DEGRADATION_RATE * cycle_count)
    
    for power, ts in zip(dc_power_series, timestamps):
        dt = 0.5  # 30 menit
        load = 50  # constant load (W)
        
        # Tapering: efisiensi charger menurun saat SOC > 0.8
        eff = CHARGE_EFF * (1 - 0.3 * max(0, soc - 0.8))
        
        net = (power * eff - load * (1/DISCHARGE_EFF)) * dt
        soc = soc + net / capacity
        soc = max(0.05, min(1.0, soc))
        
        # Organic noise (sensor + BMS rounding)
        noise = np.random.normal(0, 0.003)
        soc_series.append(np.clip(soc + noise, 0.05, 1.0) * 100)
    
    return soc_series

# Tambahkan voltage ripple non-linier (Peukert effect approximation)
def derive_voltage(soc_series, nominal_v=24.0):
    """
    Voltage non-linier mengikuti SOC curve (discharge curve Li-Ion)
    Menggunakan polynomial fit dari kurva discharge tipikal 24V LiFePO4
    """
    soc_normalized = np.array(soc_series) / 100
    # Polynomial approximation: V = a*SOC^3 + b*SOC^2 + c*SOC + d
    v = (2.1 * soc_normalized**3 - 3.8 * soc_normalized**2 
         + 2.9 * soc_normalized + 0.8) * nominal_v
    noise = np.random.normal(0, 0.15, len(v))  # sensor noise ±0.15V
    return v + noise
```

**Dataset Baterai Referensi untuk Validasi:**
- NASA Battery Dataset: https://data.nasa.gov/dataset/Li-ion-Battery-Aging-Datasets/uj5r-zjdb
- Synthetic Distributed Adaptive BMS Dataset (Kaggle): search "battery aging dataset"
- Gunakan sebagai referensi noise pattern, bukan sebagai dataset utama

### 5.3 Lima Tipe Anomali yang Diinjeksikan

| ID | Nama Anomali | Variabel Terdampak | Pola Injeksi | Proporsi | Justifikasi Operasional |
|----|-------------|-------------------|--------------|----------|------------------------|
| A1 | Panel Degradation | dc_power, mppt_w | Gradual decay 30–50% selama 6 jam | 2% data | Soiling, partial shading progresif |
| A2 | Sudden Panel Drop | dc_power, mppt_w, volt_v | Spike turun 60–80% selama 1–3 tick | 1.5% data | Bayangan awan tebal, bird dropping |
| A3 | Battery Fault | batt_pct | Penurunan abnormal cepat (>5%/tick) atau stuck di nilai tetap | 2% data | Sel rusak, sulfation, disconnection |
| A4 | Sensor Drift | volt_v atau curr_a | Offset +/-15% persistent | 1.5% data | Kalibrasi error, koneksi longgar |
| A5 | Device Offline | semua variabel | Nilai NaN atau last-value-held > 3 tick berturut | 2% data | Jaringan putus, hardware fault |
| A6 | Extended Low Irradiance (Normal!) | dc_power, mppt_w | Penurunan 50–80% selama 12–48 jam berturut | ±15% data | **SKENARIO NORMAL**: musim hujan panjang Indonesia — model HARUS tidak mendeteksi ini sebagai anomali. Digunakan untuk evaluasi FPR. |
| A7 | False Data Injection (Adversarial) | volt_v, curr_a | Nilai dimanipulasi sistematis dalam range "normal" tapi tidak konsisten secara fisik (V↑ tapi P↓) | 1% data | **SERANGAN SIBER**: manipulasi sensor yang disengaja. Sulit dideteksi rule-based. Digunakan untuk evaluasi adversarial resilience. |
| - | Normal | - | Baseline behavior | **~90% data** | - |

> ⚠️ **CATATAN KRITIS (dari peer review internal):** Total anomali nyata hanya **~9%** dari dataset.
> Ini merepresentasikan kondisi riil (benchmark SWaT/WADI: 1–5%; sistem EBT off-grid: estimasi 5–10%).
> Reviewer akan menolak paper dengan proporsi anomali >15% karena menjadi masalah klasifikasi biasa, bukan anomaly detection.
> 
> **Skenario A6 (Extended Low Irradiance) adalah inovasi kunci paper ini**: membuktikan model mampu
> membedakan "kerusakan alat" vs "cuaca buruk" — research contribution yang belum ada di paper sebelumnya.

### 5.4 SCIO-Bench Dataset Schema (Output Final)

```
Kolom Output:
- timestamp (YYYY-MM-DD HH:MM:SS)
- device_id (string, contoh: "SIM_PLANT1_INV01")
- prod_wh (float, Wh)
- batt_pct (float, %)
- volt_v (float, V)
- curr_a (float, A)
- mppt_w (float, W)
- temp_c (float, °C)
- irradiance (float, W/m², fitur tambahan)
- rssi (int, dBm)
- protocol (categorical: lora/4g)
- is_anomaly (bool)
- anomaly_type (string: normal/panel_degradation/sudden_drop/battery_fault/sensor_drift/offline)

Target: ~20.000 baris total (2 plant × 34 hari × 48 interval/hari)
Dataset dipublikasikan di Zenodo sebagai kontribusi terpisah (DOI tersendiri)
```

---

## BAGIAN 6 — METODOLOGI EKSPERIMEN

### 6.1 Tiga Metode yang Dibandingkan

#### Metode 1: Rule-Based Threshold (Baseline — M1 SCIO)

```
Justifikasi: Ini adalah implementasi aktual di SCIO M1. Menjadi baseline yang valid dan
relevan secara industrial karena merepresentasikan "current practice".

Aturan (sesuai SRS SCIO FR-066) — menggunakan MAD bukan StdDev untuk robustness:
- R1: prod_wh < 70% dari rolling 7-day median → WARNING
- R2: batt_pct < 10% → WARNING
- R3: batt_pct < 5% → CRITICAL
- R4: temp_c > 70°C → WARNING
- R5: device offline > 3 consecutive NaN → WARNING
- R6: volt_v deviasi > Median ± 3×MAD (rolling 24h) → WARNING
- R7: physics_residual (P - V×I) > Median ± 3×MAD (rolling 6h) → WARNING [FDI detection]

Implementasi MAD-based threshold (WAJIB, bukan 3-sigma):
  from scipy.stats import median_abs_deviation
  MAD = median_abs_deviation(rolling_window, scale='normal')
  # scale='normal' → konsisten dengan sigma unit untuk perbandingan
  # MAD tidak terpengaruh outlier anomali itu sendiri (robust estimator)
  # Literatur: Leys et al. (2013) merekomendasikan MAD untuk outlier detection

Implementasi: Pure Python + Pandas + scipy, no ML library
```

#### Metode 2: Classical Unsupervised ML

```
Justifikasi: Unsupervised lebih realistis untuk IoT deployment karena label anomali sangat
sulit diperoleh di lapangan. Isolation Forest dan LOF adalah state-of-the-art untuk
tabular time-series IoT (berdasarkan survey Ambat et al. 2025 dan Alkaf et al. 2026).

Algoritma yang Diuji:
a) Isolation Forest (scikit-learn IsolationForest)
   - contamination: [0.02, 0.05, 0.08, 0.10] → Grid Search di val set, optimize F1 macro
     (range disesuaikan dengan proporsi anomali nyata ~9%)
   - n_estimators: 100
   - random_state: 42
   - Feature input: [prod_wh, batt_pct, volt_v, curr_a, mppt_w, temp_c]

b) Local Outlier Factor (scikit-learn LocalOutlierFactor)
   - n_neighbors: [10, 20, 30] → Grid Search di val set
   - contamination: [0.02, 0.05, 0.08, 0.10] → Grid Search
   - novelty: True (untuk test set inference)
   - Feature input: sama dengan IF

> FAIR BENCHMARK: Semua metode HARUS dioptimalkan via Grid Search pada val set sebelum
> dievaluasi di test set. Laporkan best_params di paper (Table atau footnote). Ini mencegah
> bias komparasi yang sering jadi alasan rejection reviewer.

Preprocessing yang wajib:
- StandardScaler normalization (fit pada train set, transform pada val+test)
- Rolling window features: tambahkan mean_6h, std_6h, delta_1tick untuk setiap variabel
  (ini meningkatkan detection signifikan berdasarkan literature)
- A6 weather event flag: tambahkan binary feature 'is_low_irradiance_period'
  (rolling 6h mean irradiance < 50 W/m²) → membantu model membedakan cuaca vs fault
- **Relational Features (BARU — scale-invariant, kritis untuk FDI detection):**
  ```python
  # Rasio fisik: kebal terhadap perubahan skala dan common-mode effects
  df['ratio_power_irr'] = df['mppt_w'] / (df['irradiance'] + 1e-6)   # DC Power/Irradiance
  df['ratio_volt_curr'] = df['volt_v'] / (df['curr_a'] + 1e-6)        # V/I (impedance proxy)
  df['physics_residual'] = df['mppt_w'] - df['volt_v'] * df['curr_a'] # P - V×I (harusnya ≈0)
  df['batt_delta']       = df['batt_pct'].diff()                        # rate of change SOC
  df['prod_vs_batt']     = df['prod_wh'] - df['batt_pct'].diff() * CAPACITY_WH / 100
  # physics_residual dan ratio_volt_curr adalah fitur paling sensitif untuk A7 (FDI)
  # karena manipulator harus melanggar P=V×I untuk menipu sistem
  ```
  Sertakan fitur relasional ini di SEMUA model (Rule-Based, IF, LOF, LSTM AE, L1/L2)

**Imbalanced Data Strategy (WAJIB didokumentasikan di paper):**
> Dengan proporsi anomali ~10%, model ML akan menghadami extreme class imbalance.
> Tanpa penanganan eksplisit, model bisa bias memprediksi "Normal" terus (accuracy tinggi,
> recall anomali = 0). Strategi yang digunakan:

```python
# UNTUK TRAINING IF dan LOF: tidak perlu (unsupervised, tidak ada label saat training)
# contamination parameter sudah merefleksikan expected anomaly rate

# UNTUK EVALUASI: gunakan F1-Score (bukan Accuracy!) sebagai primary metric
# Accuracy misleading pada imbalanced data — JANGAN jadikan headline metric

# JIKA F1 masih rendah karena imbalance → coba strategi berikut (opsional):
# Opsi A: SMOTE pada feature space untuk oversampling anomali di training
from imblearn.over_sampling import SMOTE
# HANYA untuk supervised baseline jika ditambahkan
# JANGAN gunakan SMOTE pada data LSTM AE (semi-supervised, train pada normal saja)

# Opsi B: Cost-sensitive learning (weight anomali lebih tinggi)
# sklearn IF tidak support sample_weight langsung
# LOF: tidak support; gunakan sebagai-is dengan contamination tuning

# Opsi C (yang digunakan): threshold tuning via val set
# Ini adalah strategi paling tepat untuk unsupervised AD dengan imbalanced data
# threshold dipilih untuk maximize F1 di val set, bukan default 0.5

# LAPORAN di paper: cantumkan class distribution secara eksplisit
# "The dataset contains X% anomalous samples (n=XXX) out of Y total samples (n=YYYY),
#  creating a class imbalance ratio of approximately 1:Z"
```

> **Catatan untuk penulis:** Reviewer PASTI menanyakan ini. Jawab di Methodology Section 3.7:
> "Given the extreme class imbalance (~10% anomalies), we adopt threshold optimization
> on the validation set as our primary imbalance mitigation strategy, consistent with
> unsupervised anomaly detection literature [cite]. We report macro-averaged F1-Score
> as our primary metric to avoid the accuracy paradox on imbalanced datasets."
```

#### Metode 3: LSTM Autoencoder (Lightweight Deep Learning)

```
Justifikasi: Sequence-aware model dapat mendeteksi temporal anomali yang tidak bisa
ditangkap metode stateless (rule-based, IF, LOF). LSTM Autoencoder adalah pilihan paling
ringan dari deep learning family untuk IoT context.

Arsitektur:
Input  → LSTM(32 units) → LSTM(16 units) → [encoded] 
       → RepeatVector(sequence_len)
       → LSTM(16 units, return_sequences=True)
       → LSTM(32 units, return_sequences=True)
       → TimeDistributed(Dense(n_features))
       → Output (reconstruction)

Hyperparameter — dicari via Grid Search pada val set (fair benchmark):
- sequence_length: [4, 6, 8] → pilih yang maximize F1 di val
- epochs: 30 (early stopping patience=5)
- batch_size: 32
- learning_rate: 0.001 (Adam)
- loss: MSE
- threshold candidates: [percentile 90, 95, 99] dari train reconstruction error
- threshold final: dipilih di val set berdasarkan maximize F1

Anomaly Decision:
- reconstruction_error > threshold → anomaly=True
- threshold TIDAK boleh dicari di test set (data leakage!)

Training:
- Train HANYA pada data normal (is_anomaly==False, is_weather_event==False)
- Semi-supervised approach yang realistis untuk IoT deployment

Framework: TensorFlow/Keras (gratis, tersedia di Colab)
Estimasi training time: < 5 menit di Colab free T4 GPU
```

#### Metode 4: Two-Layer Hierarchical Detection (Arsitektur Utama)

> **[UPGRADE ARSITEKTUR — berdasarkan critique tingkat lanjut]**
> Mendeteksi 7 kelas sekaligus (Normal + A1–A5 + A7) memberatkan model single-layer
> dan menurunkan akurasi. Arsitektur dua lapis lebih efisien di edge dan lebih akurat.

```
LAYER 1 (L1) — Binary Detector: Normal vs. Anomaly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model    : LSTM Autoencoder (semi-supervised, train pada normal data saja)
Input    : 6 sensor features + 5 relational features (11 total)
Output   : is_anomaly (bool) berdasarkan reconstruction error > MAD-threshold
Keunggulan: Tidak perlu label → deployable tanpa labeled training data
Edge     : Jalankan SETIAP 30 detik di edge device (ESP32-S3 / RPi4)
Trigger  : Jika is_anomaly == True → kirim ke L2

LAYER 2 (L2) — Multi-Class Classifier: Anomaly Type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model    : Random Forest atau XGBoost (supervised, butuh labeled data)
Input    : Sama dengan L1 + L1 reconstruction error per feature (interpretability)
Output   : anomaly_type ∈ {A1, A2, A3, A4, A5, A7}
Training : Gunakan SMOTE pada training set untuk handle extreme class imbalance
           (SMOTE hanya pada L2 training — bukan L1 yang semi-supervised)
Edge     : Jalankan HANYA ketika L1 trigger → hemat komputasi

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42, k_neighbors=3)  # k_neighbors=3 karena anomali langka
X_l2_train, y_l2_train = sm.fit_resample(
    X_train_anomalies_only, y_train_anomaly_types)
# PENTING: SMOTE hanya pada anomali yang terdeteksi L1, tidak pada data normal
# Ini mencegah bias model memprediksi tipe anomali paling banyak

ALUR KESELURUHAN:
Sensor Data (30s) → L1 (LSTM AE) → Normal? → Log + Continue
                                  → Anomaly? → L2 (RF/XGB) → {A1,A2,A3,A4,A5,A7}
                                                             → Local Alert (LED/Buzzer)
                                                             → Cloud Notification (jika online)

KEUNTUNGAN HIERARKI:
- L1 tidak perlu label → realistis untuk deployment di lapangan
- L2 fokus hanya pada subset kecil (anomali), lebih akurat
- Edge-friendly: L2 hanya aktif saat diperlukan (event-driven)
- SMOTE di L2 tidak mengkontaminasi data normal L1
```

Implementasi di notebook:
```python
# L2 features: gabung sensor + reconstruction error per feature dari L1
recon_errors = np.mean((X_pred - X_true)**2, axis=1)  # per timestep
l2_features = np.hstack([X_anomalies, recon_errors.reshape(-1,1)])

from sklearn.ensemble import RandomForestClassifier
l2_model = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                   random_state=42)
# class_weight='balanced': backup jika SMOTE tidak cukup mengatasi imbalance
l2_model.fit(X_l2_train_smoted, y_l2_train_smoted)
```

#### Metode Tambahan: Explainable AI (XAI) — SHAP Analysis

> **[WAJIB — berdasarkan peer review internal]**
> Paper 2024–2025 yang tidak menyertakan interpretabilitas model akan ditolak karena
> operator lapangan perlu tahu SENSOR MANA yang memicu alert, bukan hanya "ada anomali".

```python
# STEP XAI-1: SHAP untuk Isolation Forest (TreeExplainer — cepat)
import shap

explainer_IF = shap.TreeExplainer(best_isolation_forest)
X_test_anomalies = X_test[y_pred_IF == -1]  # hanya sample yang diklasifikasi anomali
shap_values = explainer_IF.shap_values(X_test_anomalies)

# Output: Figure 4 — SHAP Summary Plot
shap.summary_plot(shap_values, X_test_anomalies,
                  feature_names=['prod_wh','batt_pct','volt_v','curr_a','mppt_w','temp_c',
                                 'mean_6h_prod','delta_batt','std_volt'],
                  show=False)
plt.savefig('figures/fig4_shap_isolation_forest.pdf', dpi=300, bbox_inches='tight')

# STEP XAI-2: Per-feature reconstruction error untuk LSTM AE (built-in)
feature_recon_error = np.mean((X_test_pred - X_test)**2, axis=(0, 1))
# Shape: (n_features,) — nilai lebih tinggi = fitur lebih "abnormal"
# Output: Figure 5 — bar chart feature contribution

# Narasi template untuk Section 4.3:
# "Figure 4 menunjukkan bahwa untuk anomali tipe Battery Fault (A3),
#  variabel batt_pct dan delta_batt memiliki SHAP value tertinggi (mean |SHAP| > 0.6),
#  memvalidasi bahwa model belajar representasi fisik yang benar dan tidak bergantung
#  pada spurious correlation."

# Library install:
# !pip install shap -q
# Estimasi waktu: 10–20 menit untuk TreeExplainer pada test set ukuran ~800 baris
```

### 6.2 Evaluation Framework (Extended)

```
Untuk setiap metode, hitung:

PER ANOMALY TYPE (A1–A5, EXCLUDE A6):
- Precision, Recall, F1-Score
- Confusion Matrix

KHUSUS SKENARIO A6 (Extended Low Irradiance = Normal):
- False Positive Rate (FPR) pada periode A6
- Target: FPR < 0.15 (model tidak boleh "panik" saat hujan panjang)
- Ini adalah STRESS TEST cuaca tropis Indonesia

OVERALL:
- Macro-averaged F1 (semua tipe anomali A1–A5)
- Weighted F1 (proporsional terhadap distribusi anomali)
- AUC-ROC (threshold-independent, hanya untuk metode yang menghasilkan score)
- Average Detection Latency (ADL):
  ADL = mean(timestep_detected - timestep_injected) dalam unit "ticks"
  Hanya untuk anomali yang berhasil terdeteksi (TP)

XAI METRICS (tambahan — untuk Section 4.3):
- Top-3 SHAP features per anomaly type (Isolation Forest)
- Feature reconstruction error ranking (LSTM AE)

DEPLOYMENT CONSIDERATION (Extended — Edge Hardware Constraints):
- Inference time per 1 prediction (mean dari 1000 runs, ms)
- Model size (MB): joblib.dump size untuk IF/LOF; .h5 size untuk LSTM AE
- Memory footprint (RAM usage):
    import tracemalloc
    tracemalloc.start()
    model.predict(X_test)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Peak RAM: {peak/1024/1024:.2f} MB")
- CPU load estimate (untuk Raspberry Pi 4 context):
    Catatan: Colab benchmark × faktor ~3-5x untuk RPi4 (ARM Cortex-A72 vs x86)
    Sertakan tabel konversi: "Estimated RPi4 inference: X ms"
- ESP32 feasibility note:
    IF/LOF: feasible jika model diserialisasi dan dijalankan via TFLite atau ONNX
    LSTM AE (32 units): ~50KB parameter → feasible di ESP32-S3 (8MB PSRAM)
    Cantumkan sebagai "deployment recommendation" di Section 4.5

Target metrik untuk paper:
| Metrik Edge | Rule-Based | IF | LOF | LSTM AE |
|------------|------------|----|----|---------|
| Model size | 0 MB | X MB | X MB | X MB |
| Peak RAM (Colab) | ~0 MB | X MB | X MB | X MB |
| Est. RPi4 latency | X ms | X ms | X ms | X ms |
| ESP32 feasible? | ✅ | ✅/❌ | ✅/❌ | Partial (TFLite) |

ALARM-BUDGETED EVENT-LEVEL EVALUATION (WAJIB — mencegah alarm fatigue):
```
Motivasi: Point-level F1 tidak mencerminkan pengalaman operator di lapangan.
Operator yang menerima 100 alarm per hari akan mengabaikan semua alarm (alarm fatigue).

Metrik Tambahan:
1. Missed Incident Rate (MIR) @ k alarms/hari:
   - Tentukan budget alarm: k = {5, 10, 20} alarms per hari
   - Urutkan anomaly scores dari tinggi ke rendah
   - Ambil top-k sebagai alarm aktual yang dikirim
   - MIR = jumlah kejadian anomali yang TIDAK tercakup dalam top-k
   - Target: MIR ≤ 0.15 pada budget 10 alarms/hari

2. Precision-at-k (P@k):
   - Dari top-k alarm teratas, berapa yang benar-benar anomali?
   - P@k = TP dalam top-k / k

3. Event-level F1 (bukan point-level):
   - Sebuah "kejadian anomali" = satu episode kontinu (bukan per timestep)
   - True Positive: setidaknya 1 timestep dalam episode terdeteksi
   - False Negative: seluruh episode terlewat

Implementasi:
  def alarm_budgeted_eval(scores, labels, budget_per_day=10):
      n_days = len(scores) / 48  # 48 ticks per hari (30 menit)
      k = int(budget_per_day * n_days)
      top_k_idx = np.argsort(scores)[-k:]
      detected = np.zeros(len(scores), dtype=bool)
      detected[top_k_idx] = True
      # Hitung event-level recall (berapa episode anomali yang tertangkap)
      episodes = find_anomaly_episodes(labels)  # group consecutive True labels
      caught = sum(1 for ep in episodes if any(detected[ep]))
      mir = 1 - caught / len(episodes)
      return mir, caught, len(episodes)
```

STATISTICAL TEST:
- McNemar's test untuk perbandingan biner (terdeteksi vs tidak) antar metode
- Paired t-test untuk F1 scores (jika menggunakan k-fold)
```

### 6.3 Train/Val/Test Split Detail

```
Dataset    : 34 hari × 48 interval/hari = 1.632 baris per plant
             × 2 plants = 3.264 baris total

Split (CHRONOLOGICAL — tidak random!):
- Train : Hari 1–20  → 1.920 baris (digunakan: normal data saja untuk LSTM AE)
- Val   : Hari 21–25 → 480 baris   (threshold tuning untuk LSTM AE dan IF)
- Test  : Hari 26–34 → 864 baris   (evaluation final, TIDAK DILIHAT selama development)

Anomali diinjeksikan SEBELUM split, sehingga distribusinya ada di seluruh periode.
Train set untuk Rule-Based dan IF: digunakan sebagai fitting (StandardScaler, rolling stats).
```

---

## BAGIAN 7 — STRUKTUR PAPER FINAL

### Section-by-Section Guide

```
JUDUL   : [Gunakan Opsi A atau B dari Bagian 2]
PENULIS : [Nama lengkap semua kontributor]
AFILIASI: Universitas Jenderal Soedirman, Purwokerto, Indonesia
EMAIL   : [Email instansi @unsoed.ac.id atau @student.unsoed.ac.id]
```

---

**ABSTRACT (~250 kata)**
- Kalimat 1–2: Konteks (off-grid solar IoT, 3T Indonesia)
- Kalimat 3–4: Problem (undetected anomaly, no labeled benchmark)
- Kalimat 5–6: Contribution (SCIO-Bench dataset + comparative study)
- Kalimat 7–8: Metode (3 pendekatan)
- Kalimat 9–10: Key result (F1 terbaik = XXX untuk metode YYY)
- Kalimat 11: Conclusion/implication

---

**SECTION 1: INTRODUCTION (~600 kata)**
- Paragraf 1: Konteks global EBT + IoT monitoring
- Paragraf 2: Masalah spesifik di Indonesia (3T, off-grid, kecil)
- Paragraf 3: Gap yang ada (kutip survey terbaru)
- Paragraf 4: Contribution statement (numbered list 3–4 poin)
- Paragraf 5: Paper organization

---

**SECTION 2: RELATED WORK (~500 kata)**
Subsections:
- 2.1 Anomaly Detection di Solar PV Systems (kutip 5–6 paper, identifikasi gap)
- 2.2 Machine Learning untuk IoT Anomaly Detection (kutip 4–5 paper)
- 2.3 Synthetic Dataset Generation untuk IoT Time-Series (kutip 3–4 paper)
- Akhiri dengan tabel singkat positioning paper ini vs. paper sebelumnya

Wajib kutip:
1. Alkaf et al. (2026) — Unsoed — anomaly detection PV
2. Survey MDPI Sensors 2024 (ML dan DL untuk IoT anomaly detection)
3. MDPI Energies (2022) — ML schemes for PV anomaly
4. Computer Science Review survey (2024) — IoT anomaly detection
5. Ambat et al. (2025) — Smart home energy anomaly

---

**SECTION 3: DATASET & METHODOLOGY (~800 kata)**

*3.1 Base Dataset*
- Deskripsi Kaggle dataset (Solar Power Generation Data, 2 plant India, 34 hari, 15-menit)
- Alasan pemilihan (iklim tropis, variabel relevan)
- Preprocessing steps

*3.2 Synthetic Variable Augmentation*
- Battery SOC simulation model (persamaan, asumsi)
- Voltage/current derivation
- Pseudocode atau diagram alir

*3.3 Anomaly Injection Protocol (SCIO-Bench)*
- Tabel 5 tipe anomali (ID, deskripsi, variabel, proporsi, justifikasi operasional)
- Injection code outline
- Distribusi akhir dataset

*3.4 Metode A: Rule-Based Threshold*
- Deskripsi 6 rules
- Implementasi detail

*3.5 Metode B: Classical Unsupervised ML*
- Isolation Forest (parameter, justifikasi)
- Local Outlier Factor (parameter, justifikasi)
- Feature engineering (rolling features)

*3.6 Metode C: LSTM Autoencoder*
- Arsitektur diagram (bisa ASCII art atau deskripsi)
- Training procedure (anomaly-free training)
- Anomaly scoring + threshold selection

*3.7 Evaluation Protocol*
- Metrik lengkap (Precision, Recall, F1, AUC-ROC, ADL, Inference Time)
- Train/Val/Test split rationale

---

**SECTION 4: RESULTS & DISCUSSION (~700 kata)**

*4.1 Dataset Characteristics*
- Distribusi anomali (pie chart atau bar chart)
- Contoh time-series plot dengan anomali ter-highlight

*4.2 Main Results*
Tabel wajib:
```
Table I: Comparative F1-Score per Anomaly Type
+------------------+--------+-------+---------+-------+--------+-------+
| Method           | A1     | A2    | A3      | A4    | A5     | Macro |
+==================+========+=======+=========+=======+========+=======+
| Rule-Based       | X.XX   | X.XX  | X.XX    | X.XX  | X.XX   | X.XX  |
| Isolation Forest | X.XX   | X.XX  | X.XX    | X.XX  | X.XX   | X.XX  |
| LOF              | X.XX   | X.XX  | X.XX    | X.XX  | X.XX   | X.XX  |
| LSTM Autoencoder | X.XX   | X.XX  | X.XX    | X.XX  | X.XX   | X.XX  |
+------------------+--------+-------+---------+-------+--------+-------+

Table II: Latency, Inference Time & False Positive Rate (Tropical Weather Stress Test)
+------------------+-------+------------+------------+-------------------+
| Method           | AUC   | ADL(ticks) | Time (ms)  | FPR@LowIrr (A6)  |
+==================+=======+============+============+===================+
| Rule-Based       | N/A   | X.X ± X.X  | X.X ± X.X  | X.XX              |
| Isolation Forest | X.XX  | X.X ± X.X  | X.X ± X.X  | X.XX              |
| LOF              | X.XX  | X.X ± X.X  | X.X ± X.X  | X.XX              |
| LSTM Autoencoder | X.XX  | X.X ± X.X  | X.X ± X.X  | X.XX              |
+------------------+-------+------------+------------+-------------------+
(FPR@LowIrr = False Positive Rate during extended low irradiance / hujan panjang)

Table III: Best Hyperparameters (Grid Search Results — for reproducibility)
+------------------+--------------------------------------+
| Method           | Best Parameters                      |
+==================+======================================+
| Isolation Forest | contamination=X.XX, n_estimators=100 |
| LOF              | n_neighbors=XX, contamination=X.XX   |
| LSTM AE          | seq_len=X, threshold_pct=XX          |
+------------------+--------------------------------------+
```

Visualisasi wajib:
- Figure 1: ROC curves (4 method dalam 1 plot)
- Figure 2: Sample time-series (24 jam) dengan anomali + detection marker per metode
- Figure 3: Grouped bar chart F1 per anomaly type
- Figure 4: SHAP summary plot — top-3 features per anomaly type (Isolation Forest)  **[BARU]**
- Figure 5: Per-feature reconstruction error heatmap (LSTM Autoencoder)  **[BARU]**

*4.3 Discussion*
- Jawab RQ1, RQ2, RQ3, RQ4 satu per satu
- Analisis: mengapa metode X unggul di anomali Y?
- Analisis FPR@A6: apakah model "panik" saat hujan panjang? (kunci untuk konteks Indonesia)
- Analisis A7: apakah LSTM AE unggul vs rule-based dalam mendeteksi FDI? Mengapa?
  (Hipotesis: LSTM AE mendeteksi inkonsistensi temporal V×I yang tidak terlihat rule tunggal)
- Interpretasi SHAP: apakah feature importance konsisten dengan intuisi fisik?
- Trade-off F1 vs. inference time (relevansi untuk edge deployment SCIO)

*4.4 XAI Interpretation*
- Apakah top-3 SHAP features per anomaly type konsisten dengan variabel yang dimodifikasi?
  Contoh: A3 (Battery Fault) → harusnya batt_pct dan delta_batt dominan di SHAP
- Apakah LSTM AE reconstruction error tertinggi di feature yang benar?
- Narasi: "Konsistensi antara SHAP importance dan ground-truth variabel anomali
  membuktikan bahwa model belajar relasi fisik, bukan overfitting ke noise dataset."

*4.5 Implications for SCIO Platform & 3T Local Deployment*
- Rekomendasi konkret: metode mana yang optimal untuk M2 SCIO
- Justifikasi berdasarkan F1, FPR@A6, MIR@10alarms/hari, inference time, explainability, edge feasibility
- Edge deployment path:
  * L1 (LSTM AE TFLite INT8): ESP32-S3 (8MB PSRAM) — inferensi setiap 30 detik
  * L2 (RF/XGB): Raspberry Pi 4 — aktif hanya saat L1 trigger
- **Local Physical Alarm (KRITIS untuk daerah 3T tanpa internet stabil):**
  * Jika L1 deteksi anomali → GPIO trigger → LED merah + buzzer onsite
  * Alarm code via LED pattern: 1 blink=A1, 2 blink=A3(battery), 3 blink=A7(cyber)
  * Operator lapangan tidak perlu app/internet untuk mendapat notifikasi pertama
  * Ini adalah gap besar yang belum dibahas paper sebelumnya — tambahkan sebagai
    "practical contribution" di Introduction dan Future Work
- Security implication: A7 MIR tinggi → perlu IDS layer di M3
- Connection ke roadmap M1 (rule-based) → M2 (L1+L2 hierarchical, edge) → M3 (autonomous)

---

**SECTION 5: LIMITATIONS (~200 kata)**
- Dataset sintetis belum fully represent kondisi lapangan sesungguhnya
- Battery SOC model menggunakan tapering + degradation approximation; efek suhu pada
  kapasitas (Arrhenius model) tidak dimodelkan — area untuk future work
- 2 plant dari India; belum ada validasi lapangan di Indonesia secara langsung
- Skenario A6 (Extended Low Irradiance) hanya mensimulasikan mendung; efek debu tropis
  dan partial soiling belum direpresentasikan
- LSTM AE tidak diuji pada concurrent anomaly (multiple failure modes bersamaan)
- SHAP values untuk LSTM AE menggunakan proxy (reconstruction error per feature),
  bukan true Shapley values — KernelExplainer lebih akurat tapi sangat lambat di Colab

---

**SECTION 6: CONCLUSION (~200 kata)**
- Ringkas kontribusi utama
- Jawab ringkas research questions
- Future work: (1) deployment di hardware SCIO nyata, (2) federated learning antar device, (3) M3 autonomous AI

---

**REFERENCES**
Minimum 25 referensi. Format: IEEE citation style.

Referensi wajib:
1. Alkaf et al., "Improving Solar Energy Reliability with Data-Driven Anomaly Detection Techniques," AITI Journal, 2026
2. Mao et al., "Research on Anomaly Detection Model for Power Consumption Data Based on Time-Series Reconstruction," Energies, 2024
3. [Survey MDPI Sensors] "Machine Learning and Deep Learning Techniques for IoT Network Anomaly Detection," Sensors, 2024
4. [ML Schemes PV] "Machine Learning Schemes for Anomaly Detection in Solar Power Plants," Energies, 2022
5. Ambat et al., "Anomaly detection and prediction of energy consumption for smart homes," ETRI Journal, 2025
6. Nizam et al., "Real-Time Deep Anomaly Detection Framework for Multivariate Time-Series Data in Industrial IoT," IEEE Sensors Journal, 2022
7. A. Kannal, "Solar Power Generation Data," Kaggle Dataset, 2020 — [untuk citation dataset]
8. IEEE Std 1679.1 (jika tersedia, untuk battery monitoring context)

---

## BAGIAN 8 — CHECKLIST REPRODUCIBILITY

```
[ ] Kode tersedia di GitHub repository (public)
    URL: https://github.com/[username]/scio-anomaly-benchmark

[ ] Dataset tersedia di Zenodo (DOI: 10.5281/zenodo.XXXXXXX)

[ ] Random seed ditetapkan: random_state=42 di semua eksperimen

[ ] Environment dicatat di requirements.txt:
    - Python 3.10
    - tensorflow==2.15.0
    - scikit-learn==1.4.0
    - pandas==2.1.0
    - numpy==1.26.0
    - matplotlib==3.8.0
    - seaborn==0.13.0
    - shap==0.44.0
    - kaggle==1.6.0

[ ] Google Colab notebook tersedia (link ke nbviewer atau langsung Colab)

[ ] Semua gambar dan tabel dapat di-reproduce dengan menjalankan notebook dari awal

[ ] Instruksi setup < 5 langkah di README.md
```

---

## BAGIAN 9 — INSTRUKSI EKSEKUSI UNTUK CLAUDE CODE

### Environment Setup

```bash
# Jalankan di Colab cell pertama
!pip install kaggle tensorflow scikit-learn pandas numpy matplotlib seaborn pyod shap -q

# Upload kaggle.json ke Colab (dari Kaggle Account → API → Create New Token)
from google.colab import files
files.upload()  # upload kaggle.json

!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

### Step-by-Step Execution (Claude Code harus eksekusi berurutan)

```
STEP 01: Download Dataset
├── !kaggle datasets download -d anikannal/solar-power-generation-data
└── !unzip solar-power-generation-data.zip -d data/

STEP 02: EDA & Preprocessing (with robust NaN/Inf handling)
├── Load 4 CSV files (2 plants × generation + weather)
├── Merge on DATE_TIME (inner join)
├── Resample ke 30 menit (resample('30T').mean())
├── NaN/Inf handling (WAJIB — data sensor IoT nyata pasti punya ini):
│   # Replace Infinity dengan NaN dulu
│   df.replace([np.inf, -np.inf], np.nan, inplace=True)
│   # Forward-fill maksimal 2 consecutive NaN (sesuai SRS FR-034 offline degradation)
│   df = df.fillna(method='ffill', limit=2)
│   # Sisa NaN (gap > 2): tandai sebagai A5 candidate, lalu fill dengan median kolom
│   df['is_offline'] = df.isnull().any(axis=1).astype(int)
│   df = df.fillna(df.median())
│   # Validasi: tidak ada NaN atau Inf tersisa sebelum masuk ke model
│   assert not df.isnull().values.any(), "NaN masih ada!"
│   assert not np.isinf(df.values).any(), "Inf masih ada!"
├── Drop rows dengan SEMUA nilai NaN (jika ada gap besar di awal/akhir)
└── Rename ke SCIO column convention

STEP 03: Synthetic Augmentation
├── Tambah batt_pct (gunakan simulate_soc() dari Bagian 5.2)
├── Tambah volt_v = 24 + noise (normal, std=1.5) + correction dari DC_POWER
├── Tambah curr_a = (mppt_w × 1000) / volt_v
├── Tambah rssi = np.random.normal(-70, 15, len(df)).astype(int)
└── Tambah protocol = 'lora' if rssi > -80 else '4g'

STEP 04: Anomaly Injection (A1–A5 + A7)
├── Implement inject_anomaly() function untuk setiap tipe (A1–A5)
├── A7 False Data Injection:
│   # Manipulasi fisik yang tidak konsisten: volt_v naik tapi prod_wh turun
│   # Constraint: nilai TETAP dalam range normal individu (tidak outlier sederhana!)
│   mask_a7 = select_random_segments(df, proportion=0.01)
│   df.loc[mask_a7, 'volt_v'] *= np.random.uniform(1.1, 1.2)   # naik 10-20%
│   df.loc[mask_a7, 'curr_a'] *= np.random.uniform(0.5, 0.7)   # turun 30-50%
│   # prod_wh tetap (tidak diubah) → inkonsistensi P = V×I tidak terpenuhi
│   df.loc[mask_a7, 'anomaly_type'] = 'false_data_injection'
├── Injeksi ke random segments berdasarkan proporsi (np.random.seed(42))
├── Tambah kolom is_anomaly (bool) dan anomaly_type (string)
└── Simpan ke scio_bench_dataset.csv

STEP 05: Dataset Split
├── train = df[df.timestamp < '2020-05-21']  # adjust ke tanggal aktual
├── val   = df[(df.timestamp >= '2020-05-21') & (df.timestamp < '2020-05-26')]
└── test  = df[df.timestamp >= '2020-05-26']

STEP 06: Feature Engineering
├── Compute rolling features: mean_6h, std_6h, delta_1tick untuk setiap variabel
├── StandardScaler.fit(train_normal_data)
└── Transform train, val, test

STEP 07: Method A — Rule-Based
├── Implement semua 6 rules (lihat Bagian 6.1)
├── Predict pada test set
└── Compute metrics (precision_recall_fscore_support, roc_auc_score)

STEP 08: Method B — Isolation Forest
├── Grid search contamination pada val set (metric: F1 macro)
├── Fit IsolationForest(contamination=best, random_state=42)
├── Predict pada test set
└── Compute metrics

STEP 09: Method B2 — Local Outlier Factor
├── Grid search n_neighbors pada val set
├── Fit LocalOutlierFactor(novelty=True, contamination=best)
├── Predict pada test set
└── Compute metrics

STEP 10: Method C — LSTM Autoencoder (L1) + Hierarchical L2
├── Build L1 model (arsitektur LSTM AE Bagian 6.1)
├── Train L1 HANYA pada normal train data (is_anomaly==False, is_weather_event==False)
├── Compute reconstruction error pada test set
├── MAD-threshold: median_err + 3 * MAD(val_reconstruction_errors)
├── Predict L1: is_anomaly = reconstruction_error > threshold
├── ── L2 Training (supervised, hanya pada anomali) ──
├── Filter: X_l2 = train[train.anomaly_type != 'normal']
├── SMOTE: X_smoted, y_smoted = SMOTE(k_neighbors=3).fit_resample(X_l2, y_l2)
├── Train L2: RandomForestClassifier(n_estimators=100, class_weight='balanced')
├── Predict L2 pada test anomalies (those flagged by L1)
├── ── Alarm Budget Evaluation ──
├── alarm_budgeted_eval(reconstruction_errors, y_test, budget_per_day=10)
└── Compute all metrics (point-level F1 + event-level MIR@k + A7 F1)

STEP 11: Compile Results Tables
├── Buat Table I (F1 per anomaly type per method)
├── Buat Table II (AUC, ADL, Inference Time)
└── Export ke CSV dan LaTeX format

STEP 12: Visualizations & XAI
├── Figure 1: ROC curves (4 method dalam 1 subplot atau overlay)
├── Figure 2: Time-series sample (24 jam) dengan anomali + prediction highlights
├── Figure 3: Grouped bar chart F1 per anomaly type
├── Figure 4: SHAP summary plot — !pip install shap → shap.TreeExplainer(IF)
├── Figure 5: LSTM AE per-feature reconstruction error bar chart
└── Simpan semua figure sebagai PDF (IEEE format, 300 DPI)

STEP 13: Statistical Tests
├── McNemar's test: Rule-Based vs. Best ML method
├── McNemar's test: Best ML vs. LSTM AE
└── Output: p-value, chi2 statistic, interpretasi (p<0.05 = significantly different)

STEP 14: XAI Analysis
├── !pip install shap -q
├── shap.TreeExplainer(best_isolation_forest) → shap_values pada X_test_anomalies
├── shap.summary_plot() → save Figure 4
├── LSTM per-feature reconstruction error → save Figure 5
└── Verifikasi: top SHAP features per anomaly type vs. ground truth variabel

STEP 14b: Edge Hardware Profiling + TFLite Quantization
├── tracemalloc: ukur peak RAM untuk setiap metode di X_test (full test set)
├── timeit: inference time 1000x untuk single sample (median — robust terhadap outlier)
├── joblib.dump size: model file size IF dan LOF
├── keras model.count_params(): parameter count LSTM AE
├── TFLite Quantization (INT8 — kurangi latency hingga 76% di edge):
│   converter = tf.lite.TFLiteConverter.from_keras_model(lstm_ae)
│   converter.optimizations = [tf.lite.Optimize.DEFAULT]
│   converter.target_spec.supported_types = [tf.int8]
│   tflite_model = converter.convert()
│   # Ukur: ukuran file .tflite vs .h5, dan inference time tflite interpreter
│   # Bandingkan akurasi: F1 Float32 vs F1 INT8 (harusnya delta < 1%)
│   with open('lstm_ae_int8.tflite', 'wb') as f: f.write(tflite_model)
├── Buat tabel edge deployment:
│   | Variant        | Size  | RAM   | Latency  | F1 delta |
│   |----------------|-------|-------|----------|----------|
│   | LSTM Float32   | X MB  | X MB  | X ms     | baseline |
│   | LSTM INT8      | X MB  | X MB  | X ms     | < 1%     |
│   | IF (joblib)    | X MB  | X MB  | X ms     | N/A      |
└── ESP32-S3 feasibility: INT8 LSTM AE ~50KB parameter → feasible dengan 8MB PSRAM

STEP 14c: Adversarial Robustness Evaluation
├── Isolasi test samples dengan anomaly_type == 'false_data_injection' (A7)
├── Evaluasi F1, Precision, Recall khusus A7 untuk semua metode
├── Hipotesis: Rule-Based gagal (F1 rendah); LSTM AE unggul (sequence inconsistency)
└── Tambahkan sub-tabel A7 di Table I atau lampiran

STEP 15: Export Final Package
├── scio_bench_dataset.csv → upload ke Zenodo (dataset record terpisah, DOI unik)
├── scio_anomaly_benchmark.ipynb → export + upload ke GitHub (public repo)
├── requirements.txt → catat semua library version termasuk shap==0.44.0
└── README.md → instruksi reproducibility < 5 langkah
```

### Output yang Diharapkan di Akhir Eksekusi

```
/outputs/
├── results/
│   ├── table1_f1_comparison.csv
│   ├── table2_latency_inference.csv
│   └── statistical_tests.txt
├── figures/
│   ├── fig1_roc_curves.pdf
│   ├── fig2_timeseries_example.pdf
│   └── fig3_f1_per_anomaly_type.pdf
├── dataset/
│   └── scio_bench_dataset.csv          ← Upload ke Zenodo
└── notebook/
    └── scio_anomaly_benchmark.ipynb    ← Upload ke GitHub + Colab
```

---

## BAGIAN 10 — TIMELINE 3 HARI

```
HARI 1 (Fokus: Data + Setup)
├── [2 jam] Download + EDA dataset Kaggle
├── [2 jam] Synthetic augmentation (batt_pct, volt_v, curr_a)
├── [2 jam] Anomaly injection (5 tipe)
└── [1 jam] Validasi dataset — cek distribusi, plot time-series

HARI 2 (Fokus: Eksperimen)
├── [1 jam] Method A: Rule-Based + FPR@A6 evaluation
├── [2 jam] Method B: IF + LOF (Grid Search di val set, log best_params)
├── [2 jam] Method C: LSTM Autoencoder (train + Grid Search threshold)
├── [1 jam] SHAP analysis + per-feature reconstruction error (XAI)
└── [1 jam] Compile 3 result tables + statistical tests (McNemar)

HARI 3 (Fokus: Visualisasi + Penulisan + Publikasi)
├── [1 jam] Buat 5 figures (ROC, time-series, bar F1, SHAP, recon error)
├── [3 jam] Tulis paper lengkap (gunakan kerangka Bagian 7)
├── [1 jam] Upload SCIO-Bench dataset ke Zenodo (DOI dataset terpisah)
└── [1 jam] Upload paper ke Zenodo + submit ke TechRxiv (IEEE)
```

---

## CATATAN KHUSUS UNTUK CLAUDE CODE

1. **Jangan ganti random_state** — gunakan 42 di semua eksperimen untuk reproducibility

2. **Jangan random split time-series** — selalu chronological split

3. **Battery SOC model harus deterministic** — sama untuk setiap run, tidak ada randomness

4. **Anomaly injection harus reproducible** — gunakan `np.random.seed(42)` sebelum inject

5. **Inference time measurement:**
   ```python
   import time
   times = []
   for _ in range(1000):
       start = time.perf_counter()
       model.predict(single_sample)
       times.append(time.perf_counter() - start)
   print(f"Mean inference: {np.mean(times)*1000:.2f} ms ± {np.std(times)*1000:.2f}")
   ```

6. **Jika hasil F1 < 0.5 untuk suatu metode** — jangan manipulasi threshold, dokumentasikan apa adanya. Ini justru menarik sebagai finding.

7. **Paper harus ditulis dalam English** — target reader adalah komunitas internasional

8. **Afiliasi wajib:** Universitas Jenderal Soedirman, Purwokerto, Central Java, Indonesia

9. **Saat menulis paper**, referensikan dokumen SCIO (BRD, SRS, SysRS) sebagai "prior system documentation" untuk membuktikan ini adalah penelitian yang grounded pada sistem nyata — tanpa menyebut nomor dokumen internal, cukup "the SCIO IoT platform specification"

---

*Dokumen ini adalah kerangka hidup — update saat ada perubahan metodologi*
*Last updated: April 2026*
*Dibuat berdasarkan deep research pada: IEEE/MDPI submission standards, 
Kaggle dataset landscape, research gap analysis (literature 2022–2026),
paper Alkaf et al. (Unsoed, 2026) sebagai prior work afiliasi*
