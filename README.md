# The Invisible Appliance Doctor

## Unsupervised Household Energy Anomaly Detection using Restricted Boltzmann Machines and Gibbs Sampling

---

## Abstract

This project builds an end-to-end unsupervised anomaly-detection system for household electricity consumption. A **Bernoulli Restricted Boltzmann Machine (RBM)** is trained on normal energy usage patterns extracted from the UCI Individual Household Electric Power Consumption dataset. The model assigns a free-energy-based anomaly score to each 15-minute interval and flags intervals that deviate significantly from learned normal behaviour. The system includes a statistical baseline comparison, a controlled synthetic anomaly evaluation, and an interactive Streamlit dashboard for exploration and viva demonstration.

> **Important limitation:** The dataset contains no verified appliance-fault labels. The system detects *statistical anomalies*, not confirmed faults. All alerts require human investigation.

---

## Problem Statement

Household appliances consume energy in characteristic patterns. When a device malfunctions or operates outside its normal regime, this *may* produce unusual energy readings. However:

- The UCI dataset has no fault labels.
- Supervised classification is therefore impossible.
- Unsupervised anomaly detection allows us to identify statistically unusual intervals without needing fault examples.

---

## Motivation

- Electricity anomaly detection can alert homeowners to potential inefficiencies, unusual behaviour, or devices that may deserve inspection.
- RBMs are a theoretically grounded energy-based generative model well-suited to learning distributions over discrete data.
- The project provides a complete, explainable pipeline from raw data to interactive dashboard — demonstrating skills across data engineering, deep learning, evaluation methodology, and software engineering.

---

## Objectives

1. Learn normal household electricity usage patterns from historical data.
2. Score each 15-minute interval using the RBM's free energy.
3. Flag intervals that exceed a validation-set threshold as "unusual patterns."
4. Compare the RBM against a simple statistical baseline.
5. Evaluate both detectors on controlled synthetic anomalies.
6. Present results through an interactive, explanatory dashboard.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

**Citation:** Hébrail, G. & Bérard, A. (2012). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

| Property | Value |
|----------|-------|
| Format | Semicolon-delimited text |
| Frequency | 1-minute readings |
| Date range | 16 Dec 2006 – 26 Nov 2010 |
| Rows | ~2,075,259 |
| Missing values | Encoded as `?` |

**Columns:** `Date`, `Time`, `Global_active_power` (kW), `Global_reactive_power` (kW), `Voltage` (V), `Global_intensity` (A), `Sub_metering_1` (Wh, kitchen), `Sub_metering_2` (Wh, laundry), `Sub_metering_3` (Wh, water heater/AC).

See [DATA_SETUP.md](DATA_SETUP.md) for download and placement instructions.

---

## System Architecture

```
Raw data (1-min readings, ~2M rows)
        │
        ▼
┌─────────────────────────┐
│   data_loader.py         │  Parses semicolon file, converts '?' to NaN
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   preprocessing.py       │
│   resample_to_15min()    │  Aggregate: mean (power), sum (sub-metering)
│   engineer_features()    │  Add cyclic time features, derived columns
│   chronological_split()  │  Train/Val/Test = 60/20/20 by time
│   Preprocessor.fit()     │  Learn μ, σ, quantile thresholds from TRAIN only
│   Preprocessor.transform()│  → scaled features + 21-bit binary array
└────────────┬────────────┘
             │ X_binary (n × 21)
             ▼
┌─────────────────────────┐
│   rbm.py                 │
│   BernoulliRBM           │  n_visible=21, n_hidden=k
│   .train(CD-1)           │  Gibbs sampling, mini-batch SGD
│   .free_energy(v)        │  F(v) = −v·bᵥ − Σⱼ log(1+exp(bₕⱼ+Wⱼ·v))
│   .anomaly_score(v)      │  score = +F(v)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   anomaly_detection.py   │  Threshold = percentile of VAL scores
│   flag_anomalies()       │  score > threshold → "Unusual pattern"
│   explain_interval()     │  Compare to training-hour median
└────────────┬────────────┘
             │
        ┌────┴────┐
        ▼         ▼
 baseline.py   evaluation.py
 Per-hour      Inject synthetic
 MAD Z-score   anomalies → metrics
        │         │
        └────┬────┘
             ▼
┌─────────────────────────────────────┐
│  Streamlit Dashboard (app.py)        │
│  pages/1_Energy_Usage.py             │
│  pages/2_Anomaly_Detection.py        │
│  pages/3_Model_Lab.py                │
│  pages/4_Viva_Guide.py               │
└─────────────────────────────────────┘
```

---

## Data Preprocessing

### Resampling
1-minute readings are aggregated to **15-minute intervals**:
- Power and voltage columns → **mean** (average power over the interval)
- Sub-metering columns → **sum** (total energy in the interval)

### Feature Engineering
| Feature | Derivation |
|---------|-----------|
| `hour_sin`, `hour_cos` | `sin/cos(2π × hour / 24)` — cyclic hour encoding |
| `dow_sin`, `dow_cos` | `sin/cos(2π × day_of_week / 7)` — cyclic day encoding |
| `sub_total` | `Sub_metering_1 + Sub_metering_2 + Sub_metering_3` |
| `power_ratio` | `sub_total / (Global_active_power × 1000 / 4)`, clipped [0,1] |

### Chronological Split
```
|────── 60% Train ──────|── 20% Val ──|── 20% Test ──|
        (learn model)     (threshold)   (evaluation)
```
No data shuffling — temporal order is preserved to prevent leakage.

### Binary Discretisation
Each of 7 numeric features → 3 binary bits:

| Bit | Threshold | Meaning |
|-----|-----------|---------|
| bit_0 | Q25 (training) | 1 if value > 25th percentile |
| bit_1 | Q50 (training) | 1 if value > median |
| bit_2 | Q75 (training) | 1 if value > 75th percentile |

**7 features × 3 bits = 21 visible units.** All thresholds are fitted on training data only.

---

## RBM Theory

### Model
A Bernoulli RBM is a two-layer undirected energy-based model:
- **Visible units** v ∈ {0,1}^21 — binary-encoded electricity measurements
- **Hidden units** h ∈ {0,1}^k — latent features
- **Parameters:** W (k×21), b_v (21,), b_h (k,)

### Energy Function
$$E(v, h) = -v^\top b_v - h^\top b_h - h^\top W v$$

### Joint Probability
$$P(v, h) = \frac{\exp(-E(v,h))}{Z}, \quad Z = \sum_{v,h} \exp(-E(v,h))$$

### Conditional Distributions
$$P(h_j=1 \mid v) = \sigma(b_{h_j} + W_j \cdot v)$$
$$P(v_i=1 \mid h) = \sigma(b_{v_i} + \sum_j W_{ji} h_j)$$
where $\sigma(x) = 1/(1+e^{-x})$.

### Free Energy
$$F(v) = -v \cdot b_v - \sum_j \log\!\left(1 + \exp(b_{h_j} + W_j \cdot v)\right)$$

Low F(v) = high probability = normal. High F(v) = low probability = unusual.

---

## Gibbs Sampling and Contrastive Divergence

### Gibbs Sampling
Alternate between sampling the entire hidden layer (given v) and the entire visible layer (given h). Conditional independence within each layer allows full-layer parallel sampling.

### CD-1 Algorithm
```
For each mini-batch of size B:
  Positive phase:
    h_probs⁰ = σ(b_h + W v⁰)          # P(h|data)
    h⁰ ~ Bernoulli(h_probs⁰)           # sample hidden
  Negative phase (1 Gibbs step):
    ṽ ~ Bernoulli(σ(b_v + Wᵀ h⁰))      # reconstruct visible
    h̃_probs = σ(b_h + W ṽ)             # P(h|reconstruction)
  Parameter update:
    ΔW   = η × (v⁰ᵀ h_probs⁰ − ṽᵀ h̃_probs) / B
    Δb_v = η × mean(v⁰ − ṽ)
    Δb_h = η × mean(h_probs⁰ − h̃_probs)
```
Reconstruction error = MSE(v⁰, ṽ) is tracked each epoch as a training monitor.

---

## Anomaly Scoring

### Score
`anomaly_score(v) = +F(v)` — higher score → more anomalous (lower probability under the model).

### Threshold Selection
The alert threshold = `percentile(val_scores, p)` where `p` is a user-configurable parameter (default 95). The validation set is used, never the test set.

### Explanation
For each flagged interval, we:
1. Look up the actual readings.
2. Compare against the per-hour median from training data.
3. Report which features deviate most (and by what ratio).

---

## Evaluation

### Real Data
No verified fault labels → no precision/recall on real anomalies.

### Synthetic Evaluation
Inject 4 types of simulated faults into a copy of the held-out test data:

| Type | Description |
|------|-------------|
| `spike` | Short (1–3 intervals) × 4× power spike |
| `sustained` | 2–4 intervals of 2.5× elevated consumption |
| `night` | 3× unusual consumption midnight–5am |
| `submetering` | 5× spike in Sub_metering_1 |

Metrics: Precision, Recall, F1, FPR, Confusion Matrix — for both RBM and baseline.

> Results are labelled as *synthetic evaluation only* — not representative of real-world fault detection.

---

## Installation

### Requirements
- Python 3.9+
- Windows / Linux / macOS

### Install dependencies
```powershell
pip install -r requirements.txt
```

### Run the dashboard
```powershell
cd c:\Users\LENOVO\Desktop\Appliance_Doctor
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Run the test suite
```powershell
cd c:\Users\LENOVO\Desktop\Appliance_Doctor
python -m pytest tests/ -v
```

---

## Project Structure

```
Appliance_Doctor/
├── app.py                        # Main Streamlit entry point (Overview page)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── DATA_SETUP.md                 # Dataset download guide
├── data/
│   └── household_power_consumption.txt   # Place UCI dataset here
├── models/
│   ├── rbm.npz                   # Saved RBM weights (after training)
│   ├── preprocessor.pkl          # Saved Preprocessor (after training)
│   └── pipeline_cache.pkl        # Full pipeline cache (auto-loaded on startup)
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # UCI dataset loading + synthetic demo generator
│   ├── preprocessing.py          # Resampling, features, split, Preprocessor class
│   ├── rbm.py                    # BernoulliRBM (NumPy) with CD-k
│   ├── anomaly_detection.py      # Free-energy scoring, threshold, explanation
│   ├── baseline.py               # Statistical baseline (per-hour MAD)
│   ├── evaluation.py             # Synthetic anomaly injection and metrics
│   └── visualizations.py         # Plotly chart factory functions
├── pages/
│   ├── 1_Energy_Usage.py         # Power trends, sub-metering, heatmaps
│   ├── 2_Anomaly_Detection.py    # Anomaly timeline, table, investigation panel
│   ├── 3_Model_Lab.py            # Train RBM, model info, baseline, eval
│   └── 4_Viva_Guide.py           # Study guide with 20 viva Q&A
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_rbm.py
    ├── test_anomaly_detection.py
    └── test_evaluation.py
```

---

## Ethical Considerations and Limitations

1. **No fault diagnoses:** The system must never be used to claim an appliance is faulty without human investigation.
2. **Synthetic evaluation only:** Published metrics are for simulated anomalies, not real faults.
3. **Training distribution dependency:** The model flags what is unusual *relative to training data* — if training data itself contains anomalies, those patterns will be learned as "normal."
4. **Threshold sensitivity:** The threshold is a hyperparameter — there is no universally correct value.

---

## Results Summary (after training)

After training on 90 days of data with default settings (n_hidden=15, CD-1, 50 epochs):

| Detector | Synthetic Precision | Synthetic Recall | Synthetic F1 |
|----------|--------------------| -----------------|-------------|
| RBM (free energy) | varies* | varies* | varies* |
| Statistical Baseline | varies* | varies* | varies* |

\* Results depend on training data period, hyperparameters, and injected anomaly configuration. Run the Model Lab evaluation to see actual numbers.

---

## License

This project is for academic purposes only. The UCI dataset is used under its [original terms](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
