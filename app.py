"""
app.py
------
The Invisible Appliance Doctor — Main Overview Page (Streamlit entry point).

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import joblib
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration (must be the FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Invisible Appliance Doctor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load cached pipeline on startup (if available)
# ─────────────────────────────────────────────────────────────────────────────
CACHE_PATH = Path("models/pipeline_cache.pkl")

if "trained" not in st.session_state:
    st.session_state["trained"] = False

if not st.session_state["trained"] and CACHE_PATH.exists():
    try:
        cache = joblib.load(str(CACHE_PATH))
        for key, val in cache.items():
            st.session_state[key] = val
        st.session_state["trained"] = True
    except Exception:
        st.session_state["trained"] = False

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hero gradient */
  .hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00D4FF 0%, #7B61FF 50%, #FF6B35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
  }
  .hero-sub {
    font-size: 1.1rem;
    color: #9CA3AF;
    margin-top: 0.4rem;
  }
  /* Metric cards */
  [data-testid="metric-container"] {
    background: #1F2937;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 1rem;
  }
  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #111827;
  }
  .sidebar-logo {
    font-size: 1.6rem;
    font-weight: 700;
    color: #00D4FF;
  }
  .status-ok   { color: #10B981; font-weight: 600; }
  .status-warn { color: #F59E0B; font-weight: 600; }
  .status-err  { color: #EF4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Appliance Doctor</div>', unsafe_allow_html=True)
    st.caption("Unsupervised Energy Anomaly Detection")
    st.divider()

    # Dataset status
    data_path = Path("data/household_power_consumption.txt")
    st.markdown("**Dataset**")
    if data_path.exists():
        st.markdown('<span class="status-ok">● Real dataset found ✓</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">● Dataset not found</span>', unsafe_allow_html=True)
        st.caption("Place household_power_consumption.txt in the data/ folder.")

    st.markdown("**Model**")
    if st.session_state.get("trained", False):
        st.markdown('<span class="status-ok">● RBM trained ✓</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">● Not trained yet</span>', unsafe_allow_html=True)
        st.caption("Go to Model Lab and click Train RBM.")

    st.divider()
    st.markdown("**Navigation**")
    st.page_link("pages/1_Energy_Usage.py",      label="Energy Usage")
    st.page_link("pages/2_Anomaly_Detection.py", label="Anomaly Detection")
    st.page_link("pages/3_Model_Lab.py",         label="Model Lab")
    st.page_link("pages/4_Alert_Investigator.py", label="Alert Investigator")

# ─────────────────────────────────────────────────────────────────────────────
# Hero Section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">The Invisible Appliance Doctor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Unsupervised Household Energy Anomaly Detection '
    'using Restricted Boltzmann Machines &amp; Gibbs Sampling</div>',
    unsafe_allow_html=True,
)
st.divider()

st.warning(
    "**Disclaimer:** This system detects *statistically unusual* energy patterns. "
    "An alert does **NOT** confirm an appliance fault. Human investigation is always required.",
)

# ─────────────────────────────────────────────────────────────────────────────
# Summary Metrics
# ─────────────────────────────────────────────────────────────────────────────
trained      = st.session_state.get("trained", False)
pipeline     = st.session_state.get("pipeline", {})
scores       = st.session_state.get("scores", None)
threshold    = st.session_state.get("threshold", None)

dataset_status = "Real UCI" if data_path.exists() else "Not loaded"

n_intervals    = len(scores) if scores is not None else None
n_flagged      = int(np.sum(scores > threshold)) if (scores is not None and threshold is not None) else None
flag_pct       = f"{100*n_flagged/n_intervals:.1f}%" if (n_flagged is not None and n_intervals) else None
threshold_disp = f"{threshold:.3f}" if threshold is not None else None

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Dataset", dataset_status)
with col2:
    st.metric("Intervals Analysed", f"{n_intervals:,}" if n_intervals else "N/A")
with col3:
    st.metric("Flagged Intervals", f"{n_flagged} ({flag_pct})" if n_flagged is not None else "N/A")
with col4:
    st.metric("Alert Threshold", threshold_disp or "N/A")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# About This Project
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("About This Project"):
    st.markdown("""
**The Invisible Appliance Doctor** is an academic deep-learning project that demonstrates
unsupervised anomaly detection using a **Bernoulli Restricted Boltzmann Machine (RBM)**.

### What the system does
1. Learns what *normal* electricity usage looks like from historical data (training).
2. Scores each 15-minute interval based on how surprising it is to the trained model.
3. Flags intervals that exceed a threshold as "unusual patterns".

### What the system does NOT do
- It does **NOT** diagnose appliance faults — the dataset has no fault labels.
- It does **NOT** guarantee that flagged intervals represent real problems.
- All alerts are statistical observations, not causal diagnoses.

### Technologies used
| Component | Technology |
|-----------|-----------|
| RBM (core model) | Pure NumPy — Bernoulli RBM with CD-1 |
| Anomaly score | Free energy: score = +F(v) |
| Baseline | Per-hour Median + MAD (modified Z-score) |
| Dashboard | Streamlit + Plotly |
| Dataset | UCI Individual Household Electric Power Consumption |

### Academic context
*Final-year Computer Science — Deep Learning II project.*
Builds an end-to-end unsupervised learning pipeline from data ingestion through
model training, anomaly scoring, and evaluation against controlled synthetic anomalies.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Architecture overview (always visible)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### System Architecture")
st.markdown("""
```
Raw data (1-min readings)
        │
        ▼
┌─────────────────────────┐
│   data_loader.py         │  Load UCI household power consumption dataset
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   preprocessing.py       │  Resample → Feature engineering → Binary encoding
│   Preprocessor.fit()     │  (fit on TRAIN only — no data leakage)
└────────────┬────────────┘
             │ X_binary  (n_samples × 21 bits)
             ▼
┌─────────────────────────┐
│   rbm.py                 │  Bernoulli RBM · CD-1 · Gibbs sampling
│   BernoulliRBM.train()   │  Learns P(v) ≈ P_data(v)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   anomaly_detection.py   │  score = +F(v) per interval
│   select_threshold()     │  Threshold from VALIDATION set only
└────────────┬────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│  Dashboard (Streamlit)                               │
│  ├── Energy Usage      (trends, heatmaps)            │
│  ├── Anomaly Detection (flags, investigation)         │
│  └── Model Lab         (train, compare, evaluate)    │
└────────────────────────────────────────────────────┘
```
""")

st.caption(
    "Source: [UCI Machine Learning Repository — Individual Household Electric Power Consumption]"
    "(https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) · "
    "Hébrail & Bérard, 2012"
)
