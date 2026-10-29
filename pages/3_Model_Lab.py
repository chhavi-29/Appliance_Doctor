"""
3_Model_Lab.py
--------------
Model Lab page — Train RBM, inspect architecture, compare with baseline,
run controlled synthetic evaluation.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import joblib

from src.rbm import BernoulliRBM
from src.preprocessing import Preprocessor, run_full_pipeline
from src.data_loader import load_dataset, DatasetNotFoundError
from src.anomaly_detection import score_samples, select_threshold, flag_anomalies, score_stats
from src.baseline import StatisticalBaseline
from src.evaluation import run_controlled_evaluation
from src.visualizations import (
    plot_training_loss,
    plot_rbm_weights,
    plot_score_distribution,
    plot_confusion_matrix,
)

st.set_page_config(page_title="Model Lab", page_icon=None, layout="wide")
st.markdown("## Model Lab")

CACHE_PATH = Path("models/pipeline_cache.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_train, tab_info, tab_baseline, tab_eval = st.tabs(
    ["Train RBM", "Model Info", "Baseline Comparison", "Synthetic Evaluation"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Train RBM
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    col_cfg, col_out = st.columns([1, 2])

    with col_cfg:
        st.subheader("Training Configuration")

        n_hidden     = st.slider("Hidden Units",  5, 100, 15, help="Number of hidden units (latent features).")
        epochs       = st.slider("Epochs", 10, 200, 50, help="Full passes over training data.")
        batch_size   = st.select_slider("Batch Size", options=[32, 64, 128, 256], value=64)
        learning_rate = st.select_slider(
            "Learning Rate", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01
        )
        cd_steps     = st.radio("CD Steps", [1, 2, 3], index=0, horizontal=True,
                                help="k in Contrastive Divergence-k. CD-1 is fastest.")
        max_train_days = st.slider(
            "Max Training Days", 30, 365, 90,
            help="Limit training data to this many days (for laptop speed).",
        )
        threshold_pct = st.slider(
            "Alert Threshold Percentile", 80, 99, 95,
            help="p-th percentile of validation scores is used as the detection threshold.",
        )
        seed = st.number_input("Random Seed", value=42, step=1)

        train_button = st.button("Train RBM", type="primary", use_container_width=True)

    with col_out:
        trained = st.session_state.get("trained", False)
        rbm     = st.session_state.get("rbm", None)

        # ── Status header ────────────────────────────────────────────────────
        if trained and rbm is not None:
            st.success("RBM is trained. You can retrain with new settings below.")
            with st.expander("Training Loss (previous run)"):
                try:
                    fig_loss = plot_training_loss(rbm.losses_)
                    st.plotly_chart(fig_loss, use_container_width=True)
                except Exception as e:
                    st.caption(f"Loss chart unavailable: {e}")
            if st.button("Clear and Retrain", type="secondary"):
                for key in ["trained", "rbm", "preprocessor", "pipeline",
                            "scores", "val_scores", "threshold", "baseline",
                            "baseline_scores", "df_15", "train_df", "test_df",
                            "eval_results"]:
                    st.session_state.pop(key, None)
                st.rerun()
        else:
            st.info("Configure settings on the left and click **Train RBM** to begin.")

        # ── Training ─────────────────────────────────────────────────────────
        if train_button:
            progress_bar      = st.progress(0.0, text="Starting...")
            status_placeholder = st.empty()
            loss_placeholder   = st.empty()

            try:
                # 1. Load data
                status_placeholder.info("Loading data...")
                try:
                    df_raw = load_dataset("data")
                    status_placeholder.info("Dataset loaded successfully.")
                except DatasetNotFoundError:
                    progress_bar.empty()
                    status_placeholder.error(
                        "Dataset not found.\n\n"
                        "Please place `household_power_consumption.txt` inside the `data/` folder, "
                        "then click **Train RBM** again.\n\n"
                        "See **DATA_SETUP.md** in the project folder for download instructions."
                    )
                    st.stop()

                # 2. Preprocess
                status_placeholder.info("Preprocessing and splitting data...")
                pipeline = run_full_pipeline(df_raw, max_train_days=max_train_days)
                X_train  = pipeline["X_train"]
                X_val    = pipeline["X_val"]
                X_test   = pipeline["X_test"]
                preprocessor = pipeline["preprocessor"]

                st.session_state["pipeline"]     = pipeline
                st.session_state["preprocessor"] = preprocessor
                st.session_state["df_15"]        = pipeline["df_15"]
                st.session_state["train_df"]     = pipeline["train_df"]
                st.session_state["test_df"]      = pipeline["test_df"]

                # 3. Build RBM
                n_vis = preprocessor.n_visible_
                model = BernoulliRBM(
                    n_visible=n_vis,
                    n_hidden=int(n_hidden),
                    learning_rate=float(learning_rate),
                    cd_steps=int(cd_steps),
                    seed=int(seed),
                )

                # 4. Train with progress callback
                status_placeholder.info(f"Training RBM ({epochs} epochs)...")
                accumulated_losses = []

                def epoch_callback(epoch, train_loss, val_loss):
                    pct = min(1.0, (epoch + 1) / epochs)   # float 0.0–1.0 for st.progress
                    loss_str = f"{train_loss:.5f}"
                    val_str  = f"  |  val: {val_loss:.5f}" if val_loss is not None else ""
                    progress_bar.progress(
                        pct,
                        text=f"Epoch {epoch+1}/{epochs} — loss: {loss_str}{val_str}"
                    )
                    accumulated_losses.append(train_loss)
                    if (epoch + 1) % max(1, epochs // 10) == 0:
                        loss_placeholder.caption(
                            f"Reconstruction error so far: {loss_str}"
                        )

                model.train(
                    X_train,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    X_val=X_val,
                    callback=epoch_callback,
                    verbose=False,
                )

                # 5. Score
                status_placeholder.info("Computing anomaly scores...")
                val_scores  = score_samples(model, X_val)
                test_scores = score_samples(model, X_test)
                thr         = select_threshold(val_scores, float(threshold_pct))

                # 6. Baseline
                status_placeholder.info("Fitting statistical baseline...")
                baseline = StatisticalBaseline()
                baseline.fit(pipeline["train_df"])
                baseline_val_scores  = baseline.score(pipeline["val_df"])
                baseline_threshold   = baseline.select_threshold(
                    pipeline["val_df"], float(threshold_pct)
                )
                baseline_test_scores = baseline.score(pipeline["test_df"])

                # 7. Save models
                Path("models").mkdir(exist_ok=True)
                model.save("models/rbm.npz")
                preprocessor.save("models/preprocessor.pkl")

                # 8. Update session state
                st.session_state["rbm"]               = model
                st.session_state["trained"]           = True
                st.session_state["scores"]            = test_scores
                st.session_state["val_scores"]        = val_scores
                st.session_state["threshold"]         = thr
                st.session_state["baseline"]          = baseline
                st.session_state["baseline_scores"]   = baseline_test_scores
                st.session_state["baseline_threshold"] = baseline_threshold

                # 9. Cache to disk
                cache_data = {k: st.session_state[k] for k in [
                    "pipeline", "rbm", "preprocessor", "scores", "val_scores",
                    "threshold", "baseline", "baseline_scores", "baseline_threshold",
                    "df_15", "train_df", "test_df",
                ]}
                joblib.dump(cache_data, str(CACHE_PATH))

                progress_bar.progress(1.0, text="Training complete.")
                status_placeholder.success(
                    f"RBM trained successfully. {int(np.sum(test_scores > thr))} intervals "
                    f"flagged on test set (threshold: {thr:.4f})."
                )

                # Show loss chart
                try:
                    fig_loss = plot_training_loss(model.losses_)
                    loss_placeholder.empty()
                    st.plotly_chart(fig_loss, use_container_width=True)
                except Exception:
                    pass

                st.info("Navigate to **Anomaly Detection** to explore results.")

            except Exception as exc:
                progress_bar.empty()
                status_placeholder.error(f"Training failed: {exc}")
                st.exception(exc)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Info
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    trained      = st.session_state.get("trained", False)
    rbm          = st.session_state.get("rbm", None)
    preprocessor = st.session_state.get("preprocessor", None)
    threshold    = st.session_state.get("threshold", None)

    st.markdown("""
### RBM Architecture

```
  VISIBLE LAYER (21 binary units)         HIDDEN LAYER
  ┌─────────────────────────────┐         ┌──────────────┐
  │  bit_0 .. bit_2  (GAP)      │ ←─ W ─→ │  h_0         │
  │  bit_0 .. bit_2  (GRP)      │         │  h_1         │
  │  bit_0 .. bit_2  (Voltage)  │         │  ...         │
  │  bit_0 .. bit_2  (GI)       │         │  h_{n-1}     │
  │  bit_0 .. bit_2  (SM1)      │         └──────────────┘
  │  bit_0 .. bit_2  (SM2)      │
  │  bit_0 .. bit_2  (SM3)      │
  └─────────────────────────────┘
  7 features × 3 bits = 21 visible units
```

Each bit encodes whether a feature is **above a quantile threshold** (Q25 / Q50 / Q75)
learned from training data only.

**Why binary?** Bernoulli RBMs require binary inputs.  Continuous electricity
measurements are discretised into 3-bit "thermometer" codes, making the encoding
interpretable and avoiding numerical instability.
    """)

    if trained and rbm is not None:
        st.markdown("#### Model Parameters")
        params = {
            "n_visible":     rbm.n_visible,
            "n_hidden":      rbm.n_hidden,
            "learning_rate": rbm.learning_rate,
            "cd_steps":      rbm.cd_steps,
            "batch_size":    rbm.batch_size_ if hasattr(rbm, "batch_size_") else "N/A",
            "epochs_trained": len(rbm.losses_),
            "final_loss":    round(rbm.losses_[-1], 5) if rbm.losses_ else "N/A",
            "threshold":     round(threshold, 4) if threshold else "N/A",
        }
        params_df = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        try:
            bit_names = preprocessor.bit_names_ if preprocessor else None
            fig_w = plot_rbm_weights(rbm.W, bit_names)
            st.plotly_chart(fig_w, use_container_width=True)
            st.caption(
                "Weight matrix W (n_hidden × n_visible). "
                "Bright cells = strong connection between a hidden unit and a visible bit. "
                "Each hidden unit specialises in a different pattern of energy usage."
            )
        except Exception as e:
            st.caption(f"Weight chart unavailable: {e}")
    else:
        st.info("Train the RBM first to see model parameters and weight visualisation.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Baseline Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab_baseline:
    trained           = st.session_state.get("trained", False)
    scores_rbm        = st.session_state.get("scores", None)
    scores_base       = st.session_state.get("baseline_scores", None)
    threshold_rbm     = st.session_state.get("threshold", None)
    threshold_base    = st.session_state.get("baseline_threshold", None)

    if not trained or scores_rbm is None:
        st.info("Train the RBM first to compare with the baseline.")
    else:
        try:
            fig_dist = plot_score_distribution(
                scores_rbm, scores_base, threshold_rbm, threshold_base
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Distribution chart error: {e}")

        n_rbm  = int(np.sum(scores_rbm  > threshold_rbm))  if threshold_rbm  else 0
        n_base = int(np.sum(scores_base > threshold_base)) if (scores_base is not None and threshold_base) else 0
        n      = len(scores_rbm)

        comp_data = {
            "Metric": [
                "Threshold", "Intervals Flagged", "Flagging Rate (%)",
                "Score Min", "Score Max", "Score Mean", "Score Std"
            ],
            "RBM": [
                round(threshold_rbm, 4) if threshold_rbm else "N/A",
                n_rbm, f"{100*n_rbm/n:.1f}%",
                round(scores_rbm.min(), 4), round(scores_rbm.max(), 4),
                round(scores_rbm.mean(), 4), round(scores_rbm.std(), 4),
            ],
            "Statistical Baseline": [
                round(threshold_base, 4) if threshold_base else "N/A",
                n_base, f"{100*n_base/n:.1f}%" if n else "N/A",
                round(scores_base.min(), 4) if scores_base is not None else "N/A",
                round(scores_base.max(), 4) if scores_base is not None else "N/A",
                round(scores_base.mean(), 4) if scores_base is not None else "N/A",
                round(scores_base.std(), 4) if scores_base is not None else "N/A",
            ],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.info(
            "**About the baseline:** The statistical baseline computes a modified Z-score "
            "using the per-hour median and MAD from training data. It is easy to explain "
            "but cannot capture multivariate patterns like the RBM.",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Synthetic Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.warning(
        "The anomalies below are **SIMULATED** for controlled evaluation. "
        "They are NOT real appliance failures.",
    )

    trained        = st.session_state.get("trained", False)
    rbm_model      = st.session_state.get("rbm", None)
    prep           = st.session_state.get("preprocessor", None)
    baseline_model = st.session_state.get("baseline", None)
    test_df_eval   = st.session_state.get("test_df", None)
    thr_rbm        = st.session_state.get("threshold", None)
    thr_base       = st.session_state.get("baseline_threshold", None)

    if not trained or rbm_model is None:
        st.info("Train the RBM first, then run the evaluation.")
    else:
        eval_button = st.button("Run Controlled Evaluation", type="primary")

        prev_results = st.session_state.get("eval_results", None)
        if prev_results is not None and not eval_button:
            st.success("Showing previous evaluation results. Click above to rerun.")

        if eval_button or prev_results is not None:
            if eval_button:
                with st.spinner("Running controlled synthetic anomaly evaluation..."):
                    try:
                        results = run_controlled_evaluation(
                            test_df=test_df_eval,
                            rbm=rbm_model,
                            preprocessor=prep,
                            baseline=baseline_model,
                            rbm_threshold=thr_rbm,
                            baseline_threshold=thr_base,
                        )
                        st.session_state["eval_results"] = results
                        prev_results = results
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")
                        st.exception(exc)
                        st.stop()

            if prev_results is not None:
                st.markdown("#### Comparison Table")
                try:
                    st.dataframe(
                        prev_results.get("comparison_df", pd.DataFrame()),
                        use_container_width=True,
                    )
                except Exception:
                    pass

                # Confusion matrices
                st.markdown("#### Confusion Matrices")
                cm_col1, cm_col2 = st.columns(2)
                with cm_col1:
                    st.markdown("**RBM**")
                    try:
                        fig_cm_rbm = plot_confusion_matrix(
                            prev_results["rbm_cm"], title="RBM Confusion Matrix"
                        )
                        st.plotly_chart(fig_cm_rbm, use_container_width=True)
                    except Exception as e:
                        st.caption(f"RBM CM unavailable: {e}")
                with cm_col2:
                    st.markdown("**Statistical Baseline**")
                    try:
                        fig_cm_base = plot_confusion_matrix(
                            prev_results["baseline_cm"], title="Baseline Confusion Matrix"
                        )
                        st.plotly_chart(fig_cm_base, use_container_width=True)
                    except Exception as e:
                        st.caption(f"Baseline CM unavailable: {e}")

                # Metrics
                st.markdown("#### Detailed Metrics")
                metrics_rows = []
                for model_name in ["rbm", "baseline"]:
                    m = prev_results.get(f"{model_name}_metrics", {})
                    if m:
                        metrics_rows.append({
                            "Detector":   "RBM" if model_name == "rbm" else "Statistical Baseline",
                            "Precision":  round(m.get("precision", float("nan")), 3),
                            "Recall":     round(m.get("recall", float("nan")), 3),
                            "F1 Score":   round(m.get("f1", float("nan")), 3),
                            "FPR":        round(m.get("fpr", float("nan")), 3),
                            "TP":         m.get("tp", "N/A"),
                            "FP":         m.get("fp", "N/A"),
                            "TN":         m.get("tn", "N/A"),
                            "FN":         m.get("fn", "N/A"),
                        })
                if metrics_rows:
                    st.dataframe(
                        pd.DataFrame(metrics_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.info(
                    "**Interpretation:** Metrics above are computed on *simulated* anomalies "
                    "only. They indicate how well the detectors find the specific synthetic "
                    "fault patterns injected. They do not measure real-world fault detection "
                    "performance.",
                )
