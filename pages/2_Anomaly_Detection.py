"""
2_Anomaly_Detection.py
----------------------
Anomaly Detection page — Timeline, flagged intervals table, investigation panel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import io

from src.anomaly_detection import (
    flag_anomalies,
    get_flagged_intervals_df,
    explain_interval,
    select_threshold,
    score_stats,
)
from src.visualizations import (
    plot_power_timeseries,
    plot_anomaly_scores,
    plot_interval_comparison,
)

st.set_page_config(page_title="Anomaly Detection", page_icon=None, layout="wide")

st.markdown("## Anomaly Detection")

# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────
trained   = st.session_state.get("trained", False)
scores    = st.session_state.get("scores", None)
val_scores = st.session_state.get("val_scores", None)
threshold = st.session_state.get("threshold", None)
df_15     = st.session_state.get("df_15", None)
test_df   = st.session_state.get("test_df", None)
train_df  = st.session_state.get("train_df", None)

if not trained or scores is None or test_df is None:
    st.warning(
        "**Model not trained yet.** Please go to Model Lab and click **Train RBM** first.",
    )
    st.page_link("pages/3_Model_Lab.py", label="Go to Model Lab")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — threshold control
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Detection Settings")
    pct = st.slider(
        "Alert Threshold (percentile of val scores)",
        min_value=80, max_value=99, value=95, step=1,
        help="Higher % → fewer alerts (fewer false positives, more false negatives).",
    )
    if val_scores is not None:
        current_threshold = select_threshold(val_scores, float(pct))
        st.session_state["threshold"] = current_threshold
        threshold = current_threshold
    else:
        current_threshold = threshold

    n_flagged   = int(np.sum(scores > current_threshold))
    flag_pct    = 100 * n_flagged / len(scores) if len(scores) else 0

    st.metric("Threshold value", f"{current_threshold:.4f}")
    st.metric("Intervals flagged", f"{n_flagged:,}")
    st.metric("Flagging rate", f"{flag_pct:.1f}%")
    st.caption("Adjusting this slider does NOT retrain the model — it only shifts the alert line.")

    st.divider()
    st.warning(
        "An unusual pattern does **NOT** confirm an appliance fault. "
        "Human investigation is required.",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Summary row
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total test intervals", f"{len(scores):,}")
with c2:
    st.metric("Flagged", f"{n_flagged:,}")
with c3:
    st.metric("Flagging rate", f"{flag_pct:.1f}%")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Timeline", "Flagged Intervals", "Investigate"])

flagged_mask = scores > current_threshold

# ── TAB 1: Timeline ──────────────────────────────────────────────────────────
with tab1:
    try:
        fig_ts = plot_power_timeseries(test_df, flagged_mask=flagged_mask)
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption("Orange markers = intervals where anomaly score exceeds threshold.")
    except Exception as e:
        st.error(f"Timeline chart error: {e}")

    try:
        fig_scores = plot_anomaly_scores(scores, test_df.index, current_threshold)
        st.plotly_chart(fig_scores, use_container_width=True)
        st.caption(
            "Red dashed line = current alert threshold. "
            "Bars above the line are flagged as unusual."
        )
    except Exception as e:
        st.error(f"Score chart error: {e}")

    with st.expander("How is the anomaly score computed?"):
        st.markdown("""
The anomaly score for each 15-minute interval is computed using the RBM's **free energy**:

$$\\text{score}(v) = F(v) = -v \\cdot b_v - \\sum_j \\log(1 + \\exp(b_{h_j} + W_j \\cdot v))$$

- **Higher score → interval surprised the model → more anomalous.**
- The RBM learns what *normal* looks like during training.  Intervals that look very
  different from the training distribution have high free energy under the model.
- The threshold is chosen as the **p-th percentile of validation-set scores**, ensuring
  no leakage from the test set.
        """)

# ── TAB 2: Flagged Intervals ─────────────────────────────────────────────────
with tab2:
    st.warning(
        "**IMPORTANT:** These flags indicate *statistical anomalies only*. "
        "They do **NOT** confirm appliance faults or any real-world problem.",
    )

    try:
        flagged_df = get_flagged_intervals_df(test_df, scores, current_threshold)
    except Exception as e:
        st.error(f"Could not build flagged table: {e}")
        st.stop()

    st.markdown(f"**{len(flagged_df)} intervals flagged** above threshold {current_threshold:.4f}")

    if flagged_df.empty:
        st.success("No intervals flagged at the current threshold. Try lowering the percentile.")
    else:
        st.dataframe(
            flagged_df.style.format({"anomaly_score": "{:.4f}"}),
            use_container_width=True,
            height=400,
        )

        # Download button
        csv_buf = io.StringIO()
        flagged_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download flagged intervals as CSV",
            data=csv_buf.getvalue(),
            file_name="flagged_intervals.csv",
            mime="text/csv",
        )

# ── TAB 3: Investigate ───────────────────────────────────────────────────────
with tab3:
    st.markdown("### Investigate a Flagged Interval")

    flagged_indices = np.where(flagged_mask)[0]

    if len(flagged_indices) == 0:
        st.info("No intervals are flagged at the current threshold.")
    else:
        flagged_timestamps = test_df.index[flagged_indices]
        ts_options = [str(ts) for ts in flagged_timestamps]

        selected_ts_str = st.selectbox(
            "Select a flagged interval to investigate:",
            options=ts_options,
            index=0,
        )

        selected_pos = ts_options.index(selected_ts_str)
        selected_idx = flagged_indices[selected_pos]

        # Align with full df_15 for explanation
        try:
            # Map test_df positional index to df_15 positional index
            selected_ts = test_df.index[selected_idx]
            global_idx  = df_15.index.get_loc(selected_ts)

            # We need scores aligned with df_15; scores are for test only
            # Build a full score array padded with NaN for non-test rows
            full_scores = np.full(len(df_15), np.nan)
            test_start  = df_15.index.get_loc(test_df.index[0])
            full_scores[test_start: test_start + len(scores)] = scores

            explanation = explain_interval(
                interval_idx=global_idx,
                df_15=df_15,
                scores=full_scores,
                threshold=current_threshold,
                train_df=train_df if train_df is not None else test_df,
            )

            # Display
            st.markdown("#### Explanation")
            st.markdown(f"```\n{explanation['summary']}\n```")

            st.error(explanation["disclaimer"])

            # Side-by-side comparison
            c_read, c_typ = st.columns(2)
            with c_read:
                st.markdown("**Actual Readings**")
                read_df = pd.DataFrame.from_dict(
                    explanation["readings"], orient="index", columns=["Value"]
                ).round(4)
                st.dataframe(read_df, use_container_width=True)
            with c_typ:
                st.markdown(f"**Typical at Hour {explanation['hour']}:00 (training median)**")
                typ_df = pd.DataFrame.from_dict(
                    explanation["typical"], orient="index", columns=["Typical"]
                ).round(4)
                st.dataframe(typ_df, use_container_width=True)

            # Comparison chart
            try:
                fig_comp = plot_interval_comparison(
                    explanation["readings"], explanation["typical"]
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            except Exception:
                pass

            # Deviation table
            st.markdown("#### Feature Deviations (reading / typical)")
            dev_data = [
                {
                    "Feature": f,
                    "Ratio (actual/typical)": round(r, 3) if not np.isnan(r) else "N/A",
                    "Direction": (
                        "Above normal" if r > 1.05
                        else ("Below normal" if r < 0.95 else "Normal")
                    ) if not np.isnan(r) else "N/A",
                }
                for f, r in explanation["deviations"]
            ]
            st.dataframe(pd.DataFrame(dev_data), use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Could not generate explanation: {exc}")
