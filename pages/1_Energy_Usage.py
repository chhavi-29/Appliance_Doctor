"""
1_Energy_Usage.py
-----------------
Energy Usage Explorer page — Power trends, sub-metering, hourly/daily patterns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from src.visualizations import (
    plot_power_timeseries,
    plot_submetering,
    plot_hourly_heatmap,
    plot_daily_consumption,
)

st.set_page_config(page_title="Energy Usage", page_icon=None, layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Guard: need trained pipeline
# ─────────────────────────────────────────────────────────────────────────────
pipeline = st.session_state.get("pipeline", None)
df_15    = st.session_state.get("df_15", None)

st.markdown("## Energy Usage Explorer")

if df_15 is None or pipeline is None:
    st.info(
        "**No data loaded yet.**\n\n"
        "Please go to **Model Lab** and click **Train RBM** first.",
    )
    st.page_link("pages/3_Model_Lab.py", label="Go to Model Lab")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: date range selector
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Date Range")
    min_date = df_15.index.min().date()
    max_date = df_15.index.max().date()
    date_range = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    st.divider()
    st.markdown("### Data Info")
    st.caption(f"Total intervals: **{len(df_15):,}**")
    st.caption(f"From: **{min_date}**")
    st.caption(f"To:   **{max_date}**")

# ─────────────────────────────────────────────────────────────────────────────
# Filter by date range
# ─────────────────────────────────────────────────────────────────────────────
mask     = (df_15.index.date >= start_date) & (df_15.index.date <= end_date)
df_view  = df_15[mask]

if df_view.empty:
    st.warning("No data in selected date range. Adjust the date picker.")
    st.stop()

# Get anomaly info if available
scores    = st.session_state.get("scores", None)
threshold = st.session_state.get("threshold", None)

# Try to build a flagged mask aligned with df_view
# scores are aligned to the TEST split; we use df_15 global index to align
test_df = st.session_state.get("test_df", None)
flagged_mask = None
if scores is not None and threshold is not None and test_df is not None:
    try:
        flag_series = pd.Series(scores > threshold, index=test_df.index)
        aligned     = flag_series.reindex(df_view.index, fill_value=False)
        flagged_mask = aligned.values
    except Exception:
        flagged_mask = None

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Power Over Time", "Sub-Metering", "Hourly Patterns", "Daily Summary"]
)

# ── TAB 1: Power Over Time ───────────────────────────────────────────────────
with tab1:
    try:
        fig = plot_power_timeseries(df_view, flagged_mask=flagged_mask)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

    m1, m2, m3 = st.columns(3)
    gap = df_view.get("Global_active_power")
    if gap is not None and not gap.dropna().empty:
        # kWh ≈ mean kW × (15/60) hours × n_intervals
        total_kwh = float(gap.mean()) * (15 / 60) * len(gap)
        with m1:
            st.metric("Avg Active Power", f"{gap.mean():.3f} kW")
        with m2:
            st.metric("Peak Active Power", f"{gap.max():.3f} kW")
        with m3:
            st.metric("Estimated Total", f"{total_kwh:.1f} kWh")

    if flagged_mask is not None:
        st.caption(
            f"Orange markers = {int(flagged_mask.sum())} flagged intervals "
            f"(anomaly score > threshold). These are *statistical anomalies*, "
            f"not confirmed faults."
        )

# ── TAB 2: Sub-Metering ─────────────────────────────────────────────────────
with tab2:
    try:
        fig2 = plot_submetering(df_view)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

    sub_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    sub_labels = {
        "Sub_metering_1": "Kitchen",
        "Sub_metering_2": "Laundry",
        "Sub_metering_3": "Water Heater/AC",
    }
    avail_subs = [c for c in sub_cols if c in df_view.columns]
    if avail_subs:
        st.markdown("#### Sub-metering Statistics (Wh per 15-min interval)")
        stats_data = {}
        for col in avail_subs:
            s = df_view[col].dropna()
            stats_data[sub_labels[col]] = {
                "Mean (Wh)":   round(s.mean(), 2),
                "Median (Wh)": round(s.median(), 2),
                "Max (Wh)":    round(s.max(), 2),
                "Total (Wh)":  round(s.sum(), 0),
            }
        st.dataframe(pd.DataFrame(stats_data).T, use_container_width=True)

    st.info(
        "Sub_metering_1 = Kitchen · Sub_metering_2 = Laundry room · "
        "Sub_metering_3 = Water heater / AC",
    )

# ── TAB 3: Hourly Patterns ───────────────────────────────────────────────────
with tab3:
    try:
        fig3 = plot_hourly_heatmap(df_15)   # use full data for meaningful heatmap
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

    st.markdown("""
    #### How to read this heatmap
    - **Rows** = Day of the week (Mon–Sun)
    - **Columns** = Hour of the day (0–23)
    - **Colour intensity** = Average active power (kW) for that hour/day combination
    - Bright cells → typically high consumption  |  Dark cells → typically low consumption
    - Common patterns: morning peaks, evening peaks, low overnight usage
    """)

# ── TAB 4: Daily Summary ─────────────────────────────────────────────────────
with tab4:
    try:
        fig4 = plot_daily_consumption(df_view)
        st.plotly_chart(fig4, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

    if "Global_active_power" in df_view.columns:
        daily = (
            df_view["Global_active_power"]
            .resample("D")
            .mean()
            .dropna()
        )
        if not daily.empty:
            top5 = daily.nlargest(5).reset_index()
            top5.columns = ["Date", "Avg Active Power (kW)"]
            top5["Avg Active Power (kW)"] = top5["Avg Active Power (kW)"].round(3)
            st.markdown("#### Top 5 Highest Consumption Days")
            st.dataframe(top5, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data preview
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("Raw Data Preview (last 20 intervals)"):
    preview = df_view.tail(20).copy()
    numeric_cols = preview.select_dtypes(include=np.number).columns
    st.dataframe(
        preview[numeric_cols].round(4),
        use_container_width=True,
    )
    st.caption(f"Showing last 20 of {len(df_view):,} intervals in selected date range.")
