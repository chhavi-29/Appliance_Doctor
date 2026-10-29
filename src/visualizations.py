"""
visualizations.py
-----------------
Plotly figure factory functions for the Invisible Appliance Doctor dashboard.

All functions return go.Figure objects that can be displayed with
st.plotly_chart(...) in Streamlit pages.

Design theme
------------
Dark background, electric blue / amber / red accent palette.
All charts are interactive (hover, zoom, pan).
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Color palette
# --------------------------------------------------------------------------- #
NORMAL_COLOR  = "#00D4FF"   # Electric blue
ANOMALY_COLOR = "#FF6B35"   # Amber-orange
THRESHOLD_COLOR = "#FF3366" # Red for threshold line
BG_COLOR      = "#0E1117"   # Streamlit dark bg
GRID_COLOR    = "#1F2937"   # Subtle grid
TEXT_COLOR    = "#E0E0E0"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
        ),
    )
)


def _apply_theme(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        height=height,
        margin=dict(l=60, r=30, t=50, b=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# 1. Power time series
# --------------------------------------------------------------------------- #
def plot_power_timeseries(
    df: pd.DataFrame,
    flagged_mask: Optional[np.ndarray] = None,
    col: str = "Global_active_power",
    title: str = "Global Active Power Consumption",
) -> go.Figure:
    """
    Line chart of power consumption over time with anomalies highlighted.

    Parameters
    ----------
    df           : 15-min DataFrame with DatetimeIndex
    flagged_mask : bool array aligned with df (True = flagged)
    col          : Column to plot (default Global_active_power)
    """
    if col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{col}' not found.", showarrow=False)
        return _apply_theme(fig, title)

    fig = go.Figure()

    if flagged_mask is not None and flagged_mask.any():
        # Normal intervals
        normal_mask = ~flagged_mask
        fig.add_trace(go.Scatter(
            x=df.index[normal_mask],
            y=df[col].values[normal_mask],
            mode="lines",
            name="Normal",
            line=dict(color=NORMAL_COLOR, width=1.5),
            opacity=0.85,
        ))
        # Anomalous intervals — red markers
        fig.add_trace(go.Scatter(
            x=df.index[flagged_mask],
            y=df[col].values[flagged_mask],
            mode="markers",
            name="⚠ Unusual",
            marker=dict(color=ANOMALY_COLOR, size=6, symbol="circle"),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col].values,
            mode="lines",
            name=col,
            line=dict(color=NORMAL_COLOR, width=1.5),
        ))

    fig.update_yaxes(title_text="Active Power (kW)")
    fig.update_xaxes(title_text="Time")
    return _apply_theme(fig, title, height=380)


# --------------------------------------------------------------------------- #
# 2. Anomaly scores
# --------------------------------------------------------------------------- #
def plot_anomaly_scores(
    scores:     np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold:  float,
    title:      str = "Anomaly Score Timeline",
) -> go.Figure:
    """
    Bar chart of anomaly scores with a horizontal threshold line.
    Bars above threshold are coloured red/orange.
    """
    colors = np.where(scores > threshold, ANOMALY_COLOR, NORMAL_COLOR)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=timestamps,
        y=scores,
        marker_color=colors,
        name="Anomaly Score (F(v))",
        hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color=THRESHOLD_COLOR, width=2, dash="dash"),
        annotation_text=f"Threshold: {threshold:.3f}",
        annotation_font_color=THRESHOLD_COLOR,
    )

    fig.update_yaxes(title_text="Anomaly Score (higher = more unusual)")
    fig.update_xaxes(title_text="Time")
    return _apply_theme(fig, title, height=350)


# --------------------------------------------------------------------------- #
# 3. Sub-metering trends
# --------------------------------------------------------------------------- #
def plot_submetering(
    df: pd.DataFrame,
    title: str = "Sub-Metering Energy Breakdown",
) -> go.Figure:
    """
    Line chart of all three sub-metering channels on the same axes.
    """
    sm_cols = {
        "Sub_metering_1": ("Kitchen",         "#FF6B35"),
        "Sub_metering_2": ("Laundry Room",    "#9B59B6"),
        "Sub_metering_3": ("Water Heater/AC", "#2ECC71"),
    }
    fig = go.Figure()
    for col, (label, color) in sm_cols.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col].values,
                mode="lines", name=label,
                line=dict(color=color, width=1.5),
                hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:.1f}} Wh<extra></extra>",
            ))

    fig.update_yaxes(title_text="Energy per 15-min interval (Wh)")
    fig.update_xaxes(title_text="Time")
    return _apply_theme(fig, title, height=350)


# --------------------------------------------------------------------------- #
# 4. Hourly usage heatmap
# --------------------------------------------------------------------------- #
def plot_hourly_heatmap(
    df: pd.DataFrame,
    col: str = "Global_active_power",
    title: str = "Average Consumption by Hour & Day of Week",
) -> go.Figure:
    """
    Heatmap: rows = hour of day (0–23), columns = day of week (Mon–Sun).
    """
    if col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{col}' not found.", showarrow=False)
        return _apply_theme(fig, title)

    df_copy = df[[col]].copy()
    df_copy["hour"] = df_copy.index.hour
    df_copy["dow"]  = df_copy.index.dayofweek

    pivot = df_copy.groupby(["hour", "dow"])[col].mean().unstack(fill_value=np.nan)
    pivot = pivot.reindex(columns=range(7), fill_value=np.nan)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=day_names,
        y=[f"{h:02d}:00" for h in range(24)],
        colorscale="Blues",
        hovertemplate="<b>%{x} %{y}</b><br>Avg: %{z:.3f} kW<extra></extra>",
        colorbar=dict(title="kW", tickfont=dict(color=TEXT_COLOR)),
    ))

    fig.update_yaxes(autorange="reversed", title_text="Hour of Day")
    fig.update_xaxes(title_text="Day of Week")
    return _apply_theme(fig, title, height=500)


# --------------------------------------------------------------------------- #
# 5. Training loss
# --------------------------------------------------------------------------- #
def plot_training_loss(
    losses: List[float],
    title:  str = "RBM Training: Reconstruction Error per Epoch",
) -> go.Figure:
    """
    Line chart of reconstruction error across training epochs.
    A decreasing curve indicates the RBM is learning the data distribution.
    """
    epochs = list(range(1, len(losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=losses,
        mode="lines+markers",
        name="Reconstruction Error",
        line=dict(color=NORMAL_COLOR, width=2),
        marker=dict(size=4),
        hovertemplate="Epoch %{x}<br>Error: %{y:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="MSE Reconstruction Error")
    return _apply_theme(fig, title, height=320)


# --------------------------------------------------------------------------- #
# 6. RBM weight heatmap
# --------------------------------------------------------------------------- #
def plot_rbm_weights(
    W: np.ndarray,
    bit_names: Optional[List[str]] = None,
    title: str = "RBM Weight Matrix (W)",
) -> go.Figure:
    """
    Heatmap of the RBM weight matrix W (n_hidden × n_visible).
    Positive weights indicate a hidden unit is excited by that bit.
    """
    n_hidden, n_visible = W.shape
    x_labels = bit_names or [f"v{i}" for i in range(n_visible)]
    y_labels = [f"h{j}" for j in range(n_hidden)]

    fig = go.Figure(data=go.Heatmap(
        z=W,
        x=x_labels,
        y=y_labels,
        colorscale="RdBu",
        zmid=0,
        hovertemplate="Hidden %{y} — Visible %{x}<br>Weight: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Weight", tickfont=dict(color=TEXT_COLOR)),
    ))

    fig.update_xaxes(title_text="Visible Unit (binary feature bit)", tickangle=-45)
    fig.update_yaxes(title_text="Hidden Unit")
    return _apply_theme(fig, title, height=max(300, n_hidden * 25 + 100))


# --------------------------------------------------------------------------- #
# 7. Score distribution comparison
# --------------------------------------------------------------------------- #
def plot_score_distribution(
    rbm_scores:      np.ndarray,
    baseline_scores: np.ndarray,
    rbm_threshold:      float,
    baseline_threshold: float,
    title: str = "Anomaly Score Distributions",
) -> go.Figure:
    """
    Overlaid histograms of RBM and baseline scores with threshold lines.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["RBM Scores (Free Energy)", "Baseline Scores (MAD Z-score)"],
    )

    fig.add_trace(go.Histogram(
        x=rbm_scores, name="RBM", marker_color=NORMAL_COLOR,
        opacity=0.75, nbinsx=50,
        hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=baseline_scores, name="Baseline", marker_color="#9B59B6",
        opacity=0.75, nbinsx=50,
        hovertemplate="Z-score: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ), row=1, col=2)

    # Threshold lines
    fig.add_vline(x=rbm_threshold,      line=dict(color=THRESHOLD_COLOR, dash="dash"), row=1, col=1)
    fig.add_vline(x=baseline_threshold, line=dict(color=THRESHOLD_COLOR, dash="dash"), row=1, col=2)

    fig = _apply_theme(fig, title, height=350)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    return fig


# --------------------------------------------------------------------------- #
# 8. Confusion matrix
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    cm: list,
    title: str = "Confusion Matrix",
    labels: List[str] = None,
) -> go.Figure:
    """
    Annotated heatmap confusion matrix.

    Parameters
    ----------
    cm     : 2×2 list [[TN, FP], [FN, TP]]
    labels : ['Normal', 'Anomaly'] by default
    """
    if labels is None:
        labels = ["Normal (0)", "Anomaly (1)"]

    cm_arr = np.array(cm)
    # Text annotations
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=str(cm_arr[i, j]),
                showarrow=False,
                font=dict(size=18, color="white"),
            ))

    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=labels,
        y=labels,
        colorscale=[[0, "#1F2937"], [1, ANOMALY_COLOR]],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(annotations=annotations)
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual", autorange="reversed")
    return _apply_theme(fig, title, height=320)


# --------------------------------------------------------------------------- #
# 9. Daily power consumption bar chart
# --------------------------------------------------------------------------- #
def plot_daily_consumption(
    df: pd.DataFrame,
    col: str = "Global_active_power",
    title: str = "Daily Total Active Power Consumption",
) -> go.Figure:
    """
    Bar chart: total kWh per day.
    """
    if col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{col}' not found.", showarrow=False)
        return _apply_theme(fig, title)

    daily = df[col].resample("D").mean() * 24  # kW × 24 h ≈ kWh
    daily = daily.dropna()

    fig = go.Figure(go.Bar(
        x=daily.index, y=daily.values,
        marker_color=NORMAL_COLOR, opacity=0.85,
        name="Daily kWh (approx)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>~%{y:.1f} kWh<extra></extra>",
    ))

    fig.update_yaxes(title_text="Approx. Daily Consumption (kWh)")
    fig.update_xaxes(title_text="Date")
    return _apply_theme(fig, title, height=340)


# --------------------------------------------------------------------------- #
# 10. Interval detail comparison bar chart
# --------------------------------------------------------------------------- #
def plot_interval_comparison(
    readings: dict,
    typical:  dict,
    title:    str = "Interval Readings vs. Typical (same hour)",
) -> go.Figure:
    """
    Grouped bar chart comparing one interval's readings against the
    hour-of-day median from training data.
    """
    features = [k for k in readings if not pd.isna(readings.get(k))]
    actual   = [readings[k] for k in features]
    typical_ = [typical.get(k, np.nan) for k in features]

    # Short display names
    short_names = {
        "Global_active_power":   "Active Power",
        "Global_reactive_power": "Reactive Power",
        "Voltage":               "Voltage",
        "Global_intensity":      "Intensity",
        "Sub_metering_1":        "SM1 (Kitchen)",
        "Sub_metering_2":        "SM2 (Laundry)",
        "Sub_metering_3":        "SM3 (Heater/AC)",
    }
    display_names = [short_names.get(f, f) for f in features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="This interval",
        x=display_names, y=actual,
        marker_color=ANOMALY_COLOR, opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        name=f"Typical (training median, same hour)",
        x=display_names, y=typical_,
        marker_color=NORMAL_COLOR, opacity=0.7,
    ))

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Value (mixed units)")
    return _apply_theme(fig, title, height=350)
