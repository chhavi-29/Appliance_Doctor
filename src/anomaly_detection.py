"""
anomaly_detection.py
--------------------
Anomaly scoring, threshold selection, and plain-language explanation
for the Invisible Appliance Doctor.

How anomaly scoring works
-------------------------
1. For each 15-minute interval, compute the RBM anomaly score:
       score(v) = +F(v)  where F is the free energy of the binary encoding v
   Higher score → interval is MORE surprising to the model → more anomalous.

2. Use the VALIDATION set to choose a threshold (not the test set!).
   This avoids selecting a threshold that perfectly fits the test data,
   which would give misleadingly optimistic results.

3. Flag any interval where score > threshold as "Unusual pattern".

Important caveats
-----------------
• A high anomaly score does NOT prove an appliance is faulty.
• The model has learned what is statistically typical in the training period.
  Anything that looks statistically different will be flagged.
• Flagged intervals require human investigation to determine the root cause.
• Always check whether an alert corresponds to a genuinely unusual real-world
  event (e.g., a party, a vacation return, a hot day) before assuming a fault.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_samples(rbm, X_binary: np.ndarray) -> np.ndarray:
    """
    Compute anomaly scores for binary-encoded intervals using the RBM.

    Score = +free_energy(v).  Higher = more anomalous.

    Parameters
    ----------
    rbm      : Trained BernoulliRBM instance.
    X_binary : np.ndarray, shape (n_samples, n_visible) — binary feature matrix.

    Returns
    -------
    scores : np.ndarray, shape (n_samples,) — one score per interval.
    """
    if not rbm.is_trained():
        raise RuntimeError("RBM must be trained before scoring. Call rbm.train() first.")
    scores = rbm.anomaly_score(X_binary)
    return scores.astype(np.float64)


def reconstruction_error_score(rbm, X_binary: np.ndarray) -> np.ndarray:
    """
    Alternative score: per-sample MSE between input and reconstruction.

    Useful as a sanity check alongside the free-energy score.
    High reconstruction error → the RBM struggles to reproduce the input
    → the input is unusual relative to what the model has learned.
    """
    if not rbm.is_trained():
        raise RuntimeError("RBM must be trained before scoring.")
    return rbm.reconstruction_error(X_binary)


# --------------------------------------------------------------------------- #
# Threshold selection
# --------------------------------------------------------------------------- #
def select_threshold(
    val_scores: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """
    Pick an anomaly threshold using the validation set.

    Method: use the p-th percentile of validation scores.
    By setting p=95 (default), we expect ~5% of validation intervals
    to be flagged. Adjust the percentile in the dashboard to balance
    sensitivity (recall) against false alarms (false positive rate).

    WHY the validation set?
    -----------------------
    Using the test set to pick the threshold would be "peeking" at the
    answers — our metrics on the test set would be optimistically biased.
    The validation set is unseen by the training process but is also
    NOT reported as final evaluation results.

    Parameters
    ----------
    val_scores : np.ndarray — anomaly scores for validation intervals.
    percentile : float in (0, 100) — desired flagging rate (default 95 = flag top 5%).

    Returns
    -------
    float : threshold value.
    """
    if not (0 < percentile < 100):
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    threshold = float(np.percentile(val_scores, percentile))
    logger.info(
        "Threshold selected: %.4f (%.1f-th percentile of %d val scores)",
        threshold, percentile, len(val_scores),
    )
    return threshold


# --------------------------------------------------------------------------- #
# Flagging
# --------------------------------------------------------------------------- #
def flag_anomalies(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Return a boolean mask indicating which intervals exceed the threshold.

    Parameters
    ----------
    scores    : np.ndarray, shape (n_samples,)
    threshold : float — intervals with score > threshold are flagged

    Returns
    -------
    np.ndarray of bool, shape (n_samples,)
    True = "Unusual pattern – investigate further"
    """
    return scores > threshold


def get_anomaly_summary(
    scores: np.ndarray,
    threshold: float,
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Build a summary DataFrame of anomalous intervals.

    Parameters
    ----------
    scores     : np.ndarray — anomaly scores per interval
    threshold  : float
    timestamps : optional DatetimeIndex — if provided, included as a column

    Returns
    -------
    pd.DataFrame with columns: timestamp (if provided), score, label
    """
    mask  = flag_anomalies(scores, threshold)
    df    = pd.DataFrame({"score": scores})
    df["flagged"] = mask
    df["label"]   = np.where(mask, "⚠ Unusual pattern", "✓ Normal")
    if timestamps is not None:
        df.index = timestamps
    return df[df["flagged"]].sort_values("score", ascending=False)


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #
def explain_interval(
    interval_idx: int,
    df_15: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    train_df: pd.DataFrame,
    preprocessor=None,
) -> dict:
    """
    Generate a plain-language explanation for a specific flagged interval.

    Strategy
    --------
    1. Look up the interval's original-scale electricity readings.
    2. Find all training intervals at the same hour-of-day.
    3. Compare the flagged interval's readings against the training-hour median.
    4. Report which features deviate most.

    Parameters
    ----------
    interval_idx : int — positional index into df_15 / scores array
    df_15        : pd.DataFrame — full 15-min DataFrame (original scale)
    scores       : np.ndarray — anomaly scores aligned with df_15
    threshold    : float — current anomaly threshold
    train_df     : pd.DataFrame — training split (for typical-usage stats)
    preprocessor : optional Preprocessor (for bit names)

    Returns
    -------
    dict with keys:
        timestamp      : str
        score          : float
        is_anomaly     : bool
        hour           : int
        readings       : dict[col → value] for the flagged interval
        typical        : dict[col → median] for same hour in training data
        deviations     : list of (feature, ratio) sorted by deviation magnitude
        summary        : str — plain-language summary paragraph
        disclaimer     : str — caveat about what anomalies do NOT mean
    """
    if interval_idx >= len(df_15):
        raise IndexError(f"interval_idx {interval_idx} out of range (n={len(df_15)})")

    row = df_15.iloc[interval_idx]
    ts  = df_15.index[interval_idx]
    score = float(scores[interval_idx])
    is_anomaly = score > threshold

    hour = ts.hour
    dow  = ts.dayofweek

    power_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    available_cols = [c for c in power_cols if c in df_15.columns]

    # Current readings
    readings = {col: float(row.get(col, np.nan)) for col in available_cols}

    # Typical readings at same hour in training data
    same_hour_train = train_df[train_df.index.hour == hour]
    typical = {
        col: float(same_hour_train[col].median()) if col in same_hour_train.columns
             else np.nan
        for col in available_cols
    }

    # Compute deviation ratios (reading / typical), cap at ±20
    deviations = []
    for col in available_cols:
        val = readings.get(col, np.nan)
        med = typical.get(col, np.nan)
        if pd.isna(val) or pd.isna(med) or med == 0:
            ratio = np.nan
        else:
            ratio = val / med
        deviations.append((col, ratio))

    # Sort by absolute deviation from 1.0 (= typical)
    deviations.sort(
        key=lambda x: abs(x[1] - 1.0) if not np.isnan(x[1]) else 0,
        reverse=True,
    )

    # Build plain-language summary
    flag_label = "⚠ UNUSUAL PATTERN" if is_anomaly else "✓ NORMAL"
    day_names  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    summary_parts = [
        f"{flag_label}  |  {ts.strftime('%Y-%m-%d %H:%M')}  "
        f"(Hour {hour}:00, {day_names[dow]})\n",
        f"Anomaly score: {score:.3f}  (threshold: {threshold:.3f})\n\n",
    ]

    gap = readings.get("Global_active_power", np.nan)
    typ_gap = typical.get("Global_active_power", np.nan)

    if not np.isnan(gap) and not np.isnan(typ_gap):
        direction = "higher" if gap > typ_gap else "lower"
        summary_parts.append(
            f"Global active power was {gap:.2f} kW, compared to the typical "
            f"{typ_gap:.2f} kW at this hour ({direction} than usual).\n"
        )

    top_deviations = [(f, r) for f, r in deviations[:3] if not np.isnan(r)]
    for feat, ratio in top_deviations:
        pct = (ratio - 1.0) * 100
        direction_str = f"{abs(pct):.0f}% above" if pct > 0 else f"{abs(pct):.0f}% below"
        summary_parts.append(f"  • {feat}: {direction_str} the typical value for this hour.\n")

    disclaimer = (
        "⚠ IMPORTANT: An unusual pattern does NOT confirm an appliance fault. "
        "This alert means the energy usage at this time was statistically "
        "different from what the model learned as typical. Causes may include: "
        "guests, unusual appliance usage, measurement noise, or a genuine fault. "
        "Human investigation is required to determine the actual cause."
    )

    return {
        "timestamp":   str(ts),
        "score":       score,
        "is_anomaly":  is_anomaly,
        "hour":        hour,
        "dow":         dow,
        "readings":    readings,
        "typical":     typical,
        "deviations":  deviations,
        "summary":     "".join(summary_parts),
        "disclaimer":  disclaimer,
    }


# --------------------------------------------------------------------------- #
# Batch explanation helper
# --------------------------------------------------------------------------- #
def get_flagged_intervals_df(
    df_15: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Return a display DataFrame of all flagged intervals with key readings.

    Parameters
    ----------
    df_15     : 15-minute DataFrame (original scale)
    scores    : anomaly score array (same length as df_15)
    threshold : alert threshold

    Returns
    -------
    pd.DataFrame sorted by score descending, with columns:
        timestamp, score, Global_active_power, Sub_metering_1..3, label
    """
    flags = flag_anomalies(scores, threshold)
    flagged_df = df_15[flags].copy()
    flagged_df["anomaly_score"] = scores[flags]
    flagged_df["label"] = "⚠ Unusual pattern"
    flagged_df = flagged_df.reset_index().rename(columns={"datetime": "timestamp"})
    flagged_df = flagged_df.sort_values("anomaly_score", ascending=False)

    display_cols = ["timestamp", "anomaly_score", "label"]
    power_cols   = [
        "Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
    ]
    for col in power_cols:
        if col in flagged_df.columns:
            display_cols.append(col)

    return flagged_df[display_cols].reset_index(drop=True)


def score_stats(scores: np.ndarray) -> dict:
    """Return descriptive statistics for an array of anomaly scores."""
    return {
        "min":    float(np.min(scores)),
        "q25":    float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "q75":    float(np.percentile(scores, 75)),
        "q95":    float(np.percentile(scores, 95)),
        "max":    float(np.max(scores)),
        "mean":   float(np.mean(scores)),
        "std":    float(np.std(scores)),
    }
