"""
evaluation.py
-------------
Controlled synthetic anomaly evaluation for the Invisible Appliance Doctor.

Why synthetic anomalies?
------------------------
The UCI household power consumption dataset does NOT contain verified
appliance-fault labels. We cannot compute precision/recall against real faults.

Instead, we inject KNOWN synthetic anomalies into a copy of the held-out
test data and measure how well each detector finds them.

IMPORTANT caveats:
• These anomalies are SIMULATED — they do not represent real appliance failures.
• Real anomalies are more subtle and varied than our injected patterns.
• Good performance on simulated anomalies does not guarantee real-world accuracy.
• The test data used here is NOT used to train the RBM or select its threshold.

Anomaly types injected
----------------------
1. spike      : Short (1–4 intervals) × large (3–6 ×) power spike
2. sustained  : Sustained high consumption for 30–60 min (2–4 ×)
3. night      : Unusual consumption during quiet nighttime hours (00:00–05:00)
4. submetering: Abrupt large increase in a specific sub-metering channel

All injected intervals are recorded so we know the ground truth.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)




# --------------------------------------------------------------------------- #
# Anomaly injection
# --------------------------------------------------------------------------- #

# ─────────────────────────────────────────────────────────────────────────────
# Public alias: ANOMALY_TYPES  (set after DEFAULT_ANOMALY_CONFIG is defined)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ANOMALY_CONFIG = {
    "spike": {
        "n_events":     5,
        "duration_min": 1,   # intervals (15-min each)
        "duration_max": 3,
        "multiplier":   4.0,
        "column":       "Global_active_power",
    },
    "sustained": {
        "n_events":     3,
        "duration_min": 2,
        "duration_max": 4,
        "multiplier":   2.5,
        "column":       "Global_active_power",
    },
    "night": {
        "n_events":     4,
        "night_hours":  list(range(0, 5)),   # midnight to 4:59
        "multiplier":   3.0,
        "duration_min": 1,
        "duration_max": 2,
        "column":       "Global_active_power",
    },
    "submetering": {
        "n_events":     4,
        "duration_min": 1,
        "duration_max": 3,
        "multiplier":   5.0,
        "column":       "Sub_metering_1",
    },
}

# Public alias: list of anomaly type names for tests and UI display
ANOMALY_TYPES = list(DEFAULT_ANOMALY_CONFIG.keys())


# Alias: compute_metrics = evaluate_detector (simplified signature for tests)
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Thin wrapper around evaluate_detector for use in tests.
    Accepts raw boolean/int arrays; returns the same metric dict.
    """
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    y_pred_int = np.asarray(y_pred).astype(int)
    y_true_int = np.asarray(y_true).astype(int)
    prec = float(precision_score(y_true_int, y_pred_int, zero_division=0))
    rec  = float(recall_score(y_true_int, y_pred_int, zero_division=0))
    f1   = float(f1_score(y_true_int, y_pred_int, zero_division=0))
    cm   = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr  = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def inject_anomalies(
    test_df: pd.DataFrame,
    config: Optional[dict] = None,
    seed: int = 123,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Copy the test DataFrame and inject known synthetic anomalies.

    Parameters
    ----------
    test_df : pd.DataFrame
        Held-out test period (original scale, 15-min intervals).
        This DataFrame is NOT modified — a copy is returned.
    config  : dict
        Configuration for each anomaly type (see DEFAULT_ANOMALY_CONFIG).
        If None, defaults are used.
    seed    : int
        Random seed for reproducibility.

    Returns
    -------
    df_injected : pd.DataFrame
        Copy of test_df with synthetic anomalies injected.
    true_labels : np.ndarray, shape (n_intervals,)
        1 = anomalous interval (injected), 0 = normal interval.
    """
    if config is None:
        config = DEFAULT_ANOMALY_CONFIG

    df = test_df.copy()
    n  = len(df)
    true_labels = np.zeros(n, dtype=int)
    rng = np.random.default_rng(seed)

    for anomaly_type, cfg in config.items():
        n_events     = cfg.get("n_events", 3)
        dur_min      = cfg.get("duration_min", 1)
        dur_max      = cfg.get("duration_max", 3)
        multiplier   = cfg.get("multiplier", 3.0)
        col          = cfg.get("column", "Global_active_power")
        night_hours  = cfg.get("night_hours", None)

        if col not in df.columns:
            logger.warning("Column '%s' not found; skipping %s anomalies.", col, anomaly_type)
            continue

        for _ in range(n_events):
            duration = rng.integers(dur_min, dur_max + 1)

            if anomaly_type == "night" and night_hours is not None:
                # Find intervals in the night hours
                night_mask = np.isin(df.index.hour, night_hours)
                night_idxs = np.where(night_mask)[0]
                if len(night_idxs) == 0:
                    continue
                start = int(rng.choice(night_idxs))
            else:
                # Random start, leaving a margin at each end
                margin = 10
                if n - margin < margin:
                    start = 0
                else:
                    start = int(rng.integers(margin, n - duration - margin))

            end = min(start + duration, n)

            # Apply multiplier to the target column
            df.iloc[start:end, df.columns.get_loc(col)] *= multiplier

            # Also proportionally bump Global_active_power if col is sub-metering
            if col.startswith("Sub_metering") and "Global_active_power" in df.columns:
                df.iloc[start:end, df.columns.get_loc("Global_active_power")] *= (multiplier * 0.5)

            # Mark as anomalous
            true_labels[start:end] = 1
            logger.debug("Injected %s at rows %d:%d (col=%s)", anomaly_type, start, end, col)

    n_injected = int(true_labels.sum())
    logger.info(
        "Injected %d anomalous intervals across %d total (%.1f%%)",
        n_injected, n, 100 * n_injected / n,
    )
    return df, true_labels


# --------------------------------------------------------------------------- #
# Evaluation metrics
# --------------------------------------------------------------------------- #
def evaluate_detector(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    scores:    np.ndarray,
    threshold: float,
    detector_name: str = "Detector",
) -> dict:
    """
    Compute classification metrics for a binary anomaly detector.

    Parameters
    ----------
    y_true        : np.ndarray of int — ground truth (1=anomaly, 0=normal)
    y_pred        : np.ndarray of bool/int — predicted flags
    scores        : np.ndarray — raw anomaly scores
    threshold     : float — threshold used to produce y_pred
    detector_name : str — label for reporting

    Returns
    -------
    dict with keys:
        name, precision, recall, f1, fpr, n_flagged, n_true_anomalies,
        n_samples, threshold, confusion_matrix (2×2 list)
    """
    y_pred_int = y_pred.astype(int)
    y_true_int = y_true.astype(int)

    # Guard: if only one class in y_true, metrics may be undefined
    if len(np.unique(y_true_int)) < 2:
        logger.warning("y_true has only one class — metrics may be undefined.")

    prec = float(precision_score(y_true_int, y_pred_int, zero_division=0))
    rec  = float(recall_score(y_true_int, y_pred_int, zero_division=0))
    f1   = float(f1_score(y_true_int, y_pred_int, zero_division=0))

    cm = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1])
    # cm = [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "name":              detector_name,
        "precision":         prec,
        "recall":            rec,
        "f1":                f1,
        "fpr":               fpr,
        "n_flagged":         int(y_pred_int.sum()),
        "n_true_anomalies":  int(y_true_int.sum()),
        "n_samples":         len(y_true_int),
        "threshold":         threshold,
        "confusion_matrix":  cm.tolist(),
        "tp":                int(tp),
        "fp":                int(fp),
        "tn":                int(tn),
        "fn":                int(fn),
    }


def compare_detectors(
    rbm_results:      dict,
    baseline_results: dict,
) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame for the two detectors.

    Parameters
    ----------
    rbm_results      : dict returned by evaluate_detector() for the RBM
    baseline_results : dict returned by evaluate_detector() for the baseline

    Returns
    -------
    pd.DataFrame with one row per detector and metric columns.
    """
    metrics = ["precision", "recall", "f1", "fpr", "n_flagged", "threshold"]
    rows = []
    for res in [rbm_results, baseline_results]:
        row = {"Detector": res["name"]}
        for m in metrics:
            row[m] = res.get(m, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Detector")


# --------------------------------------------------------------------------- #
# Runtime measurement wrapper
# --------------------------------------------------------------------------- #
def timed_score(score_fn, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """
    Run score_fn(*args, **kwargs) and return (scores, elapsed_seconds).
    """
    t0     = time.perf_counter()
    scores = score_fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return scores, elapsed


# --------------------------------------------------------------------------- #
# Full evaluation pipeline
# --------------------------------------------------------------------------- #
def run_controlled_evaluation(
    test_df:        pd.DataFrame,
    rbm,
    preprocessor,
    baseline,
    rbm_threshold:      float,
    baseline_threshold: float,
    anomaly_config:     Optional[dict] = None,
    seed:               int = 123,
) -> dict:
    """
    Run the complete controlled synthetic anomaly evaluation.

    Steps
    -----
    1. Inject anomalies into a copy of test_df
    2. Preprocess the injected data
    3. Score with RBM and baseline
    4. Evaluate both detectors
    5. Return metrics + comparison

    Parameters
    ----------
    test_df            : Original (unmodified) test DataFrame
    rbm                : Trained BernoulliRBM
    preprocessor       : Fitted Preprocessor
    baseline           : Fitted StatisticalBaseline
    rbm_threshold      : Anomaly threshold for RBM
    baseline_threshold : Anomaly threshold for baseline
    anomaly_config     : Injection config dict (or None for defaults)
    seed               : Random seed

    Returns
    -------
    dict with keys:
        df_injected, true_labels,
        rbm_scores, rbm_flags, rbm_metrics,
        baseline_scores, baseline_flags, baseline_metrics,
        comparison_df
    """
    from src.anomaly_detection import flag_anomalies, score_samples

    # Step 1: Inject anomalies
    df_injected, true_labels = inject_anomalies(test_df, config=anomaly_config, seed=seed)

    # Step 2: Preprocess injected data
    _, X_injected = preprocessor.transform(df_injected)

    # Step 3: Score with RBM
    rbm_scores, rbm_time = timed_score(score_samples, rbm, X_injected)
    rbm_flags = flag_anomalies(rbm_scores, rbm_threshold)

    # Step 4: Score with baseline
    baseline_scores, bl_time = timed_score(baseline.score, df_injected)
    baseline_flags = baseline_scores > baseline_threshold

    # Step 5: Evaluate
    rbm_metrics = evaluate_detector(
        true_labels, rbm_flags, rbm_scores, rbm_threshold, "RBM (Free Energy)"
    )
    rbm_metrics["runtime_s"] = round(rbm_time, 4)

    baseline_metrics = evaluate_detector(
        true_labels, baseline_flags, baseline_scores, baseline_threshold,
        "Statistical Baseline (MAD)"
    )
    baseline_metrics["runtime_s"] = round(bl_time, 4)

    comparison_df = compare_detectors(rbm_metrics, baseline_metrics)

    logger.info(
        "Evaluation complete — RBM: P=%.2f R=%.2f F1=%.2f | Baseline: P=%.2f R=%.2f F1=%.2f",
        rbm_metrics["precision"], rbm_metrics["recall"], rbm_metrics["f1"],
        baseline_metrics["precision"], baseline_metrics["recall"], baseline_metrics["f1"],
    )

    return {
        "df_injected":       df_injected,
        "true_labels":       true_labels,
        "rbm_scores":        rbm_scores,
        "rbm_flags":         rbm_flags,
        "rbm_metrics":       rbm_metrics,
        "rbm_cm":            np.array(rbm_metrics["confusion_matrix"]),
        "baseline_scores":   baseline_scores,
        "baseline_flags":    baseline_flags,
        "baseline_metrics":  baseline_metrics,
        "baseline_cm":       np.array(baseline_metrics["confusion_matrix"]),
        "comparison_df":     comparison_df,
    }
