"""
baseline.py
-----------
Simple statistical baseline for anomaly detection.

Purpose
-------
The RBM is compared against this baseline to demonstrate:
1. What a naive approach achieves
2. Whether the RBM provides any extra value
3. How to explain anomaly detection without any neural network

Method: Per-hour Median Absolute Deviation (MAD) Z-score
----------------------------------------------------------
For each hour-of-day h ∈ {0, 1, …, 23}:
    median_h = median(GAP during hour h in training data)
    mad_h    = median(|GAP - median_h| during hour h in training data)

Modified Z-score for a new observation x at hour h:
    z(x, h) = 0.6745 × (x - median_h) / (mad_h + ε)

The constant 0.6745 = Φ^{-1}(0.75) normalises MAD to be consistent
with the standard deviation of a Gaussian distribution.

WHY use median and MAD instead of mean and std?
----------------------------------------------
• Electricity data is right-skewed and has outliers.
• The mean is strongly pulled by extreme values.
• The median is robust: 50% of data can be outliers before it is affected.
• MAD is likewise resistant to outliers.
• This makes the baseline more reliable for real-world power data.

Threshold: flag intervals with |z| > threshold (default 3.5).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StatisticalBaseline:
    """
    Per-hour Median + MAD anomaly detector for Global_active_power.

    Attributes (after fit)
    ----------------------
    hourly_stats_ : pd.DataFrame with columns: hour, median, mad
    target_col    : str — the column being modelled (default Global_active_power)
    """

    # 0.6745 = scipy.stats.norm.ppf(0.75)
    _MAD_SCALE = 0.6745

    def __init__(self, target_col: str = "Global_active_power"):
        self.target_col   = target_col
        self.hourly_stats_: Optional[pd.DataFrame] = None
        self._is_fitted    = False

    # ------------------------------------------------------------------ #
    def fit(self, train_df: pd.DataFrame) -> "StatisticalBaseline":
        """
        Compute per-hour median and MAD from training data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training-period 15-minute DataFrame with DatetimeIndex.
        """
        if self.target_col not in train_df.columns:
            raise ValueError(
                f"Column '{self.target_col}' not found in DataFrame. "
                f"Available: {list(train_df.columns)}"
            )

        col = train_df[self.target_col].dropna()
        hours = col.index.hour

        stats = []
        for h in range(24):
            vals = col[hours == h].values
            if len(vals) == 0:
                med = np.nan
                mad = np.nan
            else:
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med)))
            stats.append({"hour": h, "median": med, "mad": mad})

        self.hourly_stats_ = pd.DataFrame(stats).set_index("hour")
        self._is_fitted = True

        logger.info(
            "StatisticalBaseline fitted on %d samples (%s column).",
            len(col), self.target_col,
        )
        return self

    # ------------------------------------------------------------------ #
    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute modified Z-score for each interval.

        Parameters
        ----------
        df : pd.DataFrame with DatetimeIndex and target column.

        Returns
        -------
        np.ndarray, shape (n_samples,) — absolute modified Z-score.
        """
        self._check_fitted()

        col   = df[self.target_col].fillna(df[self.target_col].median()).values
        hours = df.index.hour.values

        scores = np.zeros(len(df), dtype=np.float64)
        for i, (val, h) in enumerate(zip(col, hours)):
            med = self.hourly_stats_.loc[h, "median"]
            mad = self.hourly_stats_.loc[h, "mad"]
            if np.isnan(med) or np.isnan(mad):
                scores[i] = 0.0
            else:
                # Add small epsilon to avoid division by zero
                scores[i] = self._MAD_SCALE * abs(val - med) / (mad + 1e-8)

        return scores

    # ------------------------------------------------------------------ #
    def predict(self, df: pd.DataFrame, threshold: float = 3.5) -> np.ndarray:
        """
        Return boolean anomaly flags.

        Parameters
        ----------
        df        : DataFrame with DatetimeIndex and target column.
        threshold : Flag if |z| > threshold (default 3.5, Iglewicz & Hoaglin).

        Returns
        -------
        np.ndarray of bool — True = anomalous.
        """
        return self.score(df) > threshold

    # ------------------------------------------------------------------ #
    def select_threshold(
        self,
        val_df: pd.DataFrame,
        percentile: float = 95.0,
    ) -> float:
        """
        Pick threshold using the validation set (same approach as RBM).

        Parameters
        ----------
        val_df     : Validation-period DataFrame.
        percentile : Desired percentile of validation scores.

        Returns
        -------
        float : threshold.
        """
        val_scores = self.score(val_df)
        thr = float(np.percentile(val_scores, percentile))
        logger.info(
            "Baseline threshold: %.4f (%.1f-th percentile of %d val scores)",
            thr, percentile, len(val_scores),
        )
        return thr

    # ------------------------------------------------------------------ #
    def get_stats_table(self) -> pd.DataFrame:
        """Return the per-hour statistics as a display-ready DataFrame."""
        self._check_fitted()
        return self.hourly_stats_.reset_index().rename(
            columns={"hour": "Hour of Day", "median": "Median (kW)", "mad": "MAD (kW)"}
        )

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "StatisticalBaseline must be fitted first. Call .fit(train_df)."
            )
