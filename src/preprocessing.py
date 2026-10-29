"""
preprocessing.py
----------------
All data-preparation steps for the Invisible Appliance Doctor.

Pipeline overview
-----------------
1.  resample_to_15min(df)      → aggregate 1-minute readings into 15-min intervals
2.  engineer_features(df_15)   → add cyclic time encodings + derived power features
3.  chronological_split(df)    → train / val / test splits (60/20/20 by time)
4.  Preprocessor.fit(train)    → learn normalisation stats + binary thresholds
5.  Preprocessor.transform(df) → return (feature_df, binary_array)

Binary Discretisation Strategy
-------------------------------
Each of the 7 numeric features is converted into 3 binary bits using
quantile thresholds computed on training data only:
    bit_0 = 1 if value > Q25(train)   "above lower quartile"
    bit_1 = 1 if value > Q50(train)   "above median"
    bit_2 = 1 if value > Q75(train)   "above upper quartile"

This gives 7 × 3 = 21 visible units for the RBM.
Higher bits set → higher-than-normal consumption at that feature.

Feature meanings
----------------
Global_active_power   [kW]  Total active power consumed
Global_reactive_power [kW]  Reactive power (related to inductance in appliances)
Voltage               [V]   Household voltage (should stay near 240 V)
Global_intensity      [A]   Current intensity
Sub_metering_1        [Wh]  Kitchen area energy
Sub_metering_2        [Wh]  Laundry room energy
Sub_metering_3        [Wh]  Water heater / air-conditioning energy
hour_sin, hour_cos         Cyclic encoding of hour-of-day
dow_sin,  dow_cos          Cyclic encoding of day-of-week
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Features to feed into the RBM (must be numeric and meaningful)
RBM_FEATURES: List[str] = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# Quantile thresholds used for binary discretisation
QUANTILE_LEVELS = [0.25, 0.50, 0.75]   # → 3 bits per feature
N_VISIBLE = len(RBM_FEATURES) * len(QUANTILE_LEVELS)  # = 21

# Chronological split fractions
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
# test uses the remaining 0.20


# --------------------------------------------------------------------------- #
# Step 1: Resample
# --------------------------------------------------------------------------- #
def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute readings to 15-minute intervals.

    Power columns (kW, V, A) are averaged over the interval.
    Sub-metering columns (Wh) are *summed* because they represent
    energy consumed in each minute; summing gives energy per 15 min.

    Missing values: rows with NaN are excluded from the mean/sum
    calculation. An interval is kept even if some minutes are missing,
    unless *all* minutes in the interval are NaN (those intervals are dropped).

    Parameters
    ----------
    df : pd.DataFrame
        Raw 1-minute DataFrame with DatetimeIndex and columns in NUMERIC_COLS.

    Returns
    -------
    pd.DataFrame
        15-minute resampled DataFrame. Index is the start of each interval.
    """
    # min_count=1: a window with no readings gives NaN (not 0), so that
    # windows with no data at all are dropped by dropna(how="all") below.
    def _sum_or_nan(s):
        return s.sum(min_count=1)

    agg_map = {
        "Global_active_power":   "mean",
        "Global_reactive_power": "mean",
        "Voltage":               "mean",
        "Global_intensity":      "mean",
        "Sub_metering_1":        _sum_or_nan,
        "Sub_metering_2":        _sum_or_nan,
        "Sub_metering_3":        _sum_or_nan,
    }
    # Only include columns that exist in the input
    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}

    df_15 = df.resample("15min").agg(agg_map)

    # Drop intervals that are entirely missing
    df_15 = df_15.dropna(how="all")

    logger.info(
        "Resampled to 15-min: %d intervals (%s → %s)",
        len(df_15), df_15.index.min(), df_15.index.max(),
    )
    return df_15


# --------------------------------------------------------------------------- #
# Step 2: Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df_15: pd.DataFrame) -> pd.DataFrame:
    """
    Add interpretable engineered features to the 15-minute DataFrame.

    New columns added
    -----------------
    hour_sin / hour_cos  : cyclic sine/cosine encoding of hour-of-day
                           (avoids discontinuity between 23:00 and 00:00)
    dow_sin / dow_cos    : cyclic encoding of day-of-week (0=Mon … 6=Sun)
    power_ratio          : ratio of sub-metering total to global active power.
                           High ratio → more energy is in sub-metered circuits.
                           Close to 1 → little unmetered usage.
    sub_total            : sum of all three sub-metering channels (Wh per interval)

    Parameters
    ----------
    df_15 : pd.DataFrame
        15-minute resampled DataFrame (output of resample_to_15min).

    Returns
    -------
    pd.DataFrame
        Copy of df_15 with additional engineered columns.
    """
    df = df_15.copy()

    hour = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = df.index.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    sub_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    sub_cols_present = [c for c in sub_cols if c in df.columns]
    df["sub_total"] = df[sub_cols_present].sum(axis=1)

    # Avoid division by zero; clamp denominator
    gap_power_kw = df.get("Global_active_power", pd.Series(1, index=df.index))
    # Convert global active power (kW × 15 min) to Wh for fair ratio
    gap_wh = gap_power_kw * 15 * 60 / 3600 * 1000  # kWh → Wh per 15-min interval
    # gap_wh: kW * (15/60) h * 1000 W/kW = kW * 0.25 * 1000 Wh
    gap_wh_safe = gap_wh.replace(0, np.nan)
    df["power_ratio"] = (df["sub_total"] / gap_wh_safe).clip(0, 1).fillna(0)

    return df


# --------------------------------------------------------------------------- #
# Step 3: Chronological split
# --------------------------------------------------------------------------- #
def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float   = VAL_FRAC,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame into train / validation / test sets
    **without shuffling** (chronological order is preserved).

    WHY no shuffling?
    -----------------
    Shuffling time-series data would allow the model to "see" future data
    during training (data leakage). Chronological splitting simulates the
    real use case: train on historical data, evaluate on unseen future data.

    Parameters
    ----------
    df         : Full 15-minute DataFrame (time-sorted).
    train_frac : Fraction of intervals for training   (default 0.60).
    val_frac   : Fraction for validation              (default 0.20).
                 Remainder goes to test.

    Returns
    -------
    (train_df, val_df, test_df) — each is a non-overlapping slice.
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train: n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]

    logger.info(
        "Split: train=%d, val=%d, test=%d intervals",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# Step 4 & 5: Preprocessor class
# --------------------------------------------------------------------------- #
class Preprocessor:
    """
    Fit-then-transform preprocessor for the RBM pipeline.

    Responsibilities
    ----------------
    1. StandardScaler for continuous features (fitted on training data only)
    2. Quantile-based binary discretisation thresholds (fitted on training data)
    3. transform() returns both a scaled continuous feature DataFrame
       and a binary (0/1) numpy array for the RBM

    Attributes (after fit)
    ----------------------
    scaler_         : sklearn StandardScaler fitted on train RBM_FEATURES
    thresholds_     : dict[feature_name → list of 3 float thresholds]
    feature_names_  : list of RBM feature names used
    n_visible_      : int, number of binary visible units (= len(RBM_FEATURES) × 3)
    bit_names_      : list of bit label strings for interpretability
    """

    def __init__(self, features: Optional[List[str]] = None):
        self.features: List[str] = features or RBM_FEATURES
        self.scaler_: Optional[StandardScaler] = None
        self.thresholds_: Optional[dict] = None
        self.n_visible_: int = len(self.features) * len(QUANTILE_LEVELS)
        self.bit_names_: List[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        """
        Learn normalisation stats and binary thresholds from training data.

        IMPORTANT: Call fit() only on training data. Never on validation/test.
        This prevents data leakage.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training-period 15-minute DataFrame (output of engineer_features).
        """
        available = [f for f in self.features if f in train_df.columns]
        if not available:
            raise ValueError(
                f"None of the requested features {self.features} "
                f"found in DataFrame columns: {list(train_df.columns)}"
            )
        self.features = available
        self.n_visible_ = len(self.features) * len(QUANTILE_LEVELS)

        # Fill NaN with column median (from training data)
        train_clean = train_df[self.features].copy()
        self._train_medians = train_clean.median()
        train_clean = train_clean.fillna(self._train_medians)

        # Fit scaler
        self.scaler_ = StandardScaler()
        self.scaler_.fit(train_clean.values)

        # Compute quantile thresholds for binary discretisation
        self.thresholds_ = {}
        for feat in self.features:
            vals = train_clean[feat].values
            self.thresholds_[feat] = [
                float(np.quantile(vals, q)) for q in QUANTILE_LEVELS
            ]

        # Build interpretable bit names
        q_labels = ["Q25", "Q50", "Q75"]
        self.bit_names_ = [
            f"{feat}>{ql}" for feat in self.features for ql in q_labels
        ]

        self._is_fitted = True
        logger.info(
            "Preprocessor fitted on %d samples → %d visible units",
            len(train_df), self.n_visible_,
        )
        return self

    # ------------------------------------------------------------------ #
    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply learned preprocessing to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            15-minute interval DataFrame (may be train, val, or test).

        Returns
        -------
        feat_df : pd.DataFrame
            Scaled continuous features (n_intervals × n_features).
        binary  : np.ndarray, shape (n_intervals, n_visible)
            Binary (0/1) encoded representation for the RBM.
            Each row: 3 bits per feature (above Q25, Q50, Q75 of training).
        """
        self._check_fitted()

        subset = df[self.features].copy()
        subset = subset.fillna(self._train_medians)

        # Scaled continuous features (for visualisation / explanations)
        scaled = self.scaler_.transform(subset.values)
        feat_df = pd.DataFrame(scaled, index=df.index, columns=self.features)

        # Binary discretisation
        binary = self._discretise(subset.values)

        return feat_df, binary

    # ------------------------------------------------------------------ #
    def transform_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience: return only the binary array."""
        _, binary = self.transform(df)
        return binary

    # ------------------------------------------------------------------ #
    def _discretise(self, X: np.ndarray) -> np.ndarray:
        """
        Convert continuous feature matrix to binary indicators.

        For feature i with thresholds [Q25_i, Q50_i, Q75_i]:
            bit[3i]   = 1 if X[:, i] > Q25_i   (above lower quartile)
            bit[3i+1] = 1 if X[:, i] > Q50_i   (above median)
            bit[3i+2] = 1 if X[:, i] > Q75_i   (above upper quartile)
        """
        n_samples = X.shape[0]
        n_bits = len(self.features) * len(QUANTILE_LEVELS)
        binary = np.zeros((n_samples, n_bits), dtype=np.float32)

        for i, feat in enumerate(self.features):
            thresholds = self.thresholds_[feat]
            for j, thr in enumerate(thresholds):
                col_idx = i * len(QUANTILE_LEVELS) + j
                binary[:, col_idx] = (X[:, i] > thr).astype(np.float32)

        return binary

    # ------------------------------------------------------------------ #
    def get_threshold_description(self) -> pd.DataFrame:
        """Return a DataFrame showing all quantile thresholds for reference."""
        self._check_fitted()
        rows = []
        for feat in self.features:
            for q_level, thr in zip(QUANTILE_LEVELS, self.thresholds_[feat]):
                rows.append({
                    "feature": feat,
                    "quantile": f"Q{int(q_level*100)}",
                    "threshold": thr,
                    "bit_name": f"{feat}>Q{int(q_level*100)}",
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    def save(self, path: str = "models/preprocessor.pkl") -> None:
        """Persist the fitted preprocessor to disk."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str = "models/preprocessor.pkl") -> "Preprocessor":
        """Load a previously saved preprocessor."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Preprocessor loaded from %s", path)
        return obj

    # ------------------------------------------------------------------ #
    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted yet. "
                "Call preprocessor.fit(train_df) first."
            )


# --------------------------------------------------------------------------- #
# Convenience: full pipeline
# --------------------------------------------------------------------------- #
def run_full_pipeline(
    df_raw: pd.DataFrame,
    max_train_days: Optional[int] = None,
    save_preprocessor: bool = True,
    preprocessor_path: str = "models/preprocessor.pkl",
) -> dict:
    """
    Run the full preprocessing pipeline from raw 1-min data to binary arrays.

    Parameters
    ----------
    df_raw         : Raw 1-minute DataFrame from data_loader.
    max_train_days : If set, limits training data to the most recent N days
                     of the training period (useful for laptops).
    save_preprocessor : Whether to save the fitted Preprocessor to disk.
    preprocessor_path : Where to save the preprocessor.

    Returns
    -------
    dict with keys:
        df_15       : Full 15-min resampled DataFrame
        train_df, val_df, test_df : Split 15-min DataFrames
        preprocessor : Fitted Preprocessor object
        X_train, X_val, X_test  : Binary numpy arrays for RBM
        feat_train, feat_val, feat_test : Scaled continuous DataFrames
    """
    df_15 = resample_to_15min(df_raw)
    df_15 = engineer_features(df_15)

    train_df, val_df, test_df = chronological_split(df_15)

    # Optionally limit training data (laptop-friendly mode)
    if max_train_days is not None:
        cutoff = train_df.index[-1] - pd.Timedelta(days=max_train_days)
        train_df = train_df[train_df.index >= cutoff]
        logger.info("Training limited to last %d days: %d intervals", max_train_days, len(train_df))

    prep = Preprocessor()
    prep.fit(train_df)

    feat_train, X_train = prep.transform(train_df)
    feat_val,   X_val   = prep.transform(val_df)
    feat_test,  X_test  = prep.transform(test_df)

    if save_preprocessor:
        prep.save(preprocessor_path)

    return {
        "df_15":       df_15,
        "train_df":    train_df,
        "val_df":      val_df,
        "test_df":     test_df,
        "preprocessor": prep,
        "X_train":     X_train,
        "X_val":       X_val,
        "X_test":      X_test,
        "feat_train":  feat_train,
        "feat_val":    feat_val,
        "feat_test":   feat_test,
    }
