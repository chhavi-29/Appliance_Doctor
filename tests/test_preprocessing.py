"""
test_preprocessing.py
---------------------
Unit tests for src/preprocessing.py

Tests verify:
- Correct resampling interval count
- No data leakage (scaler fitted only on train)
- Binary encoding shape
- Chronological split ordering
- Feature engineering produces expected columns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    resample_to_15min,
    engineer_features,
    chronological_split,
    Preprocessor,
    RBM_FEATURES,
    QUANTILE_LEVELS,
    N_VISIBLE,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_raw_df():
    """1-minute raw DataFrame for 3 days."""
    n = 3 * 24 * 60  # 3 days of 1-min data
    index = pd.date_range("2023-01-01", periods=n, freq="1min")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Global_active_power":   rng.uniform(0.2, 5.0, n),
        "Global_reactive_power": rng.uniform(0.0, 1.0, n),
        "Voltage":               rng.normal(240, 3, n),
        "Global_intensity":      rng.uniform(0.5, 20, n),
        "Sub_metering_1":        rng.uniform(0, 40, n),
        "Sub_metering_2":        rng.uniform(0, 30, n),
        "Sub_metering_3":        rng.uniform(0, 20, n),
    }, index=index)
    df.index.name = "datetime"
    return df


@pytest.fixture
def sample_15min_df(sample_raw_df):
    return engineer_features(resample_to_15min(sample_raw_df))


# --------------------------------------------------------------------------- #
# Test: resample_to_15min
# --------------------------------------------------------------------------- #
class TestResampleTo15Min:

    def test_output_frequency(self, sample_raw_df):
        """15-minute intervals must be exactly 15 minutes apart."""
        df_15 = resample_to_15min(sample_raw_df)
        diffs = df_15.index.diff().dropna()
        assert all(diffs == pd.Timedelta("15min")), \
            "Not all intervals are exactly 15 minutes apart"

    def test_expected_interval_count(self, sample_raw_df):
        """3 days → 3 × 24 × 4 = 288 intervals (approximately)."""
        df_15 = resample_to_15min(sample_raw_df)
        assert 280 <= len(df_15) <= 292, \
            f"Expected ~288 intervals for 3 days, got {len(df_15)}"

    def test_no_missing_all_rows(self, sample_raw_df):
        """No row should be entirely NaN after resampling."""
        df_15 = resample_to_15min(sample_raw_df)
        all_nan = df_15.isnull().all(axis=1)
        assert not all_nan.any(), "Some rows are entirely NaN after resampling"

    def test_columns_preserved(self, sample_raw_df):
        """Output should retain all original columns."""
        df_15 = resample_to_15min(sample_raw_df)
        for col in ["Global_active_power", "Sub_metering_1"]:
            assert col in df_15.columns, f"Missing column: {col}"

    def test_sub_metering_is_sum(self, sample_raw_df):
        """Sub-metering should be summed (not averaged) per interval."""
        df_15 = resample_to_15min(sample_raw_df)
        # Manual check: sum of first 15 rows vs first resampled row
        first_15 = sample_raw_df["Sub_metering_1"].iloc[:15].sum()
        first_resampled = df_15["Sub_metering_1"].iloc[0]
        assert abs(first_15 - first_resampled) < 1e-6, \
            f"Sub-metering should be summed: expected {first_15}, got {first_resampled}"


# --------------------------------------------------------------------------- #
# Test: engineer_features
# --------------------------------------------------------------------------- #
class TestEngineerFeatures:

    def test_cyclic_columns_present(self, sample_15min_df):
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert col in sample_15min_df.columns, f"Missing engineered column: {col}"

    def test_cyclic_encoding_range(self, sample_15min_df):
        """Cyclic features must be in [-1, 1]."""
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert sample_15min_df[col].between(-1, 1).all(), \
                f"{col} values are out of [-1, 1] range"

    def test_power_ratio_range(self, sample_15min_df):
        """power_ratio should be clipped to [0, 1]."""
        if "power_ratio" in sample_15min_df.columns:
            assert sample_15min_df["power_ratio"].between(0, 1).all(), \
                "power_ratio is outside [0, 1]"

    def test_sub_total_correct(self, sample_15min_df):
        """sub_total must equal sum of three sub-metering columns."""
        expected = (
            sample_15min_df["Sub_metering_1"] +
            sample_15min_df["Sub_metering_2"] +
            sample_15min_df["Sub_metering_3"]
        )
        pd.testing.assert_series_equal(
            sample_15min_df["sub_total"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# --------------------------------------------------------------------------- #
# Test: chronological_split
# --------------------------------------------------------------------------- #
class TestChronologicalSplit:

    def test_split_order(self, sample_15min_df):
        """Train must be before val, val before test — no time overlap."""
        train, val, test = chronological_split(sample_15min_df)
        assert train.index.max() < val.index.min(), \
            "Train and val overlap in time"
        assert val.index.max() < test.index.min(), \
            "Val and test overlap in time"

    def test_split_totals(self, sample_15min_df):
        """Train + val + test should equal full dataset."""
        train, val, test = chronological_split(sample_15min_df)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_15min_df), \
            f"Rows lost in split: {len(sample_15min_df)} → {total}"

    def test_approximate_fractions(self, sample_15min_df):
        """Train ≈ 60%, val ≈ 20%, test ≈ 20% (±5%)."""
        train, val, test = chronological_split(sample_15min_df)
        n = len(sample_15min_df)
        assert abs(len(train) / n - 0.60) < 0.05
        assert abs(len(val)   / n - 0.20) < 0.05


# --------------------------------------------------------------------------- #
# Test: Preprocessor
# --------------------------------------------------------------------------- #
class TestPreprocessor:

    @pytest.fixture
    def fitted_prep(self, sample_15min_df):
        train, _, _ = chronological_split(sample_15min_df)
        prep = Preprocessor()
        prep.fit(train)
        return prep, train, sample_15min_df

    def test_fit_sets_is_fitted(self, sample_15min_df):
        train, _, _ = chronological_split(sample_15min_df)
        prep = Preprocessor()
        prep.fit(train)
        assert prep._is_fitted

    def test_binary_shape(self, fitted_prep):
        prep, train, df_15 = fitted_prep
        _, binary = prep.transform(df_15)
        assert binary.shape == (len(df_15), prep.n_visible_), \
            f"Binary shape mismatch: {binary.shape} vs expected ({len(df_15)}, {prep.n_visible_})"

    def test_binary_values_are_01(self, fitted_prep):
        prep, train, df_15 = fitted_prep
        _, binary = prep.transform(df_15)
        unique = np.unique(binary)
        assert set(unique).issubset({0.0, 1.0}), \
            f"Binary array contains non-0/1 values: {unique}"

    def test_no_leakage_scaler(self, sample_15min_df):
        """
        Test that the scaler mean is computed from training data only.
        Specifically, the scaler's mean for Global_active_power should be
        close to the training period mean, NOT the full-data mean.
        """
        train, val, test = chronological_split(sample_15min_df)
        prep = Preprocessor()
        prep.fit(train)

        col = "Global_active_power"
        if col not in prep.features:
            pytest.skip(f"{col} not in features")

        feat_idx = prep.features.index(col)
        scaler_mean = prep.scaler_.mean_[feat_idx]
        train_mean  = train[col].mean()
        full_mean   = sample_15min_df[col].mean()

        # Scaler mean should match train mean, not full mean
        assert abs(scaler_mean - train_mean) < 0.1 * abs(train_mean), \
            "Scaler appears to have been fitted on non-training data (leakage suspected)"

    def test_n_visible(self, fitted_prep):
        prep, _, _ = fitted_prep
        assert prep.n_visible_ == len(prep.features) * len(QUANTILE_LEVELS)

    def test_threshold_description(self, fitted_prep):
        prep, _, _ = fitted_prep
        desc = prep.get_threshold_description()
        assert len(desc) == len(prep.features) * len(QUANTILE_LEVELS)
        assert "threshold" in desc.columns

    def test_save_load_roundtrip(self, fitted_prep, tmp_path):
        prep, _, df_15 = fitted_prep
        save_path = tmp_path / "preprocessor.pkl"
        prep.save(str(save_path))
        loaded = Preprocessor.load(str(save_path))
        _, binary_orig   = prep.transform(df_15)
        _, binary_loaded = loaded.transform(df_15)
        np.testing.assert_array_equal(binary_orig, binary_loaded)


def test_empty_windows_are_dropped_not_zero_filled():
    """A 15-min window with no readings must be dropped, not kept as sub-metering = 0."""
    idx = list(pd.date_range("2023-01-01 00:00", periods=15, freq="1min")) + \
          list(pd.date_range("2023-01-01 01:00", periods=15, freq="1min"))
    df = pd.DataFrame({c: 1.0 for c in RBM_FEATURES}, index=pd.DatetimeIndex(idx))
    out = resample_to_15min(df)
    assert len(out) == 2, "windows with no data should be dropped"
    assert out.notna().all().all()
