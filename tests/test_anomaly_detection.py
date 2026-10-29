"""
test_anomaly_detection.py
-------------------------
Unit tests for src/anomaly_detection.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    score_samples,
    select_threshold,
    flag_anomalies,
    get_anomaly_summary,
    get_flagged_intervals_df,
    explain_interval,
    score_stats,
)
from src.rbm import BernoulliRBM


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
N_VIS = 21
N_HID = 6


@pytest.fixture
def trained_rbm():
    rng = np.random.default_rng(0)
    X   = rng.integers(0, 2, size=(100, N_VIS)).astype(np.float64)
    rbm = BernoulliRBM(n_visible=N_VIS, n_hidden=N_HID, seed=0)
    rbm.train(X, epochs=5, batch_size=16, verbose=False)
    return rbm, X


@pytest.fixture
def sample_scores():
    rng = np.random.default_rng(1)
    return rng.normal(5.0, 1.5, 200)


@pytest.fixture
def sample_df_15():
    """Small 15-min DataFrame for explanation tests."""
    n   = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", name="datetime")
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "Global_active_power":   rng.uniform(0.2, 5.0, n),
        "Global_reactive_power": rng.uniform(0.0, 1.0, n),
        "Voltage":               rng.normal(240, 3, n),
        "Global_intensity":      rng.uniform(0.5, 20, n),
        "Sub_metering_1":        rng.uniform(0, 40, n),
        "Sub_metering_2":        rng.uniform(0, 30, n),
        "Sub_metering_3":        rng.uniform(0, 20, n),
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: score_samples
# ─────────────────────────────────────────────────────────────────────────────
class TestScoreSamples:

    def test_output_shape(self, trained_rbm):
        rbm, X = trained_rbm
        scores = score_samples(rbm, X)
        assert scores.shape == (len(X),)

    def test_output_finite(self, trained_rbm):
        rbm, X = trained_rbm
        scores = score_samples(rbm, X)
        assert np.all(np.isfinite(scores))

    def test_output_dtype(self, trained_rbm):
        rbm, X = trained_rbm
        scores = score_samples(rbm, X)
        assert scores.dtype == np.float64

    def test_untrained_raises(self):
        rbm = BernoulliRBM(n_visible=N_VIS, n_hidden=N_HID, seed=0)
        X   = np.zeros((5, N_VIS))
        with pytest.raises(RuntimeError, match="trained"):
            score_samples(rbm, X)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: select_threshold
# ─────────────────────────────────────────────────────────────────────────────
class TestSelectThreshold:

    def test_returns_float(self, sample_scores):
        t = select_threshold(sample_scores, 95.0)
        assert isinstance(t, float)

    def test_correct_percentile(self, sample_scores):
        t = select_threshold(sample_scores, 95.0)
        assert abs(t - np.percentile(sample_scores, 95.0)) < 1e-10

    def test_50th_percentile_is_median(self, sample_scores):
        t = select_threshold(sample_scores, 50.0)
        assert abs(t - np.median(sample_scores)) < 1e-10

    def test_invalid_percentile_raises(self, sample_scores):
        with pytest.raises(ValueError):
            select_threshold(sample_scores, 0.0)
        with pytest.raises(ValueError):
            select_threshold(sample_scores, 100.0)
        with pytest.raises(ValueError):
            select_threshold(sample_scores, -5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: flag_anomalies
# ─────────────────────────────────────────────────────────────────────────────
class TestFlagAnomalies:

    def test_boolean_output(self, sample_scores):
        t    = select_threshold(sample_scores, 95.0)
        mask = flag_anomalies(sample_scores, t)
        assert mask.dtype == bool

    def test_expected_flag_count(self, sample_scores):
        t    = select_threshold(sample_scores, 95.0)
        mask = flag_anomalies(sample_scores, t)
        # Approximately 5% of 200 samples → ~10 flags
        assert 5 <= mask.sum() <= 25

    def test_nothing_flagged_above_max(self, sample_scores):
        t    = sample_scores.max() + 1.0
        mask = flag_anomalies(sample_scores, t)
        assert mask.sum() == 0

    def test_all_flagged_below_min(self, sample_scores):
        t    = sample_scores.min() - 1.0
        mask = flag_anomalies(sample_scores, t)
        assert mask.sum() == len(sample_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: get_flagged_intervals_df
# ─────────────────────────────────────────────────────────────────────────────
class TestGetFlaggedIntervalsDf:

    def test_returns_dataframe(self, sample_df_15, sample_scores):
        t  = select_threshold(sample_scores, 95.0)
        df = get_flagged_intervals_df(sample_df_15, sample_scores, t)
        assert isinstance(df, pd.DataFrame)

    def test_contains_anomaly_score_col(self, sample_df_15, sample_scores):
        t  = select_threshold(sample_scores, 95.0)
        df = get_flagged_intervals_df(sample_df_15, sample_scores, t)
        assert "anomaly_score" in df.columns

    def test_all_rows_are_flagged(self, sample_df_15, sample_scores):
        t  = select_threshold(sample_scores, 95.0)
        df = get_flagged_intervals_df(sample_df_15, sample_scores, t)
        assert (df["anomaly_score"] > t).all()

    def test_sorted_by_score_descending(self, sample_df_15, sample_scores):
        t  = select_threshold(sample_scores, 90.0)
        df = get_flagged_intervals_df(sample_df_15, sample_scores, t)
        if len(df) > 1:
            assert df["anomaly_score"].is_monotonic_decreasing


# ─────────────────────────────────────────────────────────────────────────────
# Tests: explain_interval
# ─────────────────────────────────────────────────────────────────────────────
class TestExplainInterval:

    def test_returns_dict(self, sample_df_15, sample_scores):
        t   = select_threshold(sample_scores, 95.0)
        exp = explain_interval(0, sample_df_15, sample_scores, t, sample_df_15)
        assert isinstance(exp, dict)

    def test_required_keys(self, sample_df_15, sample_scores):
        t    = select_threshold(sample_scores, 95.0)
        exp  = explain_interval(0, sample_df_15, sample_scores, t, sample_df_15)
        for key in ["timestamp", "score", "is_anomaly", "readings", "typical",
                    "deviations", "summary", "disclaimer"]:
            assert key in exp, f"Missing key: {key}"

    def test_disclaimer_present(self, sample_df_15, sample_scores):
        t    = select_threshold(sample_scores, 95.0)
        exp  = explain_interval(0, sample_df_15, sample_scores, t, sample_df_15)
        assert len(exp["disclaimer"]) > 50

    def test_is_anomaly_correct(self, sample_df_15, sample_scores):
        t   = select_threshold(sample_scores, 95.0)
        exp = explain_interval(0, sample_df_15, sample_scores, t, sample_df_15)
        assert exp["is_anomaly"] == (sample_scores[0] > t)

    def test_out_of_range_raises(self, sample_df_15, sample_scores):
        t = select_threshold(sample_scores, 95.0)
        with pytest.raises(IndexError):
            explain_interval(10000, sample_df_15, sample_scores, t, sample_df_15)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: score_stats
# ─────────────────────────────────────────────────────────────────────────────
class TestScoreStats:

    def test_returns_dict(self, sample_scores):
        stats = score_stats(sample_scores)
        assert isinstance(stats, dict)

    def test_required_keys(self, sample_scores):
        stats = score_stats(sample_scores)
        for key in ["min", "q25", "median", "q75", "q95", "max", "mean", "std"]:
            assert key in stats

    def test_min_le_max(self, sample_scores):
        stats = score_stats(sample_scores)
        assert stats["min"] <= stats["max"]

    def test_median_between_q25_q75(self, sample_scores):
        stats = score_stats(sample_scores)
        assert stats["q25"] <= stats["median"] <= stats["q75"]
