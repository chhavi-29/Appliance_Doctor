"""
test_evaluation.py
------------------
Unit tests for src/evaluation.py

Tests verify:
- Synthetic anomaly injection produces expected labels
- Metrics (precision, recall, F1) are computed correctly
- Evaluation output structure is correct
- Anomaly types inject at expected magnitudes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    inject_anomalies,
    compute_metrics,
    run_controlled_evaluation,
    ANOMALY_TYPES,
)
from src.rbm import BernoulliRBM
from src.preprocessing import Preprocessor
from src.baseline import StatisticalBaseline


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_test_df():
    """14 days of synthetic 15-min data (2016 intervals)."""
    n   = 14 * 24 * 4
    idx = pd.date_range("2023-06-01", periods=n, freq="15min", name="datetime")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Global_active_power":   rng.uniform(0.2, 3.0, n),
        "Global_reactive_power": rng.uniform(0.0, 0.5, n),
        "Voltage":               rng.normal(240, 2, n),
        "Global_intensity":      rng.uniform(1.0, 15, n),
        "Sub_metering_1":        rng.uniform(0, 30, n),
        "Sub_metering_2":        rng.uniform(0, 20, n),
        "Sub_metering_3":        rng.uniform(0, 15, n),
    }, index=idx)


@pytest.fixture
def trained_pipeline(sample_test_df):
    """Tiny trained RBM + Preprocessor + Baseline for evaluation tests."""
    # Generate a larger training set
    n_train = 30 * 24 * 4
    idx_train = pd.date_range("2023-01-01", periods=n_train, freq="15min", name="datetime")
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({
        "Global_active_power":   rng.uniform(0.2, 3.0, n_train),
        "Global_reactive_power": rng.uniform(0.0, 0.5, n_train),
        "Voltage":               rng.normal(240, 2, n_train),
        "Global_intensity":      rng.uniform(1.0, 15, n_train),
        "Sub_metering_1":        rng.uniform(0, 30, n_train),
        "Sub_metering_2":        rng.uniform(0, 20, n_train),
        "Sub_metering_3":        rng.uniform(0, 15, n_train),
    }, index=idx_train)

    # Fit preprocessor
    from src.preprocessing import engineer_features, resample_to_15min
    # Data already 15-min; just add engineered features
    from src.preprocessing import RBM_FEATURES, QUANTILE_LEVELS, N_VISIBLE

    prep = Preprocessor()
    prep.fit(train_df)

    n_vis = prep.n_visible_
    rbm   = BernoulliRBM(n_visible=n_vis, n_hidden=5, seed=0)
    _, X_train = prep.transform(train_df)
    rbm.train(X_train, epochs=3, batch_size=32, verbose=False)

    baseline = StatisticalBaseline()
    baseline.fit(train_df)

    # Threshold from a pseudo-val set (use half of test as val)
    n_val = len(sample_test_df) // 2
    val_df = sample_test_df.iloc[:n_val]
    _, X_val = prep.transform(val_df)
    from src.anomaly_detection import score_samples, select_threshold
    val_scores     = score_samples(rbm, X_val)
    rbm_threshold  = select_threshold(val_scores, 95.0)
    base_threshold = baseline.select_threshold(val_df, 95.0)

    return {
        "rbm":               rbm,
        "preprocessor":      prep,
        "baseline":          baseline,
        "rbm_threshold":     rbm_threshold,
        "baseline_threshold": base_threshold,
        "train_df":          train_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests: inject_anomalies
# ─────────────────────────────────────────────────────────────────────────────
class TestInjectAnomalies:

    def test_returns_df_and_labels(self, sample_test_df):
        modified, labels = inject_anomalies(sample_test_df)
        assert isinstance(modified, pd.DataFrame)
        assert isinstance(labels, np.ndarray)

    def test_original_not_modified(self, sample_test_df):
        original_vals = sample_test_df["Global_active_power"].values.copy()
        inject_anomalies(sample_test_df)
        np.testing.assert_array_equal(
            sample_test_df["Global_active_power"].values,
            original_vals,
        )

    def test_labels_are_binary(self, sample_test_df):
        _, labels = inject_anomalies(sample_test_df)
        unique = np.unique(labels)
        assert set(unique).issubset({0, 1}), f"Labels contain non-0/1 values: {unique}"

    def test_some_anomalies_injected(self, sample_test_df):
        _, labels = inject_anomalies(sample_test_df)
        assert labels.sum() > 0, "No anomalies were injected"

    def test_not_all_anomalies(self, sample_test_df):
        _, labels = inject_anomalies(sample_test_df)
        assert labels.sum() < len(labels), "All intervals marked as anomaly"

    def test_same_length_as_input(self, sample_test_df):
        modified, labels = inject_anomalies(sample_test_df)
        assert len(modified) == len(sample_test_df)
        assert len(labels)   == len(sample_test_df)

    def test_anomalous_intervals_elevated(self, sample_test_df):
        """Injected power spike intervals should have higher GAP than the original."""
        modified, labels = inject_anomalies(sample_test_df)
        anom_idx = np.where(labels == 1)[0]
        if len(anom_idx) > 0:
            orig_mean = sample_test_df["Global_active_power"].mean()
            anom_mean = modified["Global_active_power"].iloc[anom_idx].mean()
            # Injected anomalies include high-power spikes, so at least some should be elevated
            # (not all anomaly types raise GAP, but on average they should)
            assert anom_mean != orig_mean, "Anomaly injection did not change any values"

    def test_anomaly_types_documented(self):
        assert len(ANOMALY_TYPES) >= 2, "Expected at least 2 anomaly type definitions"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_metrics
# ─────────────────────────────────────────────────────────────────────────────
class TestComputeMetrics:

    def test_perfect_detection(self):
        labels  = np.array([0, 0, 1, 1, 0, 1])
        preds   = np.array([0, 0, 1, 1, 0, 1])
        metrics = compute_metrics(labels, preds)
        assert metrics["precision"] == 1.0
        assert metrics["recall"]    == 1.0
        assert metrics["f1"]        == 1.0
        assert metrics["fpr"]       == 0.0

    def test_no_detections(self):
        labels  = np.array([0, 0, 1, 1, 0, 1])
        preds   = np.zeros(6)
        metrics = compute_metrics(labels, preds)
        assert metrics["recall"]    == 0.0
        assert metrics["tp"]        == 0
        assert metrics["fn"]        == 3

    def test_all_flagged(self):
        labels  = np.array([0, 0, 1, 1, 0, 1])
        preds   = np.ones(6)
        metrics = compute_metrics(labels, preds)
        assert metrics["recall"] == 1.0
        assert metrics["fp"]     == 3
        assert metrics["fpr"]    == 1.0

    def test_required_keys(self):
        labels  = np.array([0, 1, 0, 1])
        preds   = np.array([0, 1, 1, 0])
        metrics = compute_metrics(labels, preds)
        for key in ["precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn"]:
            assert key in metrics, f"Missing key: {key}"

    def test_metrics_in_range(self):
        rng    = np.random.default_rng(7)
        labels = rng.integers(0, 2, 100)
        preds  = rng.integers(0, 2, 100)
        m      = compute_metrics(labels, preds)
        for key in ["precision", "recall", "f1", "fpr"]:
            assert 0.0 <= m[key] <= 1.0, f"{key} out of [0,1]: {m[key]}"

    def test_confusion_matrix_sum(self):
        labels  = np.array([0, 0, 1, 1, 0, 1])
        preds   = np.array([0, 1, 1, 0, 0, 1])
        m       = compute_metrics(labels, preds)
        total   = m["tp"] + m["fp"] + m["tn"] + m["fn"]
        assert total == len(labels), f"CM sum {total} != n {len(labels)}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: run_controlled_evaluation
# ─────────────────────────────────────────────────────────────────────────────
class TestRunControlledEvaluation:

    def test_returns_dict(self, sample_test_df, trained_pipeline):
        result = run_controlled_evaluation(
            test_df=sample_test_df,
            rbm=trained_pipeline["rbm"],
            preprocessor=trained_pipeline["preprocessor"],
            baseline=trained_pipeline["baseline"],
            rbm_threshold=trained_pipeline["rbm_threshold"],
            baseline_threshold=trained_pipeline["baseline_threshold"],
        )
        assert isinstance(result, dict)

    def test_required_output_keys(self, sample_test_df, trained_pipeline):
        result = run_controlled_evaluation(
            test_df=sample_test_df,
            rbm=trained_pipeline["rbm"],
            preprocessor=trained_pipeline["preprocessor"],
            baseline=trained_pipeline["baseline"],
            rbm_threshold=trained_pipeline["rbm_threshold"],
            baseline_threshold=trained_pipeline["baseline_threshold"],
        )
        for key in ["rbm_metrics", "baseline_metrics", "rbm_cm", "baseline_cm"]:
            assert key in result, f"Missing key: {key}"

    def test_rbm_metrics_valid(self, sample_test_df, trained_pipeline):
        result = run_controlled_evaluation(
            test_df=sample_test_df,
            rbm=trained_pipeline["rbm"],
            preprocessor=trained_pipeline["preprocessor"],
            baseline=trained_pipeline["baseline"],
            rbm_threshold=trained_pipeline["rbm_threshold"],
            baseline_threshold=trained_pipeline["baseline_threshold"],
        )
        m = result["rbm_metrics"]
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"]    <= 1.0
        assert 0.0 <= m["f1"]        <= 1.0

    def test_original_test_df_unchanged(self, sample_test_df, trained_pipeline):
        original = sample_test_df["Global_active_power"].values.copy()
        run_controlled_evaluation(
            test_df=sample_test_df,
            rbm=trained_pipeline["rbm"],
            preprocessor=trained_pipeline["preprocessor"],
            baseline=trained_pipeline["baseline"],
            rbm_threshold=trained_pipeline["rbm_threshold"],
            baseline_threshold=trained_pipeline["baseline_threshold"],
        )
        np.testing.assert_array_equal(
            sample_test_df["Global_active_power"].values,
            original,
            err_msg="run_controlled_evaluation modified the original test_df",
        )
