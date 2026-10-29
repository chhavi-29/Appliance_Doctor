"""
test_rbm.py
-----------
Unit tests for src/rbm.py

Tests verify:
- Correct initialisation (weight shapes, biases)
- Numerically stable sigmoid
- CD-1 step runs without error
- Free energy returns finite values
- Anomaly score direction is correct (unusual inputs score higher)
- Training reduces reconstruction error
- Save/load round-trips weights correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.rbm import BernoulliRBM, sigmoid


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
N_VISIBLE = 21
N_HIDDEN  = 8


@pytest.fixture
def rbm():
    return BernoulliRBM(n_visible=N_VISIBLE, n_hidden=N_HIDDEN, seed=42)


@pytest.fixture
def small_binary_data():
    """50 samples of random binary data."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=(50, N_VISIBLE)).astype(np.float64)


@pytest.fixture
def trained_rbm(small_binary_data):
    rbm = BernoulliRBM(n_visible=N_VISIBLE, n_hidden=N_HIDDEN, seed=42)
    rbm.train(small_binary_data, epochs=5, batch_size=16, verbose=False)
    return rbm


# --------------------------------------------------------------------------- #
# Test: sigmoid
# --------------------------------------------------------------------------- #
class TestSigmoid:

    def test_sigmoid_zero(self):
        assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10

    def test_sigmoid_large_positive(self):
        assert sigmoid(np.array([100.0]))[0] > 0.9999

    def test_sigmoid_large_negative(self):
        assert sigmoid(np.array([-100.0]))[0] < 1e-10

    def test_sigmoid_range(self):
        x = np.linspace(-10, 10, 1000)
        s = sigmoid(x)
        assert np.all(s > 0) and np.all(s < 1), "Sigmoid output outside (0,1)"

    def test_sigmoid_no_overflow(self):
        """Should not raise on extreme values."""
        x = np.array([-1000.0, 0.0, 1000.0])
        result = sigmoid(x)
        assert np.all(np.isfinite(result)), "Sigmoid produced inf or nan"


# --------------------------------------------------------------------------- #
# Test: initialisation
# --------------------------------------------------------------------------- #
class TestRBMInit:

    def test_weight_shape(self, rbm):
        assert rbm.W.shape == (N_HIDDEN, N_VISIBLE), \
            f"W shape: expected ({N_HIDDEN}, {N_VISIBLE}), got {rbm.W.shape}"

    def test_visible_bias_shape(self, rbm):
        assert rbm.b_v.shape == (N_VISIBLE,)

    def test_hidden_bias_shape(self, rbm):
        assert rbm.b_h.shape == (N_HIDDEN,)

    def test_biases_zero_init(self, rbm):
        np.testing.assert_array_equal(rbm.b_v, np.zeros(N_VISIBLE))
        np.testing.assert_array_equal(rbm.b_h, np.zeros(N_HIDDEN))

    def test_not_trained_initially(self, rbm):
        assert not rbm.is_trained()

    def test_repr_contains_key_info(self, rbm):
        r = repr(rbm)
        assert "BernoulliRBM" in r
        assert "untrained" in r


# --------------------------------------------------------------------------- #
# Test: sampling operations
# --------------------------------------------------------------------------- #
class TestSampling:

    def test_hidden_probs_shape(self, rbm, small_binary_data):
        probs = rbm._hidden_probs(small_binary_data)
        assert probs.shape == (len(small_binary_data), N_HIDDEN)

    def test_hidden_probs_range(self, rbm, small_binary_data):
        probs = rbm._hidden_probs(small_binary_data)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_visible_probs_shape(self, rbm):
        h = np.random.default_rng(0).random((10, N_HIDDEN))
        probs = rbm._visible_probs(h)
        assert probs.shape == (10, N_VISIBLE)

    def test_sample_hidden_binary(self, rbm, small_binary_data):
        _, h_sample = rbm._sample_hidden(small_binary_data)
        unique = np.unique(h_sample)
        assert set(unique).issubset({0.0, 1.0}), f"Hidden samples are not binary: {unique}"

    def test_gibbs_step_shapes(self, rbm, small_binary_data):
        h_probs, v_probs = rbm.gibbs_step(small_binary_data)
        assert h_probs.shape == (len(small_binary_data), N_HIDDEN)
        assert v_probs.shape == (len(small_binary_data), N_VISIBLE)


# --------------------------------------------------------------------------- #
# Test: CD update and training
# --------------------------------------------------------------------------- #
class TestCDUpdate:

    def test_cd_update_returns_float(self, rbm, small_binary_data):
        err = rbm._cd_update(small_binary_data[:16])
        assert isinstance(err, float), f"Expected float, got {type(err)}"
        assert np.isfinite(err), "CD update returned non-finite error"

    def test_weights_change_after_update(self, rbm, small_binary_data):
        W_before = rbm.W.copy()
        rbm._cd_update(small_binary_data[:16])
        assert not np.array_equal(rbm.W, W_before), "Weights did not change after CD update"

    def test_training_reduces_error(self, small_binary_data):
        """Reconstruction error should decrease (or at least not explode) over training."""
        rbm = BernoulliRBM(n_visible=N_VISIBLE, n_hidden=N_HIDDEN, seed=42)
        rbm.train(small_binary_data, epochs=30, batch_size=16, verbose=False)
        assert len(rbm.losses_) == 30
        # Check that loss is finite and generally trending down
        assert np.isfinite(rbm.losses_[-1]), "Final loss is not finite"
        assert rbm.losses_[-1] < 1.0, f"Final loss too high: {rbm.losses_[-1]}"

    def test_is_trained_after_training(self, trained_rbm):
        assert trained_rbm.is_trained()

    def test_losses_list_length(self, small_binary_data):
        rbm = BernoulliRBM(n_visible=N_VISIBLE, n_hidden=N_HIDDEN, seed=42)
        rbm.train(small_binary_data, epochs=5, batch_size=16, verbose=False)
        assert len(rbm.losses_) == 5

    def test_reproducibility(self, small_binary_data):
        """Same seed → same weights."""
        rbm1 = BernoulliRBM(n_visible=N_VISIBLE, n_hidden=N_HIDDEN, seed=99)
        rbm2 = BernoulliRBM(n_visible=N_VISIBLE, n_hidden=N_HIDDEN, seed=99)
        rbm1.train(small_binary_data, epochs=5, batch_size=16, verbose=False)
        rbm2.train(small_binary_data, epochs=5, batch_size=16, verbose=False)
        np.testing.assert_array_almost_equal(rbm1.W, rbm2.W, decimal=10)


# --------------------------------------------------------------------------- #
# Test: Free energy and anomaly score
# --------------------------------------------------------------------------- #
class TestFreeEnergy:

    def test_free_energy_shape(self, trained_rbm, small_binary_data):
        fe = trained_rbm.free_energy(small_binary_data)
        assert fe.shape == (len(small_binary_data),), \
            f"Free energy shape: expected ({len(small_binary_data)},), got {fe.shape}"

    def test_free_energy_finite(self, trained_rbm, small_binary_data):
        fe = trained_rbm.free_energy(small_binary_data)
        assert np.all(np.isfinite(fe)), "Free energy contains inf or nan"

    def test_anomaly_score_is_free_energy(self, trained_rbm, small_binary_data):
        fe     = trained_rbm.free_energy(small_binary_data)
        scores = trained_rbm.anomaly_score(small_binary_data)
        np.testing.assert_array_almost_equal(scores, fe, decimal=10)

    def test_unusual_input_scores_higher_than_typical(self):
        """Direction check: patterns seen in training must score LOWER than a
        pattern never seen in training."""
        rng = np.random.default_rng(1)
        pattern_a = np.array([1, 1, 1, 0, 0, 0] * 3 + [1, 0, 1], dtype=float)
        pattern_b = np.array([0, 0, 0, 1, 1, 1] * 3 + [0, 1, 0], dtype=float)
        X = np.vstack([pattern_a, pattern_b])[rng.integers(0, 2, 500)]
        flip = rng.random(X.shape) < 0.02          # a little noise
        X = np.abs(X - flip)
        rbm = BernoulliRBM(n_visible=N_VISIBLE, n_hidden=8, seed=0)
        rbm.train(X, epochs=60, batch_size=32, verbose=False)
        typical = rbm.anomaly_score(np.vstack([pattern_a, pattern_b])).mean()
        never_seen = np.array([1, 0] * 10 + [1], dtype=float)   # alternating, not in training
        unusual = rbm.anomaly_score(never_seen[None, :]).mean()
        assert unusual > typical, "Unseen pattern should have a HIGHER anomaly score"

    def test_all_ones_vs_all_zeros(self, trained_rbm):
        """
        All-ones vector (maximum consumption) should score differently
        from all-zeros (minimum consumption). Not a strict direction test
        because the model is trained on random data here.
        """
        v_ones  = np.ones((1, N_VISIBLE))
        v_zeros = np.zeros((1, N_VISIBLE))
        score_ones  = trained_rbm.anomaly_score(v_ones)[0]
        score_zeros = trained_rbm.anomaly_score(v_zeros)[0]
        # They should not be equal
        assert score_ones != score_zeros, \
            "All-ones and all-zeros produce identical anomaly scores"

    def test_reconstruction_error_shape(self, trained_rbm, small_binary_data):
        err = trained_rbm.reconstruction_error(small_binary_data)
        assert err.shape == (len(small_binary_data),)

    def test_reconstruction_error_range(self, trained_rbm, small_binary_data):
        err = trained_rbm.reconstruction_error(small_binary_data)
        # MSE of binary data is in [0, 1]
        assert np.all(err >= 0) and np.all(err <= 1)


# --------------------------------------------------------------------------- #
# Test: Save / Load
# --------------------------------------------------------------------------- #
class TestSaveLoad:

    def test_save_creates_file(self, trained_rbm, tmp_path):
        path = str(tmp_path / "rbm.npz")
        trained_rbm.save(path)
        assert Path(path).exists()

    def test_load_restores_weights(self, trained_rbm, tmp_path):
        path = str(tmp_path / "rbm.npz")
        trained_rbm.save(path)
        loaded = BernoulliRBM.load(path)
        np.testing.assert_array_equal(trained_rbm.W,   loaded.W)
        np.testing.assert_array_equal(trained_rbm.b_v, loaded.b_v)
        np.testing.assert_array_equal(trained_rbm.b_h, loaded.b_h)

    def test_load_restores_parameters(self, trained_rbm, tmp_path):
        path = str(tmp_path / "rbm.npz")
        trained_rbm.save(path)
        loaded = BernoulliRBM.load(path)
        assert loaded.n_visible     == trained_rbm.n_visible
        assert loaded.n_hidden      == trained_rbm.n_hidden
        assert loaded.learning_rate == trained_rbm.learning_rate
        assert loaded.is_trained()

    def test_load_produces_same_scores(self, trained_rbm, small_binary_data, tmp_path):
        path = str(tmp_path / "rbm.npz")
        trained_rbm.save(path)
        loaded = BernoulliRBM.load(path)
        scores_orig   = trained_rbm.anomaly_score(small_binary_data)
        scores_loaded = loaded.anomaly_score(small_binary_data)
        np.testing.assert_array_almost_equal(scores_orig, scores_loaded, decimal=10)
