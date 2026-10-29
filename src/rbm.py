"""
rbm.py
------
Bernoulli Restricted Boltzmann Machine (RBM) — pure NumPy implementation.

Theory overview
---------------
An RBM is a two-layer undirected probabilistic graphical model with:
  • visible units v  ∈ {0,1}^n_visible  — represent observed (binary) data
  • hidden units  h  ∈ {0,1}^n_hidden   — learn latent features

Parameters
  W  : weight matrix, shape (n_hidden × n_visible)
  b_v: visible bias,  shape (n_visible,)
  b_h: hidden bias,   shape (n_hidden,)

Conditional distributions (Bernoulli)
  P(h_j = 1 | v) = σ(b_h[j] + Σ_i W[j,i] · v[i])   — visible → hidden
  P(v_i = 1 | h) = σ(b_v[i] + Σ_j W[j,i] · h[j])   — hidden → visible
where σ(x) = 1 / (1 + exp(-x)) is the logistic sigmoid.

Training: Contrastive Divergence (CD-k)
  CD-k approximates the gradient of the log-likelihood without computing
  the partition function Z (which is intractable for large models).

  Algorithm for one mini-batch:
  1. POSITIVE PHASE: observe real data v^(0)
     a. Compute P(h=1 | v^(0))  →  h^(0)_probs
     b. Sample  h^(0) ~ Bernoulli(h^(0)_probs)    ← Gibbs step
  2. RUN k GIBBS STEPS starting from h^(0):
     a. Sample  v^(k) ~ Bernoulli(P(v=1 | h^(k-1)))
     b. Compute h^(k)_probs = P(h=1 | v^(k))
     (for CD-1, k=1)
  3. NEGATIVE PHASE: v^(k) is the "model's fantasy"
  4. WEIGHT UPDATE (gradient ascent on log-likelihood):
     ΔW   = η × (v^(0)ᵀ · h^(0)_probs - v^(k)ᵀ · h^(k)_probs) / batch_size
     Δb_v = η × mean(v^(0) - v^(k))
     Δb_h = η × mean(h^(0)_probs - h^(k)_probs)

Free Energy
  F(v) = -v·b_v - Σ_j log(1 + exp(b_h[j] + W[j,:]·v))
  Lower free energy = more probable (= more "normal") under the model.
  We use +F(v) as the anomaly score: higher score = more anomalous.

Reconstruction Error
  MSE between v and the reconstructed v̂ = P(v=1 | h(v)).
  Used as training loss to monitor convergence.
"""

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Numerically stable sigmoid
# --------------------------------------------------------------------------- #
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Element-wise logistic sigmoid σ(x) = 1 / (1 + exp(-x)).

    Numerically stable: uses -expm1 trick so that np.where does not
    raise overflow warnings on extreme inputs (both branches are always
    evaluated by numpy, but results outside the condition are discarded).
    """
    # Clip to avoid overflow in exp; result outside [-500, 500] is 0 or 1
    x = np.clip(x, -500, 500)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# --------------------------------------------------------------------------- #
# BernoulliRBM
# --------------------------------------------------------------------------- #
class BernoulliRBM:
    """
    Bernoulli Restricted Boltzmann Machine trained with Contrastive Divergence.

    Parameters
    ----------
    n_visible    : int  — number of visible units (= n_features × n_bits)
    n_hidden     : int  — number of hidden units (hyperparameter; try 10–50)
    learning_rate: float — step size η for weight updates (default 0.01)
    cd_steps     : int  — number of Gibbs steps k in CD-k (default 1 = CD-1)
    seed         : int  — random seed for reproducibility
    momentum     : float — momentum coefficient for smoother training (0 = off)
    weight_decay : float — L2 regularisation to prevent weight explosion

    Attributes (after training)
    ---------------------------
    W       : np.ndarray, shape (n_hidden, n_visible)
    b_v     : np.ndarray, shape (n_visible,)   — visible biases
    b_h     : np.ndarray, shape (n_hidden,)    — hidden biases
    losses_ : List[float] — reconstruction error per epoch
    """

    def __init__(
        self,
        n_visible:     int   = 21,
        n_hidden:      int   = 10,
        learning_rate: float = 0.01,
        cd_steps:      int   = 1,
        seed:          int   = 42,
        momentum:      float = 0.5,
        weight_decay:  float = 1e-4,
    ):
        self.n_visible     = n_visible
        self.n_hidden      = n_hidden
        self.learning_rate = learning_rate
        self.cd_steps      = cd_steps
        self.seed          = seed
        self.momentum      = momentum
        self.weight_decay  = weight_decay

        # Initialise parameters
        rng = np.random.default_rng(seed)

        # Small random weights — Gaussian with small std is standard practice
        # (Hinton, 2010: "A Practical Guide to Training RBMs")
        self.W   = rng.normal(0, 0.01, size=(n_hidden, n_visible)).astype(np.float64)
        self.b_v = np.zeros(n_visible, dtype=np.float64)
        self.b_h = np.zeros(n_hidden,  dtype=np.float64)

        # Momentum buffers (same shape as parameters)
        self._dW   = np.zeros_like(self.W)
        self._db_v = np.zeros_like(self.b_v)
        self._db_h = np.zeros_like(self.b_h)

        self.losses_: List[float] = []
        self._rng = rng
        self._trained = False

    # ------------------------------------------------------------------ #
    # Core Gibbs sampling operations
    # ------------------------------------------------------------------ #
    def _hidden_probs(self, v: np.ndarray) -> np.ndarray:
        """
        Compute P(h_j = 1 | v) for all hidden units j.

        Pre-activation: a_j = b_h[j] + Σ_i W[j,i] · v[i]
        Result: sigmoid(a)  →  shape (batch, n_hidden)
        """
        # v: (batch, n_visible)   W: (n_hidden, n_visible)
        # activation = v @ W.T + b_h
        return sigmoid(v @ self.W.T + self.b_h)  # (batch, n_hidden)

    def _sample_hidden(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ancestral sampling: h ~ Bernoulli(P(h=1|v)).

        Returns
        -------
        h_probs  : (batch, n_hidden) — probabilities
        h_sample : (batch, n_hidden) — 0/1 samples drawn from h_probs
        """
        h_probs  = self._hidden_probs(v)
        h_sample = (self._rng.random(h_probs.shape) < h_probs).astype(np.float64)
        return h_probs, h_sample

    def _visible_probs(self, h: np.ndarray) -> np.ndarray:
        """
        Compute P(v_i = 1 | h) for all visible units i.

        Pre-activation: a_i = b_v[i] + Σ_j W[j,i] · h[j]
        Result: sigmoid(a)  →  shape (batch, n_visible)
        """
        # h: (batch, n_hidden)   W: (n_hidden, n_visible)
        # activation = h @ W + b_v
        return sigmoid(h @ self.W + self.b_v)    # (batch, n_visible)

    def _sample_visible(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample v ~ Bernoulli(P(v=1|h)).

        Returns
        -------
        v_probs  : (batch, n_visible) — probabilities
        v_sample : (batch, n_visible) — 0/1 samples
        """
        v_probs  = self._visible_probs(h)
        v_sample = (self._rng.random(v_probs.shape) < v_probs).astype(np.float64)
        return v_probs, v_sample

    def gibbs_step(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        One complete Gibbs step: v → h → v'.

        Parameters
        ----------
        v : np.ndarray, shape (batch, n_visible)

        Returns
        -------
        h_sample : (batch, n_hidden)  — hidden sample
        v_probs  : (batch, n_visible) — reconstructed visible probabilities
        """
        h_probs, h_sample = self._sample_hidden(v)
        v_probs, _        = self._sample_visible(h_sample)
        return h_probs, v_probs

    # ------------------------------------------------------------------ #
    # Contrastive Divergence update
    # ------------------------------------------------------------------ #
    def _cd_update(self, v_data: np.ndarray) -> float:
        """
        Perform one CD-k weight update on a mini-batch.

        Parameters
        ----------
        v_data : np.ndarray, shape (batch_size, n_visible)
            Binary training samples (one mini-batch).

        Returns
        -------
        float : mean squared reconstruction error for this batch.
        """
        batch_size = v_data.shape[0]

        # ---- POSITIVE PHASE ---------------------------------------- #
        # Clamp visible to real data and sample hidden units
        h0_probs, h0_sample = self._sample_hidden(v_data)

        # Positive statistics: ⟨v h⟩_data
        pos_W   = h0_probs.T @ v_data               # (n_hidden, n_visible)
        pos_b_h = h0_probs.mean(axis=0)             # (n_hidden,)
        pos_b_v = v_data.mean(axis=0)               # (n_visible,)

        # ---- NEGATIVE PHASE (k Gibbs steps) ------------------------ #
        v_model = v_data.copy()
        h_model_probs = h0_probs

        for _ in range(self.cd_steps):
            # Sample visible from current hidden
            _, v_model    = self._sample_visible(
                (self._rng.random(h_model_probs.shape) < h_model_probs).astype(np.float64)
            )
            # Sample hidden from reconstructed visible
            h_model_probs, _ = self._sample_hidden(v_model)

        # Negative statistics: ⟨v h⟩_model
        neg_W   = h_model_probs.T @ v_model          # (n_hidden, n_visible)
        neg_b_h = h_model_probs.mean(axis=0)
        neg_b_v = v_model.mean(axis=0)

        # ---- WEIGHT UPDATE (gradient + momentum + weight decay) ---- #
        lr = self.learning_rate

        # Gradient: positive - negative, normalised by batch size
        grad_W   = (pos_W   - neg_W)   / batch_size
        grad_b_h = (pos_b_h - neg_b_h)
        grad_b_v = (pos_b_v - neg_b_v)

        # Momentum update: velocity = momentum × velocity + learning_rate × gradient
        self._dW   = self.momentum * self._dW   + lr * grad_W
        self._db_h = self.momentum * self._db_h + lr * grad_b_h
        self._db_v = self.momentum * self._db_v + lr * grad_b_v

        # Apply updates with L2 weight decay on W only
        self.W   += self._dW   - lr * self.weight_decay * self.W
        self.b_h += self._db_h
        self.b_v += self._db_v

        # ---- Reconstruction error (MSE between data and reconstruction) #
        # Use the probability (soft reconstruction) for a smoother loss signal
        v0_probs = self._visible_probs(h0_sample)
        recon_error = float(np.mean((v_data - v0_probs) ** 2))
        return recon_error

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        X_val: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, float, Optional[float]], None]] = None,
        verbose: bool = True,
    ) -> "BernoulliRBM":
        """
        Train the RBM using mini-batch Contrastive Divergence.

        Parameters
        ----------
        X_train    : np.ndarray, shape (n_train, n_visible) — binary training data
        epochs     : Number of full passes over training data
        batch_size : Mini-batch size
        X_val      : Optional validation binary array for monitoring
        callback   : Optional function(epoch, train_loss, val_loss) called each epoch
        verbose    : Print progress every 10 epochs

        Returns
        -------
        self : for chaining
        """
        n_samples = X_train.shape[0]
        self.losses_ = []
        val_losses   = []

        t_start = time.time()
        logger.info(
            "Starting RBM training: %d samples, %d epochs, batch_size=%d",
            n_samples, epochs, batch_size,
        )

        for epoch in range(1, epochs + 1):
            # Shuffle training data each epoch
            idx = self._rng.permutation(n_samples)
            X_shuffled = X_train[idx]

            epoch_loss = 0.0
            n_batches  = 0

            # Mini-batch loop
            for start in range(0, n_samples, batch_size):
                batch = X_shuffled[start: start + batch_size]
                if len(batch) == 0:
                    continue
                epoch_loss += self._cd_update(batch)
                n_batches  += 1

            train_loss = epoch_loss / max(n_batches, 1)
            self.losses_.append(train_loss)

            # Validation loss (free-energy based)
            val_loss = None
            if X_val is not None and len(X_val) > 0:
                val_fe   = self.free_energy(X_val)
                val_loss = float(np.mean(val_fe))
                val_losses.append(val_loss)

            if callback:
                callback(epoch, train_loss, val_loss)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                elapsed = time.time() - t_start
                val_str = f"  val_free_energy={val_loss:.4f}" if val_loss is not None else ""
                logger.info(
                    "Epoch %3d/%d  train_recon=%.4f%s  [%.1fs elapsed]",
                    epoch, epochs, train_loss, val_str, elapsed,
                )

        self._trained = True
        logger.info(
            "Training complete. Final reconstruction error: %.4f", self.losses_[-1]
        )
        return self

    # ------------------------------------------------------------------ #
    # Free energy and anomaly scoring
    # ------------------------------------------------------------------ #
    def free_energy(self, v: np.ndarray) -> np.ndarray:
        """
        Compute the free energy F(v) for each sample.

        Formula
        -------
        F(v) = -v · b_v  -  Σ_j log(1 + exp(b_h[j] + W[j,:] · v))

        Interpretation
        --------------
        • Lower F(v) = v is MORE probable under the model = more "normal"
        • Higher F(v) = v is LESS probable = more "anomalous"

        The anomaly score is +F(v) itself (see anomaly_score): a HIGH free
        energy means a LOW probability, so high score = high anomaly.

        Parameters
        ----------
        v : np.ndarray, shape (n_samples, n_visible)

        Returns
        -------
        np.ndarray, shape (n_samples,) — free energy per sample
        """
        # Term 1: visible bias contribution
        term_v = v @ self.b_v                       # (n_samples,)

        # Term 2: hidden unit log-partition contribution
        # pre_h[i, j] = b_h[j] + Σ_k W[j,k] · v[i,k]
        pre_h  = v @ self.W.T + self.b_h            # (n_samples, n_hidden)

        # Numerically stable log(1 + exp(x)) = softplus
        # Uses: log(1 + exp(x)) = max(x,0) + log(1 + exp(-|x|))
        softplus = np.logaddexp(0, pre_h)            # (n_samples, n_hidden)
        term_h   = softplus.sum(axis=1)             # (n_samples,)

        free_energy = -term_v - term_h              # (n_samples,)
        return free_energy

    def anomaly_score(self, v: np.ndarray) -> np.ndarray:
        """
        Return anomaly score = +F(v).  Higher → more anomalous.

        Free energy is low for probable (normal) inputs and high for
        improbable ones, so using F(v) directly gives the intuitive direction:
        normal samples have low scores, anomalies have high scores.
        """
        return self.free_energy(v)

    def reconstruct(self, v: np.ndarray) -> np.ndarray:
        """
        One-step reconstruction: v → h_probs → v̂_probs.

        Useful for visualisation and debugging.

        Parameters
        ----------
        v : np.ndarray, shape (n_samples, n_visible)

        Returns
        -------
        v_hat : np.ndarray, shape (n_samples, n_visible) — soft reconstruction
        """
        h_probs = self._hidden_probs(v)
        v_hat   = self._visible_probs(h_probs)
        return v_hat

    def reconstruction_error(self, v: np.ndarray) -> np.ndarray:
        """Per-sample MSE between input and soft reconstruction."""
        v_hat = self.reconstruct(v)
        return np.mean((v - v_hat) ** 2, axis=1)

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #
    def save(self, path: str = "models/rbm.npz") -> None:
        """
        Save RBM parameters to a compressed NumPy archive.

        Saved arrays: W, b_v, b_h, losses
        Saved metadata: n_visible, n_hidden, learning_rate, cd_steps, seed
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W           = self.W,
            b_v         = self.b_v,
            b_h         = self.b_h,
            losses      = np.array(self.losses_),
            # Metadata stored as 0-d arrays
            n_visible    = np.array(self.n_visible),
            n_hidden     = np.array(self.n_hidden),
            learning_rate= np.array(self.learning_rate),
            cd_steps     = np.array(self.cd_steps),
            seed         = np.array(self.seed),
            momentum     = np.array(self.momentum),
            weight_decay = np.array(self.weight_decay),
        )
        logger.info("RBM saved to %s", path)

    @classmethod
    def load(cls, path: str = "models/rbm.npz") -> "BernoulliRBM":
        """Load a previously saved RBM from disk."""
        data = np.load(path)
        rbm = cls(
            n_visible     = int(data["n_visible"]),
            n_hidden      = int(data["n_hidden"]),
            learning_rate = float(data["learning_rate"]),
            cd_steps      = int(data["cd_steps"]),
            seed          = int(data["seed"]),
            momentum      = float(data["momentum"]),
            weight_decay  = float(data["weight_decay"]),
        )
        rbm.W       = data["W"]
        rbm.b_v     = data["b_v"]
        rbm.b_h     = data["b_h"]
        rbm.losses_ = data["losses"].tolist()
        rbm._trained = True
        logger.info("RBM loaded from %s", path)
        return rbm

    def is_trained(self) -> bool:
        return self._trained

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return (
            f"BernoulliRBM(n_visible={self.n_visible}, n_hidden={self.n_hidden}, "
            f"lr={self.learning_rate}, cd_steps={self.cd_steps}, "
            f"status={status})"
        )
