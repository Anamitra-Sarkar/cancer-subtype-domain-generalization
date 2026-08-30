"""Domain-Adversarial Neural Network (DANN) -- Ganin et al. 2016, JMLR 17:2096-2030.

Implements gradient-reversal layer + domain classifier to encourage
domain-invariant features. Pure numpy/sklearn-style MLP for sandbox compatibility
(no PyTorch dependency required for tests). A PyTorch implementation is
documented as the production alternative; this numpy version is functionally
equivalent for the synthetic DG verification.

Architecture:
  Feature extractor: hidden layer (ReLU)
  Label predictor: linear -> softmax
  Domain classifier: gradient-reversed hidden -> linear -> softmax over domains

Training: alternate minimizing label loss and maximizing domain classification
loss via gradient reversal (implemented as explicit gradient sign flip).

Reference:
  Ganin, Y. et al. "Domain-Adversarial Training of Neural Networks." JMLR 2016.
  ArXiv:1505.07818.
"""

from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


class DANNClassifier:
    """Simple DANN with one hidden feature-extractor layer.

    Args:
        n_features: input dimension (genes)
        n_classes: number of subtypes
        n_domains: number of domains (for domain classifier)
        hidden_dim: feature extractor hidden size
        lambda_domain: weight of domain-adversarial loss (gradient reversal strength)
        lr: learning rate
        n_epochs: training epochs
        batch_size: minibatch size
        seed: RNG seed
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_domains: int,
        hidden_dim: int = 32,
        lambda_domain: float = 0.1,
        lr: float = 0.01,
        n_epochs: int = 200,
        batch_size: int = 32,
        seed: int = 0,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_domains = n_domains
        self.hidden_dim = hidden_dim
        self.lambda_domain = lambda_domain
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Xavier init
        scale_f = np.sqrt(2.0 / (n_features + hidden_dim))
        self.W_f = self.rng.normal(0, scale_f, size=(n_features, hidden_dim))
        self.b_f = np.zeros(hidden_dim)
        scale_y = np.sqrt(2.0 / (hidden_dim + n_classes))
        self.W_y = self.rng.normal(0, scale_y, size=(hidden_dim, n_classes))
        self.b_y = np.zeros(n_classes)
        scale_d = np.sqrt(2.0 / (hidden_dim + n_domains))
        self.W_d = self.rng.normal(0, scale_d, size=(hidden_dim, n_domains))
        self.b_d = np.zeros(n_domains)

        self._fitted = False
        self.domain_to_idx: dict[str, int] = {}

    def _forward(self, X: np.ndarray):
        h_pre = X @ self.W_f + self.b_f  # (B, H)
        h = _relu(h_pre)  # (B, H)
        logits_y = h @ self.W_y + self.b_y  # (B, C)
        logits_d = h @ self.W_d + self.b_d  # (B, D)
        return h_pre, h, logits_y, logits_d

    def fit(self, X: np.ndarray, y: np.ndarray, domains: np.ndarray) -> "DANNClassifier":
        # Map domain strings to indices
        unique_domains = sorted(set(domains.tolist()))
        self.domain_to_idx = {d: i for i, d in enumerate(unique_domains)}
        # If n_domains mismatched, update
        if len(unique_domains) != self.n_domains:
            # Re-init domain head if domain count differs
            self.n_domains = len(unique_domains)
            scale_d = np.sqrt(2.0 / (self.hidden_dim + self.n_domains))
            self.W_d = self.rng.normal(0, scale_d, size=(self.hidden_dim, self.n_domains))
            self.b_d = np.zeros(self.n_domains)

        d_idx = np.array([self.domain_to_idx[d] for d in domains], dtype=int)
        n = X.shape[0]

        for epoch in range(self.n_epochs):
            perm = self.rng.permutation(n)
            X_sh, y_sh, d_sh = X[perm], y[perm], d_idx[perm]
            # Gradual increase of lambda (as in Ganin et al.)
            p = epoch / max(self.n_epochs - 1, 1)
            lamb = self.lambda_domain * (2.0 / (1.0 + np.exp(-10 * p)) - 1)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                xb, yb, db = X_sh[start:end], y_sh[start:end], d_sh[start:end]
                B = xb.shape[0]

                h_pre, h, logits_y, logits_d = self._forward(xb)
                py = _softmax(logits_y)
                pd = _softmax(logits_d)

                # One-hot
                y_onehot = np.zeros_like(py)
                y_onehot[np.arange(B), yb] = 1.0
                d_onehot = np.zeros_like(pd)
                d_onehot[np.arange(B), db] = 1.0

                # Gradients for label predictor (standard cross-entropy)
                d_logits_y = (py - y_onehot) / B
                grad_W_y = h.T @ d_logits_y
                grad_b_y = d_logits_y.sum(axis=0)

                # Gradients for domain classifier
                d_logits_d = (pd - d_onehot) / B
                grad_W_d = h.T @ d_logits_d
                grad_b_d = d_logits_d.sum(axis=0)

                # Backprop to hidden: label path (+) and domain path (reversed sign)
                dh_y = d_logits_y @ self.W_y.T  # (B, H)
                dh_d = d_logits_d @ self.W_d.T  # (B, H)
                # Gradient reversal: domain gradient is subtracted (negated) weighted by lamb
                dh = dh_y - lamb * dh_d
                dh = dh * _relu_grad(h_pre)

                grad_W_f = xb.T @ dh
                grad_b_f = dh.sum(axis=0)

                # SGD step
                self.W_y -= self.lr * grad_W_y
                self.b_y -= self.lr * grad_b_y
                self.W_d -= self.lr * grad_W_d
                self.b_d -= self.lr * grad_b_d
                self.W_f -= self.lr * grad_W_f
                self.b_f -= self.lr * grad_b_f

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, _, logits_y, _ = self._forward(X)
        return _softmax(logits_y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())
