"""Preprocessing: per-domain standardization + global fallback.

Domain-generalization technique: per-domain batch normalization / feature
standardization within domain (well-established, simple alternative to DANN).
Each domain's features are standardized to zero mean/unit variance using
ONLY that domain's statistics. At test time on an unseen domain, we use
the held-out domain's own statistics (computed from its test samples) OR
fallback to global training stats if single-sample inference.

See also: src/dann.py for domain-adversarial alternative (Ganin et al. 2016).
"""

from __future__ import annotations

import numpy as np

from data_pipeline.dataset import ExpressionDataset


class DomainStandardizer:
    """Per-domain z-score standardization.

    Fit on training domains: computes mean/std per domain.
    Transform: standardizes each sample using its domain's stats.
    For unseen domains at test time, uses global mean/std (from all train domains)
    or optionally computes stats from the test batch itself (transductive).
    """

    def __init__(self, eps: float = 1e-8, transductive: bool = False):
        self.eps = eps
        self.transductive = transductive
        self.domain_means: dict[str, np.ndarray] = {}
        self.domain_stds: dict[str, np.ndarray] = {}
        self.global_mean: np.ndarray | None = None
        self.global_std: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: np.ndarray, domains: np.ndarray) -> "DomainStandardizer":
        unique = sorted(set(domains.tolist()))
        for d in unique:
            mask = domains == d
            Xd = X[mask]
            self.domain_means[d] = Xd.mean(axis=0)
            self.domain_stds[d] = Xd.std(axis=0)
            # Avoid division by zero
            self.domain_stds[d] = np.where(self.domain_stds[d] < self.eps, 1.0, self.domain_stds[d])
        self.global_mean = X.mean(axis=0)
        self.global_std = X.std(axis=0)
        self.global_std = np.where(self.global_std < self.eps, 1.0, self.global_std)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        assert self._fitted, "Must fit before transform"
        assert self.global_mean is not None and self.global_std is not None
        if self.transductive:
            # Compute stats from the batch itself (per-domain within batch)
            # Useful for LODO eval where test domain statistics are available
            result = np.zeros_like(X, dtype=float)
            unique_batch = sorted(set(domains.tolist()))
            for d in unique_batch:
                mask = domains == d
                if d in self.domain_means:
                    m, s = self.domain_means[d], self.domain_stds[d]
                else:
                    # Unseen domain: compute from batch
                    Xd = X[mask]
                    m = Xd.mean(axis=0)
                    s = Xd.std(axis=0)
                    s = np.where(s < self.eps, 1.0, s)
                result[mask] = (X[mask] - m) / s
            return result
        else:
            result = np.zeros_like(X, dtype=float)
            for i, d in enumerate(domains):
                if d in self.domain_means:
                    m, s = self.domain_means[d], self.domain_stds[d]
                else:
                    m, s = self.global_mean, self.global_std
                result[i] = (X[i] - m) / s
            return result

    def fit_transform(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        return self.fit(X, domains).transform(X, domains)


def preprocess(
    train_dataset: ExpressionDataset,
    test_dataset: ExpressionDataset | None = None,
    standardizer: DomainStandardizer | None = None,
    transductive: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, DomainStandardizer]:
    """Preprocess train (and optionally test) datasets.

    Returns:
        X_train_processed, X_test_processed (or None), fitted standardizer
    """
    if standardizer is None:
        standardizer = DomainStandardizer(transductive=transductive)
        X_train_p = standardizer.fit_transform(train_dataset.X, train_dataset.domains)
    else:
        X_train_p = standardizer.transform(train_dataset.X, train_dataset.domains)

    if test_dataset is not None:
        X_test_p = standardizer.transform(test_dataset.X, test_dataset.domains)
    else:
        X_test_p = None

    return X_train_p, X_test_p, standardizer
