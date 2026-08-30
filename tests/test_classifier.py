"""Classifier correctness on clean separable synthetic data."""

import numpy as np
from data_pipeline.dataset import make_synthetic_dataset
from src.classifier import make_classifier


def test_logreg_clean_separable():
    # Low batch shift, high signal -> should be highly accurate
    ds = make_synthetic_dataset(n_genes=20, n_per_domain_per_class=15, seed=0, batch_shift_scale=0.1, signal_scale=4.0, noise_scale=0.3)
    clf = make_classifier("logreg", seed=0)
    # Simple global standardization
    mean = ds.X.mean(axis=0)
    std = ds.X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X = (ds.X - mean) / std
    clf.fit(X, ds.y)
    acc = clf.score(X, ds.y)
    assert acc > 0.90, f"Expected >90% on clean separable data, got {acc:.3f}"


def test_mlp_clean_separable():
    ds = make_synthetic_dataset(n_genes=20, n_per_domain_per_class=15, seed=1, batch_shift_scale=0.1, signal_scale=4.0, noise_scale=0.3)
    clf = make_classifier("mlp", seed=0)
    mean = ds.X.mean(axis=0)
    std = ds.X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X = (ds.X - mean) / std
    clf.fit(X, ds.y)
    acc = clf.score(X, ds.y)
    assert acc > 0.85, f"MLP expected >85%, got {acc:.3f}"
