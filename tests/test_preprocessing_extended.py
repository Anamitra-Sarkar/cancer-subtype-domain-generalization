"""Extended preprocessing edge-case tests."""

import numpy as np
import pytest

from data_pipeline.dataset import make_synthetic_dataset
from data_pipeline.preprocessing import DomainStandardizer, preprocess


def test_transductive_unseen_domain():
    ds = make_synthetic_dataset(n_genes=6, n_per_domain_per_class=5, seed=0, domains=["A", "B", "C"])
    # Fit on A,B only, transform C as unseen in transductive mode -> should use batch stats, not global
    train_mask = ds.domains != "C"
    test_mask = ds.domains == "C"
    std = DomainStandardizer(transductive=True)
    std.fit(ds.X[train_mask], ds.domains[train_mask])
    X_test = std.transform(ds.X[test_mask], ds.domains[test_mask])
    # After transductive, test batch should be ~zero-mean per gene
    assert np.allclose(X_test.mean(axis=0), 0, atol=1e-6)


def test_non_transductive_unseen_fallback_to_global():
    ds = make_synthetic_dataset(n_genes=4, n_per_domain_per_class=5, seed=1, domains=["A", "B", "C"])
    train_mask = ds.domains != "C"
    test_mask = ds.domains == "C"
    std = DomainStandardizer(transductive=False)
    std.fit(ds.X[train_mask], ds.domains[train_mask])
    X_test = std.transform(ds.X[test_mask], ds.domains[test_mask])
    # Should not be exactly zero-mean (uses train global stats)
    # But after global standardization, it shouldn't be NaN
    assert np.all(np.isfinite(X_test))
    # For this synthetic where C has shifted mean, applying global should leave residual shift
    # Check that transforming with global != transductive batch result
    std_t = DomainStandardizer(transductive=True)
    std_t.fit(ds.X[train_mask], ds.domains[train_mask])
    X_test_t = std_t.transform(ds.X[test_mask], ds.domains[test_mask])
    # They should differ for unseen domain
    assert not np.allclose(X_test, X_test_t)


def test_zero_std_gene_handling():
    # Gene with constant value across a domain (zero std) should not divide by zero
    X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0], [2.0, 8.0], [2.0, 9.0], [2.0, 10.0]])
    domains = np.array(["A", "A", "A", "B", "B", "B"])
    std = DomainStandardizer()
    Xp = std.fit_transform(X, domains)
    assert np.all(np.isfinite(Xp))
    # Constant gene in domain A should be zero after standardization (since we replace std 0 with 1.0)
    # For domain A, gene0 is constant 1.0 -> after subtract mean 1.0, /1.0 => 0
    assert np.allclose(Xp[0:3, 0], 0)


def test_single_sample_inference_uses_global():
    ds = make_synthetic_dataset(n_genes=4, n_per_domain_per_class=5, seed=2)
    std = DomainStandardizer(transductive=False)
    std.fit(ds.X, ds.domains)
    # Single sample from unseen domain
    X_single = np.array([[0.5, 0.2, -0.3, 1.1]])
    domains_single = np.array(["UNSEEN_COHORT"])
    Xp = std.transform(X_single, domains_single)
    assert Xp.shape == (1, 4)
    assert np.all(np.isfinite(Xp))
    # Should equal (X - global_mean)/global_std
    expected = (X_single - std.global_mean) / std.global_std
    np.testing.assert_allclose(Xp, expected)


def test_preprocess_train_test_split():
    ds = make_synthetic_dataset(n_genes=6, n_per_domain_per_class=5, seed=3)
    train_ds = ds.subset_by_indices(np.arange(10))
    test_ds = ds.subset_by_indices(np.arange(10, 15))
    X_train, X_test, std = preprocess(train_ds, test_ds, transductive=True)
    assert X_train.shape == (10, 6)
    assert X_test.shape == (5, 6)
    assert np.all(np.isfinite(X_train))
    assert np.all(np.isfinite(X_test))


def test_fit_transform_idempotent():
    ds = make_synthetic_dataset(n_genes=5, n_per_domain_per_class=4, seed=4)
    std = DomainStandardizer()
    X1 = std.fit_transform(ds.X, ds.domains)
    # Second transform with same data should be identical
    X2 = std.transform(ds.X, ds.domains)
    np.testing.assert_allclose(X1, X2)
