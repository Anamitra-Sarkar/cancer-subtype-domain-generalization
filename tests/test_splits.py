"""Test LODO split correctness: no cross-domain leakage."""

import numpy as np
from data_pipeline.dataset import make_synthetic_dataset
from data_pipeline.splits import leave_one_domain_out_splits, random_splits


def test_lodo_no_leakage():
    ds = make_synthetic_dataset(n_genes=10, n_per_domain_per_class=5, seed=0)
    splits = leave_one_domain_out_splits(ds)
    assert len(splits) == len(ds.unique_domains) == 3
    for s in splits:
        train_domains = set(ds.domains[s.train_idx].tolist())
        test_domains = set(ds.domains[s.test_idx].tolist())
        assert train_domains.isdisjoint(test_domains), f"Leakage: {train_domains} vs {test_domains}"
        assert len(test_domains) == 1
        # All samples accounted for
        assert len(s.train_idx) + len(s.test_idx) == ds.n_samples
        # Held-out domain is exactly test domain
        assert s.test_domains == sorted(test_domains)
        assert s.train_domains == sorted(train_domains)


def test_lodo_covers_all_domains():
    ds = make_synthetic_dataset(domains=["A", "B", "C", "D"], n_per_domain_per_class=3, seed=1)
    splits = leave_one_domain_out_splits(ds)
    held_out = {s.test_domains[0] for s in splits}
    assert held_out == set(ds.unique_domains)


def test_random_splits_shape():
    ds = make_synthetic_dataset(seed=2)
    splits = random_splits(ds, n_splits=3, seed=0)
    assert len(splits) == 3
    for s in splits:
        assert len(s.train_idx) + len(s.test_idx) == ds.n_samples
