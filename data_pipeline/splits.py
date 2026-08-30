"""Splitting protocols: leave-one-domain-out (required) and random-split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold

from data_pipeline.dataset import ExpressionDataset


@dataclass
class StratifiedDomainSplit:
    """One split with train/test indices."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    train_domains: list[str]
    test_domains: list[str]
    fold: int


def leave_one_domain_out_splits(dataset: ExpressionDataset) -> list[StratifiedDomainSplit]:
    """Leave-one-domain-out splits.

    Each split: train on N-1 domains, test on the held-out domain.
    Ensures NO data leakage across domains (samples from test domain never in train).

    Returns:
        list of splits, one per unique domain.
    """
    unique = dataset.unique_domains
    splits: list[StratifiedDomainSplit] = []
    for i, held_out in enumerate(unique):
        mask_test = dataset.domains == held_out
        mask_train = ~mask_test
        train_idx = np.where(mask_train)[0]
        test_idx = np.where(mask_test)[0]
        assert len(train_idx) > 0 and len(test_idx) > 0, f"Empty split for domain {held_out}"
        # Verify no domain leakage
        train_dom_set = set(dataset.domains[train_idx].tolist())
        test_dom_set = set(dataset.domains[test_idx].tolist())
        assert train_dom_set.isdisjoint(test_dom_set), f"Domain leakage: {train_dom_set} ∩ {test_dom_set}"
        assert held_out in test_dom_set
        splits.append(StratifiedDomainSplit(
            train_idx=train_idx, test_idx=test_idx,
            train_domains=sorted(train_dom_set),
            test_domains=sorted(test_dom_set),
            fold=i,
        ))
    return splits


def random_splits(
    dataset: ExpressionDataset,
    n_splits: int = 5,
    test_size: float = 0.2,
    seed: int = 0,
) -> list[StratifiedDomainSplit]:
    """Standard stratified K-fold ignoring domains (optimistic, same-domain).

    This deliberately mixes domains in train and test, giving an upper-bound
    estimate that overstates cross-domain robustness.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[StratifiedDomainSplit] = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset.X, dataset.y)):
        # Record which domains appear (for reporting)
        train_dom_set = sorted(set(dataset.domains[train_idx].tolist()))
        test_dom_set = sorted(set(dataset.domains[test_idx].tolist()))
        splits.append(StratifiedDomainSplit(
            train_idx=train_idx, test_idx=test_idx,
            train_domains=train_dom_set,
            test_domains=test_dom_set,
            fold=fold,
        ))
    return splits


def train_test_split_within_domains(
    dataset: ExpressionDataset,
    test_frac: float = 0.2,
    seed: int = 0,
) -> StratifiedDomainSplit:
    """Single random split stratified by label (ignoring domain)."""
    from sklearn.model_selection import train_test_split as sk_tts
    rng_indices = np.arange(dataset.n_samples)
    train_idx, test_idx = sk_tts(
        rng_indices, test_size=test_frac, stratify=dataset.y, random_state=seed
    )
    return StratifiedDomainSplit(
        train_idx=np.array(train_idx),
        test_idx=np.array(test_idx),
        train_domains=sorted(set(dataset.domains[train_idx].tolist())),
        test_domains=sorted(set(dataset.domains[test_idx].tolist())),
        fold=0,
    )
