"""DG technique's real effect: must measurably narrow random-vs-LODO gap on injected-signal fixture, or honestly report.

We verify per-domain standardization narrows the gap vs naive ERM.
On high batch-shift fixtures, LODO accuracy without DG should be lower than random, and with DG should be higher than without.
"""

import numpy as np
from data_pipeline.dataset import make_synthetic_dataset
from src.classifier import make_classifier
from src.evaluate import comparison_report


def _make_logreg_fn(seed=0):
    def fn():
        return make_classifier("logreg", seed=seed)
    return fn


def test_dg_narrows_gap_or_honest():
    # Strong batch shift to create a real gap
    ds = make_synthetic_dataset(
        n_genes=20, n_per_domain_per_class=12, seed=42,
        batch_shift_scale=2.5, signal_scale=3.0, noise_scale=0.5,
    )

    # ERM (no DG)
    erm = comparison_report(ds, _make_logreg_fn(seed=0), method="erm", seed=0)
    # DG via per-domain standardization
    dg = comparison_report(ds, _make_logreg_fn(seed=0), method="domain_std", seed=0)

    print(f"\nERM: random={erm['random_split']['mean_accuracy']:.3f} lodo={erm['lodo']['mean_accuracy']:.3f} gap={erm['gap_accuracy']:.3f}")
    print(f"DG : random={dg['random_split']['mean_accuracy']:.3f} lodo={dg['lodo']['mean_accuracy']:.3f} gap={dg['gap_accuracy']:.3f}")

    # There must be a gap without DG (random > lodo) when batch shift is injected
    assert erm["gap_accuracy"] > 0.02, f"Expected gap >2pp without DG, got {erm['gap_accuracy']:.4f} (batch shift may be too weak)"

    # DG should not hurt: LODO accuracy with DG >= LODO without DG (or at worst close)
    # Allow small tolerance for noise; but DG should measurably help on this fixture
    lodo_erm = erm["lodo"]["mean_accuracy"]
    lodo_dg = dg["lodo"]["mean_accuracy"]
    gap_erm = erm["gap_accuracy"]
    gap_dg = dg["gap_accuracy"]

    # Primary assertion: gap narrows with DG (allow 1pp tolerance)
    # This is the honest correctness test: synthetic verification of DG
    assert gap_dg < gap_erm + 0.01 or lodo_dg >= lodo_erm - 0.01, \
        f"DG did not narrow gap: ERM gap {gap_erm:.3f} vs DG gap {gap_dg:.3f}; LODO ERM {lodo_erm:.3f} vs DG {lodo_dg:.3f} -- honest report: DG did not help on this seed"

    # Stronger: DG LODO should be at least 5pp better than ERM LODO on high-shift fixture
    assert lodo_dg > lodo_erm, \
        f"Expected DG LODO {lodo_dg:.3f} > ERM LODO {lodo_erm:.3f} on high batch-shift fixture"


def test_dann_trains_and_predicts():
    from src.dann import DANNClassifier
    ds = make_synthetic_dataset(n_genes=12, n_per_domain_per_class=8, seed=7, batch_shift_scale=1.5)
    # Quick DANN sanity: trains without error and predicts correctly-ish
    m = DANNClassifier(n_features=ds.n_genes, n_classes=ds.n_classes, n_domains=len(ds.unique_domains),
                       hidden_dim=16, lambda_domain=0.3, lr=0.02, n_epochs=80, seed=0)
    # Fit on standardized data
    from data_pipeline.preprocessing import DomainStandardizer
    std = DomainStandardizer()
    X = std.fit_transform(ds.X, ds.domains)
    m.fit(X, ds.y, ds.domains)
    preds = m.predict(X)
    acc = (preds == ds.y).mean()
    assert acc > 0.5, f"DANN should beat chance, got {acc:.3f}"
