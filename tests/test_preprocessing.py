import numpy as np
from data_pipeline.dataset import make_synthetic_dataset
from data_pipeline.preprocessing import DomainStandardizer


def test_domain_standardizer_per_domain():
    ds = make_synthetic_dataset(n_genes=6, n_per_domain_per_class=10, seed=0, batch_shift_scale=2.0)
    std = DomainStandardizer()
    Xp = std.fit_transform(ds.X, ds.domains)
    # Per-domain means should be ~0 after transform
    for d in ds.unique_domains:
        mask = ds.domains == d
        m = Xp[mask].mean(axis=0)
        assert np.allclose(m, 0, atol=1e-6), f"Domain {d} mean not zero: {m}"
