"""Training entry point and artifact saving."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from data_pipeline.dataset import ExpressionDataset
from data_pipeline.preprocessing import DomainStandardizer
from src.classifier import make_classifier
from src.dann import DANNClassifier
from src.evaluate import comparison_report


def make_model_fn_for_method(method: str, n_features: int, n_classes: int, n_domains: int, seed: int = 0):
    if method == "dann":
        def fn():
            return DANNClassifier(
                n_features=n_features, n_classes=n_classes, n_domains=n_domains,
                hidden_dim=32, lambda_domain=0.5, lr=0.02, n_epochs=300, seed=seed,
            )
        return fn
    else:
        def fn():
            return make_classifier(kind="logreg", seed=seed, C=1.0)
        return fn


def train_and_evaluate(dataset: ExpressionDataset, method: str = "domain_std", seed: int = 0) -> dict:
    """Run full comparison and return report dict."""
    make_fn = make_model_fn_for_method(method, dataset.n_genes, dataset.n_classes, len(dataset.unique_domains), seed)
    report = comparison_report(dataset, make_fn, method=method, seed=seed)
    return report


def train_final_model(
    dataset: ExpressionDataset,
    method: str = "domain_std",
    seed: int = 0,
    out_dir: str | Path = "model_artifacts",
):
    """Train on full dataset and save artifact (for backend release gate)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if method == "domain_std":
        std = DomainStandardizer(transductive=False)
        X = std.fit_transform(dataset.X, dataset.domains)
    else:
        mean = dataset.X.mean(axis=0)
        stdv = dataset.X.std(axis=0)
        stdv = np.where(stdv < 1e-8, 1.0, stdv)
        X = (dataset.X - mean) / stdv
        std = {"mean": mean, "std": stdv, "global": True}

    if method == "dann":
        model = DANNClassifier(
            n_features=dataset.n_genes, n_classes=dataset.n_classes,
            n_domains=len(dataset.unique_domains),
            hidden_dim=32, lambda_domain=0.5, lr=0.02, n_epochs=300, seed=seed,
        )
        model.fit(X, dataset.y, dataset.domains)
    else:
        model = make_classifier(kind="logreg", seed=seed)
        model.fit(X, dataset.y)

    # Save
    artifact = {
        "model": model,
        "standardizer": std,
        "subtype_names": dataset.subtype_names,
        "gene_names": dataset.gene_names,
        "method": method,
        "n_genes": dataset.n_genes,
    }
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(artifact, f)
    # Save metadata for release gate verification
    meta = {
        "method": method,
        "subtypes": dataset.subtype_names,
        "genes": dataset.gene_names,
        "n_genes": dataset.n_genes,
        "domains": dataset.unique_domains,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model to {out / 'model.pkl'}")
    return artifact
