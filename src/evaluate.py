"""Evaluation: random-split vs leave-one-domain-out comparison."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from data_pipeline.dataset import ExpressionDataset
from data_pipeline.splits import leave_one_domain_out_splits, random_splits
from data_pipeline.preprocessing import DomainStandardizer


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": float(acc), "macro_f1": float(mf1), "confusion_matrix": cm, "n": int(len(y_true))}


def evaluate_lodo(
    dataset: ExpressionDataset,
    make_model_fn,
    use_domain_std: bool = False,
    transductive: bool = False,
) -> dict:
    """Run LODO evaluation.

    make_model_fn: callable returning a fresh unfitted classifier with .fit(X,y) / .predict(X)
                   For DANN, it must accept (X,y,domains).
    use_domain_std: if True, apply per-domain standardization before training
    """
    splits = leave_one_domain_out_splits(dataset)
    per_fold = []
    all_true, all_pred = [], []
    for s in splits:
        train_ds = dataset.subset_by_indices(s.train_idx)
        test_ds = dataset.subset_by_indices(s.test_idx)
        if use_domain_std:
            std = DomainStandardizer(transductive=transductive)
            X_train = std.fit_transform(train_ds.X, train_ds.domains)
            X_test = std.transform(test_ds.X, test_ds.domains)
        else:
            # Simple global standardization (fit on train only to avoid leakage)
            mean = train_ds.X.mean(axis=0)
            stdv = train_ds.X.std(axis=0)
            stdv = np.where(stdv < 1e-8, 1.0, stdv)
            X_train = (train_ds.X - mean) / stdv
            X_test = (test_ds.X - mean) / stdv

        model = make_model_fn()
        # DANN needs domains; detect via try
        try:
            # Check if model is DANN-like (has lambda_domain or expects domains)
            import inspect
            sig = inspect.signature(model.fit)
            if "domains" in sig.parameters:
                model.fit(X_train, train_ds.y, train_ds.domains)
            else:
                model.fit(X_train, train_ds.y)
        except TypeError:
            model.fit(X_train, train_ds.y)

        y_pred = model.predict(X_test)
        m = compute_metrics(test_ds.y, y_pred)
        m["held_out_domain"] = s.test_domains[0]
        per_fold.append(m)
        all_true.extend(test_ds.y.tolist())
        all_pred.extend(y_pred.tolist())

    pooled = compute_metrics(np.array(all_true), np.array(all_pred))
    mean_acc = float(np.mean([f["accuracy"] for f in per_fold]))
    mean_f1 = float(np.mean([f["macro_f1"] for f in per_fold]))
    return {"per_fold": per_fold, "pooled": pooled, "mean_accuracy": mean_acc, "mean_macro_f1": mean_f1}


def evaluate_random(
    dataset: ExpressionDataset,
    make_model_fn,
    use_domain_std: bool = False,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Standard stratified K-fold ignoring domains."""
    splits = random_splits(dataset, n_splits=n_splits, seed=seed)
    per_fold = []
    all_true, all_pred = [], []
    for s in splits:
        train_ds = dataset.subset_by_indices(s.train_idx)
        test_ds = dataset.subset_by_indices(s.test_idx)
        if use_domain_std:
            std = DomainStandardizer(transductive=False)
            X_train = std.fit_transform(train_ds.X, train_ds.domains)
            X_test = std.transform(test_ds.X, test_ds.domains)
        else:
            mean = train_ds.X.mean(axis=0)
            stdv = train_ds.X.std(axis=0)
            stdv = np.where(stdv < 1e-8, 1.0, stdv)
            X_train = (train_ds.X - mean) / stdv
            X_test = (test_ds.X - mean) / stdv

        model = make_model_fn()
        try:
            import inspect
            sig = inspect.signature(model.fit)
            if "domains" in sig.parameters:
                model.fit(X_train, train_ds.y, train_ds.domains)
            else:
                model.fit(X_train, train_ds.y)
        except TypeError:
            model.fit(X_train, train_ds.y)
        y_pred = model.predict(X_test)
        m = compute_metrics(test_ds.y, y_pred)
        per_fold.append(m)
        all_true.extend(test_ds.y.tolist())
        all_pred.extend(y_pred.tolist())
    pooled = compute_metrics(np.array(all_true), np.array(all_pred))
    mean_acc = float(np.mean([f["accuracy"] for f in per_fold]))
    mean_f1 = float(np.mean([f["macro_f1"] for f in per_fold]))
    return {"per_fold": per_fold, "pooled": pooled, "mean_accuracy": mean_acc, "mean_macro_f1": mean_f1}


def comparison_report(
    dataset: ExpressionDataset,
    make_model_fn,
    method: str = "erm",
    seed: int = 0,
) -> dict:
    """Full comparison: random-split (optimistic) vs LODO (realistic).

    method:
      erm: standard global standardization, no DG
      domain_std: per-domain standardization (simple DG)
      dann: domain-adversarial (requires DANN model from make_model_fn)
    """
    if method == "domain_std":
        use_ds = True
        transductive = True  # for LODO, use test-domain stats (realistic batch-aware)
    else:
        use_ds = False
        transductive = False

    random_res = evaluate_random(dataset, make_model_fn, use_domain_std=use_ds, seed=seed)
    lodo_res = evaluate_lodo(dataset, make_model_fn, use_domain_std=use_ds, transductive=transductive)

    gap_acc = random_res["mean_accuracy"] - lodo_res["mean_accuracy"]
    gap_f1 = random_res["mean_macro_f1"] - lodo_res["mean_macro_f1"]

    return {
        "method": method,
        "random_split": random_res,
        "lodo": lodo_res,
        "gap_accuracy": float(gap_acc),
        "gap_macro_f1": float(gap_f1),
        "n_domains": len(dataset.unique_domains),
        "domains": dataset.unique_domains,
        "subtypes": dataset.subtype_names,
    }
