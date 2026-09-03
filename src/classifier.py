"""Base classifier: regularized multinomial logistic regression + small MLP option."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def make_classifier(
    kind: str = "logreg",
    n_classes: int = 3,
    seed: int = 0,
    **kwargs,
):
    """Factory for classifier.

    kind: 'logreg' or 'mlp'
    """
    if kind == "logreg":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=kwargs.get("C", 1.0),
            random_state=seed,
        )
    elif kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (64, 32)),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown classifier kind: {kind}")
