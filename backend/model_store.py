"""Fail-closed release gate: model only loaded if env vars explicitly approved."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional


class ModelStore:
    """Holds loaded model artifact or None if not released."""

    def __init__(self):
        self.artifact: Optional[dict[str, Any]] = None
        self.loaded: bool = False
        self.revision: Optional[str] = None
        self.error: Optional[str] = None

    def try_load(self) -> bool:
        """Attempt to load model artifact per release gate.

        Required env vars:
          MODEL_RELEASE_APPROVED=true
          APPROVED_ARTIFACT_REVISION=<non-empty>
        Artifact path: MODEL_ARTIFACT_PATH or ./model_artifacts/model.pkl

        Returns True if loaded, False otherwise (fail-closed, no exception).
        """
        approved = os.getenv("MODEL_RELEASE_APPROVED", "false").lower() == "true"
        revision = os.getenv("APPROVED_ARTIFACT_REVISION", "")
        artifact_path = os.getenv("MODEL_ARTIFACT_PATH", "model_artifacts/model.pkl")

        if not approved:
            self.error = "MODEL_RELEASE_APPROVED != true -- model not loaded (fail-closed)"
            self.loaded = False
            return False
        if not revision:
            self.error = "APPROVED_ARTIFACT_REVISION not set -- model not loaded"
            self.loaded = False
            return False
        p = Path(artifact_path)
        if not p.exists():
            self.error = f"Artifact not found at {p} (revision {revision})"
            self.loaded = False
            return False
        try:
            with open(p, "rb") as f:
                artifact = pickle.load(f)
            self.artifact = artifact
            self.loaded = True
            self.revision = revision
            self.error = None
            return True
        except Exception as e:
            self.error = f"Failed to load artifact: {e}"
            self.loaded = False
            return False

    def is_ready(self) -> bool:
        return self.loaded and self.artifact is not None

    def predict(self, expression: list[float]) -> dict:
        if not self.is_ready():
            raise RuntimeError("Model not loaded -- release gate not approved")
        artifact = self.artifact
        assert artifact is not None
        model = artifact["model"]
        std = artifact["standardizer"]
        subtype_names = artifact["subtype_names"]
        n_genes = artifact["n_genes"]

        import numpy as np

        if len(expression) != n_genes:
            raise ValueError(f"Expected {n_genes} genes, got {len(expression)}")

        X = np.array(expression, dtype=float).reshape(1, -1)
        # Apply standardizer: handle both DomainStandardizer and dict (global)
        if hasattr(std, "transform"):
            # DomainStandardizer without domain -> use global stats
            if hasattr(std, "global_mean") and std.global_mean is not None:
                mean, s = std.global_mean, std.global_std  # type: ignore
                Xp = (X - mean) / s
            else:
                Xp = X
        elif isinstance(std, dict) and "mean" in std:
            mean, s = std["mean"], std["std"]
            Xp = (X - mean) / s
        else:
            Xp = X

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xp)[0]
        else:
            # Fallback
            pred = model.predict(Xp)[0]
            proba = np.zeros(len(subtype_names))
            proba[int(pred)] = 1.0

        pred_idx = int(np.argmax(proba))
        return {
            "subtype": subtype_names[pred_idx],
            "confidence": float(proba[pred_idx]),
            "probabilities": {subtype_names[i]: float(proba[i]) for i in range(len(subtype_names))},
        }


# Singleton
store = ModelStore()
