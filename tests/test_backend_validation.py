"""Backend validation hardening: 400/422 for malformed inputs, fail-closed gates."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from data_pipeline.dataset import make_synthetic_dataset
from src.train import train_final_model


def _get_client_with_model(tmp_path):
    ds = make_synthetic_dataset(n_genes=6, n_per_domain_per_class=5, seed=0)
    train_final_model(ds, method="domain_std", seed=0, out_dir=str(tmp_path / "art"))
    artifact_path = str(tmp_path / "art" / "model.pkl")
    from backend.model_store import ModelStore
    import backend.app as app_module

    orig = app_module.store
    # Set up env & store
    env = {"MODEL_RELEASE_APPROVED": "true", "APPROVED_ARTIFACT_REVISION": "test-rev", "MODEL_ARTIFACT_PATH": artifact_path}
    with patch.dict(os.environ, env, clear=False):
        s = ModelStore()
        ok = s.try_load()
        assert ok
        app_module.store = s
        from backend.app import app
        client = TestClient(app)
        return client, app_module, orig, ds


def test_predict_empty_vector_returns_422(tmp_path):
    client, app_module, orig, ds = _get_client_with_model(tmp_path)
    try:
        r = client.post("/predict", json={"expression": []})
        # Pydantic min_length violation -> 422
        assert r.status_code in (400, 422), r.text
    finally:
        app_module.store = orig


def test_predict_nan_returns_422_or_400(tmp_path):
    client, app_module, orig, ds = _get_client_with_model(tmp_path)
    try:
        # JSON NaN is not valid JSON; send string that parses to NaN via python float('nan') not JSON serializable
        # Instead send a large number that will be parsed as inf, or directly use null handling?
        # We can test the model_store directly for NaN
        from backend.model_store import ModelStore
        # Use string "NaN" won't be parsed as float by pydantic; need to test via direct model_store
        # Test that non-finite via string "inf" style: pydantic may coerce
        r = client.post("/predict", json={"expression": [float("inf")] * 6})
        # json dumps inf may be null or error; if inf is serialized, pydantic validator should catch
        assert r.status_code in (400, 422), f"Expected 400/422 for inf, got {r.status_code}: {r.text}"
        # Also test model_store directly
        with patch.dict(os.environ, {"MODEL_RELEASE_APPROVED": "true", "APPROVED_ARTIFACT_REVISION": "x", "MODEL_ARTIFACT_PATH": str(tmp_path / "art" / "model.pkl")}):
            s = ModelStore()
            s.try_load()
            with pytest.raises(ValueError, match="not finite"):
                s.predict([float("nan")] * 6)
            with pytest.raises(ValueError, match="not finite"):
                s.predict([float("inf")] * 6)
    finally:
        app_module.store = orig


def test_predict_wrong_gene_count_returns_400(tmp_path):
    client, app_module, orig, ds = _get_client_with_model(tmp_path)
    try:
        r = client.post("/predict", json={"expression": [0.1] * 2})
        assert r.status_code == 400
        assert "Expected 6 genes" in r.json()["detail"]
    finally:
        app_module.store = orig


def test_predict_503_when_gate_closed_still_fail_closed(tmp_path):
    from backend.model_store import ModelStore
    import backend.app as app_module
    orig = app_module.store
    try:
        with patch.dict(os.environ, {"MODEL_RELEASE_APPROVED": "false", "APPROVED_ARTIFACT_REVISION": ""}, clear=False):
            s = ModelStore()
            s.try_load()
            app_module.store = s
            from backend.app import app
            with TestClient(app) as client:
                r = client.post("/predict", json={"expression": [0.1] * 6})
                assert r.status_code == 503
                # Wrong length should still be 503 when gate closed (fail-closed before validation)
                r2 = client.post("/predict", json={"expression": [0.1] * 2})
                assert r2.status_code == 503
    finally:
        app_module.store = orig


def test_comparison_handles_corrupt_json(tmp_path):
    import json
    # Create a corrupt metrics file at outputs/metrics.json and expect 500, not crash
    # We need to monkey-patch cwd or Path check; easiest: write to ./outputs/metrics.json temporarily
    out = Path("outputs/metrics.json")
    backup = None
    if out.exists():
        backup = out.read_text()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{ not valid json", encoding="utf-8")
        from backend.app import app
        with TestClient(app) as client:
            r = client.get("/comparison")
            assert r.status_code == 500
            assert "Failed to read" in r.json()["detail"]
    finally:
        if backup is not None:
            out.write_text(backup, encoding="utf-8")
        else:
            if out.exists():
                out.unlink()


def test_model_info_503_when_not_ready():
    from backend.model_store import ModelStore
    import backend.app as app_module
    orig = app_module.store
    try:
        with patch.dict(os.environ, {"MODEL_RELEASE_APPROVED": "false"}, clear=False):
            s = ModelStore()
            s.try_load()
            app_module.store = s
            from backend.app import app
            with TestClient(app) as client:
                r = client.get("/model-info")
                assert r.status_code == 503
    finally:
        app_module.store = orig
