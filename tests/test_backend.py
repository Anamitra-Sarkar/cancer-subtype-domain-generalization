"""Backend API tests: release gate on/off, auth stub."""

import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from data_pipeline.dataset import make_synthetic_dataset
from src.train import train_final_model


@pytest.fixture
def client_no_gate():
    # Ensure gate closed
    with patch.dict(os.environ, {"MODEL_RELEASE_APPROVED": "false", "APPROVED_ARTIFACT_REVISION": ""}, clear=False):
        # Need to reimport to reset store? Instead patch store directly
        from backend.model_store import ModelStore
        import backend.app as app_module
        # Reset store
        app_module.store = ModelStore()
        app_module.store.try_load()
        from backend.app import app
        with TestClient(app) as c:
            yield c


def test_health():
    from backend.app import app
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_readiness_gate_closed():
    with patch.dict(os.environ, {"MODEL_RELEASE_APPROVED": "false"}, clear=False):
        from backend.model_store import ModelStore
        import backend.app as app_module
        orig = app_module.store
        tmp_store = ModelStore()
        tmp_store.try_load()
        app_module.store = tmp_store
        try:
            from backend.app import app
            with TestClient(app) as client:
                r = client.get("/readiness")
                assert r.status_code == 200
                data = r.json()
                assert data["ready"] is False
                assert data["model_loaded"] is False
                assert "MODEL_RELEASE_APPROVED" in data["error"]
        finally:
            app_module.store = orig


def test_predict_503_when_gate_closed():
    with patch.dict(os.environ, {"MODEL_RELEASE_APPROVED": "false", "APPROVED_ARTIFACT_REVISION": ""}, clear=False):
        from backend.model_store import ModelStore
        import backend.app as app_module
        orig = app_module.store
        s = ModelStore()
        s.try_load()
        app_module.store = s
        try:
            from backend.app import app
            with TestClient(app) as client:
                r = client.post("/predict", json={"expression": [0.1] * 20})
                assert r.status_code == 503
        finally:
            app_module.store = orig


def test_predict_success_when_gate_open(tmp_path):
    # Train a tiny model and save artifact
    ds = make_synthetic_dataset(n_genes=8, n_per_domain_per_class=5, seed=0)
    train_final_model(ds, method="domain_std", seed=0, out_dir=str(tmp_path / "art"))
    artifact_path = str(tmp_path / "art" / "model.pkl")
    assert Path(artifact_path).exists()

    from backend.model_store import ModelStore
    import backend.app as app_module
    orig = app_module.store
    with patch.dict(os.environ, {"MODEL_RELEASE_APPROVED": "true", "APPROVED_ARTIFACT_REVISION": "test-rev-1", "MODEL_ARTIFACT_PATH": artifact_path}, clear=False):
        s = ModelStore()
        ok = s.try_load()
        assert ok is True
        assert s.is_ready()
        app_module.store = s
        try:
            from backend.app import app
            with TestClient(app) as client:
                # Readiness should be ready
                r = client.get("/readiness")
                assert r.json()["ready"] is True
                # Predict
                expr = ds.X[0].tolist()  # length 8
                r = client.post("/predict", json={"expression": expr})
                assert r.status_code == 200, r.text
                data = r.json()
                assert "subtype" in data
                assert "confidence" in data
                assert 0 <= data["confidence"] <= 1
                assert data["subtype"] in ds.subtype_names
                # Wrong length -> 400
                r = client.post("/predict", json={"expression": [0.1] * 3})
                assert r.status_code == 400
        finally:
            app_module.store = orig


def test_auth_stub_no_auth_required():
    # When REQUIRE_AUTH != true and no service account, verify_bearer_token returns None (pass-through)
    with patch.dict(os.environ, {"REQUIRE_AUTH": "false", "FIREBASE_SERVICE_ACCOUNT_PATH": "/nonexistent/path.json"}, clear=False):
        from backend.auth import verify_bearer_token
        result = verify_bearer_token(authorization=None)
        assert result is None


def test_auth_stub_requires_token():
    with patch.dict(os.environ, {"REQUIRE_AUTH": "true", "FIREBASE_SERVICE_ACCOUNT_PATH": ""}, clear=False):
        from backend.auth import verify_bearer_token
        from fastapi import HTTPException
        # Missing header should raise 401 when REQUIRE_AUTH=true
        try:
            verify_bearer_token(authorization=None)
            assert False, "Should have raised 401"
        except HTTPException as e:
            assert e.status_code == 401
        # Valid token passes
        # Need FIREBASE_SERVICE_ACCOUNT_PATH to trigger require path? With empty path and REQUIRE_AUTH true, it still requires token
        # Actually with sa None but require_auth true, missing auth raises; valid token still needs to be checked
        # Create dummy sa file to exercise that path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"type":"service_account"}')
            f.flush()
            with patch.dict(os.environ, {"FIREBASE_SERVICE_ACCOUNT_PATH": f.name}, clear=False):
                res = verify_bearer_token(authorization="Bearer valid-token-12345")
                assert res is not None
                assert res["verified"] is True
            Path(f.name).unlink()


def test_comparison_endpoint():
    from backend.app import app
    with TestClient(app) as c:
        r = c.get("/comparison")
        assert r.status_code == 200
        data = r.json()
        # Either has metrics or honest placeholder
        assert isinstance(data, dict)
