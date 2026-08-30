"""FastAPI app for cancer subtype classification."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.auth import verify_bearer_token
from backend.model_store import store


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.try_load()
    yield


app = FastAPI(
    title="Cancer Subtype Domain-Generalized Classifier",
    description="PAM50 breast cancer subtyping with domain generalization. Model served only if release gate approved.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    expression: list[float] = Field(..., description="Gene expression vector (length must match model n_genes)")
    sample_id: str | None = None


class PredictResponse(BaseModel):
    sample_id: str | None = None
    subtype: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "service": "cancer-subtype-api"}


@app.get("/readiness")
def readiness():
    """Honestly reflects whether a real approved model is loaded."""
    return {
        "ready": store.is_ready(),
        "model_loaded": store.loaded,
        "revision": store.revision,
        "error": store.error,
    }


@app.get("/model-info")
def model_info():
    if not store.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded -- release gate not approved. Set MODEL_RELEASE_APPROVED=true and APPROVED_ARTIFACT_REVISION.")
    art = store.artifact
    assert art is not None
    return {
        "method": art.get("method"),
        "subtypes": art.get("subtype_names"),
        "n_genes": art.get("n_genes"),
        "gene_names": art.get("gene_names", [])[:10],
        "revision": store.revision,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, user=Depends(verify_bearer_token)):
    if not store.is_ready():
        raise HTTPException(status_code=503, detail="Model not released -- prediction unavailable (fail-closed release gate)")
    try:
        result = store.predict(req.expression)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return PredictResponse(sample_id=req.sample_id, **result)


@app.get("/comparison")
def comparison():
    """Return precomputed random-split vs LODO comparison if available.

    Looks for outputs/metrics.json or model_artifacts/metrics.json.
    """
    import json
    from pathlib import Path

    for p in [Path("outputs/metrics.json"), Path("model_artifacts/metrics.json"), Path("outputs/comparison.json")]:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    # Return honest placeholder when not yet computed
    return {
        "message": "No comparison metrics computed yet. Run: python -m src.train or data_pipeline.cli with real data.",
        "available": False,
        "method": None,
    }
