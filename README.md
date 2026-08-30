# Cancer Subtype Domain-Generalized Classification

Domain-generalized learning for PAM50 breast cancer molecular subtype classification (Parker et al. 2009 JCO) with leave-one-domain-out evaluation and per-domain standardization + DANN (Ganin et al. 2016).

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -v
python -m data_pipeline.cli --expression-path data/expression.csv --subtype-labels-path data/subtypes.csv --domain-labels-path data/domains.csv --method domain_std --output-dir outputs
uvicorn backend.app:app --reload --port 8000
# frontend
cd frontend && npm install && npm run dev
```

Release gate: `MODEL_RELEASE_APPROVED=true APPROVED_ARTIFACT_REVISION=v1 uvicorn backend.app:app`

See `docs/architecture.md` and `docs/data_sources.md`.
