# Architecture

## Overview
Domain-generalized PAM50 breast cancer subtyping (Parker et al. 2009 JCO). Goal: classifier generalizes to unseen cohorts/platforms (domains), not just random split of same cohort.

## Components

### `data_pipeline/`
- `dataset.py`: `ExpressionDataset` (X, y, domains, gene_names, subtype_names), `load_dataset()` for CSV/TSV with `--expression-path --subtype-labels-path --domain-labels-path`, `make_synthetic_dataset()` with injected subtype signal + domain batch shift. Parser hardened for real-world file quirks: auto-sniffed delimiter (comma/tab/semicolon), BOM stripping, header auto-detection with/without leading `sample_id` column, quoted fields, whitespace-trimmed values, comment lines (`#`, `//`), blank lines, trailing commas, gene-names-from-header or `GENE_i` fallback, robust 1-col/2-col label file handling (header variants like `sample`/`id`/`label`/`cohort`/`batch` case-insensitive), sample-id–aware reordering, and explicit `ValueError` with row-level diagnostics (instead of opaque `assert`) for mismatched counts, non-numeric, or non-finite values. `gene_names_path` also handles comments and inline comma lists.
- `splits.py`: `leave_one_domain_out_splits()` (required DG evaluation: train N-1, test held-out, no leakage — verified by disjoint domain sets) and `random_splits()` (optimistic stratified K-fold mixing domains).
- `preprocessing.py`: `DomainStandardizer` — per-domain z-score (fit per domain, transform per domain; unseen domain fallback to global or transductive batch stats; zero-std safeguard `eps` -> 1.0). Simple, well-established DG technique. Supports single-sample inference and mixed batch sizes.
- `cli.py`: real-data entry point; fails gracefully if files absent and prints TCGA/cBioPortal/GDC instructions; raises `FileNotFoundError`/`ValueError` with clear messages for malformed inputs.

### `src/` (model)
- `classifier.py`: regularized multinomial logistic regression (`lbfgs`, `multi_class=multinomial`) and small MLP option.
- `dann.py`: Domain-Adversarial Neural Network (Ganin et al. 2016, JMLR 17:2096-2030, arXiv:1505.07818). Gradient-reversal layer via explicit sign flip, gradual lambda schedule `lambda * (2/(1+exp(-10*p))-1)`, domain classifier over domains. Numpy implementation for sandbox; PyTorch is the production alternative with identical architecture.
- `evaluate.py`: `evaluate_lodo`, `evaluate_random`, `comparison_report` — honest gap reporting (random accuracy minus LODO accuracy).
- `train.py`: `train_and_evaluate`, `train_final_model` (saves `model_artifacts/model.pkl` + `metadata.json` for release gate).

### `backend/` (FastAPI)
- `app.py`: endpoints `GET /health`, `GET /readiness` (honest model-loaded state), `GET /model-info` (503 if not released), `POST /predict` (503 if gate closed, 400 on gene-count mismatch, 422 on empty/non-finite/invalid expression via `field_validator`; strict `math.isfinite` check), `GET /comparison` (serves `outputs/metrics.json` or honest placeholder; returns 500 with detail if JSON is corrupt instead of crashing). Validation errors map to clean `400`/`422` JSON, not `500` stacks. `sample_id` length-capped at 256 chars.
- `model_store.py`: fail-closed `ModelStore.try_load()` — only loads if `MODEL_RELEASE_APPROVED=true` and `APPROVED_ARTIFACT_REVISION` set and artifact exists; otherwise `loaded=false` and `error` string. Predictions abstain with 503 when not loaded; `predict()` validates finite, non-empty, exact `n_genes` match with `ValueError` → 400.
- `auth.py`: Firebase-auth-shaped stub `verify_bearer_token` reading `FIREBASE_SERVICE_ACCOUNT_PATH` JSON; if absent and `REQUIRE_AUTH!=true`, open (sandbox); if required, 401 on missing/invalid bearer. Unit-tested with mocked verifier.

### `frontend/` (React + Vite + TypeScript)
- `src/App.tsx`: scientific dashboard — header with model-ready badge, fail-closed banner matching `/readiness`, predict panel (expression paste, probability bars), comparison panel (random vs LODO accuracy/gap with honest "not yet computed" state), About box with citations. Hardened for a11y/responsive: skip-link, `<label>` + `aria-label`/`aria-describedby` on textarea, `role="alert"` + `aria-live` for errors/banners, `role="progressbar"` with `aria-valuenow` for probability bars, `aria-busy` during fetch, semantic `<main>`/`<section>` with `aria-labelledby`, keyboard focus-visible outlines, 480px/900px breakpoints with header `flex-wrap`.
- Modern dark-header + card design, responsive grid, PAM50 pill colors, WCAG-improved contrast (`#0284c7` on white, `#475569` body copy, pill colors meet AA), focus rings (`outline: 3px solid #0284c7`), `sr-only` helpers.

## DG Methodology Choice

**Required LODO protocol** is the evaluation backbone. **Model-side DG**:
- Primary: per-domain standardization (simple, defensible, verifiably narrows gap on synthetic fixtures).
- Optional DANN: implemented correctly per Ganin et al. 2016; selectable via `method="dann"` in training.

Citations are real and verified (see `docs/data_sources.md`).

## Data Flow
```
TCGA/cBioPortal/GDC (or synthetic fixture)
  -> load_dataset / make_synthetic_dataset
  -> DomainStandardizer (per-domain fit)
  -> classifier (logreg/DANN)
  -> evaluate (random vs LODO) -> comparison_report
  -> train_final_model -> model_artifacts/model.pkl
  -> backend release gate -> /predict
  -> frontend dashboard
```

## Real-Run Procedure (Kaggle/Modal)
```bash
pip install -r requirements.txt
# Download TCGA BRCA + METABRIC from cBioPortal/GDC (see docs/data_sources.md)
python -m data_pipeline.cli --expression-path data/expression.csv --subtype-labels-path data/subtypes.csv --domain-labels-path data/domains.csv --method domain_std --output-dir outputs
MODEL_RELEASE_APPROVED=true APPROVED_ARTIFACT_REVISION=$(cat model_artifacts/metadata.json | python -c "import json,sys;print(json.load(open('model_artifacts/metadata.json'))['method'])")-v1 uvicorn backend.app:app --host 0.0.0.0 --port 8000
cd frontend && npm install && npm run dev
```

## Constraints
- No heavy compute/downloads in sandbox — synthetic fixtures only for CI.
- All metrics/synthetic disclaimers are honest; no fabricated clinical findings.
