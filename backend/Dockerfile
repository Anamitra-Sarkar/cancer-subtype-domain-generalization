# ---- builder: install deps ----
FROM python:3.11-slim AS builder
WORKDIR /app
# system deps for scipy/sklearn build if needed
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime ----
FROM python:3.11-slim AS runtime
WORKDIR /app
# copy installed deps
COPY --from=builder /install /usr/local
# copy application code (repo root context expects backend/, data_pipeline/, src/)
# When building with context = repo root, adjust; when context = backend, only backend is available.
# Support both: try repo-root layout first, fallback to backend-only.
COPY requirements.txt ./requirements.txt
COPY backend/ ./backend/
COPY data_pipeline/ ./data_pipeline/
COPY src/ ./src/
# ensure model_artifacts dir exists (empty until release gate approved)
RUN mkdir -p model_artifacts outputs
# Fail-closed gate: no default approval; env vars must be explicitly set to enable model.
# MODEL_RELEASE_APPROVED defaults to "false" in code; APPROVED_ARTIFACT_REVISION defaults to "".
ENV MODEL_RELEASE_APPROVED=false
ENV APPROVED_ARTIFACT_REVISION=""
ENV MODEL_ARTIFACT_PATH=model_artifacts/model.pkl
ENV PORT=8000
EXPOSE 8000
# Run uvicorn on the real entrypoint backend.app:app
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
