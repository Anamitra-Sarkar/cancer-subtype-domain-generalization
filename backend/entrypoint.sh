#!/bin/sh
# Downloads the approved model.pkl from HF Hub before starting the server,
# if MODEL_RELEASE_APPROVED + APPROVED_ARTIFACT_REVISION are set. If not
# approved, starts anyway -- backend/model_store.py correctly fails closed.
set -e

if [ "$MODEL_RELEASE_APPROVED" = "true" ] && [ -n "$APPROVED_ARTIFACT_REVISION" ]; then
  echo "[entrypoint] Downloading approved artifact revision $APPROVED_ARTIFACT_REVISION"
  python3 -c "
import os
import shutil
from huggingface_hub import hf_hub_download

rev = os.environ['APPROVED_ARTIFACT_REVISION']
token = os.environ.get('HF_TOKEN')
repo_id = 'bhumika-tewari-282006/cancer-subtype-domain-generalization'
dest = os.environ.get('MODEL_ARTIFACT_PATH', 'model_artifacts/model.pkl')
os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)

p = hf_hub_download(repo_id=repo_id, filename='model.pkl', repo_type='model', revision=rev, token=token)
shutil.copy(p, dest)
print(f'[entrypoint] model.pkl -> {dest}')
"
else
  echo "[entrypoint] No approved release configured; starting in fail-closed abstention mode."
fi

exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-8000}"
