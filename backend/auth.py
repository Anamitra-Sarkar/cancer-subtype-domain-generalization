"""Firebase-auth-shaped auth dependency stub.

Real deployment: set FIREBASE_SERVICE_ACCOUNT_PATH to a JSON service account file.
If no such file is present (sandbox), auth is bypassed unless REQUIRE_AUTH=true.
Unit tests mock the verifier.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, Header, HTTPException


def _load_service_account(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def verify_bearer_token(authorization: Optional[str] = Header(None)) -> dict | None:
    """Verify Firebase-like bearer token.

    Behavior:
    - If FIREBASE_SERVICE_ACCOUNT_PATH is not set or file missing and REQUIRE_AUTH != 'true':
        returns None (auth optional, for sandbox/demo).
    - If REQUIRE_AUTH=true and no valid token: raises 401.
    - If service account file exists: validates token format (stub: checks non-empty bearer).
      Real implementation would call firebase_admin.auth.verify_id_token.
    """
    require_auth = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
    sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")
    sa = _load_service_account(sa_path) if sa_path else None

    if not require_auth and sa is None:
        # Sandbox: auth not required, no service account -> pass through
        return None

    if authorization is None or not authorization.startswith("Bearer "):
        if require_auth:
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        return None

    token = authorization[len("Bearer "):].strip()
    if not token:
        if require_auth:
            raise HTTPException(status_code=401, detail="Empty bearer token")
        return None

    # Stub verification: in real deployment, call Firebase Admin SDK
    # Here we do a minimal check: token must be non-empty and not obviously invalid
    if token == "invalid" or len(token) < 5:
        raise HTTPException(status_code=401, detail="Invalid token")

    # If service account exists, pretend to verify with it
    # (real: firebase_admin.auth.verify_id_token(token))
    return {"uid": "mock-user", "token": token[:8] + "...", "verified": True}


def get_current_user(payload: dict | None = Depends(verify_bearer_token)) -> dict | None:
    return payload
