"""API-key auth dependency (DECISIONS.md §14).

Single bearer token compared against `settings.api_key` in constant time.
Attached at router scope in `app/main.py` — `/health` is naturally exempt
because it lives directly on the FastAPI app, not on a protected router.
"""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.settings import settings

# auto_error=False so missing-header returns None and we control the
# response shape — FastAPI's default would emit 403, but DECISIONS.md §11
# specifies 401 for missing/bad API keys.
_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Reject unless `Authorization: Bearer <key>` matches `API_KEY`.

    Same 401 for missing-header, wrong-key, and unset-server-key paths —
    operators distinguish via deployment state, clients don't need to.
    """
    expected = settings.api_key
    presented = creds.credentials if creds is not None else ""
    if not expected or not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
