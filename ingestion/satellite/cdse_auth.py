"""
CDSE (Copernicus Data Space Ecosystem) OAuth2 token manager.

Uses the Resource Owner Password Credentials (password grant) flow with the
public `cdse-public` client — no client secret required.

Caches the access token module-level and auto-refreshes it 60 s before expiry
so callers never have to worry about token state.

The CDSE *catalog* API (OData metadata search) is publicly accessible without
auth.  Authentication is only needed when downloading actual scene products.
`get_token()` is therefore only called by code that downloads scene data;
the catalog ingestors call it optionally and degrade gracefully if credentials
are absent.
"""

import time

import httpx

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
CLIENT_ID = "cdse-public"

# Module-level token cache (per-process)
_token: str | None = None
_expires_at: float = 0.0


def get_token() -> str:
    """
    Return a valid CDSE access token, refreshing if expired or within 60 s of
    expiry.  Raises httpx.HTTPStatusError on bad credentials.
    """
    global _token, _expires_at

    if _token and time.time() < _expires_at - 60:
        return _token

    if not settings.cdse_username or not settings.cdse_password:
        raise RuntimeError(
            "CDSE credentials not configured (CDSE_USERNAME / CDSE_PASSWORD in .env)"
        )

    resp = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "password",
            "client_id": CLIENT_ID,
            "username": settings.cdse_username,
            "password": settings.cdse_password,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    _token = data["access_token"]
    _expires_at = time.time() + data.get("expires_in", 600)

    logger.info("cdse_token_refreshed", expires_in=data.get("expires_in"))
    return _token


def get_auth_headers() -> dict[str, str]:
    """Convenience wrapper — returns Bearer auth header dict."""
    return {"Authorization": f"Bearer {get_token()}"}
