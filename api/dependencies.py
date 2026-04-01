"""FastAPI shared dependencies."""

from typing import Generator

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from shared.config import settings
from shared.db import get_session

# ── API key auth ──────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """
    FastAPI dependency that enforces API key authentication when MCEI_API_KEY
    is configured.  If MCEI_API_KEY is empty (development mode), all requests
    are allowed and a warning is logged at startup via validate_secrets().
    """
    if not settings.mcei_api_key:
        # Auth disabled — development mode, validate_secrets() warns at startup
        return
    if api_key != settings.mcei_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Supply it as the X-API-Key header.",
        )


# ── DB session ────────────────────────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """FastAPI Depends injection: yields a SQLAlchemy session."""
    with get_session() as session:
        yield session
