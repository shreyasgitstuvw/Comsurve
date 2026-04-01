"""
Pytest fixtures for MCEI test suite.

Provides:
  - db_session  : in-memory SQLite session (all tables created fresh per test)
  - patch_db    : monkeypatches get_session in all relevant modules
  - mock_gemini : a MagicMock replacement for GeminiClient
  - api_client  : TestClient with API auth disabled
"""

import json
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import Session, sessionmaker

from shared.models import Base

# All module paths that import get_session as a local name.
# When patching, we must patch the name at its point-of-use (not just in shared.db)
# because `from shared.db import get_session` creates an independent local binding.
_GET_SESSION_TARGETS = [
    "shared.db.get_session",
    "processor.anomaly_detector.get_session",
    "processor.ais_anomaly_detector.get_session",
    "processor.satellite_anomaly_detector.get_session",
    "api.dependencies.get_session",
]


# ── In-memory database ────────────────────────────────────────────────────────

@pytest.fixture()
def engine():
    """Fresh in-memory SQLite engine with all MCEI tables."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=pool.StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def db_session(engine):
    """Transactional SQLAlchemy session — rolled back after each test."""
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture()
def patch_db(db_session):
    """
    Patches get_session at every point-of-use so all production code
    receives the in-memory test session, regardless of how they imported it.
    """
    @contextmanager
    def _fake_session():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    with ExitStack() as stack:
        for target in _GET_SESSION_TARGETS:
            try:
                stack.enter_context(patch(target, side_effect=_fake_session))
            except (AttributeError, ModuleNotFoundError):
                pass  # module not imported yet or path doesn't exist
        yield db_session


# ── Mock Gemini client ────────────────────────────────────────────────────────

@pytest.fixture()
def mock_gemini():
    """
    Returns a MagicMock GeminiClient.

    Default behaviours (override per-test as needed):
      .embed(text)          → list of 3072 zeros
      .generate_text(prompt) → JSON string with a plausible structure
    """
    client = MagicMock()
    client.embed.return_value = [0.0] * 3072
    client.generate_text.return_value = json.dumps({
        "cause_category": "mock_cause",
        "confidence": 0.9,
        "price_impact_pct": 2.5,
        "summary": "Mock Gemini response for testing.",
    })
    return client


# ── FastAPI test client (auth disabled) ───────────────────────────────────────

@pytest.fixture()
def api_client(engine):
    """
    FastAPI TestClient with:
      - API key auth disabled (MCEI_API_KEY left empty)
      - shared.db.get_session patched to the in-memory engine
    """
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=engine)

    @contextmanager
    def _fake_session():
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    with patch("shared.db.get_session", side_effect=_fake_session):
        with patch("shared.config.settings") as mock_settings:
            mock_settings.mcei_api_key = ""
            mock_settings.cors_origins_list.return_value = ["http://localhost:8501"]
            mock_settings.db_path = ":memory:"

            # Import after patching so module-level calls use mock settings
            from api.main import app
            client = TestClient(app, raise_server_exceptions=True)
            yield client
