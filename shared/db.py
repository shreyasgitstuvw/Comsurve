"""
SQLAlchemy engine and session factory.
All services use get_session() — no module touches SQLite directly.

SQLite concurrency notes:
  - WAL mode: readers never block writers; writers never block readers.
  - busy_timeout=10 000 ms: SQLite will retry locked writes for up to 10 s
    before raising OperationalError, preventing spurious failures when the
    scheduler runs multiple jobs concurrently.
  - StaticPool with check_same_thread=False: safe for APScheduler background
    threads sharing one in-process connection.
  - synchronous=NORMAL: good durability/performance balance for WAL.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import Session, sessionmaker

from shared.config import settings
from shared.models import Base

_engine = create_engine(
    f"sqlite:///{settings.db_path}",
    connect_args={
        "check_same_thread": False,
        "timeout": 10,           # seconds SQLite waits on a locked table
    },
    poolclass=pool.StaticPool,   # single shared connection — correct for SQLite
    echo=False,
)


@event.listens_for(_engine, "connect")
def _set_pragmas(dbapi_conn, _):
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA foreign_keys=ON")
    dbapi_conn.execute("PRAGMA synchronous=NORMAL")
    dbapi_conn.execute("PRAGMA busy_timeout=10000")  # ms
    dbapi_conn.execute("PRAGMA cache_size=-32000")   # 32 MB page cache


_SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    Base.metadata.create_all(_engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager yielding a SQLAlchemy session with automatic commit/rollback."""
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
