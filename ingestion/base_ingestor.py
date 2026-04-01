"""
Abstract base class for all ingestors.
Contract: fetch() returns RawIngestionRecord list → save_to_db() bulk-inserts idempotently → run() orchestrates.

All ingestors inherit this. The idempotency guarantee (INSERT OR IGNORE on unique constraint)
is enforced here so no individual ingestor needs to implement it.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger
from shared.schemas import RawIngestionRecord

logger = get_logger(__name__)


class BaseIngestor(ABC):
    """
    Abstract ingestor. Subclasses implement fetch() only.
    save_to_db() and run() are provided and must not be overridden.
    """

    source: str  # must be set as class attribute in each subclass

    @abstractmethod
    def fetch(self) -> list[RawIngestionRecord]:
        """Fetch data from external source. Returns list of records to store."""
        ...

    def save_to_db(self, records: list[RawIngestionRecord]) -> int:
        """
        Bulk-insert records using INSERT OR IGNORE.
        Returns count of newly inserted rows (ignores duplicates silently).
        """
        if not records:
            return 0

        inserted = 0
        with get_session() as session:
            for rec in records:
                result = session.execute(
                    text("""
                        INSERT OR IGNORE INTO raw_ingestion
                            (source, commodity, symbol, timestamp, data_type, raw_json, ingested_at, processed)
                        VALUES
                            (:source, :commodity, :symbol, :timestamp, :data_type, :raw_json, :ingested_at, 0)
                    """),
                    {
                        "source": rec.source,
                        "commodity": rec.commodity,
                        "symbol": rec.symbol,
                        "timestamp": rec.timestamp,
                        "data_type": rec.data_type,
                        "raw_json": rec.raw_json,
                        "ingested_at": datetime.utcnow(),
                    },
                )
                inserted += result.rowcount

        return inserted

    def run(self) -> dict:
        """
        Orchestrates fetch → save. Returns a summary dict for logging/scheduling.
        Catches and logs exceptions so a single ingestor failure doesn't crash the scheduler.
        """
        start = datetime.utcnow()
        try:
            records = self.fetch()
            inserted = self.save_to_db(records)
            duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            summary = {
                "source": self.source,
                "fetched": len(records),
                "inserted": inserted,
                "duplicates_skipped": len(records) - inserted,
                "duration_ms": duration_ms,
                "status": "ok",
            }
            logger.info("ingestor_run", **summary)
            return summary
        except Exception as exc:
            duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            logger.error("ingestor_run_failed", source=self.source, error=str(exc), duration_ms=duration_ms)
            return {"source": self.source, "status": "error", "error": str(exc)}
