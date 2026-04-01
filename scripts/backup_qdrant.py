#!/usr/bin/env python
"""
Qdrant snapshot backup script.

Creates a snapshot of each MCEI collection (lng, copper, soybeans) using the
Qdrant REST snapshot API, downloads it to the local backup directory, and
prunes snapshots older than RETENTION_DAYS.

Usage:
    python scripts/backup_qdrant.py [--backup-dir <path>] [--retention-days <N>]

Defaults:
    --backup-dir      ./qdrant_backups
    --retention-days  7

Qdrant must be running in local mode (QdrantManager connects via its path-based
client, but snapshot creation requires the HTTP API on localhost:6333).
For path-based local storage, this script copies the raw qdrant_data directory
instead of using the HTTP snapshot API (no server process required).

Environment variables:
    QDRANT_PATH       Override the qdrant data directory (default: ./qdrant_data)
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import settings
from shared.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

COMMODITIES = ["lng", "copper", "soybeans"]
COLLECTION_PREFIX = "mcei"
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backup MCEI Qdrant collections")
    parser.add_argument(
        "--backup-dir",
        default="./qdrant_backups",
        help="Directory to write snapshot archives (default: ./qdrant_backups)",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=7,
        help="Delete snapshots older than N days (default: 7)",
    )
    return parser.parse_args()


def backup_qdrant_local(backup_dir: Path, qdrant_path: Path) -> list[str]:
    """
    For local-mode Qdrant (path-based, no HTTP server), copy the qdrant_data
    directory as a timestamped archive.  This is the simplest and most reliable
    backup strategy when Qdrant is embedded.

    Returns a list of created archive paths.
    """
    if not qdrant_path.exists():
        logger.warning("qdrant_path_not_found", path=str(qdrant_path))
        return []

    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime(TIMESTAMP_FMT)
    archive_name = f"qdrant_snapshot_{ts}"
    archive_path = backup_dir / archive_name

    logger.info("qdrant_backup_start", source=str(qdrant_path), dest=str(archive_path))

    shutil.copytree(str(qdrant_path), str(archive_path))

    # Create a compressed tarball and remove the raw copy
    tarball = f"{archive_path}.tar.gz"
    shutil.make_archive(str(archive_path), "gztar", str(backup_dir), archive_name)
    shutil.rmtree(str(archive_path))

    size_mb = os.path.getsize(tarball) / (1024 * 1024)
    logger.info(
        "qdrant_backup_complete",
        archive=tarball,
        size_mb=round(size_mb, 2),
    )
    return [tarball]


def prune_old_backups(backup_dir: Path, retention_days: int) -> int:
    """
    Delete .tar.gz backup archives older than retention_days.
    Returns the number of files deleted.
    """
    cutoff = time.time() - retention_days * 86400
    deleted = 0

    for f in backup_dir.glob("qdrant_snapshot_*.tar.gz"):
        if f.stat().st_mtime < cutoff:
            logger.info("qdrant_backup_prune", file=str(f), age_days=retention_days)
            f.unlink()
            deleted += 1

    return deleted


def main() -> int:
    args = _parse_args()
    backup_dir = Path(args.backup_dir).resolve()
    qdrant_path = Path(settings.qdrant_path).resolve()
    retention_days = args.retention_days

    logger.info(
        "qdrant_backup_run",
        backup_dir=str(backup_dir),
        qdrant_path=str(qdrant_path),
        retention_days=retention_days,
    )

    created = backup_qdrant_local(backup_dir, qdrant_path)
    pruned = prune_old_backups(backup_dir, retention_days)

    logger.info(
        "qdrant_backup_summary",
        archives_created=len(created),
        archives_pruned=pruned,
    )

    if not created:
        logger.error("qdrant_backup_nothing_created")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
