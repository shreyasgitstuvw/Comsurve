"""
Structured JSON logging configuration with optional file rotation.

All services call get_logger(__name__) to get a configured logger.

Log output:
  - Always: stdout (JSON, one line per event)
  - If LOG_FILE is set: rotating file handler (10 MB max, 5 backups kept)

Environment variables:
  LOG_LEVEL   DEBUG | INFO | WARNING | ERROR (default: INFO)
  LOG_FILE    Path to the rotating log file (default: empty = stdout only)
"""

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog

from shared.config import settings

_configured = False


def configure_logging() -> None:
    """
    Call once at service startup to configure structlog + stdlib logging.

    Idempotent — subsequent calls are no-ops so get_logger() callers don't
    need to guard against double-configuration.
    """
    global _configured
    if _configured:
        return
    _configured = True

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ── Root handler: stdout (always) ─────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(log_level)

    if not root.handlers:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(stdout_handler)

    # ── Optional rotating file handler ────────────────────────────────────────
    log_file = getattr(settings, "log_file", "")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,              # keep 5 rotated files (.1 … .5)
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)
