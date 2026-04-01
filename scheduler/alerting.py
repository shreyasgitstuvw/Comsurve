"""
MCEI alert dispatcher.

Sends webhook notifications when:
  - A job fails for the Nth consecutive time (CONSECUTIVE_FAILURE_THRESHOLD)
  - Gemini quota is exhausted (detected from error string)
  - Any job has been silent (no successful run) for more than SILENCE_HOURS

Configure MCEI_ALERT_WEBHOOK_URL in .env to enable.
If the URL is empty, alerts are only logged (no HTTP call).

Webhook payload (POST, application/json):
  {
    "level": "error" | "warning",
    "event": "<event_type>",
    "job": "<job_name>",
    "message": "<human readable>",
    "consecutive_failures": <int>,
    "timestamp": "<ISO 8601>"
  }
"""

import json
from datetime import datetime, timedelta

import httpx

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

CONSECUTIVE_FAILURE_THRESHOLD = 3   # alert after this many back-to-back failures
SILENCE_HOURS = 26                  # alert if a job hasn't succeeded in this long

# In-process failure counters — reset to 0 on success
_failure_counts: dict[str, int] = {}


def _post_webhook(payload: dict) -> None:
    url = settings.mcei_alert_webhook_url
    if not url:
        logger.warning("alert_no_webhook_configured", **payload)
        return
    try:
        httpx.post(url, json=payload, timeout=10)
        logger.info("alert_webhook_sent", event=payload.get("event"), job=payload.get("job"))
    except Exception as exc:
        logger.error("alert_webhook_failed", error=str(exc), payload=payload)


def record_job_success(job_name: str) -> None:
    """Call after a job completes successfully to reset its failure counter."""
    _failure_counts[job_name] = 0


def record_job_failure(job_name: str, error: str) -> None:
    """
    Call after a job fails.  Fires a webhook when consecutive failures
    reach CONSECUTIVE_FAILURE_THRESHOLD, or immediately if Gemini quota
    is exhausted (since that blocks the entire AI pipeline).
    """
    count = _failure_counts.get(job_name, 0) + 1
    _failure_counts[job_name] = count

    # Immediate alert for Gemini quota exhaustion — impacts all AI jobs
    if "RESOURCE_EXHAUSTED" in error and "limit: 0" in error:
        _post_webhook({
            "level": "warning",
            "event": "gemini_quota_exhausted",
            "job": job_name,
            "message": (
                "Gemini daily free-tier quota exhausted. AI pipeline (embedding, "
                "causality, prediction, evaluation) will use fallback models until "
                "quota resets at midnight Pacific time."
            ),
            "consecutive_failures": count,
            "timestamp": datetime.utcnow().isoformat(),
        })
        return

    if count >= CONSECUTIVE_FAILURE_THRESHOLD:
        _post_webhook({
            "level": "error",
            "event": "consecutive_job_failures",
            "job": job_name,
            "message": (
                f"Job '{job_name}' has failed {count} consecutive times. "
                f"Latest error: {error[:300]}"
            ),
            "consecutive_failures": count,
            "timestamp": datetime.utcnow().isoformat(),
        })


def check_job_silence(recent_jobs: list[dict]) -> None:
    """
    Called periodically (e.g., by the scheduler health check job) to detect
    jobs that haven't succeeded recently.  `recent_jobs` is the job_runs
    table snapshot from get_recent_runs().
    """
    silence_cutoff = datetime.utcnow() - timedelta(hours=SILENCE_HOURS)

    # Build last-success map
    last_success: dict[str, datetime] = {}
    for row in recent_jobs:
        if row.get("status") == "ok" and row.get("finished_at"):
            try:
                ts = datetime.fromisoformat(row["finished_at"])
            except ValueError:
                continue
            job = row["job_name"]
            if job not in last_success or ts > last_success[job]:
                last_success[job] = ts

    expected_jobs = [
        "news", "price_realtime", "price_historical", "ais",
        "processor", "ai_engine", "causality", "prediction", "evaluation",
    ]
    for job in expected_jobs:
        last = last_success.get(job)
        if last is None or last < silence_cutoff:
            _post_webhook({
                "level": "warning",
                "event": "job_silence",
                "job": job,
                "message": (
                    f"Job '{job}' has not completed successfully in the last "
                    f"{SILENCE_HOURS} hours. Last success: "
                    f"{last.isoformat() if last else 'never'}."
                ),
                "consecutive_failures": _failure_counts.get(job, 0),
                "timestamp": datetime.utcnow().isoformat(),
            })
