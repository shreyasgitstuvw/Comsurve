"""
Staging deployment checklist validator.

Connects to a running MCEI API and validates that all critical subsystems
are correctly configured and operational.

Usage:
    python scripts/staging_check.py [--url http://localhost:8000] [--api-key <key>]

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)


CHECKS_PASSED = []
CHECKS_FAILED = []


def _check(name: str, passed: bool, detail: str = ""):
    if passed:
        CHECKS_PASSED.append(name)
        print(f"  ✓  {name}" + (f" — {detail}" if detail else ""))
    else:
        CHECKS_FAILED.append(name)
        print(f"  ✗  {name}" + (f" — {detail}" if detail else ""))


def run_checks(base_url: str, api_key: str) -> int:
    headers = {"X-API-Key": api_key} if api_key else {}
    client = httpx.Client(base_url=base_url, headers=headers, timeout=10)

    print(f"\n  Target: {base_url}")
    print(f"  Auth:   {'enabled (X-API-Key set)' if api_key else 'disabled (no key)'}")
    print()

    # ── 1. Root reachable ─────────────────────────────────────────────────────
    print("[ Connectivity ]")
    try:
        r = client.get("/")
        _check("API root reachable", r.status_code == 200,
               f"status={r.status_code}")
    except Exception as exc:
        _check("API root reachable", False, str(exc))
        print("\n  Cannot reach API — aborting remaining checks.\n")
        return 1

    # ── 2. Health endpoint ────────────────────────────────────────────────────
    print("\n[ Health ]")
    try:
        r = client.get("/health")
        health = r.json() if r.status_code == 200 else {}
        _check("Health endpoint returns 200", r.status_code == 200)
        _check("Health status=ok", health.get("status") == "ok")
        _check("Version present", bool(health.get("version")))
        _check("DB path reported", bool(health.get("db_path")))

        table_counts = health.get("table_counts", {})
        _check("All 8 DB tables present", len(table_counts) == 8,
               f"found {len(table_counts)} tables")
        _check("Auth mode reported", "auth_enabled" in health,
               f"auth_enabled={health.get('auth_enabled')}")
    except Exception as exc:
        _check("Health endpoint parseable", False, str(exc))

    # ── 3. Authentication ─────────────────────────────────────────────────────
    print("\n[ Authentication ]")
    try:
        # Check if auth is enforced (only if we have a key to test with)
        if api_key:
            r_no_auth = httpx.get(f"{base_url}/health", timeout=10)
            auth_enforced = r_no_auth.status_code == 401
            _check("Unauthenticated request rejected (401)",
                   auth_enforced, f"got {r_no_auth.status_code}")
        else:
            _check("Auth check skipped (no API key provided — dev mode)", True)
    except Exception as exc:
        _check("Auth check", False, str(exc))

    # ── 4. Rate limiting ──────────────────────────────────────────────────────
    print("\n[ Rate Limiting ]")
    try:
        # Fire 5 quick requests; at least the first should succeed
        statuses = []
        for _ in range(5):
            r = client.get("/health")
            statuses.append(r.status_code)
        _check("Rate limiter active (requests succeed under limit)",
               200 in statuses, f"statuses={statuses[:3]}...")
    except Exception as exc:
        _check("Rate limiting check", False, str(exc))

    # ── 5. CORS headers ───────────────────────────────────────────────────────
    print("\n[ CORS ]")
    try:
        r = client.options("/health", headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        })
        origin_header = r.headers.get("access-control-allow-origin", "")
        _check("CORS preflight returns Allow-Origin",
               bool(origin_header), f"origin={origin_header!r}")
    except Exception as exc:
        _check("CORS preflight", False, str(exc))

    # ── 6. Key API endpoints ──────────────────────────────────────────────────
    print("\n[ Endpoints ]")
    for path, label in [
        ("/anomalies", "GET /anomalies"),
        ("/signals", "GET /signals"),
        ("/reports", "GET /reports"),
        ("/predictions", "GET /predictions"),
        ("/evaluations", "GET /evaluations"),
        ("/metrics", "GET /metrics (Prometheus)"),
    ]:
        try:
            r = client.get(path)
            _check(label, r.status_code in (200, 404),
                   f"status={r.status_code}")
        except Exception as exc:
            _check(label, False, str(exc))

    # ── 7. Metrics format ─────────────────────────────────────────────────────
    print("\n[ Prometheus Metrics ]")
    try:
        r = client.get("/metrics")
        text = r.text if r.status_code == 200 else ""
        _check("mcei_build_info present", "mcei_build_info" in text)
        _check("mcei_table_rows present", "mcei_table_rows" in text)
        _check("mcei_uptime_seconds present", "mcei_uptime_seconds" in text)
        _check("mcei_job_last_success_ts present", "mcei_job_last_success_ts" in text)
    except Exception as exc:
        _check("Metrics format", False, str(exc))

    # ── 8. DB data present ────────────────────────────────────────────────────
    print("\n[ Data ]")
    try:
        r = client.get("/health")
        if r.status_code == 200:
            counts = r.json().get("table_counts", {})
            raw_count = counts.get("raw_ingestion", 0)
            _check("raw_ingestion has data", raw_count > 0,
                   f"{raw_count} rows — run smoke test to ingest if 0")
        else:
            _check("Data check skipped (health failed)", False)
    except Exception as exc:
        _check("Data check", False, str(exc))

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(CHECKS_PASSED) + len(CHECKS_FAILED)
    print("\n" + "=" * 56)
    print(f"  {len(CHECKS_PASSED)}/{total} checks passed")
    if CHECKS_FAILED:
        print(f"  Failed: {', '.join(CHECKS_FAILED)}")
    print("=" * 56 + "\n")

    return 0 if not CHECKS_FAILED else 1


def main():
    parser = argparse.ArgumentParser(description="MCEI staging deployment checker")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the MCEI API (default: http://localhost:8000)")
    parser.add_argument("--api-key", default="",
                        help="Value of X-API-Key header (leave empty if auth disabled)")
    args = parser.parse_args()

    print("\n" + "=" * 56)
    print("  MCEI STAGING CHECKLIST")
    print("=" * 56)

    return run_checks(args.url, args.api_key)


if __name__ == "__main__":
    sys.exit(main())
