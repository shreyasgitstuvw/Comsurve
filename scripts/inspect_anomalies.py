"""
CLI tool: pretty-print recent anomaly_events rows.
Shows commodity, type, severity, status, and source count.

Usage:
    python scripts/inspect_anomalies.py
    python scripts/inspect_anomalies.py --commodity lng --limit 50
    python scripts/inspect_anomalies.py --status new
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.db import get_session, init_db
from shared.models import AnomalyEvent


def main():
    parser = argparse.ArgumentParser(description="Inspect MCEI anomaly events")
    parser.add_argument("--commodity", help="Filter by commodity (lng/copper/soybeans)")
    parser.add_argument("--status", help="Filter by status (new/embedding_queued/processed)")
    parser.add_argument("--limit", type=int, default=20, help="Max rows to display (default 20)")
    args = parser.parse_args()

    init_db()

    rows = []
    with get_session() as session:
        q = session.query(AnomalyEvent).order_by(AnomalyEvent.detected_at.desc())
        if args.commodity:
            q = q.filter(AnomalyEvent.commodity == args.commodity)
        if args.status:
            q = q.filter(AnomalyEvent.status == args.status)
        for row in q.limit(args.limit).all():
            try:
                source_count = len(json.loads(row.source_ids))
            except Exception:
                source_count = "?"
            rows.append({
                "id": row.id,
                "commodity": row.commodity,
                "anomaly_type": row.anomaly_type,
                "severity": row.severity,
                "status": row.status,
                "source_count": source_count,
                "detected_at": row.detected_at,
            })

    if not rows:
        print("No anomaly events found.")
        return

    print(f"\n{'ID':>5}  {'Commodity':10}  {'Type':22}  {'Severity':8}  {'Status':18}  {'Sources':7}  {'Detected At'}")
    print("-" * 110)
    for row in rows:
        print(
            f"{row['id']:>5}  {row['commodity']:10}  {row['anomaly_type']:22}  "
            f"{row['severity']:8.3f}  {row['status']:18}  {str(row['source_count']):>7}  "
            f"{row['detected_at'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
    print(f"\nShowing {len(rows)} of most recent anomalies.")
    print()


if __name__ == "__main__":
    main()
