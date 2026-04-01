"""
Satellite ingestion runner — runs Sentinel-1 and Sentinel-2 ingestors
sequentially and returns a combined summary dict.

Called by the scheduler every 6 hours (satellite revisit periods are 5–12 days,
so hourly polling would be wasteful; 6 h catches new acquisitions promptly).
"""

from ingestion.satellite.sentinel1_ingestor import Sentinel1Ingestor
from ingestion.satellite.sentinel2_ingestor import Sentinel2Ingestor
from shared.db import init_db
from shared.logger import get_logger

logger = get_logger(__name__)


def run() -> dict:
    s1 = Sentinel1Ingestor(lookback_days=7).run()
    s2 = Sentinel2Ingestor(lookback_days=14, max_cloud_pct=95.0).run()

    summary = {
        "sentinel1": s1,
        "sentinel2": s2,
        "total_inserted": (s1.get("inserted", 0) + s2.get("inserted", 0)),
    }
    logger.info("satellite_run_complete", **summary)
    return summary


if __name__ == "__main__":
    init_db()
    result = run()
    print(result)
