"""GET /prices/{commodity} — recent price history from processed_features."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.dependencies import get_db

router = APIRouter(prefix="/prices", tags=["prices"])

WINDOW_MAP = {
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
    "1m": timedelta(days=30),
    "3m": timedelta(days=90),
    "1y": timedelta(days=365),
}
VALID_COMMODITIES = {"lng", "copper", "soybeans"}


@router.get("/{commodity}")
def get_prices(
    commodity: str = Path(description="lng | copper | soybeans"),
    window: str = Query("1m", description="1d | 1w | 1m | 3m | 1y"),
    db: Session = Depends(get_db),
):
    if commodity not in VALID_COMMODITIES:
        raise HTTPException(status_code=400, detail=f"commodity must be one of {VALID_COMMODITIES}")
    if window not in WINDOW_MAP:
        raise HTTPException(status_code=400, detail=f"window must be one of {list(WINDOW_MAP.keys())}")

    since = datetime.utcnow() - WINDOW_MAP[window]

    rows = db.execute(
        text("""
            SELECT ri.timestamp, ri.source, ri.symbol, pf.value
            FROM processed_features pf
            JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
            WHERE ri.commodity = :commodity
              AND pf.feature_type = 'price'
              AND ri.timestamp >= :since
            ORDER BY ri.timestamp ASC
        """),
        {"commodity": commodity, "since": since},
    ).fetchall()

    return {
        "commodity": commodity,
        "window": window,
        "count": len(rows),
        "prices": [
            {
                "timestamp": str(r[0]),
                "source": r[1],
                "symbol": r[2],
                "price": r[3],
            }
            for r in rows
        ],
    }
