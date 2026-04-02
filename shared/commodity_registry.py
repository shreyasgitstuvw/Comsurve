"""
Central registry for commodity definitions, API symbols, and schedule constants.
All ingestors import from here — no hardcoded commodity strings anywhere else.
"""

from typing import Final

# ── Commodity slugs ───────────────────────────────────────────────────────────
COMMODITY_LIST: Final[list[str]] = ["lng", "copper", "soybeans"]

# ── FRED series IDs ───────────────────────────────────────────────────────────
# LNG: use Henry Hub natural gas spot as proxy (EIA provides LNG-specific data)
FRED_SERIES: Final[dict[str, str]] = {
    "lng": "DHHNGSP",        # Henry Hub Natural Gas Spot Price ($/MMBtu)
    "copper": "PCOPPUSDM",   # Global price of Copper (USD/metric ton, monthly)
    "soybeans": "PSOYBUSDM", # Global price of Soybeans (USD/metric ton, monthly)
}

# ── EIA series IDs ────────────────────────────────────────────────────────────
# lng_exports uses the /v2/natural-gas/move/expc/data/ route (seriesid route gives 404)
# ng_storage uses the /v2/seriesid/ route (still valid)
EIA_EXPC_SERIES: Final[dict[str, str]] = {
    "lng_exports": "N9133US2",           # Liquefied US Natural Gas Exports (MMcf, monthly)
}
EIA_SERIES: Final[dict[str, str]] = {
    "ng_storage": "NG.NW2_EPG0_SWO_R48_BCF.W",  # Working gas in storage (Bcf/week)
}

# ── yfinance tickers ──────────────────────────────────────────────────────────
YFINANCE_TICKERS: Final[dict[str, str]] = {
    "lng": "NG=F",      # Natural Gas futures
    "copper": "HG=F",   # Copper futures
    "soybeans": "ZS=F", # Soybean futures
}

# ── Commodities-API symbols ───────────────────────────────────────────────────
COMMODITIES_API_SYMBOLS: Final[dict[str, str]] = {
    "lng": "NG",
    "copper": "XCU",
    "soybeans": "SOYB",
}

# ── News search keywords ──────────────────────────────────────────────────────
NEWS_KEYWORDS: Final[dict[str, list[str]]] = {
    "lng": ["LNG", "liquefied natural gas", "LNG terminal", "LNG export"],
    "copper": ["copper", "copper mine", "copper supply", "COMEX copper"],
    "soybeans": ["soybeans", "soybean", "soy export", "soybean harvest"],
}

# ── Per-commodity anomaly detection thresholds ────────────────────────────────
# Tuned for each commodity's typical volatility regime.
# LNG (Henry Hub) is highly volatile — needs higher Z threshold to avoid noise.
# Copper and Soybeans are more stable with seasonal patterns.
ANOMALY_THRESHOLDS: Final[dict[str, dict[str, float]]] = {
    "lng": {
        "price_zscore": 2.5,         # Henry Hub frequently spikes > 2.0 on weather alone
        "sentiment_spike": 0.55,     # LNG news sentiment tends to be more measured
        "vessel_drop_pct": 0.50,     # 50% vessel count drop
        "idle_moored_ratio": 0.85,   # 85% vessels moored at terminal
        "min_similarity": 0.70,      # Qdrant cosine floor for historical matches
    },
    "copper": {
        "price_zscore": 1.8,         # Copper is sensitive to macro; lower threshold catches early signals
        "sentiment_spike": 0.60,
        "vessel_drop_pct": 0.50,
        "idle_moored_ratio": 0.85,
        "min_similarity": 0.70,
    },
    "soybeans": {
        "price_zscore": 2.0,         # Seasonal baseline; 2.0 is reasonable
        "sentiment_spike": 0.65,     # Soybean news has high baseline positivity (harvest coverage)
        "vessel_drop_pct": 0.45,     # Soybean exports are time-critical; smaller drop is meaningful
        "idle_moored_ratio": 0.80,   # Port congestion is more common in soybean season
        "min_similarity": 0.68,
    },
}

# ── Price display precision per commodity ────────────────────────────────────
# Used by dashboard for formatting and API for rounding.
PRICE_DECIMALS: Final[dict[str, int]] = {
    "lng": 3,        # $/MMBtu — e.g. 2.847
    "copper": 4,     # $/lb futures — e.g. 3.8420
    "soybeans": 2,   # cents/bushel — e.g. 1178.25
}

# ── Ingestion schedules (seconds) — used by APScheduler ──────────────────────
SCHEDULE_MAP: Final[dict[str, int]] = {
    "news_interval_hours": 6,
    "price_realtime_interval_hours": 1,
    "price_historical_cron_hour": 2,   # 02:00 daily
    "ais_interval_minutes": 30,
    "processor_interval_minutes": 30,
    "processor_start_offset_minutes": 30,  # fires 30 min after AIS
    "ai_engine_cron_hour": 3,          # 03:00 daily
    "causality_cron_hour": 4,          # 04:00 daily
}
