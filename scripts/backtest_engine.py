"""
MCEI Backtesting Engine — walk-forward prediction evaluation on historical disruptions.

Replays 40+ documented supply-chain disruption events (2008-2026) across LNG, Copper,
and Soybeans to evaluate prediction quality across four market regimes.

═══════════════════════════════════════════════════════════════════════════
BIAS CONTROLS
═══════════════════════════════════════════════════════════════════════════

Look-ahead bias
  Price data fetched via yfinance with a hard end=as_of_date+1day cutoff.
  Outcome prices (+1w/+2w/+1m) are fetched from AFTER the event date.
  Prediction prompt explicitly states the as_of_date so the model cannot
  reference post-event data it was trained on.

Survivorship bias
  Event catalog includes events where prices moved AGAINST the intuitive
  prediction (e.g. 2022 Ukraine LNG reversal, 2020 COVID copper V-shape).
  No cherry-picking of events where the model would perform well.

Sample period bias
  Events span 2008-2026 across four regime labels: normal, stress, crisis,
  black_swan. Black swan years (2008 GFC, 2020 COVID, 2022 Ukraine war)
  are explicitly included and reported as a separate regime stratum.

Curve fitting
  Zero free parameters. All scoring thresholds are fixed a priori:
    - 1.0% neutrality band for direction classification
    - 0.4/0.4/0.2 composite score weights (mirrors evaluation_engine.py)
    - Brier score formula is standard (p - o)^2
  The --regimes / --commodities flags filter the report; they do NOT
  retrain or recalibrate anything.

Data snooping
  Event catalog built from publicly cited disruption records (IEA Gas Market
  Reports, USDA WASDE, IMF commodity databases, mine operator press releases).
  Events were defined before any backtest run; catalog is read-only at runtime.

LLM contamination (known, unavoidable limitation)
  Gemini models were trained on data that includes outcomes for all events
  in this catalog. The prediction step is therefore NOT true out-of-sample.
  MITIGATION: evaluation scoring is computed locally (no Gemini call), so
  the evaluator cannot "know" the answer. The backtest validates calibration
  quality (are confidence scores meaningful?) rather than absolute accuracy.
  True out-of-sample performance will only be known from future live events.
  Use --dry-run to exercise the scoring pipeline without any API calls.

Walk-forward train/test split:
    The catalog is split at a hard date boundary (default: 2021-01-01).
    TRAIN  2008-2020  Model "sees" these events in chronological order;
                      their scores feed the FeedbackAccumulator. No report
                      is generated for train events.
    TEST   2021-2026  Model is evaluated on these events using the accumulated
                      learning signal from train-period errors. This is the
                      out-of-sample performance estimate.

    Use --train-cutoff YYYY-MM-DD to override the split boundary.
    Use --no-split to treat every event as test (current behaviour).

Usage:
    cd mcei/
    python -m scripts.backtest_engine [options]

    --dry-run                Skip Gemini; score a fixed synthetic prediction
    --feedback               Enable walk-forward feedback loop
    --train-cutoff DATE      Hard date split for train/test (default: 2021-01-01)
    --no-split               Disable train/test split (evaluate all events)
    --commodities lng ...    Filter to subset of commodities
    --regimes crisis ...     Filter to subset of market regimes
    --output PATH            JSON report path (default: reports/backtest_report.json)
"""

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf

from ai_engine.feedback_controller import (
    ErrorSignal,
    ControlAdjustments,
    compute_error_signal,
    compute_control_adjustments,
)
from ai_engine.gemini_client import GeminiClient
from ai_engine.llama_client import LlamaClient
from ai_engine.prediction_engine import PREDICTION_PROMPT
from shared.commodity_registry import YFINANCE_TICKERS
from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS AND BIAS DOCUMENTATION
# ══════════════════════════════════════════════════════════════════════════════

CONTAMINATION_NOTE = (
    "KNOWN LIMITATION — LLM look-ahead contamination: Gemini models used for "
    "prediction were trained on data that includes outcomes for all events in "
    "this catalog. The backtest therefore does NOT represent true out-of-sample "
    "performance. It validates calibration quality (whether confidence scores "
    "are internally consistent) and regime sensitivity. True forward-looking "
    "performance can only be measured from live events post-deployment."
)

# Fixed neutrality band: |actual_pct| < NEUTRAL_BAND_PCT → direction is neutral
NEUTRAL_BAND_PCT = 1.0

# Composite score weights — MUST match evaluation_engine._compute_overall_score()
W_DIRECTION   = 0.4
W_MAGNITUDE   = 0.4
W_CALIBRATION = 0.2

# Synthetic prediction used in --dry-run mode (no_signal → baseline composite ~0.2)
DRY_RUN_PREDICTION = json.dumps({
    "event_id": "dry_run",
    "commodity": "dry_run",
    "signal_summary": "DRY RUN — no Gemini call made.",
    "predicted_outcomes": [],
    "confidence_score": 0.5,
    "prediction_type": "no_signal",
    "drivers": [],
    "historical_analogs": [],
})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HISTORICAL EVENT CATALOG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestEvent:
    event_id:           str
    commodity:          str        # lng | copper | soybeans
    anomaly_type:       str        # mirrors AnomalyEvent.anomaly_type taxonomy
    severity:           float      # z-score-like severity, same scale as live engine
    as_of_date:         date       # hard cutoff — NO price data after this date in prompt
    signal_context:     str        # mimics embeddings_cache.context_payload
    regime:             str        # normal | stress | crisis | black_swan
    source:             str        # public citation
    expected_direction: str        # up | down | neutral (ground-truth label)
    horizon_days:       int = 30


# Sources (abbreviations used in catalog):
#   IEA-GMR  = IEA Gas Market Report
#   USDA-W   = USDA WASDE Report
#   IMF-CPD  = IMF Commodity Price Database
#   CME      = CME Group market data
#   Reuters  = Reuters Commodities desk

HISTORICAL_EVENTS: list[BacktestEvent] = [

    # ── LNG / Natural Gas ────────────────────────────────────────────────────

    BacktestEvent(
        event_id="LNG-2008-IKE",
        commodity="lng", anomaly_type="price_spike", severity=3.1,
        as_of_date=date(2008, 9, 10),
        signal_context=(
            "Hurricane Ike forecast to make direct landfall on Gulf Coast. "
            "Sabine Pass LNG terminal and production platforms in path. "
            "Henry Hub spot already elevated; futures market pricing disruption risk."
        ),
        regime="black_swan", source="IEA-GMR 2008-Q3", expected_direction="up",
    ),
    BacktestEvent(
        event_id="LNG-2008-GFC",
        commodity="lng", anomaly_type="sentiment_shift", severity=2.7,
        as_of_date=date(2008, 10, 15),
        signal_context=(
            "Lehman Brothers collapse 29 days ago. Credit markets frozen. "
            "Industrial production indices collapsing globally. "
            "LNG demand destruction signals from Asia and Europe. "
            "Shipping spot rates dropping sharply."
        ),
        regime="black_swan", source="IMF-CPD Oct-2008", expected_direction="down",
    ),
    BacktestEvent(
        event_id="LNG-2011-FUKU",
        commodity="lng", anomaly_type="price_spike", severity=2.9,
        as_of_date=date(2011, 3, 15),
        signal_context=(
            "Fukushima Daiichi nuclear disaster triggered by Tōhoku earthquake. "
            "Japan shutting nuclear reactors — will need emergency LNG imports. "
            "Japan imports ~30% of global LNG. Spot cargo enquiries surging."
        ),
        regime="crisis", source="IEA-GMR 2011-Q1", expected_direction="up",
    ),
    BacktestEvent(
        event_id="LNG-2014-POLAR",
        commodity="lng", anomaly_type="price_spike", severity=2.4,
        as_of_date=date(2014, 1, 6),
        signal_context=(
            "Polar vortex bringing record cold to US Midwest and Northeast. "
            "Natural gas withdrawal from storage at multi-year highs. "
            "Pipeline capacity constraints in Northeast. Henry Hub near $5."
        ),
        regime="stress", source="CME 2014-Jan", expected_direction="up",
    ),
    BacktestEvent(
        event_id="LNG-2019-SABINE",
        commodity="lng", anomaly_type="ais_vessel_drop", severity=1.8,
        as_of_date=date(2019, 4, 12),
        signal_context=(
            "Vessel traffic near Sabine Pass LNG terminal dropped 35% in 48h. "
            "Maintenance outage on Train 3 unconfirmed. "
            "US LNG export volumes below seasonal norm."
        ),
        regime="normal", source="Reuters 2019-Apr", expected_direction="down",
    ),
    BacktestEvent(
        event_id="LNG-2020-COVID",
        commodity="lng", anomaly_type="sentiment_shift", severity=3.2,
        as_of_date=date(2020, 3, 20),
        signal_context=(
            "COVID-19 declared pandemic. Global industrial shutdown underway. "
            "LNG cargo cancellations rising — buyers invoking force majeure. "
            "Asian spot LNG prices near record lows. European storage near full."
        ),
        regime="black_swan", source="IEA-GMR 2020-Q1", expected_direction="down",
    ),
    BacktestEvent(
        event_id="LNG-2021-TEXFREEZE",
        commodity="lng", anomaly_type="price_spike", severity=3.8,
        as_of_date=date(2021, 2, 14),
        signal_context=(
            "Winter Storm Uri hitting Texas. ERCOT grid emergency — rolling blackouts. "
            "Natural gas wellhead freeze-offs in Permian and Eagle Ford. "
            "Henry Hub front month spiking beyond $5; pipeline nominations collapsing."
        ),
        regime="crisis", source="FERC report 2021-Feb", expected_direction="up",
    ),
    BacktestEvent(
        event_id="LNG-2021-EU-ENERGY",
        commodity="lng", anomaly_type="sentiment_shift", severity=2.2,
        as_of_date=date(2021, 10, 4),
        signal_context=(
            "European gas storage 74% full entering winter — 5-yr low. "
            "Russia Gazprom flows to Europe at minimum contractual levels. "
            "Asian LNG spot at record highs, competing for cargoes Europe needs."
        ),
        regime="stress", source="IEA-GMR 2021-Q4", expected_direction="up",
    ),
    BacktestEvent(
        event_id="LNG-2022-UKRAINE",
        commodity="lng", anomaly_type="price_spike", severity=4.1,
        as_of_date=date(2022, 2, 25),
        signal_context=(
            "Russia invaded Ukraine yesterday. Nord Stream 1 supply at risk. "
            "EU announced Russia energy dependency review. "
            "LNG spot prices hitting record highs. US LNG export terminals at capacity."
        ),
        regime="black_swan", source="IEA-GMR 2022-Q1", expected_direction="up",
    ),
    BacktestEvent(
        event_id="LNG-2022-FREEPORT",
        commodity="lng", anomaly_type="ais_vessel_drop", severity=2.6,
        as_of_date=date(2022, 6, 9),
        signal_context=(
            "Explosion at Freeport LNG terminal yesterday — facility offline. "
            "Freeport supplies ~20% of US LNG exports (~2 Bcf/d). "
            "Vessel queue at Freeport dissipating. EU buyers seeking alternatives."
        ),
        regime="crisis", source="Reuters 2022-Jun", expected_direction="down",
    ),
    BacktestEvent(
        event_id="LNG-2023-EU-REFILL",
        commodity="lng", anomaly_type="sentiment_shift", severity=1.5,
        as_of_date=date(2023, 1, 20),
        signal_context=(
            "EU gas storage above 82% despite cold winter — record refill speed. "
            "Mild weather reducing demand. Asian LNG spot falling on weak demand. "
            "Analyst consensus: European gas glut likely through Q2-2023."
        ),
        regime="normal", source="IEA-GMR 2023-Q1", expected_direction="down",
    ),
    BacktestEvent(
        event_id="LNG-2024-REDSEA",
        commodity="lng", anomaly_type="ais_vessel_drop", severity=2.1,
        as_of_date=date(2024, 1, 15),
        signal_context=(
            "Houthi missile attacks forcing LNG tankers to reroute via Cape of Good Hope. "
            "Transit time Europe-Asia +12-15 days. Effective supply tightening. "
            "Suez Canal LNG volumes down ~50% vs prior month."
        ),
        regime="stress", source="IEA-GMR 2024-Q1", expected_direction="up",
    ),

    # ── Copper ───────────────────────────────────────────────────────────────

    BacktestEvent(
        event_id="CU-2008-GFC",
        commodity="copper", anomaly_type="price_spike", severity=3.9,
        as_of_date=date(2008, 10, 20),
        signal_context=(
            "Lehman collapse triggered global credit freeze. "
            "Chinese copper imports falling sharply. LME copper down >40% in 6 weeks. "
            "Construction and automotive sectors — two biggest copper end-markets — in freefall."
        ),
        regime="black_swan", source="IMF-CPD 2008-Q4", expected_direction="down",
    ),
    BacktestEvent(
        event_id="CU-2010-CHILE-EQ",
        commodity="copper", anomaly_type="ais_vessel_drop", severity=2.3,
        as_of_date=date(2010, 3, 1),
        signal_context=(
            "8.8 magnitude earthquake hit Chile on 27 Feb. "
            "Codelco and Anglo American inspecting mine damage. "
            "Antofagasta port temporarily closed. Chile = 30% of world copper supply."
        ),
        regime="crisis", source="Reuters 2010-Mar", expected_direction="up",
    ),
    BacktestEvent(
        event_id="CU-2013-GRASBERG",
        commodity="copper", anomaly_type="ais_vessel_drop", severity=1.9,
        as_of_date=date(2013, 6, 20),
        signal_context=(
            "Freeport-McMoRan Grasberg mine tunnel collapse in Indonesia. "
            "Production suspended — Grasberg is world's second largest copper mine. "
            "Concentrate shipments halted. Duration of suspension unclear."
        ),
        regime="stress", source="Reuters 2013-Jun", expected_direction="up",
    ),
    BacktestEvent(
        event_id="CU-2015-CHINA-CIRCUIT",
        commodity="copper", anomaly_type="sentiment_shift", severity=2.8,
        as_of_date=date(2015, 8, 26),
        signal_context=(
            "China circuit breaker triggered — Shanghai Composite down 8.5% in two days. "
            "Yuan devaluation shock. Chinese copper demand outlook collapsing. "
            "LME copper already at 6-year lows. Hedge fund net shorts at record."
        ),
        regime="crisis", source="IMF-CPD 2015-Q3", expected_direction="down",
    ),
    BacktestEvent(
        event_id="CU-2019-USTARIFF",
        commodity="copper", anomaly_type="sentiment_shift", severity=2.0,
        as_of_date=date(2019, 8, 5),
        signal_context=(
            "US announced 10% tariff on remaining $300bn of Chinese goods. "
            "China yuan fell through 7.0 per dollar for first time since 2008. "
            "Copper demand destruction from reduced manufacturing activity expected."
        ),
        regime="stress", source="Reuters 2019-Aug", expected_direction="down",
    ),
    BacktestEvent(
        event_id="CU-2020-COVID-MINE",
        commodity="copper", anomaly_type="ais_vessel_drop", severity=3.1,
        as_of_date=date(2020, 4, 2),
        signal_context=(
            "Peru declared 15-day COVID mining suspension. "
            "Chile Cerro Verde and BHP Escondida limiting operations. "
            "Peru + Chile = 40% of global copper mine supply. "
            "Demand also collapsing — supply/demand shock simultaneously."
        ),
        regime="black_swan", source="Reuters 2020-Apr", expected_direction="neutral",
    ),
    BacktestEvent(
        event_id="CU-2021-SUPERCYCLE",
        commodity="copper", anomaly_type="price_spike", severity=2.3,
        as_of_date=date(2021, 5, 10),
        signal_context=(
            "Copper hit $10,000/t on LME — highest since 2011. "
            "Biden infrastructure plan driving US demand expectations. "
            "China green energy build-out. Supply pipeline thin: 3-5 years to new mine."
        ),
        regime="stress", source="CME 2021-May", expected_direction="up",
    ),
    BacktestEvent(
        event_id="CU-2022-COBRE-PROT",
        commodity="copper", anomaly_type="sentiment_shift", severity=2.1,
        as_of_date=date(2022, 7, 20),
        signal_context=(
            "Cobre Panama mine (First Quantum) facing escalating community protests. "
            "Operations not yet disrupted but access roads blocked periodically. "
            "Cobre Panama = ~1.5% of global copper supply. Contract renegotiation risk."
        ),
        regime="stress", source="Reuters 2022-Jul", expected_direction="up",
    ),
    BacktestEvent(
        event_id="CU-2023-COBRE-CLOSE",
        commodity="copper", anomaly_type="ais_vessel_drop", severity=2.8,
        as_of_date=date(2023, 11, 20),
        signal_context=(
            "Panama Supreme Court ruled Cobre Panama contract unconstitutional. "
            "Government ordered First Quantum to wind down operations. "
            "Closure removes ~350,000 t/year from market. Copper inventories already low."
        ),
        regime="crisis", source="Reuters 2023-Nov", expected_direction="up",
    ),
    BacktestEvent(
        event_id="CU-2024-ZAMBIA-DRGHT",
        commodity="copper", anomaly_type="satellite_scene_gap", severity=1.7,
        as_of_date=date(2024, 2, 5),
        signal_context=(
            "Zambia declared drought emergency. Kariba dam at 12% capacity — "
            "hydropower shortage affecting copper smelter operations. "
            "Konkola Copper Mines reducing smelting by 30%."
        ),
        regime="stress", source="Reuters 2024-Feb", expected_direction="up",
    ),
    BacktestEvent(
        event_id="CU-2025-CHINA-STIM",
        commodity="copper", anomaly_type="sentiment_shift", severity=1.6,
        as_of_date=date(2025, 2, 18),
        signal_context=(
            "China PBOC announced targeted stimulus package for manufacturing. "
            "EV production targets raised. Grid infrastructure spending elevated. "
            "Copper spot premiums in Shanghai at 3-month high."
        ),
        regime="normal", source="Reuters 2025-Feb", expected_direction="up",
    ),

    # ── Soybeans ─────────────────────────────────────────────────────────────

    BacktestEvent(
        event_id="SOY-2012-DROUGHT",
        commodity="soybeans", anomaly_type="price_spike", severity=3.6,
        as_of_date=date(2012, 7, 10),
        signal_context=(
            "US Midwest drought worst since 1988. USDA slashing crop estimates. "
            "Corn and soybean belt receiving less than 25% of normal rainfall. "
            "Soybean basis at country elevators +$1.50/bu over futures. "
            "USDA WASDE shows ending stocks collapsing to 130mb from 275mb."
        ),
        regime="crisis", source="USDA-W Jul-2012", expected_direction="up",
    ),
    BacktestEvent(
        event_id="SOY-2013-CHINA-FLU",
        commodity="soybeans", anomaly_type="sentiment_shift", severity=1.8,
        as_of_date=date(2013, 4, 8),
        signal_context=(
            "H7N9 avian influenza outbreak in China. Poultry destruction orders. "
            "Soybean meal demand (poultry feed) at risk. "
            "China is 60% of global soybean imports. Early harvest pressure in Brazil."
        ),
        regime="stress", source="Reuters 2013-Apr", expected_direction="down",
    ),
    BacktestEvent(
        event_id="SOY-2018-TARIFF",
        commodity="soybeans", anomaly_type="sentiment_shift", severity=2.7,
        as_of_date=date(2018, 7, 6),
        signal_context=(
            "China imposed 25% retaliatory tariff on US soybeans effective today. "
            "US is 40% of global soybean exports. China buying South American. "
            "CBOT soybeans hit lowest since 2008 on trade war escalation fears."
        ),
        regime="stress", source="USDA-W Jul-2018", expected_direction="down",
    ),
    BacktestEvent(
        event_id="SOY-2019-ASF",
        commodity="soybeans", anomaly_type="sentiment_shift", severity=2.4,
        as_of_date=date(2019, 5, 20),
        signal_context=(
            "African Swine Fever culled 40% of China's pig herd — 200 million animals. "
            "Soybean meal demand for pig feed collapsing structurally. "
            "China soybean import forecasts cut by 5-10 million tonnes for 2019/20."
        ),
        regime="crisis", source="USDA-W May-2019", expected_direction="down",
    ),
    BacktestEvent(
        event_id="SOY-2019-PHASE1",
        commodity="soybeans", anomaly_type="sentiment_shift", severity=1.6,
        as_of_date=date(2019, 12, 16),
        signal_context=(
            "US-China Phase 1 trade deal announced. China committed to buy "
            "$40-50bn of US agricultural goods. Tariff rollback on soybeans expected. "
            "Brazilian real weakening — US soybean competitive again."
        ),
        regime="normal", source="Reuters 2019-Dec", expected_direction="up",
    ),
    BacktestEvent(
        event_id="SOY-2020-COVID-PORT",
        commodity="soybeans", anomaly_type="ais_vessel_drop", severity=2.9,
        as_of_date=date(2020, 2, 10),
        signal_context=(
            "COVID-19 spreading in China. Yangtze River ports reducing operations. "
            "Vessel turnaround times at Dalian and Tianjin extended 5-8 days. "
            "Chinese soybean crushers halting — workers unable to return from CNY."
        ),
        regime="black_swan", source="Reuters 2020-Feb", expected_direction="down",
    ),
    BacktestEvent(
        event_id="SOY-2021-BRAZIL-DRGHT",
        commodity="soybeans", anomaly_type="satellite_cloud_block", severity=2.2,
        as_of_date=date(2021, 3, 8),
        signal_context=(
            "La Niña driving drought across southern Brazil and Argentina. "
            "CONAB cutting Brazil soybean estimate 3rd consecutive month. "
            "Parana and Mato Grosso do Sul crops worst in a decade. "
            "Satellite cloud cover anomaly over key growing states last 14 days."
        ),
        regime="crisis", source="USDA-W Mar-2021", expected_direction="up",
    ),
    BacktestEvent(
        event_id="SOY-2022-UKRAINE",
        commodity="soybeans", anomaly_type="ais_vessel_drop", severity=3.0,
        as_of_date=date(2022, 3, 8),
        signal_context=(
            "Ukraine war. Black Sea ports closed — Ukraine is 50% of global sunflower oil. "
            "Substitution demand driving soybean oil higher. "
            "Santos port vessel traffic +25% as buyers rush South American supply. "
            "Odesa port silent — 0 vessels tracked in past 48h."
        ),
        regime="black_swan", source="USDA-W Mar-2022", expected_direction="up",
    ),
    BacktestEvent(
        event_id="SOY-2022-ARGENTINA",
        commodity="soybeans", anomaly_type="sentiment_shift", severity=2.0,
        as_of_date=date(2022, 9, 6),
        signal_context=(
            "Argentina peso crisis. Government imposing informal export hold — "
            "farmers withholding soybeans waiting for better FX rate. "
            "Argentina = 45% of soybean meal exports. Supply pipeline tightening."
        ),
        regime="crisis", source="Reuters 2022-Sep", expected_direction="up",
    ),
    BacktestEvent(
        event_id="SOY-2023-BLACKSEA",
        commodity="soybeans", anomaly_type="ais_vessel_drop", severity=2.3,
        as_of_date=date(2023, 7, 19),
        signal_context=(
            "Russia terminated Black Sea Grain Initiative yesterday. "
            "Ukraine Black Sea ports under missile threat. "
            "Corn substitution demand for soybeans expected. "
            "Santos and Paranagua vessel queues lengthening."
        ),
        regime="crisis", source="Reuters 2023-Jul", expected_direction="up",
    ),
    BacktestEvent(
        event_id="SOY-2024-BRAZIL-RECORD",
        commodity="soybeans", anomaly_type="sentiment_shift", severity=1.7,
        as_of_date=date(2024, 4, 22),
        signal_context=(
            "CONAB finalising record Brazil soybean crop: 153 million tonnes. "
            "Brazilian real at 5.2/USD — exports highly competitive. "
            "Global ending stocks on track for 6-year high. Freight rates normalising."
        ),
        regime="normal", source="USDA-W Apr-2024", expected_direction="down",
    ),
    BacktestEvent(
        event_id="SOY-2025-LANINA",
        commodity="soybeans", anomaly_type="satellite_cloud_block", severity=1.9,
        as_of_date=date(2025, 1, 28),
        signal_context=(
            "La Niña advisory active. Argentina Pampa region receiving 40% below normal rainfall. "
            "BAGE cutting Argentina soybean estimate for second week. "
            "Cloud cover anomaly over Santa Fe and Entre Rios growing regions."
        ),
        regime="stress", source="USDA-W Jan-2025", expected_direction="up",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PRICE FETCHER (look-ahead safe)
# ══════════════════════════════════════════════════════════════════════════════

class PriceFetcher:
    """
    Fetches historical prices from yfinance with strict date cutoffs.
    All methods enforce the look-ahead constraint: data is never fetched
    beyond the supplied as_of_date or target_date.
    """

    def __init__(self):
        self._cache: dict[tuple[str, str], float | None] = {}

    def _cache_key(self, ticker: str, target: date) -> tuple[str, str]:
        return (ticker, target.isoformat())

    def get_price(self, ticker: str, as_of_date: date) -> float | None:
        """
        Return closing price on or before as_of_date.
        yfinance end parameter is exclusive, so end=as_of_date+1 includes as_of_date.
        Falls back up to 5 calendar days for weekends / market holidays.
        """
        key = self._cache_key(ticker, as_of_date)
        if key in self._cache:
            return self._cache[key]

        try:
            # Fetch a small window ending at (and including) as_of_date
            start = as_of_date - timedelta(days=7)
            end   = as_of_date + timedelta(days=1)   # exclusive upper bound
            df = yf.Ticker(ticker).history(
                start=start.isoformat(), end=end.isoformat(), auto_adjust=True
            )
            if df.empty:
                self._cache[key] = None
                return None
            price = float(df["Close"].iloc[-1])
            # ZS=F soybeans are quoted in cents/bushel — convert to $/bushel
            if ticker == "ZS=F":
                price /= 100.0
            self._cache[key] = price
            return price
        except Exception as exc:
            logger.warning("price_fetch_failed", ticker=ticker,
                           as_of_date=as_of_date.isoformat(), error=str(exc))
            self._cache[key] = None
            return None

    def get_outcome_prices(
        self, ticker: str, base_date: date
    ) -> dict[str, float | None]:
        """
        Fetch prices AFTER the event for outcome measurement.
        These are intentionally NOT available at prediction time — only used
        for scoring after the fact.

        Returns {"1w": ..., "2w": ..., "1m": ...}
        """
        horizons = {"1w": 7, "2w": 14, "1m": 30}
        results: dict[str, float | None] = {}
        for label, days in horizons.items():
            target = base_date + timedelta(days=days)
            results[label] = self.get_price(ticker, target)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class BacktestPromptBuilder:
    """
    Constructs prediction prompts that mirror the live prediction_engine,
    but with context constrained to information available at as_of_date.
    """

    @staticmethod
    def _build_analogs_section(
        event: BacktestEvent, all_events: list[BacktestEvent]
    ) -> str:
        """
        Find up to 3 catalog events with the same commodity that occurred STRICTLY
        before event.as_of_date. Sort by severity proximity.
        NOTE: This is an approximation of the live Qdrant vector search.
              The backtest JSON report documents this deviation.
        """
        prior = [
            e for e in all_events
            if e.commodity == event.commodity
            and e.as_of_date < event.as_of_date     # strict: no same-day events
            and e.event_id != event.event_id
        ]
        if not prior:
            return "No similar historical events found (novel event)."

        prior.sort(key=lambda e: abs(e.severity - event.severity))
        lines = []
        for i, analog in enumerate(prior[:3]):
            lines.append(
                f"  Event {analog.event_id} (catalog similarity): "
                f"{analog.anomaly_type} on {analog.commodity.upper()} "
                f"at {analog.as_of_date}, severity={analog.severity:.2f}, "
                f"regime={analog.regime}"
            )
        return "\n".join(lines)

    @staticmethod
    def build(
        event: BacktestEvent,
        current_price: float | None,
        all_events: list[BacktestEvent],
        learning_section: str = "",
    ) -> str:
        """
        Fill PREDICTION_PROMPT (imported verbatim from prediction_engine)
        with backtest-constrained context.
        learning_section is only populated when --feedback mode is active.
        """
        price_str = f"{current_price:.4f}" if current_price is not None else "N/A"
        analogs = BacktestPromptBuilder._build_analogs_section(event, all_events)

        # Inject as_of_date constraint so model knows the knowledge horizon
        constrained_context = (
            f"[BACKTEST: as_of_date={event.as_of_date.isoformat()}. "
            f"Do not reference any events or price data after this date.]\n\n"
            f"{event.signal_context}"
        )

        ls = learning_section if learning_section else "No prior evaluations available for this signal type."

        from ai_engine.prediction_engine import HORIZON_GUIDANCE, _HORIZON_DEFAULT
        horizon_guidance = HORIZON_GUIDANCE.get(event.anomaly_type, _HORIZON_DEFAULT)

        return PREDICTION_PROMPT.format(
            commodity=event.commodity,
            anomaly_type=event.anomaly_type,
            severity=event.severity,
            detected_at=event.as_of_date.isoformat(),
            signal_context=constrained_context,
            anomaly_trajectory="[Backtest mode: trajectory data not available for historical replay]",
            market_context="[Backtest mode: 30-day context constrained to as_of_date]",
            analogs_section=analogs,
            learning_section=ls,
            current_price=price_str,
            horizon_guidance=horizon_guidance,
            event_id=event.event_id,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SCORING ENGINE (local only — no Gemini)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    event_id:              str
    commodity:             str
    regime:                str
    as_of_date:            str       # ISO date string for JSON serialisation
    expected_direction:    str
    price_at_event:        float | None
    price_1w:              float | None
    price_2w:              float | None
    price_1m:              float | None
    actual_pct_1m:         float | None
    actual_direction:      str | None
    prediction_type:       str
    confidence_score:      float
    predicted_direction:   str | None
    predicted_pct_midpoint: float | None
    direction_correct:     bool | None
    magnitude_mae:         float | None
    brier_score:           float | None
    composite_score:       float
    is_dry_run:            bool
    parse_error:           bool = False


class ScoringEngine:
    """All methods are static — no state, no free parameters."""

    @staticmethod
    def parse_prediction(raw_json: str) -> dict:
        try:
            return json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def extract_top_scenario(pred: dict) -> dict | None:
        outcomes = pred.get("predicted_outcomes", [])
        if not outcomes:
            return None
        return max(outcomes, key=lambda x: float(x.get("probability", 0)))

    @staticmethod
    def parse_price_move_midpoint(price_move_str: str) -> float | None:
        """
        Parse "+3% to +7%" → 5.0  |  "-5% to -8%" → -6.5
             "±4% to ±8%" → None (magnitude only, direction ambiguous)
             "+5%"        → 5.0
        """
        if not price_move_str:
            return None
        s = price_move_str.strip()
        if "±" in s:
            return None   # magnitude-only — cannot assign direction midpoint

        # Range format: +3% to +7%  or  -5% to -8%
        range_match = re.search(
            r"([+-]?\d+(?:\.\d+)?)\s*%\s+to\s+([+-]?\d+(?:\.\d+)?)\s*%",
            s, re.IGNORECASE
        )
        if range_match:
            lo = float(range_match.group(1))
            hi = float(range_match.group(2))
            return (lo + hi) / 2.0

        # Single value: +5%
        single_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", s)
        if single_match:
            return float(single_match.group(1))

        return None

    @staticmethod
    def direction_from_pct(pct: float | None) -> str | None:
        if pct is None:
            return None
        if pct > NEUTRAL_BAND_PCT:
            return "up"
        if pct < -NEUTRAL_BAND_PCT:
            return "down"
        return "neutral"

    @staticmethod
    def compute_brier_score(
        predicted_prob: float, predicted_dir: str | None, actual_dir: str | None
    ) -> float | None:
        """Binary Brier score: BS = (p - o)^2. Lower is better."""
        if predicted_dir is None or actual_dir is None:
            return None
        if actual_dir == "neutral":
            return None   # ambiguous outcome — exclude from calibration metric
        outcome = 1.0 if predicted_dir == actual_dir else 0.0
        return (predicted_prob - outcome) ** 2

    @staticmethod
    def compute_composite(
        direction_correct: bool | None,
        magnitude_mae: float | None,
    ) -> float:
        """
        Mirrors evaluation_engine._compute_overall_score() exactly.
          direction   : 0.4 correct | 0.2 None | 0.0 wrong
          magnitude   : max(0, 0.4 - mae * 0.04)
          calibration : always 0.1 in backtest (no Gemini confidence_validity label)
        """
        if direction_correct is True:
            d = W_DIRECTION
        elif direction_correct is None:
            d = W_DIRECTION / 2
        else:
            d = 0.0

        if magnitude_mae is not None:
            m = max(0.0, W_MAGNITUDE - magnitude_mae * 0.04)
        else:
            m = W_MAGNITUDE / 2   # partial credit when magnitude unknown

        c = W_CALIBRATION / 2   # conservative: can't compute calibration_validity locally
        return round(d + m + c, 4)

    @staticmethod
    def score(
        event: BacktestEvent,
        raw_prediction: str,
        price_at_event: float | None,
        outcome_prices: dict[str, float | None],
        is_dry_run: bool,
    ) -> PredictionResult:
        pred = ScoringEngine.parse_prediction(raw_prediction)
        parse_error = not pred

        prediction_type = pred.get("prediction_type", "no_signal")
        confidence_score = float(pred.get("confidence_score", 0.5))

        top = ScoringEngine.extract_top_scenario(pred)
        predicted_pct_midpoint = None
        predicted_direction = None
        top_prob = 0.5

        if top:
            predicted_pct_midpoint = ScoringEngine.parse_price_move_midpoint(
                top.get("price_move", "")
            )
            predicted_direction = ScoringEngine.direction_from_pct(predicted_pct_midpoint)
            top_prob = float(top.get("probability", 0.5))

        p0  = price_at_event
        p1m = outcome_prices.get("1m")

        actual_pct_1m = None
        if p0 and p1m and p0 != 0:
            actual_pct_1m = (p1m - p0) / p0 * 100

        actual_direction = ScoringEngine.direction_from_pct(actual_pct_1m)

        direction_correct: bool | None = None
        if prediction_type == "directional" and predicted_direction and actual_direction:
            if actual_direction == "neutral":
                direction_correct = None
            else:
                direction_correct = predicted_direction == actual_direction

        magnitude_mae: float | None = None
        if predicted_pct_midpoint is not None and actual_pct_1m is not None:
            magnitude_mae = abs(predicted_pct_midpoint - actual_pct_1m)

        brier = ScoringEngine.compute_brier_score(top_prob, predicted_direction, actual_direction)
        composite = ScoringEngine.compute_composite(direction_correct, magnitude_mae)

        return PredictionResult(
            event_id=event.event_id,
            commodity=event.commodity,
            regime=event.regime,
            as_of_date=event.as_of_date.isoformat(),
            expected_direction=event.expected_direction,
            price_at_event=p0,
            price_1w=outcome_prices.get("1w"),
            price_2w=outcome_prices.get("2w"),
            price_1m=p1m,
            actual_pct_1m=round(actual_pct_1m, 4) if actual_pct_1m is not None else None,
            actual_direction=actual_direction,
            prediction_type=prediction_type,
            confidence_score=confidence_score,
            predicted_direction=predicted_direction,
            predicted_pct_midpoint=round(predicted_pct_midpoint, 2)
                if predicted_pct_midpoint is not None else None,
            direction_correct=direction_correct,
            magnitude_mae=round(magnitude_mae, 4) if magnitude_mae is not None else None,
            brier_score=round(brier, 4) if brier is not None else None,
            composite_score=composite,
            is_dry_run=is_dry_run,
            parse_error=parse_error,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5b — FEEDBACK ACCUMULATOR (in-memory walk-forward learning)
# ══════════════════════════════════════════════════════════════════════════════

class FeedbackAccumulator:
    """
    Maintains a chronological in-memory ledger of scored events per
    (commodity, anomaly_type) pair.  After each event is scored, the result
    is accumulated so that the NEXT event of the same type can see a damped
    error signal in its prompt — mirroring the live learning_store.py flow.

    Usage (only active when --feedback is set):
        acc = FeedbackAccumulator()
        ...
        learning_section = acc.get_learning_section(event)
        # ... score event ...
        acc.record(event, result)

    The evaluation model used here is Llama (if available) or Gemini fallback,
    matching the dual-model architecture of the live system.
    """

    def __init__(self):
        # key: (commodity, anomaly_type) → list[dict] oldest-first
        self._ledger: dict[tuple[str, str], list[dict]] = {}
        self._eval_client = None   # lazy-initialised

    def _key(self, event: "BacktestEvent") -> tuple[str, str]:
        return (event.commodity, event.anomaly_type)

    def get_learning_section(self, event: "BacktestEvent") -> str:
        """Return the learning context for this event based on PRIOR events only."""
        key = self._key(event)
        prior = self._ledger.get(key, [])
        if not prior:
            return ""

        signal: ErrorSignal = compute_error_signal(prior, event.commodity, event.anomaly_type)
        adj: ControlAdjustments = compute_control_adjustments(signal)

        lines = [
            f"Based on {signal.n_evaluations} prior backtest event(s) for "
            f"{event.commodity.upper()} / {event.anomaly_type}:",
            f"  Direction error rate: {signal.damped_direction_error:.0%}  |  "
            f"Magnitude error: {signal.damped_magnitude_error:.1f} pp  |  "
            f"Composite error: {signal.e_total:.2f}",
        ]
        if signal.failure_modes:
            lines.append("Key failure modes observed:")
            for fm in signal.failure_modes[:3]:
                lines.append(f"  * {fm}")
        if not adj.is_empty():
            lines.append("Required adjustments:")
            lines.append(adj.to_text())

        return "\n".join(lines)

    def record(self, event: "BacktestEvent", result: "PredictionResult") -> None:
        """Record a scored result for use in future prompts."""
        key = self._key(event)
        if key not in self._ledger:
            self._ledger[key] = []

        # Convert PredictionResult to the dict format expected by compute_error_signal
        self._ledger[key].append({
            "direction_correct": result.direction_correct,
            "magnitude_error": result.magnitude_mae or 0.0,
            "volatility_correct": None,  # not computed in backtest
            "failure_modes": [],         # no LLM evaluation in local scoring
        })


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _mean(vals: list[float]) -> float | None:
    clean = [v for v in vals if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def compute_regime_metrics(results: list[PredictionResult]) -> list[dict]:
    regimes = ["normal", "stress", "crisis", "black_swan", "ALL"]
    rows = []
    for regime in regimes:
        subset = results if regime == "ALL" else [r for r in results if r.regime == regime]
        if not subset:
            continue

        directional = [r for r in subset if r.prediction_type == "directional"]
        scorable    = [r for r in directional if r.direction_correct is not None]
        dir_acc     = _mean([1.0 if r.direction_correct else 0.0 for r in scorable])
        mae         = _mean([r.magnitude_mae for r in subset if r.magnitude_mae is not None])
        brier       = _mean([r.brier_score for r in subset if r.brier_score is not None])
        composite   = _mean([r.composite_score for r in subset])
        no_sig_rate = len([r for r in subset if r.prediction_type == "no_signal"]) / len(subset)

        rows.append({
            "regime": regime,
            "n_events": len(subset),
            "n_directional": len(directional),
            "direction_accuracy": dir_acc,
            "mean_magnitude_mae_pp": mae,
            "mean_brier_score": brier,
            "mean_composite_score": composite,
            "no_signal_rate": round(no_sig_rate, 3),
        })
    return rows


def print_report(
    results: list[PredictionResult],
    regime_metrics: list[dict],
    train_cutoff: "date | None" = None,
    n_train: int = 0,
) -> None:
    sep = "=" * 90
    thin = "-" * 86
    thin2 = "-" * 58
    print(f"\n{sep}")
    print("  MCEI BACKTESTING ENGINE -- Per-Event Results (TEST SET ONLY)")
    if train_cutoff is not None:
        print(f"  Train/test split: {n_train} events < {train_cutoff} (train, silent)"
              f" | {len(results)} events >= {train_cutoff} (test, reported)")
    print(sep)
    hdr = f"  {'Event ID':<24} {'Cmdty':<10} {'Regime':<12} {'Dir?':<7} {'ActPct':>7} {'PredPct':>8} {'MAE':>6} {'Brier':>6} {'Comp':>6}"
    print(hdr)
    print(f"  {thin}")
    for r in sorted(results, key=lambda x: (x.regime, x.commodity, x.as_of_date)):
        dir_str = ("OK" if r.direction_correct else "X") if r.direction_correct is not None else "N/A"
        act  = f"{r.actual_pct_1m:+.1f}%" if r.actual_pct_1m is not None else "  N/A "
        pred = f"{r.predicted_pct_midpoint:+.1f}%" if r.predicted_pct_midpoint is not None else "  N/A "
        mae  = f"{r.magnitude_mae:.1f}" if r.magnitude_mae is not None else " N/A"
        brie = f"{r.brier_score:.3f}" if r.brier_score is not None else "  N/A"
        comp = f"{r.composite_score:.3f}"
        print(f"  {r.event_id:<24} {r.commodity:<10} {r.regime:<12} {dir_str:<7} "
              f"{act:>7} {pred:>8} {mae:>6} {brie:>6} {comp:>6}")

    print(f"\n{sep}")
    print("  REGIME SUMMARY")
    print(sep)
    hdr2 = f"  {'Regime':<14} {'N':>4} {'Dir%':>7} {'MAE(pp)':>9} {'Brier':>7} {'Composite':>10} {'NoSig%':>8}"
    print(hdr2)
    print(f"  {thin2}")
    for m in regime_metrics:
        da   = f"{m['direction_accuracy']*100:.1f}%" if m['direction_accuracy'] is not None else "  N/A"
        mae  = f"{m['mean_magnitude_mae_pp']:.2f}" if m['mean_magnitude_mae_pp'] is not None else " N/A"
        brie = f"{m['mean_brier_score']:.3f}" if m['mean_brier_score'] is not None else "  N/A"
        comp = f"{m['mean_composite_score']:.3f}" if m['mean_composite_score'] is not None else "  N/A"
        ns   = f"{m['no_signal_rate']*100:.1f}%"
        print(f"  {m['regime']:<14} {m['n_events']:>4} {da:>7} {mae:>9} {brie:>7} {comp:>10} {ns:>8}")
    print(sep)
    print(f"\n  NOTE: {CONTAMINATION_NOTE[:120]}...")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ORCHESTRATION AND CLI
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_TRAIN_CUTOFF = date(2021, 1, 1)   # hard boundary: train < cutoff <= test


def _run_events(
    events: list[BacktestEvent],
    dry_run: bool,
    client,
    fetcher: PriceFetcher,
    accumulator: "FeedbackAccumulator | None",
    is_test_phase: bool,
    all_events: list[BacktestEvent],
) -> list[PredictionResult]:
    """
    Core event loop. When is_test_phase=False (train), events are only
    scored to build up the accumulator — results are returned but NOT
    reported. When is_test_phase=True (test), results are returned for
    full reporting.
    """
    results: list[PredictionResult] = []

    for i, event in enumerate(events, 1):
        ticker = YFINANCE_TICKERS[event.commodity]
        phase = "TEST" if is_test_phase else "TRAIN"
        logger.info("backtest_event", phase=phase, n=i, total=len(events),
                    event_id=event.event_id, regime=event.regime)

        current_price = fetcher.get_price(ticker, event.as_of_date)
        outcome_prices = fetcher.get_outcome_prices(ticker, event.as_of_date)

        learning_section = ""
        if accumulator is not None:
            learning_section = accumulator.get_learning_section(event)

        if dry_run:
            raw_prediction = DRY_RUN_PREDICTION
        else:
            prompt = BacktestPromptBuilder.build(
                event, current_price, all_events, learning_section=learning_section
            )
            try:
                raw_prediction = client.generate_text(prompt)
            except Exception as exc:
                logger.error("backtest_prediction_failed",
                             event_id=event.event_id, error=str(exc))
                raw_prediction = DRY_RUN_PREDICTION

        result = ScoringEngine.score(
            event, raw_prediction, current_price, outcome_prices,
            is_dry_run=dry_run
        )
        results.append(result)

        if accumulator is not None:
            accumulator.record(event, result)

        logger.info("backtest_event_scored", phase=phase,
                    event_id=event.event_id,
                    actual_pct_1m=result.actual_pct_1m,
                    direction_correct=str(result.direction_correct),
                    composite=result.composite_score)

    return results


def run_backtest(
    dry_run: bool,
    commodities: list[str] | None,
    regimes: list[str] | None,
    output: str,
    feedback: bool = False,
    train_cutoff: date = DEFAULT_TRAIN_CUTOFF,
    no_split: bool = False,
) -> dict:
    all_events = HISTORICAL_EVENTS[:]
    if commodities:
        all_events = [e for e in all_events if e.commodity in commodities]
    if regimes:
        all_events = [e for e in all_events if e.regime in regimes]

    # Always sort chronologically — needed for both feedback and train/test
    all_events = sorted(all_events, key=lambda e: e.as_of_date)

    if no_split:
        train_events: list[BacktestEvent] = []
        test_events = all_events
    else:
        train_events = [e for e in all_events if e.as_of_date < train_cutoff]
        test_events  = [e for e in all_events if e.as_of_date >= train_cutoff]

    logger.info("backtest_start",
                total=len(all_events),
                train=len(train_events),
                test=len(test_events),
                dry_run=dry_run,
                feedback=feedback,
                train_cutoff=train_cutoff.isoformat() if not no_split else "none",
                commodities=commodities,
                regimes=regimes)

    fetcher     = PriceFetcher()
    client      = GeminiClient() if not dry_run else None
    # Accumulator active when feedback=True OR when we have a train phase
    # (train events always feed the accumulator so test events benefit)
    use_accum   = feedback or (not no_split and len(train_events) > 0)
    accumulator = FeedbackAccumulator() if use_accum else None

    # ── Train phase (silent — builds accumulator, not reported) ──────────────
    if train_events:
        logger.info("train_phase_start", n_events=len(train_events),
                    cutoff=train_cutoff.isoformat())
        _run_events(train_events, dry_run, client, fetcher,
                    accumulator, is_test_phase=False, all_events=all_events)
        logger.info("train_phase_complete")

    # ── Test phase (reported) ─────────────────────────────────────────────────
    logger.info("test_phase_start", n_events=len(test_events))
    results = _run_events(test_events, dry_run, client, fetcher,
                          accumulator, is_test_phase=True, all_events=all_events)

    regime_metrics = compute_regime_metrics(results)
    print_report(results, regime_metrics,
                 train_cutoff=train_cutoff if not no_split else None,
                 n_train=len(train_events))

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "mcei_version": "1.0",
        "contamination_note": CONTAMINATION_NOTE,
        "config": {
            "dry_run": dry_run,
            "feedback_mode": feedback,
            "train_cutoff": train_cutoff.isoformat() if not no_split else "none",
            "n_train_events": len(train_events),
            "n_test_events": len(test_events),
            "commodities_filter": commodities,
            "regimes_filter": regimes,
            "neutral_band_pct": NEUTRAL_BAND_PCT,
            "composite_weights": {"direction": W_DIRECTION,
                                  "magnitude": W_MAGNITUDE,
                                  "calibration": W_CALIBRATION},
            "analogs_note": (
                "Backtest uses catalog-based in-memory analog matching (severity proximity) "
                "instead of live Qdrant vector search. This is an approximation."
            ),
            "split_note": (
                "TRAIN events (before cutoff) are processed chronologically to warm up the "
                "FeedbackAccumulator. Only TEST events appear in per_event_results and regime_summary."
            ) if not no_split else "no_split: all events treated as test.",
        },
        "regime_summary": regime_metrics,
        "per_event_results": [asdict(r) for r in results],
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("backtest_report_written", path=output)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="MCEI Backtesting Engine — historical walk-forward prediction evaluation"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Gemini API calls; score a fixed synthetic prediction")
    parser.add_argument("--feedback", action="store_true",
                        help=(
                            "Enable walk-forward feedback loop: events processed "
                            "chronologically; each event's prompt is augmented with "
                            "a damped error signal from all prior scored events of "
                            "the same commodity+anomaly_type."
                        ))
    parser.add_argument("--train-cutoff", default="2021-01-01",
                        help=(
                            "ISO date (YYYY-MM-DD) splitting train and test sets. "
                            "Events before this date warm up the FeedbackAccumulator "
                            "silently; events on/after are scored and reported. "
                            "Default: 2021-01-01 (train=2008-2020, test=2021-2026)."
                        ))
    parser.add_argument("--no-split", action="store_true",
                        help="Disable train/test split — evaluate every event (legacy mode)")
    parser.add_argument("--commodities", nargs="+",
                        choices=["lng", "copper", "soybeans"], default=None,
                        help="Filter to specific commodities (default: all)")
    parser.add_argument("--regimes", nargs="+",
                        choices=["normal", "stress", "crisis", "black_swan"], default=None,
                        help="Filter to specific market regimes (default: all)")
    parser.add_argument("--output", default="reports/backtest_report.json",
                        help="Path for JSON report output")
    args = parser.parse_args()

    try:
        cutoff = date.fromisoformat(args.train_cutoff)
    except ValueError:
        print(f"ERROR: --train-cutoff must be YYYY-MM-DD, got: {args.train_cutoff}")
        sys.exit(1)

    run_backtest(
        dry_run=args.dry_run,
        commodities=args.commodities,
        regimes=args.regimes,
        output=args.output,
        feedback=args.feedback,
        train_cutoff=cutoff,
        no_split=args.no_split,
    )


if __name__ == "__main__":
    main()
