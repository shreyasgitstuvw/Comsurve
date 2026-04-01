"""
OpenSky Network ADS-B aircraft ingestor.

Queries the OpenSky REST API for live aircraft state vectors within each
monitored port bounding box.  The signal of interest is unusual aircraft
density over commodity terminals and corridors:

  - Private / corporate jet activity surge at copper mining airports
    (potential insider / logistics signal)
  - Supply helicopter / cargo aircraft increase at LNG terminals
    (maintenance or incident response)
  - Agricultural survey aircraft over soybean export hubs

Authentication uses the OAuth2 client_credentials flow with the
OpenSky API client (clientId / clientSecret from .env).  If auth fails,
the ingestor falls back to the unauthenticated endpoint (lower rate limit).

Each stored row represents one snapshot of the aircraft count in a port zone:
  source    = "opensky"
  data_type = "aircraft"
  symbol    = "{port_slug}__aircraft_count"
  timestamp = UTC time of the API call (floored to the minute)
  raw_json  = {port_slug, commodity, aircraft_count, states_sample: [...5]}
"""

import json
from datetime import datetime

import httpx

from ingestion.base_ingestor import BaseIngestor
from ingestion.ais.port_registry import PORT_REGISTRY
from shared.config import settings
from shared.schemas import RawIngestionRecord
from shared.logger import get_logger

logger = get_logger(__name__)

OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)
REQUEST_TIMEOUT = 20

# Module-level token cache
_opensky_token: str | None = None
_opensky_token_expires: float = 0.0


def _get_opensky_token() -> str | None:
    """
    Obtain an OAuth2 access token via client_credentials flow.
    Returns None (not raises) if credentials are absent or auth fails — the
    caller degrades to unauthenticated access.
    """
    import time

    global _opensky_token, _opensky_token_expires

    if _opensky_token and time.time() < _opensky_token_expires - 60:
        return _opensky_token

    if not settings.opensky_client_id or not settings.opensky_client_secret:
        return None

    try:
        resp = httpx.post(
            OPENSKY_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": settings.opensky_client_id,
                "client_secret": settings.opensky_client_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        _opensky_token = data["access_token"]
        import time as _t
        _opensky_token_expires = _t.time() + data.get("expires_in", 3600)
        logger.info("opensky_token_refreshed")
        return _opensky_token
    except Exception as exc:
        logger.warning("opensky_auth_failed", error=str(exc))
        return None


class OpenSkyIngestor(BaseIngestor):
    source = "opensky"

    def fetch(self) -> list[RawIngestionRecord]:
        records: list[RawIngestionRecord] = []
        ts = datetime.utcnow().replace(second=0, microsecond=0)

        token = _get_opensky_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        for port_slug, port_info in PORT_REGISTRY.items():
            min_lat, min_lon, max_lat, max_lon = port_info["bbox"]
            commodity = port_info["commodity"]

            try:
                resp = httpx.get(
                    OPENSKY_STATES_URL,
                    params={
                        "lamin": min_lat,
                        "lomin": min_lon,
                        "lamax": max_lat,
                        "lomax": max_lon,
                    },
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning(
                    "opensky_query_failed",
                    port=port_slug,
                    error=str(exc),
                )
                continue

            states = data.get("states") or []
            count = len(states)

            # Sample up to 5 aircraft for context (icao24, callsign, altitude)
            sample = []
            for sv in states[:5]:
                if sv and len(sv) >= 8:
                    sample.append({
                        "icao24": sv[0],
                        "callsign": (sv[1] or "").strip(),
                        "altitude_m": sv[7],
                        "on_ground": sv[8] if len(sv) > 8 else None,
                    })

            symbol = f"{port_slug}__aircraft_count"
            raw = {
                "port_slug": port_slug,
                "port_name": port_info["name"],
                "commodity": commodity,
                "aircraft_count": count,
                "authenticated": token is not None,
                "states_sample": sample,
                "fetched_at": ts.isoformat(),
            }

            records.append(RawIngestionRecord(
                source=self.source,
                commodity=commodity,
                symbol=symbol,
                timestamp=ts,
                data_type="aircraft",
                raw_json=json.dumps(raw),
            ))

            logger.info(
                "opensky_zone_polled",
                port=port_slug,
                aircraft_count=count,
            )

        return records
