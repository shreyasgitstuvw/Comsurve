"""
aisstream.io AIS vessel tracking ingestor.
Connects via WebSocket, subscribes to all monitored port bounding boxes,
collects vessel position reports for a fixed window, then saves one
raw_ingestion row per port that had vessel activity.

Schedule: every 30 minutes.

aisstream.io message format:
  {
    "MessageType": "PositionReport" | "StandardClassBPositionReport" | ...,
    "Message": { <MessageType>: { Latitude, Longitude, Sog, Cog, ... } },
    "MetaData": { MMSI, ShipName, latitude, longitude, time_utc }
  }

Note: position coordinates are in MetaData (lowercase lat/lon), not Message.
"""

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone

import websockets

from ingestion.ais.port_registry import PORT_REGISTRY, find_port_for_position, get_all_bboxes
from ingestion.base_ingestor import BaseIngestor
from shared.config import settings
from shared.logger import get_logger
from shared.schemas import RawIngestionRecord

logger = get_logger(__name__)

AISSTREAM_URI = "wss://stream.aisstream.io/v0/stream"

# Position-bearing message types we care about
POSITION_MESSAGE_TYPES = {
    "PositionReport",
    "StandardClassBPositionReport",
    "ExtendedClassBPositionReport",
}


class AISStreamIngestor(BaseIngestor):
    source = "aisstream"

    def __init__(self, collection_seconds: int = 60):
        """
        collection_seconds: how long to listen on the WebSocket per run.
        60s gives a reasonable snapshot without keeping the connection open.
        """
        self.collection_seconds = collection_seconds

    def fetch(self) -> list[RawIngestionRecord]:
        if not settings.aisstream_api_key:
            raise ValueError("AISSTREAM_API_KEY not set in .env")
        return asyncio.run(self._collect())

    async def _collect(self) -> list[RawIngestionRecord]:
        """Connect, collect for collection_seconds, return RawIngestionRecord list."""
        bboxes = get_all_bboxes()
        subscription = {
            "APIKey": settings.aisstream_api_key,
            "BoundingBoxes": [bboxes],   # aisstream expects list of lists
            "FilterMessageTypes": list(POSITION_MESSAGE_TYPES),
        }

        # port_slug → list of vessel snapshots
        port_vessels: dict[str, list[dict]] = defaultdict(list)
        window_start = datetime.now(timezone.utc)
        message_count = 0

        try:
            async with websockets.connect(
                AISSTREAM_URI,
                ping_interval=20,
                ping_timeout=30,
                open_timeout=15,
            ) as ws:
                await ws.send(json.dumps(subscription))
                logger.info("aisstream_connected", bboxes=len(bboxes), window_seconds=self.collection_seconds)

                deadline = asyncio.get_event_loop().time() + self.collection_seconds
                while asyncio.get_event_loop().time() < deadline:
                    remaining = deadline - asyncio.get_event_loop().time()
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
                    except asyncio.TimeoutError:
                        continue

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("MessageType") not in POSITION_MESSAGE_TYPES:
                        continue

                    meta = msg.get("MetaData", {})
                    lat = meta.get("latitude")
                    lon = meta.get("longitude")
                    if lat is None or lon is None:
                        continue

                    port_slug = find_port_for_position(lat, lon)
                    if port_slug is None:
                        continue  # vessel not in any monitored zone

                    # Extract speed from the inner message
                    inner = msg.get("Message", {}).get(msg["MessageType"], {})
                    sog = inner.get("Sog", 0.0)       # speed over ground (knots)
                    cog = inner.get("Cog", 0.0)       # course over ground
                    nav_status = inner.get("NavigationalStatus", -1)  # 0=underway, 1=anchored, 5=moored

                    time_utc_str = meta.get("time_utc", "")
                    port_vessels[port_slug].append({
                        "mmsi": meta.get("MMSI"),
                        "ship_name": meta.get("ShipName", ""),
                        "lat": lat,
                        "lon": lon,
                        "sog": sog,
                        "cog": cog,
                        "nav_status": nav_status,
                        "msg_type": msg["MessageType"],
                        "time_utc": time_utc_str,
                    })
                    message_count += 1

        except (websockets.exceptions.WebSocketException, OSError) as exc:
            logger.error("aisstream_connection_error", error=str(exc))
            raise

        logger.info(
            "aisstream_collection_complete",
            messages_received=message_count,
            ports_with_activity=len(port_vessels),
            window_seconds=self.collection_seconds,
        )

        return self._build_records(port_vessels, window_start)

    def _build_records(
        self,
        port_vessels: dict[str, list[dict]],
        window_start: datetime,
    ) -> list[RawIngestionRecord]:
        """One RawIngestionRecord per port that had vessel activity."""
        records = []

        for port_slug, vessels in port_vessels.items():
            port_info = PORT_REGISTRY.get(port_slug, {})

            # Deduplicate by MMSI — keep latest position per vessel
            seen: dict[int, dict] = {}
            for v in vessels:
                mmsi = v.get("mmsi")
                if mmsi is not None:
                    seen[mmsi] = v

            unique_vessels = list(seen.values())
            vessel_count = len(unique_vessels)
            avg_sog = sum(v["sog"] for v in unique_vessels) / vessel_count if vessel_count else 0.0
            moored_count = sum(1 for v in unique_vessels if v.get("nav_status") in (1, 5))

            records.append(RawIngestionRecord(
                source=self.source,
                commodity=port_info.get("commodity", "unknown"),
                symbol=port_slug,
                timestamp=window_start.replace(tzinfo=None),
                data_type="ais",
                raw_json=json.dumps({
                    "port_slug": port_slug,
                    "port_name": port_info.get("name", port_slug),
                    "commodity": port_info.get("commodity", ""),
                    "window_start": window_start.isoformat(),
                    "vessel_count": vessel_count,
                    "avg_sog_knots": round(avg_sog, 2),
                    "moored_count": moored_count,
                    "vessels": unique_vessels,
                }),
            ))

        return records


if __name__ == "__main__":
    from shared.db import init_db
    init_db()
    ingestor = AISStreamIngestor(collection_seconds=30)
    summary = ingestor.run()
    print(summary)
