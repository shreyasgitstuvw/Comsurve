"""
Sentinel-1 SAR scene catalog ingestor.

Queries the CDSE OData catalog for Sentinel-1 GRD (Ground Range Detected)
scenes that intersect each monitored port bounding box over the past N days.
No full product download is performed — only scene metadata is stored.

Each stored row represents one S1 scene intersecting one port:
  source    = "sentinel1"
  data_type = "satellite"
  symbol    = "{port_slug}__{scene_id_prefix}"   (unique per port × scene)
  timestamp = scene acquisition start time (UTC)
  raw_json  = {scene_id, scene_name, port_slug, commodity, orbit_direction,
               size_bytes, s3_path, acquisition_start, acquisition_end}

The satellite_feature_extractor later aggregates these rows into per-port
scene-count features.

CDSE catalog API is publicly accessible without authentication for metadata
queries — no token is required here.
"""

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx

from ingestion.base_ingestor import BaseIngestor
from ingestion.ais.port_registry import PORT_REGISTRY
from shared.schemas import RawIngestionRecord
from shared.logger import get_logger

logger = get_logger(__name__)

CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
REQUEST_TIMEOUT = 30


def _bbox_to_wkt(bbox: list[float]) -> str:
    """
    Convert [min_lat, min_lon, max_lat, max_lon] to a WKT POLYGON string.
    WKT uses (lon lat) coordinate order.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    return (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )


def _build_filter(wkt: str, since_iso: str, until_iso: str) -> str:
    geo = f"geography'SRID=4326;{wkt}'"
    return (
        f"Collection/Name eq 'SENTINEL-1'"
        f" and OData.CSC.Intersects(area={geo})"
        f" and ContentDate/Start gt {since_iso}"
        f" and ContentDate/Start lt {until_iso}"
        f" and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType'"
        f" and att/OData.CSC.StringAttribute/Value eq 'GRD')"
    )


def _extract_attribute(attributes: list[dict], name: str) -> str | None:
    for attr in attributes:
        if attr.get("Name") == name:
            return str(attr.get("Value", ""))
    return None


class Sentinel1Ingestor(BaseIngestor):
    source = "sentinel1"

    def __init__(self, lookback_days: int = 7):
        self.lookback_days = lookback_days

    def fetch(self) -> list[RawIngestionRecord]:
        records: list[RawIngestionRecord] = []

        now = datetime.now(timezone.utc)
        since = now - timedelta(days=self.lookback_days)
        since_iso = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        until_iso = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        for port_slug, port_info in PORT_REGISTRY.items():
            bbox = port_info["bbox"]
            commodity = port_info["commodity"]
            wkt = _bbox_to_wkt(bbox)
            odata_filter = _build_filter(wkt, since_iso, until_iso)

            try:
                resp = httpx.get(
                    CATALOG_URL,
                    params={
                        "$filter": odata_filter,
                        "$expand": "Attributes",
                        "$top": 50,
                        "$orderby": "ContentDate/Start desc",
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                scenes = resp.json().get("value", [])
            except Exception as exc:
                logger.warning(
                    "sentinel1_catalog_query_failed",
                    port=port_slug,
                    error=str(exc),
                )
                continue

            logger.info(
                "sentinel1_scenes_found",
                port=port_slug,
                count=len(scenes),
            )

            for scene in scenes:
                scene_id = scene.get("Id", "")
                scene_name = scene.get("Name", "")
                attrs = scene.get("Attributes") or []
                orbit_dir = _extract_attribute(attrs, "orbitDirection") or "unknown"

                content_date = scene.get("ContentDate", {})
                acq_start_str = content_date.get("Start", "")
                acq_end_str = content_date.get("End", "")

                # Parse acquisition start as the row timestamp
                try:
                    ts = datetime.fromisoformat(
                        acq_start_str.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception:
                    ts = datetime.utcnow()

                # symbol = port_slug + first 8 chars of scene UUID for uniqueness
                symbol = f"{port_slug}__{scene_id[:8]}"

                raw = {
                    "scene_id": scene_id,
                    "scene_name": scene_name,
                    "port_slug": port_slug,
                    "port_name": port_info["name"],
                    "commodity": commodity,
                    "orbit_direction": orbit_dir,
                    "size_bytes": scene.get("Size"),
                    "s3_path": scene.get("S3Path", ""),
                    "acquisition_start": acq_start_str,
                    "acquisition_end": acq_end_str,
                    "online": scene.get("Online", False),
                }

                records.append(RawIngestionRecord(
                    source=self.source,
                    commodity=commodity,
                    symbol=symbol,
                    timestamp=ts,
                    data_type="satellite",
                    raw_json=json.dumps(raw),
                ))

        return records
