"""
Sentinel-2 optical scene catalog ingestor.

Queries the CDSE OData catalog for Sentinel-2 L2A (atmospherically corrected)
scenes that intersect each monitored port bounding box over the past N days.
Filters to scenes with cloud cover below `max_cloud_pct` (default 95%) so we
store both clear and cloudy scenes — the cloud cover value itself is the signal.

Each stored row represents one S2 scene intersecting one port:
  source    = "sentinel2"
  data_type = "satellite"
  symbol    = "{port_slug}__{scene_id_prefix}"
  timestamp = scene acquisition start time (UTC)
  raw_json  = {scene_id, scene_name, port_slug, commodity, cloud_cover_pct,
               size_bytes, s3_path, acquisition_start, acquisition_end}

The satellite_feature_extractor uses cloud_cover_pct to generate weather-proxy
features.  High persistent cloud cover at agricultural ports signals potential
flooding / adverse growing conditions.
"""

import json
from datetime import datetime, timedelta, timezone

import httpx

from ingestion.base_ingestor import BaseIngestor
from ingestion.ais.port_registry import PORT_REGISTRY
from shared.schemas import RawIngestionRecord
from shared.logger import get_logger

logger = get_logger(__name__)

CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
REQUEST_TIMEOUT = 30


def _bbox_to_wkt(bbox: list[float]) -> str:
    min_lat, min_lon, max_lat, max_lon = bbox
    return (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )


def _build_filter(wkt: str, since_iso: str, until_iso: str, max_cloud: float) -> str:
    geo = f"geography'SRID=4326;{wkt}'"
    return (
        f"Collection/Name eq 'SENTINEL-2'"
        f" and OData.CSC.Intersects(area={geo})"
        f" and ContentDate/Start gt {since_iso}"
        f" and ContentDate/Start lt {until_iso}"
        f" and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType'"
        f" and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
        f" and Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover'"
        f" and att/OData.CSC.DoubleAttribute/Value lt {max_cloud})"
    )


def _extract_double_attribute(attributes: list[dict], name: str) -> float | None:
    for attr in attributes:
        if attr.get("Name") == name:
            try:
                return float(attr["Value"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


class Sentinel2Ingestor(BaseIngestor):
    source = "sentinel2"

    def __init__(self, lookback_days: int = 14, max_cloud_pct: float = 95.0):
        self.lookback_days = lookback_days
        self.max_cloud_pct = max_cloud_pct  # upper ceiling — we store most scenes

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
            odata_filter = _build_filter(
                wkt, since_iso, until_iso, self.max_cloud_pct
            )

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
                    "sentinel2_catalog_query_failed",
                    port=port_slug,
                    error=str(exc),
                )
                continue

            logger.info(
                "sentinel2_scenes_found",
                port=port_slug,
                count=len(scenes),
            )

            for scene in scenes:
                scene_id = scene.get("Id", "")
                scene_name = scene.get("Name", "")
                attrs = scene.get("Attributes") or []
                cloud_cover = _extract_double_attribute(attrs, "cloudCover")

                content_date = scene.get("ContentDate", {})
                acq_start_str = content_date.get("Start", "")
                acq_end_str = content_date.get("End", "")

                try:
                    ts = datetime.fromisoformat(
                        acq_start_str.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception:
                    ts = datetime.utcnow()

                symbol = f"{port_slug}__{scene_id[:8]}"

                raw = {
                    "scene_id": scene_id,
                    "scene_name": scene_name,
                    "port_slug": port_slug,
                    "port_name": port_info["name"],
                    "commodity": commodity,
                    "cloud_cover_pct": cloud_cover,
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
