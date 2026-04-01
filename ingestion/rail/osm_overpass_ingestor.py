"""
OSM Overpass API rail corridor ingestor.

Queries OpenStreetMap via the Overpass API for railway=rail ways within
defined commodity corridor bounding boxes.  The resulting GeoJSON is stored
in raw_ingestion as a static reference geometry that:

  1. Enriches Gemini causality prompts with rail infrastructure context
     ("this anomaly port is served by the Antofagasta-Calama corridor")
  2. Provides corridor length baselines for satellite SAR change detection
     (future: backscatter deviation along a known rail segment = disruption)
  3. Enables news cross-referencing — when a news item mentions a corridor
     name, the system can link it to the affected commodity port

Rail geometry is semi-static (months between significant changes), so this
ingestor runs weekly rather than hourly.

Each stored row represents one corridor snapshot:
  source    = "osm_rail"
  data_type = "rail"
  symbol    = corridor_slug
  timestamp = UTC time of the query (floor to day)
  raw_json  = {corridor_slug, commodity, description, bbox,
               way_count, total_length_km, nodes, geojson_features}

The unique constraint (source, symbol, timestamp) uses day-floor timestamps
so once-per-day re-runs are silently skipped via INSERT OR IGNORE.
"""

import json
import time
from datetime import datetime, timezone

import httpx

from ingestion.base_ingestor import BaseIngestor
from shared.geo_utils import bbox_to_overpass, polyline_length_km
from shared.schemas import RawIngestionRecord
from shared.logger import get_logger

logger = get_logger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
REQUEST_TIMEOUT = 60   # Overpass can be slow for large bboxes

# Key commodity rail corridors
RAIL_CORRIDORS: dict[str, dict] = {

    # ── Copper — Chile ────────────────────────────────────────────────────────
    "antofagasta_calama": {
        "commodity": "copper",
        "country": "CL",
        "description": (
            "FC Antofagasta Bolivia — copper concentrate rail from Atacama "
            "highland mines (Codelco, BHP) to Antofagasta port"
        ),
        "bbox": [-24.5, -70.5, -22.0, -68.0],
    },
    "iquique_collahuasi": {
        "commodity": "copper",
        "country": "CL",
        "description": (
            "Northern Chile rail — Collahuasi and Cerro Colorado copper mines "
            "to Iquique and Patache ports"
        ),
        "bbox": [-21.5, -70.3, -19.5, -68.5],
    },

    # ── Copper — Southern Africa ───────────────────────────────────────────────
    "zambia_durban": {
        "commodity": "copper",
        "country": "ZA/ZM",
        "description": (
            "Zambia Railways / TransNET — DRC and Zambia copper belt "
            "southbound to Durban port via Zimbabwe"
        ),
        "bbox": [-30.5, 27.0, -15.0, 32.5],
    },
    "benguela_walvis": {
        "commodity": "copper",
        "country": "AO/ZM/NA",
        "description": (
            "Benguela Railway (CFB) — DRC/Zambia copper westbound "
            "to Lobito and onward trans-shipment to Walvis Bay"
        ),
        "bbox": [-15.5, 12.0, -8.5, 24.0],
    },

    # ── Soybeans — Brazil ────────────────────────────────────────────────────
    "santos_mato_grosso": {
        "commodity": "soybeans",
        "country": "BR",
        "description": (
            "Rumo Logística ALL network — Mato Grosso and Paraná soybean "
            "producing states to Santos port via Campinas hub"
        ),
        "bbox": [-24.5, -53.0, -15.0, -45.0],
    },
    "paranagua_pr": {
        "commodity": "soybeans",
        "country": "BR",
        "description": (
            "FERROPAR / Rumo — Paraná state grain corridor "
            "to Paranaguá port"
        ),
        "bbox": [-26.5, -53.5, -24.5, -48.0],
    },

    # ── Soybeans — Argentina ─────────────────────────────────────────────────
    "rosario_cordoba": {
        "commodity": "soybeans",
        "country": "AR",
        "description": (
            "NCA Group rail — Córdoba soybean producing region "
            "to Rosario Paraná River terminal complex"
        ),
        "bbox": [-33.5, -65.0, -31.0, -60.0],
    },
}


def _build_overpass_query(bbox: list[float]) -> str:
    """Build an Overpass QL query to fetch railway=rail ways within a bbox."""
    bb = bbox_to_overpass(bbox)
    return f"""
[out:json][timeout:50];
(
  way["railway"="rail"]{bb};
  way["railway"="narrow_gauge"]{bb};
);
(._;>;);
out body;
""".strip()


def _parse_geojson_features(overpass_data: dict) -> tuple[list[dict], int, float]:
    """
    Parse Overpass JSON into GeoJSON-style feature list.
    Returns (features, way_count, total_length_km).
    """
    elements = overpass_data.get("elements", [])

    # Build node lookup: id → (lat, lon)
    node_map: dict[int, tuple[float, float]] = {}
    for el in elements:
        if el.get("type") == "node":
            node_map[el["id"]] = (el["lat"], el["lon"])

    features = []
    total_length_km = 0.0

    for el in elements:
        if el.get("type") != "way":
            continue

        node_ids = el.get("nodes", [])
        coords = []
        lat_lon_nodes = []
        for nid in node_ids:
            if nid in node_map:
                lat, lon = node_map[nid]
                coords.append([lon, lat])   # GeoJSON: [lon, lat]
                lat_lon_nodes.append((lat, lon))

        if len(coords) < 2:
            continue

        seg_len = polyline_length_km(lat_lon_nodes)
        total_length_km += seg_len

        tags = el.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "osm_id": el["id"],
                "name": tags.get("name", ""),
                "railway": tags.get("railway", "rail"),
                "gauge": tags.get("gauge", ""),
                "operator": tags.get("operator", ""),
                "length_km": round(seg_len, 3),
            },
        })

    return features, len(features), round(total_length_km, 2)


class OSMRailIngestor(BaseIngestor):
    source = "osm_rail"

    def fetch(self) -> list[RawIngestionRecord]:
        records: list[RawIngestionRecord] = []

        # Day-floor timestamp so once-per-day re-runs are idempotent
        today = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )

        for i, (corridor_slug, corridor_info) in enumerate(RAIL_CORRIDORS.items()):
            # Overpass API allows ~2 concurrent slots; 8s between sequential
            # requests keeps us within the fair-use policy
            if i > 0:
                time.sleep(8)

            bbox = corridor_info["bbox"]
            commodity = corridor_info["commodity"]
            query = _build_overpass_query(bbox)

            try:
                resp = httpx.post(
                    OVERPASS_URL,
                    data={"data": query},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    logger.warning("osm_rate_limited", corridor=corridor_slug,
                                   retry_after=60)
                    time.sleep(60)
                    resp = httpx.post(OVERPASS_URL, data={"data": query},
                                      timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                overpass_data = resp.json()
            except Exception as exc:
                logger.warning(
                    "osm_rail_query_failed",
                    corridor=corridor_slug,
                    error=str(exc),
                )
                continue

            features, way_count, total_km = _parse_geojson_features(overpass_data)

            logger.info(
                "osm_rail_corridor_fetched",
                corridor=corridor_slug,
                ways=way_count,
                total_km=total_km,
            )

            raw = {
                "corridor_slug": corridor_slug,
                "commodity": commodity,
                "country": corridor_info["country"],
                "description": corridor_info["description"],
                "bbox": bbox,
                "way_count": way_count,
                "total_length_km": total_km,
                "geojson": {
                    "type": "FeatureCollection",
                    "features": features,
                },
            }

            records.append(RawIngestionRecord(
                source=self.source,
                commodity=commodity,
                symbol=corridor_slug,
                timestamp=today,
                data_type="rail",
                raw_json=json.dumps(raw),
            ))

        return records
