"""
Bounding box registry for all monitored commodity ports and terminals.

Format per entry:
    "port_slug": {
        "name":      human-readable name
        "commodity": lng | copper | soybeans
        "bbox":      [min_lat, min_lon, max_lat, max_lon]   (aisstream format)
        "country":   ISO country code
        "note":      optional context
    }

Used by:
  - aisstream_ingestor.py  (builds WebSocket subscription)
  - ais_feature_extractor.py (assigns vessel positions to zones)
  - sentinel1_ingestor.py  (Phase 13 — defines SAR scene AOIs)
"""

from typing import Final

PORT_REGISTRY: Final[dict[str, dict]] = {

    # ── LNG Export Terminals (US) ─────────────────────────────────────────────
    "sabine_pass": {
        "name": "Sabine Pass LNG",
        "commodity": "lng",
        "bbox": [29.68, -93.92, 29.80, -93.75],
        "country": "US",
        "note": "Largest US LNG export terminal; Cheniere Energy",
    },
    "freeport_lng": {
        "name": "Freeport LNG",
        "commodity": "lng",
        "bbox": [28.90, -95.42, 29.02, -95.26],
        "country": "US",
        "note": "Freeport LNG Development LP; Gulf of Mexico",
    },
    "cameron_lng": {
        "name": "Cameron LNG",
        "commodity": "lng",
        "bbox": [29.90, -93.42, 30.02, -93.26],
        "country": "US",
        "note": "Sempra Infrastructure; Louisiana",
    },
    "corpus_christi_lng": {
        "name": "Corpus Christi LNG",
        "commodity": "lng",
        "bbox": [27.78, -97.32, 27.92, -97.18],
        "country": "US",
        "note": "Cheniere Corpus Christi; Texas",
    },
    "cove_point_lng": {
        "name": "Cove Point LNG",
        "commodity": "lng",
        "bbox": [38.36, -76.48, 38.46, -76.36],
        "country": "US",
        "note": "Dominion Energy Cove Point; Maryland",
    },

    # ── LNG Import Terminals (Europe — key demand side) ───────────────────────
    "gate_rotterdam": {
        "name": "Gate Terminal Rotterdam",
        "commodity": "lng",
        "bbox": [51.92, 4.00, 51.98, 4.12],
        "country": "NL",
        "note": "Major European LNG regasification hub",
    },
    "zeebrugge_lng": {
        "name": "Zeebrugge LNG",
        "commodity": "lng",
        "bbox": [51.32, 3.16, 51.38, 3.24],
        "country": "BE",
        "note": "Fluxys Belgium; Bruges–Zeebrugge",
    },
    "grain_lng_uk": {
        "name": "Grain LNG",
        "commodity": "lng",
        "bbox": [51.42, 0.68, 51.48, 0.76],
        "country": "GB",
        "note": "National Grid; Isle of Grain, Kent",
    },

    # ── Copper Export Ports ───────────────────────────────────────────────────
    "antofagasta": {
        "name": "Antofagasta Port",
        "commodity": "copper",
        "bbox": [-23.68, -70.47, -23.58, -70.38],
        "country": "CL",
        "note": "Primary copper export port; Codelco/BHP shipments",
    },
    "iquique": {
        "name": "Puerto Iquique",
        "commodity": "copper",
        "bbox": [-20.28, -70.20, -20.18, -70.10],
        "country": "CL",
        "note": "Northern Chile copper; secondary export point",
    },
    "puerto_ventanas": {
        "name": "Puerto Ventanas",
        "commodity": "copper",
        "bbox": [-32.80, -71.57, -32.70, -71.48],
        "country": "CL",
        "note": "Central Chile copper concentrate export",
    },
    "walvis_bay": {
        "name": "Walvis Bay",
        "commodity": "copper",
        "bbox": [-23.00, 14.48, -22.92, 14.56],
        "country": "NA",
        "note": "DRC copper transit port via Namibia corridor",
    },
    "durban": {
        "name": "Durban Port",
        "commodity": "copper",
        "bbox": [-30.02, 30.88, -29.88, 31.02],
        "country": "ZA",
        "note": "Southern Africa copper; DRC Zambia corridor",
    },

    # ── Soybean Export Ports ──────────────────────────────────────────────────
    "santos": {
        "name": "Port of Santos",
        "commodity": "soybeans",
        "bbox": [-23.98, -46.42, -23.88, -46.28],
        "country": "BR",
        "note": "Largest soybean export port in the world",
    },
    "paranagua": {
        "name": "Port of Paranagua",
        "commodity": "soybeans",
        "bbox": [-25.58, -48.58, -25.48, -48.48],
        "country": "BR",
        "note": "Major Brazil soybean/grain export terminal",
    },
    "new_orleans": {
        "name": "Port of New Orleans (Grain Elevators)",
        "commodity": "soybeans",
        "bbox": [29.92, -90.12, 30.05, -89.92],
        "country": "US",
        "note": "Mississippi River grain/soybean exports; NOLA barge terminals",
    },
    "rosario": {
        "name": "Rosario Grain Complex",
        "commodity": "soybeans",
        "bbox": [-33.02, -60.75, -32.88, -60.58],
        "country": "AR",
        "note": "Largest soybean processing/export hub globally; Parana River",
    },
}


def get_ports_for_commodity(commodity: str) -> dict[str, dict]:
    """Return all port entries for a given commodity slug."""
    return {slug: info for slug, info in PORT_REGISTRY.items() if info["commodity"] == commodity}


def get_all_bboxes() -> list[list[float]]:
    """
    Return all bounding boxes as a flat list for aisstream subscription.
    aisstream format: [[min_lat, min_lon, max_lat, max_lon], ...]
    """
    return [info["bbox"] for info in PORT_REGISTRY.values()]


def find_port_for_position(lat: float, lon: float) -> str | None:
    """
    Returns the port slug if (lat, lon) falls within any registered bounding box.
    Returns None if the position doesn't match any port.
    """
    for slug, info in PORT_REGISTRY.items():
        min_lat, min_lon, max_lat, max_lon = info["bbox"]
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return slug
    return None
