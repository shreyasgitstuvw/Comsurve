"""
Geographic utility functions shared across ingestion and processing modules.

Used by:
  - ingestion/ais/aisstream_ingestor.py  (port bounding box matching)
  - ingestion/satellite/sentinel1_ingestor.py  (bbox → WKT polygon)
  - ingestion/rail/osm_overpass_ingestor.py    (corridor bbox operations)
  - processor/satellite_feature_extractor.py   (scene-to-port intersection)
"""

import math
from typing import NamedTuple


class BBox(NamedTuple):
    """Axis-aligned bounding box in geographic coordinates."""
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


def point_in_bbox(lat: float, lon: float, bbox: BBox | list | tuple) -> bool:
    """Return True if (lat, lon) falls within the bounding box (inclusive)."""
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def bbox_to_wkt(bbox: BBox | list | tuple) -> str:
    """
    Convert a [min_lat, min_lon, max_lat, max_lon] bbox to a WKT POLYGON string.
    WKT uses (lon lat) coordinate order per the OGC standard.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    return (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )


def bbox_to_overpass(bbox: BBox | list | tuple) -> str:
    """
    Format a bbox as an Overpass API bounding-box string: (min_lat,min_lon,max_lat,max_lon).
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    return f"({min_lat},{min_lon},{max_lat},{max_lon})"


def bboxes_overlap(a: BBox | list | tuple, b: BBox | list | tuple) -> bool:
    """Return True if two bounding boxes overlap (share any area)."""
    a_min_lat, a_min_lon, a_max_lat, a_max_lon = a
    b_min_lat, b_min_lon, b_max_lat, b_max_lon = b
    return not (
        a_max_lat < b_min_lat or a_min_lat > b_max_lat
        or a_max_lon < b_min_lon or a_min_lon > b_max_lon
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in kilometres between two (lat, lon) points.
    Uses the Haversine formula.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polyline_length_km(nodes: list[tuple[float, float]]) -> float:
    """
    Approximate length in km of a polyline given as [(lat, lon), ...] node list.
    Sums Haversine distances between consecutive nodes.
    """
    if len(nodes) < 2:
        return 0.0
    total = 0.0
    for i in range(len(nodes) - 1):
        total += haversine_km(nodes[i][0], nodes[i][1], nodes[i + 1][0], nodes[i + 1][1])
    return round(total, 3)


def centroid(bbox: BBox | list | tuple) -> tuple[float, float]:
    """Return the (lat, lon) centroid of a bounding box."""
    min_lat, min_lon, max_lat, max_lon = bbox
    return ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)
