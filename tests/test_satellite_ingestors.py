"""
Tests for ingestion/satellite/sentinel1_ingestor.py and sentinel2_ingestor.py.

Strategy:
  - Helper functions (_bbox_to_wkt, _build_filter, _extract_attribute) are pure → no mocking.
  - fetch() calls httpx.get → patched with MagicMock returning realistic CDSE OData payloads.
  - PORT_REGISTRY is patched to a single-port stub so tests are fast and deterministic.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ingestion.satellite.sentinel1_ingestor import (
    Sentinel1Ingestor,
    _bbox_to_wkt,
    _build_filter,
    _extract_attribute,
)
from ingestion.satellite.sentinel2_ingestor import (
    Sentinel2Ingestor,
    _build_filter as _s2_build_filter,
    _extract_double_attribute,
)


# ── Shared constants ──────────────────────────────────────────────────────────

_BBOX = [29.68, -93.92, 29.80, -93.75]   # [min_lat, min_lon, max_lat, max_lon]

_FAKE_REGISTRY = {
    "sabine_pass": {
        "name": "Sabine Pass LNG",
        "commodity": "lng",
        "bbox": _BBOX,
        "country": "US",
    }
}

_SCENE_ID = "abcd1234-0000-0000-0000-000000000000"
_ACQ_START = "2026-04-10T06:00:00.000Z"
_ACQ_END   = "2026-04-10T06:06:00.000Z"


def _s1_scene(orbit_dir="ascending") -> dict:
    return {
        "Id": _SCENE_ID,
        "Name": "S1A_IW_GRDH_1SDV_20260410T060000",
        "Size": 1024 * 1024 * 800,
        "S3Path": "/eodata/Sentinel-1/SAR/GRD/2026/04/10/...",
        "Online": True,
        "ContentDate": {"Start": _ACQ_START, "End": _ACQ_END},
        "Attributes": [
            {"Name": "orbitDirection", "Value": orbit_dir},
            {"Name": "productType", "Value": "GRD"},
        ],
    }


def _s2_scene(cloud_cover: float = 12.5) -> dict:
    return {
        "Id": _SCENE_ID,
        "Name": "S2A_MSIL2A_20260410T160000",
        "Size": 1024 * 1024 * 600,
        "S3Path": "/eodata/Sentinel-2/MSI/L2A/2026/04/10/...",
        "Online": True,
        "ContentDate": {"Start": _ACQ_START, "End": _ACQ_END},
        "Attributes": [
            {"Name": "cloudCover", "Value": cloud_cover},
            {"Name": "productType", "Value": "S2MSI2A"},
        ],
    }


def _mock_response(scenes: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"value": scenes}
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# Helper function tests (pure — no mocking needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestBboxToWkt:
    def test_coordinate_order_is_lon_lat(self):
        wkt = _bbox_to_wkt([10.0, 20.0, 30.0, 40.0])
        # WKT: (lon lat) → first coord should be (min_lon min_lat)
        assert "20.0 10.0" in wkt

    def test_polygon_is_closed(self):
        wkt = _bbox_to_wkt([10.0, 20.0, 30.0, 40.0])
        # Closing coord matches first coord
        assert wkt.count("20.0 10.0") >= 2

    def test_returns_polygon_string(self):
        wkt = _bbox_to_wkt(_BBOX)
        assert wkt.startswith("POLYGON((")
        assert wkt.endswith("))")


class TestExtractAttribute:
    def test_returns_matching_value(self):
        attrs = [{"Name": "orbitDirection", "Value": "ascending"}]
        assert _extract_attribute(attrs, "orbitDirection") == "ascending"

    def test_returns_none_for_missing_key(self):
        assert _extract_attribute([], "orbitDirection") is None

    def test_converts_value_to_str(self):
        attrs = [{"Name": "relativeOrbit", "Value": 42}]
        result = _extract_attribute(attrs, "relativeOrbit")
        assert result == "42"
        assert isinstance(result, str)


class TestExtractDoubleAttribute:
    def test_returns_float_for_valid_entry(self):
        attrs = [{"Name": "cloudCover", "Value": 23.7}]
        assert _extract_double_attribute(attrs, "cloudCover") == pytest.approx(23.7)

    def test_returns_none_for_missing(self):
        assert _extract_double_attribute([], "cloudCover") is None

    def test_returns_none_for_invalid_value(self):
        attrs = [{"Name": "cloudCover", "Value": "bad"}]
        assert _extract_double_attribute(attrs, "cloudCover") is None


class TestBuildFilter:
    def test_s1_filter_contains_sentinel1_collection(self):
        f = _build_filter("POLYGON(())", "2026-04-01T00:00:00.000Z", "2026-04-10T00:00:00.000Z")
        assert "SENTINEL-1" in f

    def test_s1_filter_contains_grd_product_type(self):
        f = _build_filter("POLYGON(())", "2026-04-01T00:00:00.000Z", "2026-04-10T00:00:00.000Z")
        assert "GRD" in f

    def test_s2_filter_contains_sentinel2_collection(self):
        f = _s2_build_filter("POLYGON(())", "2026-04-01T00:00:00.000Z", "2026-04-10T00:00:00.000Z", 95.0)
        assert "SENTINEL-2" in f

    def test_s2_filter_contains_cloud_cover_threshold(self):
        f = _s2_build_filter("POLYGON(())", "2026-04-01T00:00:00.000Z", "2026-04-10T00:00:00.000Z", 75.0)
        assert "75.0" in f


# ══════════════════════════════════════════════════════════════════════════════
# Sentinel1Ingestor.fetch()
# ══════════════════════════════════════════════════════════════════════════════

class TestSentinel1IngestorFetch:

    def _fetch(self, scenes: list[dict] | None = None, http_error: bool = False):
        mock_resp = _mock_response([_s1_scene()] if scenes is None else scenes)
        if http_error:
            mock_resp.raise_for_status.side_effect = Exception("HTTP 503")

        with patch("ingestion.satellite.sentinel1_ingestor.PORT_REGISTRY", _FAKE_REGISTRY):
            with patch("ingestion.satellite.sentinel1_ingestor.httpx.get", return_value=mock_resp):
                return Sentinel1Ingestor(lookback_days=7).fetch()

    def test_returns_one_record_per_scene(self):
        records = self._fetch([_s1_scene(), _s1_scene()])
        assert len(records) == 2

    def test_record_source_is_sentinel1(self):
        records = self._fetch()
        assert records[0].source == "sentinel1"

    def test_record_data_type_is_satellite(self):
        records = self._fetch()
        assert records[0].data_type == "satellite"

    def test_record_commodity_matches_port_registry(self):
        records = self._fetch()
        assert records[0].commodity == "lng"

    def test_record_symbol_contains_port_slug(self):
        records = self._fetch()
        assert "sabine_pass" in records[0].symbol

    def test_raw_json_contains_orbit_direction(self):
        records = self._fetch([_s1_scene(orbit_dir="descending")])
        raw = json.loads(records[0].raw_json)
        assert raw["orbit_direction"] == "descending"

    def test_raw_json_contains_scene_id(self):
        records = self._fetch()
        raw = json.loads(records[0].raw_json)
        assert raw["scene_id"] == _SCENE_ID

    def test_empty_response_returns_no_records(self):
        records = self._fetch(scenes=[])
        assert records == []

    def test_http_error_per_port_continues_gracefully(self):
        # Single port raises HTTP error → should return empty list, not raise
        records = self._fetch(http_error=True)
        assert records == []

    def test_timestamp_parsed_from_acquisition_start(self):
        records = self._fetch()
        expected = datetime(2026, 4, 10, 6, 0, 0)
        assert records[0].timestamp == expected

    def test_multiple_ports_all_queried(self):
        registry_two = {
            "port_a": {"name": "Port A", "commodity": "lng", "bbox": _BBOX, "country": "US"},
            "port_b": {"name": "Port B", "commodity": "copper", "bbox": _BBOX, "country": "CL"},
        }
        mock_resp = _mock_response([_s1_scene()])
        with patch("ingestion.satellite.sentinel1_ingestor.PORT_REGISTRY", registry_two):
            with patch("ingestion.satellite.sentinel1_ingestor.httpx.get", return_value=mock_resp) as mock_get:
                Sentinel1Ingestor().fetch()
        assert mock_get.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# Sentinel2Ingestor.fetch()
# ══════════════════════════════════════════════════════════════════════════════

class TestSentinel2IngestorFetch:

    def _fetch(self, scenes: list[dict] | None = None, http_error: bool = False):
        mock_resp = _mock_response([_s2_scene()] if scenes is None else scenes)
        if http_error:
            mock_resp.raise_for_status.side_effect = Exception("HTTP 500")

        with patch("ingestion.satellite.sentinel2_ingestor.PORT_REGISTRY", _FAKE_REGISTRY):
            with patch("ingestion.satellite.sentinel2_ingestor.httpx.get", return_value=mock_resp):
                return Sentinel2Ingestor(lookback_days=14).fetch()

    def test_returns_one_record_per_scene(self):
        records = self._fetch([_s2_scene(), _s2_scene(cloud_cover=80.0)])
        assert len(records) == 2

    def test_record_source_is_sentinel2(self):
        records = self._fetch()
        assert records[0].source == "sentinel2"

    def test_record_data_type_is_satellite(self):
        records = self._fetch()
        assert records[0].data_type == "satellite"

    def test_cloud_cover_stored_in_raw_json(self):
        records = self._fetch([_s2_scene(cloud_cover=45.3)])
        raw = json.loads(records[0].raw_json)
        assert raw["cloud_cover_pct"] == pytest.approx(45.3)

    def test_none_cloud_cover_stored_as_null(self):
        scene = _s2_scene()
        scene["Attributes"] = []   # No cloudCover attribute
        records = self._fetch([scene])
        raw = json.loads(records[0].raw_json)
        assert raw["cloud_cover_pct"] is None

    def test_empty_response_returns_no_records(self):
        assert self._fetch(scenes=[]) == []

    def test_http_error_continues_gracefully(self):
        records = self._fetch(http_error=True)
        assert records == []

    def test_symbol_format_is_port_slug_plus_scene_prefix(self):
        records = self._fetch()
        assert records[0].symbol == f"sabine_pass__{_SCENE_ID[:8]}"

    def test_raw_json_contains_port_name(self):
        records = self._fetch()
        raw = json.loads(records[0].raw_json)
        assert raw["port_name"] == "Sabine Pass LNG"
