"""
Tests for ingestion/aircraft/opensky_ingestor.py and ingestion/rail/osm_overpass_ingestor.py.

Strategy:
  - OpenSkyIngestor.fetch(): mock httpx.get; PORT_REGISTRY patched to single port.
  - _get_opensky_token(): tested for auth-absent graceful degradation.
  - OSMRailIngestor.fetch(): mock httpx.post; RAIL_CORRIDORS patched to single entry.
  - _parse_geojson_features(): pure function, no mocking needed.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ingestion.aircraft.opensky_ingestor import OpenSkyIngestor
from ingestion.rail.osm_overpass_ingestor import (
    OSMRailIngestor,
    _build_overpass_query,
    _parse_geojson_features,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

_BBOX = [29.68, -93.92, 29.80, -93.75]

_FAKE_PORT_REGISTRY = {
    "sabine_pass": {
        "name": "Sabine Pass LNG",
        "commodity": "lng",
        "bbox": _BBOX,
        "country": "US",
    }
}

_FAKE_CORRIDOR = {
    "antofagasta_calama": {
        "commodity": "copper",
        "country": "CL",
        "description": "Test corridor",
        "bbox": [-24.5, -70.5, -22.0, -68.0],
    }
}


def _mock_http(json_body: dict, status_code: int = 200, raise_on_raise_for_status: bool = False) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    if raise_on_raise_for_status:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# OpenSkyIngestor.fetch()
# ══════════════════════════════════════════════════════════════════════════════

class TestOpenSkyIngestorFetch:

    def _opensky_states(self, n: int) -> dict:
        # OpenSky states: list of [icao24, callsign, origin_country, ..., altitude_m, on_ground, ...]
        state = ["abc123", "TEST001", "US", None, None, -93.8, 29.7, 3048.0, False]
        return {"states": [state] * n}

    def _fetch(self, n_aircraft: int = 3, http_error: bool = False, auth: bool = False):
        mock_resp = _mock_http(self._opensky_states(n_aircraft), raise_on_raise_for_status=http_error)

        patches = [
            patch("ingestion.aircraft.opensky_ingestor.PORT_REGISTRY", _FAKE_PORT_REGISTRY),
            patch("ingestion.aircraft.opensky_ingestor.httpx.get", return_value=mock_resp),
            patch("ingestion.aircraft.opensky_ingestor._get_opensky_token",
                  return_value="fake-token" if auth else None),
        ]
        with patches[0], patches[1], patches[2]:
            return OpenSkyIngestor().fetch()

    def test_returns_one_record_per_port(self):
        records = self._fetch()
        assert len(records) == 1

    def test_record_source_is_opensky(self):
        records = self._fetch()
        assert records[0].source == "opensky"

    def test_record_data_type_is_aircraft(self):
        records = self._fetch()
        assert records[0].data_type == "aircraft"

    def test_aircraft_count_stored_in_raw_json(self):
        records = self._fetch(n_aircraft=5)
        raw = json.loads(records[0].raw_json)
        assert raw["aircraft_count"] == 5

    def test_zero_aircraft_still_produces_record(self):
        records = self._fetch(n_aircraft=0)
        assert len(records) == 1
        raw = json.loads(records[0].raw_json)
        assert raw["aircraft_count"] == 0

    def test_symbol_contains_aircraft_count_suffix(self):
        records = self._fetch()
        assert records[0].symbol == "sabine_pass__aircraft_count"

    def test_commodity_matches_port_registry(self):
        records = self._fetch()
        assert records[0].commodity == "lng"

    def test_states_sample_capped_at_five(self):
        records = self._fetch(n_aircraft=10)
        raw = json.loads(records[0].raw_json)
        assert len(raw["states_sample"]) <= 5

    def test_authenticated_flag_true_when_token_present(self):
        records = self._fetch(auth=True)
        raw = json.loads(records[0].raw_json)
        assert raw["authenticated"] is True

    def test_authenticated_flag_false_when_no_token(self):
        records = self._fetch(auth=False)
        raw = json.loads(records[0].raw_json)
        assert raw["authenticated"] is False

    def test_http_error_per_port_continues_gracefully(self):
        records = self._fetch(http_error=True)
        assert records == []

    def test_null_states_field_treated_as_zero(self):
        mock_resp = _mock_http({"states": None})
        with patch("ingestion.aircraft.opensky_ingestor.PORT_REGISTRY", _FAKE_PORT_REGISTRY):
            with patch("ingestion.aircraft.opensky_ingestor.httpx.get", return_value=mock_resp):
                with patch("ingestion.aircraft.opensky_ingestor._get_opensky_token", return_value=None):
                    records = OpenSkyIngestor().fetch()
        raw = json.loads(records[0].raw_json)
        assert raw["aircraft_count"] == 0

    def test_multiple_ports_all_polled(self):
        registry_two = {
            "port_a": {"name": "A", "commodity": "lng", "bbox": _BBOX, "country": "US"},
            "port_b": {"name": "B", "commodity": "copper", "bbox": _BBOX, "country": "CL"},
        }
        mock_resp = _mock_http({"states": []})
        with patch("ingestion.aircraft.opensky_ingestor.PORT_REGISTRY", registry_two):
            with patch("ingestion.aircraft.opensky_ingestor.httpx.get", return_value=mock_resp) as mock_get:
                with patch("ingestion.aircraft.opensky_ingestor._get_opensky_token", return_value=None):
                    OpenSkyIngestor().fetch()
        assert mock_get.call_count == 2


class TestGetOpenSkyTokenAbsent:
    def test_returns_none_when_no_credentials(self):
        from ingestion.aircraft.opensky_ingestor import _get_opensky_token
        with patch("ingestion.aircraft.opensky_ingestor.settings") as mock_settings:
            mock_settings.opensky_client_id = ""
            mock_settings.opensky_client_secret = ""
            result = _get_opensky_token()
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# _parse_geojson_features (pure)
# ══════════════════════════════════════════════════════════════════════════════

class TestParseGeojsonFeatures:

    def _overpass_data(self, n_ways: int = 2) -> dict:
        """Build minimal Overpass JSON with n_ways connected by 3 nodes each."""
        nodes = [
            {"type": "node", "id": 1, "lat": -24.0, "lon": -69.0},
            {"type": "node", "id": 2, "lat": -23.5, "lon": -68.5},
            {"type": "node", "id": 3, "lat": -23.0, "lon": -68.0},
        ]
        ways = [
            {
                "type": "way",
                "id": 100 + i,
                "nodes": [1, 2, 3],
                "tags": {"railway": "rail", "name": f"Way {i}", "gauge": "1000"},
            }
            for i in range(n_ways)
        ]
        return {"elements": nodes + ways}

    def test_returns_correct_feature_count(self):
        features, way_count, _ = _parse_geojson_features(self._overpass_data(n_ways=3))
        assert way_count == 3
        assert len(features) == 3

    def test_total_length_km_positive(self):
        _, _, total_km = _parse_geojson_features(self._overpass_data())
        assert total_km > 0.0

    def test_feature_has_linestring_geometry(self):
        features, _, _ = _parse_geojson_features(self._overpass_data(n_ways=1))
        assert features[0]["geometry"]["type"] == "LineString"

    def test_coordinates_are_lon_lat_order(self):
        features, _, _ = _parse_geojson_features(self._overpass_data(n_ways=1))
        first_coord = features[0]["geometry"]["coordinates"][0]
        # first coord should be [lon=-69.0, lat=-24.0]
        assert first_coord[0] == pytest.approx(-69.0)
        assert first_coord[1] == pytest.approx(-24.0)

    def test_way_with_single_node_excluded(self):
        data = {
            "elements": [
                {"type": "node", "id": 1, "lat": -24.0, "lon": -69.0},
                {"type": "way", "id": 200, "nodes": [1], "tags": {}},
            ]
        }
        features, way_count, _ = _parse_geojson_features(data)
        assert way_count == 0
        assert features == []

    def test_empty_elements_returns_zeros(self):
        features, way_count, total_km = _parse_geojson_features({"elements": []})
        assert features == []
        assert way_count == 0
        assert total_km == 0.0

    def test_properties_include_osm_id_and_railway(self):
        features, _, _ = _parse_geojson_features(self._overpass_data(n_ways=1))
        props = features[0]["properties"]
        assert "osm_id" in props
        assert props["railway"] == "rail"


# ══════════════════════════════════════════════════════════════════════════════
# OSMRailIngestor.fetch()
# ══════════════════════════════════════════════════════════════════════════════

class TestOSMRailIngestorFetch:

    def _overpass_response(self, n_ways: int = 2) -> dict:
        nodes = [
            {"type": "node", "id": 1, "lat": -24.0, "lon": -69.0},
            {"type": "node", "id": 2, "lat": -23.5, "lon": -68.5},
            {"type": "node", "id": 3, "lat": -23.0, "lon": -68.0},
        ]
        ways = [
            {"type": "way", "id": 100 + i, "nodes": [1, 2, 3],
             "tags": {"railway": "rail", "name": f"Rail {i}"}}
            for i in range(n_ways)
        ]
        return {"elements": nodes + ways}

    def _fetch(self, n_ways: int = 2, http_error: bool = False, rate_limit: bool = False):
        resp = _mock_http(self._overpass_response(n_ways), raise_on_raise_for_status=http_error)
        if rate_limit:
            resp.status_code = 429
            # Second call succeeds
            resp2 = _mock_http(self._overpass_response(n_ways))
            responses = [resp, resp2]
            mock_post = MagicMock(side_effect=responses)
        else:
            mock_post = MagicMock(return_value=resp)

        with patch("ingestion.rail.osm_overpass_ingestor.RAIL_CORRIDORS", _FAKE_CORRIDOR):
            with patch("ingestion.rail.osm_overpass_ingestor.httpx.post", mock_post):
                with patch("ingestion.rail.osm_overpass_ingestor.time.sleep"):
                    return OSMRailIngestor().fetch()

    def test_returns_one_record_per_corridor(self):
        records = self._fetch()
        assert len(records) == 1

    def test_record_source_is_osm_rail(self):
        records = self._fetch()
        assert records[0].source == "osm_rail"

    def test_record_data_type_is_rail(self):
        records = self._fetch()
        assert records[0].data_type == "rail"

    def test_record_commodity_matches_corridor(self):
        records = self._fetch()
        assert records[0].commodity == "copper"

    def test_record_symbol_is_corridor_slug(self):
        records = self._fetch()
        assert records[0].symbol == "antofagasta_calama"

    def test_raw_json_contains_way_count(self):
        records = self._fetch(n_ways=3)
        raw = json.loads(records[0].raw_json)
        assert raw["way_count"] == 3

    def test_raw_json_contains_total_length_km(self):
        records = self._fetch()
        raw = json.loads(records[0].raw_json)
        assert raw["total_length_km"] > 0.0

    def test_raw_json_contains_geojson_feature_collection(self):
        records = self._fetch()
        raw = json.loads(records[0].raw_json)
        assert raw["geojson"]["type"] == "FeatureCollection"
        assert isinstance(raw["geojson"]["features"], list)

    def test_http_error_continues_gracefully(self):
        records = self._fetch(http_error=True)
        assert records == []

    def test_timestamp_is_day_floor(self):
        records = self._fetch()
        ts = records[0].timestamp
        assert ts.hour == 0
        assert ts.minute == 0
        assert ts.second == 0

    def test_rate_limit_retried(self):
        # 429 on first call → sleep → second call succeeds
        records = self._fetch(rate_limit=True)
        assert len(records) == 1


class TestBuildOverpassQuery:
    def test_query_contains_railway_rail(self):
        q = _build_overpass_query([-24.5, -70.5, -22.0, -68.0])
        assert 'railway"="rail' in q

    def test_query_is_non_empty_string(self):
        q = _build_overpass_query([-24.5, -70.5, -22.0, -68.0])
        assert len(q) > 20
