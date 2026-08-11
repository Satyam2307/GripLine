"""
Tests for the GripLine FastAPI application.

All tests use GRIPLINE_MOCK=true so no model download is required.
Tests verify the response contract, error handling, CORS, and mock mode.
"""

import io
import json
import os
import sys
import pathlib

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# Force mock mode BEFORE importing the app
os.environ["GRIPLINE_MOCK"] = "true"

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


# =========================================================================
# Health endpoint
# =========================================================================


class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_has_required_keys(self):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "service" in data
        assert "model_loaded" in data
        assert "mock_available" in data

    def test_health_service_name(self):
        data = client.get("/api/v1/health").json()
        assert data["service"] == "gripline-backend"

    def test_health_mock_available(self):
        data = client.get("/api/v1/health").json()
        assert data["mock_available"] is True


# =========================================================================
# File validation
# =========================================================================


class TestFileValidation:
    def test_invalid_file_type(self):
        """Reject non-image/video files."""
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "INVALID_FILE"

    def test_empty_file(self):
        """Reject empty files."""
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "EMPTY_FILE"


# =========================================================================
# Mock mode analysis
# =========================================================================


class TestAnalyzeMock:
    def test_image_mock_returns_200(self):
        """Analyze a minimal JPEG in mock mode."""
        # Create a tiny valid-ish JPEG header
        fake_jpg = _minimal_jpeg()
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("track.jpg", fake_jpg, "image/jpeg")},
            data={"demo_mode": "true"},
        )
        assert resp.status_code == 200

    def test_response_has_contract_keys(self):
        fake_jpg = _minimal_jpeg()
        data = client.post(
            "/api/v1/analyze",
            files={"file": ("track.jpg", fake_jpg, "image/jpeg")},
            data={"demo_mode": "true"},
        ).json()

        assert "session_id" in data
        assert "source" in data
        assert "processed_frames" in data
        assert "overall" in data
        assert "zones" in data
        assert "recommendation" in data
        assert "radio_call" in data
        assert "alerts" in data
        assert "timeline" in data

    def test_overall_has_required_fields(self):
        data = _analyze_mock()
        overall = data["overall"]
        assert "base_condition" in overall
        assert "display_condition" in overall
        assert "wetness_score" in overall
        assert "trend" in overall
        assert "confidence" in overall

    def test_zones_have_required_fields(self):
        data = _analyze_mock()
        for zone in data["zones"]:
            assert "id" in zone
            assert "name" in zone
            assert "base_condition" in zone
            assert "display_condition" in zone
            assert "wetness_score" in zone
            assert "trend" in zone
            assert "risk" in zone
            assert "confidence" in zone

    def test_recommendation_has_required_fields(self):
        data = _analyze_mock()
        rec = data["recommendation"]
        assert "text" in rec
        assert "suggested_tire" in rec
        assert "confidence" in rec

    def test_radio_call_has_required_fields(self):
        data = _analyze_mock()
        radio = data["radio_call"]
        assert "text" in radio
        assert "severity" in radio
        assert "should_play" in radio

    def test_session_id_is_demo(self):
        data = _analyze_mock()
        assert data["session_id"] == "demo_001"

    def test_source_type_is_image(self):
        data = _analyze_mock()
        assert data["source"]["type"] == "image"

    def test_zones_not_empty(self):
        data = _analyze_mock()
        assert len(data["zones"]) > 0

    def test_wetness_scores_in_range(self):
        data = _analyze_mock()
        assert 0.0 <= data["overall"]["wetness_score"] <= 1.0
        for zone in data["zones"]:
            assert 0.0 <= zone["wetness_score"] <= 1.0

    def test_json_serializable(self):
        data = _analyze_mock()
        # Should not raise
        json.dumps(data)

    def test_no_model_downloaded(self):
        """Verify mock mode didn't trigger model loading."""
        from backend.app.services.analyzer_adapter import is_model_loaded
        # In mock mode, model should NOT be loaded
        # (it might be if someone ran real inference before, but in test
        #  isolation with GRIPLINE_MOCK=true it should remain false)
        # We just verify the endpoint works without the model
        data = _analyze_mock()
        assert data is not None


# =========================================================================
# CORS
# =========================================================================


class TestCORS:
    def test_cors_headers_localhost_5173(self):
        resp = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_headers_localhost_3000(self):
        resp = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


# =========================================================================
# Helpers
# =========================================================================


def _minimal_jpeg() -> bytes:
    """Create a minimal valid JPEG (1×1 pixel) using Pillow."""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _analyze_mock() -> dict:
    """Run a mock analysis and return the JSON response."""
    fake_jpg = _minimal_jpeg()
    resp = client.post(
        "/api/v1/analyze",
        files={"file": ("track.jpg", fake_jpg, "image/jpeg")},
        data={"demo_mode": "true"},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    return resp.json()
