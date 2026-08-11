"""
Tests for backend.app.services.alert_engine.

No Hugging Face model required — all deterministic.
"""

import sys
import pathlib

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pytest
from backend.app.services.alert_engine import generate_alerts


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _zone(
    zid: str = "turn_4",
    name: str = "Turn 4",
    wetness: float = 0.82,
    risk: str = "High",
    trend: str = "Stable",
    confidence: float = 0.91,
) -> dict:
    return {
        "id": zid,
        "name": name,
        "wetness_score": wetness,
        "risk": risk,
        "trend": trend,
        "confidence": confidence,
        "base_condition": "Wet" if wetness >= 0.75 else "Damp" if wetness >= 0.45 else "Dry",
        "display_condition": trend if trend in ("Drying", "Worsening") else (
            "Wet" if wetness >= 0.75 else "Damp" if wetness >= 0.45 else "Dry"
        ),
    }


CORNER_META = {
    "main_straight": {"is_corner": False},
    "turn_2": {"is_corner": True},
    "turn_4": {"is_corner": True},
    "final_corner": {"is_corner": True},
}


# =========================================================================
# High-risk zone alerts
# =========================================================================


class TestHighRisk:
    def test_high_risk_generates_alert(self):
        zones = [_zone("turn_4", "Turn 4", 0.82, "High")]
        alerts = generate_alerts(zones, zone_meta=CORNER_META)
        high_risk = [a for a in alerts if a["id"] == "high_risk_turn_4"]
        assert len(high_risk) == 1
        assert "high-risk" in high_risk[0]["message"].lower()

    def test_low_risk_no_alert(self):
        zones = [_zone("turn_2", "Turn 2", 0.20, "Low")]
        alerts = generate_alerts(zones, zone_meta=CORNER_META)
        high_risk = [a for a in alerts if "high_risk" in a["id"]]
        assert len(high_risk) == 0

    def test_high_risk_corner_is_high_severity(self):
        zones = [_zone("turn_4", "Turn 4", 0.85, "High")]
        alerts = generate_alerts(zones, zone_meta=CORNER_META)
        hr = [a for a in alerts if a["id"] == "high_risk_turn_4"]
        assert hr[0]["severity"] == "high"

    def test_high_risk_straight_is_medium_severity(self):
        zones = [_zone("main_straight", "Main Straight", 0.80, "High")]
        alerts = generate_alerts(zones, zone_meta=CORNER_META)
        hr = [a for a in alerts if a["id"] == "high_risk_main_straight"]
        assert hr[0]["severity"] == "medium"


# =========================================================================
# Drying / worsening alerts
# =========================================================================


class TestTrendAlerts:
    def test_drying_alert(self):
        zones = [_zone("turn_2", "Turn 2", 0.50, "Medium")]
        alerts = generate_alerts(zones, overall_trend="Drying", zone_meta=CORNER_META)
        drying = [a for a in alerts if a["id"] == "drying"]
        assert len(drying) == 1
        assert "drying" in drying[0]["message"].lower()

    def test_worsening_per_zone(self):
        zones = [_zone("turn_2", "Turn 2", 0.60, "Medium", trend="Worsening")]
        alerts = generate_alerts(zones, zone_meta=CORNER_META)
        worsening = [a for a in alerts if a["id"] == "worsening_turn_2"]
        assert len(worsening) == 1
        assert "worsening" in worsening[0]["message"].lower()


# =========================================================================
# Low confidence
# =========================================================================


class TestLowConfidence:
    def test_low_confidence_alert(self):
        zones = [_zone("turn_2", "Turn 2", 0.50, "Medium", confidence=0.3)]
        alerts = generate_alerts(zones, zone_meta=CORNER_META)
        low = [a for a in alerts if "low_confidence" in a["id"]]
        assert len(low) >= 1

    def test_high_confidence_no_alert(self):
        zones = [_zone("turn_2", "Turn 2", 0.50, "Medium", confidence=0.9)]
        alerts = generate_alerts(zones, overall_confidence=0.9, zone_meta=CORNER_META)
        low = [a for a in alerts if "low_confidence" in a["id"]]
        assert len(low) == 0


# =========================================================================
# Deduplication
# =========================================================================


class TestDeduplication:
    def test_no_duplicate_alerts(self):
        zones = [
            _zone("turn_4", "Turn 4", 0.82, "High"),
            _zone("turn_4", "Turn 4", 0.82, "High"),  # duplicate zone
        ]
        alerts = generate_alerts(zones, zone_meta=CORNER_META)
        ids = [a["id"] for a in alerts]
        assert len(ids) == len(set(ids)), f"Duplicate alert IDs: {ids}"
