"""
Tests for backend.app.services.recommendation_engine.

No Hugging Face model required — all deterministic.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pytest
from backend.app.services.recommendation_engine import (
    generate_recommendation,
    generate_radio_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zone(
    zid: str = "turn_4",
    name: str = "Turn 4",
    wetness: float = 0.82,
    risk: str = "High",
    trend: str = "Stable",
) -> dict:
    return {
        "id": zid,
        "name": name,
        "wetness_score": wetness,
        "risk": risk,
        "trend": trend,
    }


# =========================================================================
# Tire recommendation rules
# =========================================================================


class TestTireRecommendation:
    def test_wet_tires(self):
        """Rule 1: overall >= 0.75 → Wet tires."""
        rec = generate_recommendation(0.82, "Stable", [_zone()])
        assert rec["suggested_tire"] == "Wet"
        assert "Wet tires" in rec["text"]

    def test_intermediate_tires(self):
        """Rule 2: 0.45 <= overall < 0.75 and not Drying."""
        rec = generate_recommendation(0.55, "Stable", [_zone(wetness=0.55, risk="Medium")])
        assert rec["suggested_tire"] == "Intermediate"
        assert "Intermediate" in rec["text"]

    def test_drying_tire_window(self):
        """Rule 3: 0.25–0.55 and Drying → Intermediate to Slick."""
        rec = generate_recommendation(0.45, "Drying", [_zone(wetness=0.45, risk="Medium")])
        assert rec["suggested_tire"] == "Intermediate to Slick"
        assert "window" in rec["text"].lower() or "approaching" in rec["text"].lower()

    def test_slick_tires(self):
        """Rule 4: < 0.25 and Stable → Slick."""
        rec = generate_recommendation(0.15, "Stable", [_zone(wetness=0.15, risk="Low")])
        assert rec["suggested_tire"] == "Slick"
        assert "Slick" in rec["text"]

    def test_high_risk_corner_modifier(self):
        """If a corner is high-risk, the recommendation should mention it."""
        zones = [
            _zone("turn_4", "Turn 4", 0.82, "High"),
            _zone("main_straight", "Main Straight", 0.20, "Low"),
        ]
        # Overall is damp-ish but Turn 4 is very wet
        rec = generate_recommendation(0.51, "Drying", zones)
        assert "Turn 4" in rec["text"]
        assert "high risk" in rec["text"].lower()

    def test_confidence_passed_through(self):
        rec = generate_recommendation(0.50, "Stable", [], overall_confidence=0.77)
        assert rec["confidence"] == 0.77


# =========================================================================
# Radio call generation
# =========================================================================


class TestRadioCall:
    def test_contains_radio_text(self):
        call = generate_radio_call(
            [_zone()], 0.82, "Stable"
        )
        assert len(call["text"]) > 0

    def test_drying_call(self):
        call = generate_radio_call(
            [_zone(wetness=0.50, risk="Medium")], 0.50, "Drying"
        )
        assert "drying" in call["text"].lower()
        assert call["severity"] in ("medium", "high")

    def test_worsening_call(self):
        call = generate_radio_call(
            [_zone()], 0.80, "Worsening"
        )
        assert "worsening" in call["text"].lower()
        assert call["severity"] == "high"

    def test_stable_low_wetness(self):
        call = generate_radio_call(
            [_zone(wetness=0.20, risk="Low")], 0.20, "Stable"
        )
        assert "stable" in call["text"].lower()
        assert call["should_play"] is False

    def test_high_risk_zone_mentioned(self):
        call = generate_radio_call(
            [_zone("turn_4", "Turn 4", 0.85, "High")], 0.85, "Stable"
        )
        assert "Turn 4" in call["text"]
        assert call["severity"] == "high"
        assert call["should_play"] is True

    def test_should_play_false_for_low(self):
        call = generate_radio_call(
            [_zone(wetness=0.10, risk="Low")], 0.10, "Stable"
        )
        assert call["should_play"] is False
