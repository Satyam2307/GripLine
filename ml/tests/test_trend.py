"""
Tests for ml.trend — exponential smoothing, trend detection, thresholds.

All tests are offline: no Hugging Face model required.
"""

import pytest
import numpy as np

from ml.trend import (
    exponential_smooth,
    trend_from_delta,
    display_condition,
    calculate_zone_trend,
    calculate_overall_prediction,
    tire_recommendation,
    radio_call,
)
from ml.inference import (
    wetness_score_from_probs,
    base_condition_from_score,
    risk_level_from_score,
    confidence_from_probs,
    crop_zone_from_image,
)
import numpy as np


# =========================================================================
# Wetness score calculation
# =========================================================================


class TestWetnessScore:
    def test_fully_dry(self):
        probs = {"dry": 1.0, "damp": 0.0, "wet": 0.0}
        assert wetness_score_from_probs(probs) == 0.0

    def test_fully_wet(self):
        probs = {"dry": 0.0, "damp": 0.0, "wet": 1.0}
        assert wetness_score_from_probs(probs) == 1.0

    def test_fully_damp(self):
        probs = {"dry": 0.0, "damp": 1.0, "wet": 0.0}
        assert wetness_score_from_probs(probs) == 0.5

    def test_mixed(self):
        probs = {"dry": 0.2, "damp": 0.5, "wet": 0.3}
        expected = 0.2 * 0.0 + 0.5 * 0.5 + 0.3 * 1.0  # = 0.55
        assert wetness_score_from_probs(probs) == pytest.approx(expected, abs=1e-3)


# =========================================================================
# Threshold mapping
# =========================================================================


class TestBaseCondition:
    def test_dry_low(self):
        assert base_condition_from_score(0.0) == "Dry"

    def test_dry_boundary(self):
        assert base_condition_from_score(0.39) == "Dry"

    def test_damp_low(self):
        assert base_condition_from_score(0.40) == "Damp"

    def test_damp_boundary(self):
        assert base_condition_from_score(0.69) == "Damp"

    def test_wet_low(self):
        assert base_condition_from_score(0.70) == "Wet"

    def test_wet_high(self):
        assert base_condition_from_score(1.0) == "Wet"


class TestRiskLevel:
    def test_low(self):
        assert risk_level_from_score(0.30) == "Low"

    def test_low_boundary(self):
        assert risk_level_from_score(0.44) == "Low"

    def test_medium(self):
        assert risk_level_from_score(0.50) == "Medium"

    def test_medium_boundary(self):
        assert risk_level_from_score(0.74) == "Medium"

    def test_high(self):
        assert risk_level_from_score(0.80) == "High"


class TestConfidence:
    def test_high_confidence(self):
        probs = {"dry": 0.05, "damp": 0.05, "wet": 0.90}
        assert confidence_from_probs(probs) == 0.90

    def test_uniform_confidence(self):
        probs = {"dry": 0.33, "damp": 0.34, "wet": 0.33}
        assert confidence_from_probs(probs) == 0.34


# =========================================================================
# Exponential smoothing
# =========================================================================


class TestExponentialSmoothing:
    def test_single_value(self):
        assert exponential_smooth([0.8]) == [0.8]

    def test_constant_series(self):
        result = exponential_smooth([0.5, 0.5, 0.5])
        assert result == [0.5, 0.5, 0.5]

    def test_decreasing(self):
        result = exponential_smooth([0.8, 0.6, 0.4], alpha=0.5)
        # s0 = 0.8
        # s1 = 0.5*0.6 + 0.5*0.8 = 0.7
        # s2 = 0.5*0.4 + 0.5*0.7 = 0.55
        assert result[0] == pytest.approx(0.8, abs=1e-3)
        assert result[1] == pytest.approx(0.7, abs=1e-3)
        assert result[2] == pytest.approx(0.55, abs=1e-3)

    def test_empty_list(self):
        assert exponential_smooth([]) == []


# =========================================================================
# Trend detection
# =========================================================================


class TestTrendFromDelta:
    def test_drying(self):
        assert trend_from_delta(-0.10) == "Drying"

    def test_drying_boundary(self):
        assert trend_from_delta(-0.08) == "Drying"

    def test_stable(self):
        assert trend_from_delta(0.0) == "Stable"

    def test_stable_small_negative(self):
        assert trend_from_delta(-0.05) == "Stable"

    def test_stable_small_positive(self):
        assert trend_from_delta(0.05) == "Stable"

    def test_worsening(self):
        assert trend_from_delta(0.12) == "Worsening"

    def test_worsening_boundary(self):
        assert trend_from_delta(0.08) == "Worsening"


class TestDisplayCondition:
    def test_drying_override(self):
        assert display_condition("Wet", "Drying") == "Drying"

    def test_worsening_override(self):
        assert display_condition("Damp", "Worsening") == "Worsening"

    def test_stable_keeps_base(self):
        assert display_condition("Wet", "Stable") == "Wet"

    def test_dry_stable(self):
        assert display_condition("Dry", "Stable") == "Dry"


# =========================================================================
# Zone trend (integrated)
# =========================================================================


class TestCalculateZoneTrend:
    def test_drying_history(self):
        history = [
            {"wetness_score": 0.85},
            {"wetness_score": 0.65},
            {"wetness_score": 0.45},
        ]
        result = calculate_zone_trend(history)
        assert result["trend"] == "Drying"
        assert result["delta"] < 0
        assert len(result["smoothed_scores"]) == 3

    def test_worsening_history(self):
        history = [
            {"wetness_score": 0.20},
            {"wetness_score": 0.50},
            {"wetness_score": 0.80},
        ]
        result = calculate_zone_trend(history)
        assert result["trend"] == "Worsening"
        assert result["delta"] > 0

    def test_stable_history(self):
        history = [
            {"wetness_score": 0.50},
            {"wetness_score": 0.52},
            {"wetness_score": 0.51},
        ]
        result = calculate_zone_trend(history)
        assert result["trend"] == "Stable"

    def test_empty_history(self):
        result = calculate_zone_trend([])
        assert result["trend"] == "Stable"
        assert result["smoothed_scores"] == []

    def test_single_entry(self):
        result = calculate_zone_trend([{"wetness_score": 0.60}])
        assert result["trend"] == "Stable"
        assert result["delta"] == 0.0


# =========================================================================
# Overall prediction
# =========================================================================


class TestCalculateOverallPrediction:
    def test_basic(self):
        zones = [
            {"id": "z1", "wetness_score": 0.80, "risk_level": "High"},
            {"id": "z2", "wetness_score": 0.40, "risk_level": "Low"},
        ]
        result = calculate_overall_prediction(zones)
        assert result["overall_wetness"] == pytest.approx(0.60, abs=1e-3)
        assert result["highest_risk_zone"] == "z1"

    def test_empty(self):
        result = calculate_overall_prediction([])
        assert result["overall_wetness"] == 0.0
        assert result["highest_risk_zone"] is None


# =========================================================================
# Tire recommendation
# =========================================================================


class TestTireRecommendation:
    def test_very_wet(self):
        assert "Wet tires" in tire_recommendation(0.80)

    def test_intermediate(self):
        assert "Intermediate" in tire_recommendation(0.60)

    def test_window(self):
        assert "window" in tire_recommendation(0.45).lower()

    def test_slicks(self):
        assert "Slick" in tire_recommendation(0.30)

    def test_no_change(self):
        assert "No immediate" in tire_recommendation(0.10)


# =========================================================================
# Radio call
# =========================================================================


class TestRadioCall:
    def test_drying_call(self):
        zones = [{"id": "turn_4", "name": "Turn 4", "wetness_score": 0.70,
                  "display_condition": "Drying"}]
        msg = radio_call(zones, 0.70, "Drying")
        assert "drying" in msg.lower()

    def test_wet_stable(self):
        zones = [{"id": "turn_4", "name": "Turn 4", "wetness_score": 0.85,
                  "display_condition": "Wet", "base_condition": "Wet"}]
        msg = radio_call(zones, 0.85, "Stable")
        assert "Turn 4" in msg

    def test_empty_zones(self):
        msg = radio_call([], 0.0, "Stable")
        assert "No zone data" in msg


# =========================================================================
# Crop conversion (normalised coords → pixel coords)
# =========================================================================


class TestCropZone:
    def test_full_image_crop(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = crop_zone_from_image(
            img, {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
        )
        assert crop.shape == (100, 200, 3)

    def test_quarter_crop(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = crop_zone_from_image(
            img, {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5}
        )
        assert crop.shape == (50, 100, 3)

    def test_out_of_bounds_clamped(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = crop_zone_from_image(
            img, {"x_min": -0.1, "y_min": -0.1, "x_max": 1.2, "y_max": 1.2}
        )
        assert crop.shape == (100, 200, 3)
