"""
Tests for ml.mock_analyzer — deterministic mock mode.

No Hugging Face model required.
"""

import json

import pytest

from ml.mock_analyzer import MockAnalyzer


# =========================================================================
# Single-frame mock analysis
# =========================================================================


class TestMockAnalyzeImage:
    def setup_method(self):
        self.mock = MockAnalyzer()

    def test_returns_four_zones(self):
        result = self.mock.analyze_image_mock(timestamp_seconds=0.0)
        assert len(result["zones"]) == 4

    def test_zone_ids(self):
        result = self.mock.analyze_image_mock()
        ids = {z["id"] for z in result["zones"]}
        assert ids == {"main_straight", "turn_2", "turn_4", "final_corner"}

    def test_has_required_keys(self):
        result = self.mock.analyze_image_mock()
        for zone in result["zones"]:
            assert "id" in zone
            assert "name" in zone
            assert "base_condition" in zone
            assert "display_condition" in zone
            assert "wetness_score" in zone
            assert "trend" in zone
            assert "confidence" in zone
            assert "class_probabilities" in zone
            assert "risk_level" in zone

    def test_probabilities_sum_to_one(self):
        result = self.mock.analyze_image_mock()
        for zone in result["zones"]:
            total = sum(zone["class_probabilities"].values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_wetness_score_in_range(self):
        result = self.mock.analyze_image_mock()
        for zone in result["zones"]:
            assert 0.0 <= zone["wetness_score"] <= 1.0

    def test_first_frame_is_wet(self):
        result = self.mock.analyze_image_mock(frame_index=0)
        assert result["overall_wetness"] > 0.70

    def test_turn_4_wettest(self):
        result = self.mock.analyze_image_mock(frame_index=0)
        scores = {z["id"]: z["wetness_score"] for z in result["zones"]}
        assert scores["turn_4"] >= max(
            scores["main_straight"],
            scores["turn_2"],
            scores["final_corner"],
        )

    def test_drying_over_calls(self):
        """Successive calls without explicit frame_index should dry."""
        mock = MockAnalyzer()
        r0 = mock.analyze_image_mock(timestamp_seconds=0.0)
        r1 = mock.analyze_image_mock(timestamp_seconds=1.0)
        r2 = mock.analyze_image_mock(timestamp_seconds=2.0)
        assert r0["overall_wetness"] > r1["overall_wetness"]
        assert r1["overall_wetness"] > r2["overall_wetness"]

    def test_json_serialisable(self):
        result = self.mock.analyze_image_mock()
        # Should not raise
        json.dumps(result)


# =========================================================================
# Multi-frame mock analysis
# =========================================================================


class TestMockAnalyzeVideo:
    def setup_method(self):
        self.mock = MockAnalyzer()

    def test_frames_returned(self):
        result = self.mock.analyze_video_mock(num_frames=3)
        assert len(result["frames"]) == 3

    def test_summary_keys(self):
        result = self.mock.analyze_video_mock(num_frames=3)
        summary = result["summary"]
        assert "zones" in summary
        assert "overall_wetness" in summary
        assert "overall_trend" in summary
        assert "highest_risk_zone" in summary
        assert "tire_recommendation" in summary
        assert "radio_call" in summary

    def test_timeline(self):
        result = self.mock.analyze_video_mock(num_frames=4)
        assert len(result["timeline"]) == 4
        for entry in result["timeline"]:
            assert "timestamp_seconds" in entry
            assert "overall_wetness" in entry

    def test_overall_trend_drying(self):
        result = self.mock.analyze_video_mock(num_frames=5)
        assert result["summary"]["overall_trend"] == "Drying"

    def test_turn_4_highest_risk(self):
        result = self.mock.analyze_video_mock(num_frames=5)
        assert result["summary"]["highest_risk_zone"] == "turn_4"

    def test_radio_call_not_empty(self):
        result = self.mock.analyze_video_mock(num_frames=3)
        assert len(result["summary"]["radio_call"]) > 0

    def test_json_serialisable(self):
        result = self.mock.analyze_video_mock(num_frames=5)
        json.dumps(result)


# =========================================================================
# Condition / threshold consistency
# =========================================================================


class TestMockConditionLabels:
    def test_wet_frame_zero(self):
        mock = MockAnalyzer()
        result = mock.analyze_image_mock(frame_index=0)
        # At frame 0, all zones should be Wet or very close to the boundary.
        for zone in result["zones"]:
            assert zone["base_condition"] in ("Wet", "Damp")

    def test_dry_frame_four(self):
        mock = MockAnalyzer()
        result = mock.analyze_image_mock(frame_index=4)
        # After drying, most zones should be Dry or Damp.
        conditions = {z["base_condition"] for z in result["zones"]}
        assert "Dry" in conditions or "Damp" in conditions
