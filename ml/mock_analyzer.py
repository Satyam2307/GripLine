"""
ml.mock_analyzer — Deterministic mock inference for GripLine.

Activated when ``GRIPLINE_MOCK=true``.  Returns predictable, JSON-serialisable
results that simulate a wet track drying over time, with Turn 4 remaining
wetter than other zones.

Public class
------------
    MockAnalyzer()
        .analyze_image_mock(timestamp_seconds)
        .analyze_video_mock(num_frames, fps)
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

_LOGGER = logging.getLogger("gripline.mock")
_CONFIG_PATH = pathlib.Path(__file__).parent / "zone_config.json"


def _load_zone_defs() -> list[dict]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)["zones"]


# ---------------------------------------------------------------------------
# Pre-baked per-zone drying curves
# ---------------------------------------------------------------------------

# Each entry is (dry, damp, wet) probabilities at a nominal frame index.
# Turn 4 stays wetter than the rest.

_MOCK_CURVES: dict[str, list[tuple[float, float, float]]] = {
    "main_straight": [
        (0.05, 0.13, 0.82),  # frame 0  → wetness ≈ 0.885
        (0.15, 0.35, 0.50),  # frame 1  → wetness ≈ 0.675
        (0.40, 0.40, 0.20),  # frame 2  → wetness ≈ 0.400
        (0.60, 0.30, 0.10),  # frame 3  → wetness ≈ 0.250
        (0.75, 0.20, 0.05),  # frame 4  → wetness ≈ 0.150
    ],
    "turn_2": [
        (0.06, 0.14, 0.80),
        (0.18, 0.32, 0.50),
        (0.38, 0.42, 0.20),
        (0.55, 0.35, 0.10),
        (0.70, 0.24, 0.06),
    ],
    "turn_4": [
        (0.04, 0.10, 0.86),  # wettest at every step
        (0.08, 0.22, 0.70),
        (0.15, 0.30, 0.55),
        (0.22, 0.33, 0.45),
        (0.30, 0.35, 0.35),
    ],
    "final_corner": [
        (0.06, 0.12, 0.82),
        (0.20, 0.30, 0.50),
        (0.42, 0.38, 0.20),
        (0.58, 0.32, 0.10),
        (0.72, 0.22, 0.06),
    ],
}


class MockAnalyzer:
    """Deterministic mock analyser for integration testing and demos."""

    def __init__(self) -> None:
        self._zone_defs = _load_zone_defs()
        self._call_count = 0  # tracks successive calls for drying sim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _probs_to_dict(probs: tuple[float, float, float]) -> dict[str, float]:
        return {
            "dry": round(probs[0], 4),
            "damp": round(probs[1], 4),
            "wet": round(probs[2], 4),
        }

    @staticmethod
    def _wetness(probs: dict[str, float]) -> float:
        return round(
            probs["dry"] * 0.0 + probs["damp"] * 0.5 + probs["wet"] * 1.0,
            4,
        )

    @staticmethod
    def _base_condition(score: float) -> str:
        if score <= 0.39:
            return "Dry"
        if score <= 0.69:
            return "Damp"
        return "Wet"

    @staticmethod
    def _risk_level(score: float) -> str:
        if score <= 0.44:
            return "Low"
        if score <= 0.74:
            return "Medium"
        return "High"

    def _zone_prediction(
        self,
        zone_id: str,
        zone_name: str,
        frame_idx: int,
    ) -> dict[str, Any]:
        curve = _MOCK_CURVES[zone_id]
        idx = min(frame_idx, len(curve) - 1)
        probs = self._probs_to_dict(curve[idx])
        w = self._wetness(probs)
        cond = self._base_condition(w)
        return {
            "id": zone_id,
            "name": zone_name,
            "base_condition": cond,
            "display_condition": cond,
            "wetness_score": w,
            "trend": "Stable",
            "confidence": round(max(probs.values()), 4),
            "risk_level": self._risk_level(w),
            "class_probabilities": probs,
        }

    # ------------------------------------------------------------------
    # Public API (mirrors inference.analyze_image return shape)
    # ------------------------------------------------------------------

    def analyze_image_mock(
        self,
        timestamp_seconds: float = 0.0,
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Return a deterministic prediction for one frame.

        Parameters
        ----------
        timestamp_seconds : float
        frame_index : int, optional
            If not given, uses the internal call counter so successive
            calls simulate drying.
        """
        idx = frame_index if frame_index is not None else self._call_count
        self._call_count += 1

        zones = [
            self._zone_prediction(zd["id"], zd["name"], idx)
            for zd in self._zone_defs
        ]

        overall = round(
            sum(z["wetness_score"] for z in zones) / len(zones), 4
        )

        return {
            "timestamp_seconds": timestamp_seconds,
            "zones": zones,
            "overall_wetness": overall,
        }

    def analyze_video_mock(
        self,
        num_frames: int = 5,
        fps: float = 30.0,
    ) -> dict[str, Any]:
        """
        Return a deterministic multi-frame analysis simulating drying.

        Parameters
        ----------
        num_frames : int
            Number of frames to simulate.
        fps : float
            Frames per second for timestamp calculation.
        """
        from ml.trend import (
            calculate_zone_trend,
            calculate_overall_prediction,
            display_condition as trend_display,
            radio_call,
        )

        frames: list[dict[str, Any]] = []
        for i in range(num_frames):
            ts = round(i * (1.0 / fps) * 30, 2)  # simulated stride
            frame = self.analyze_image_mock(
                timestamp_seconds=ts, frame_index=i
            )
            frames.append(frame)

        # --- Build per-zone histories and compute trends ---------------
        zone_ids = [zd["id"] for zd in self._zone_defs]
        final_zones: list[dict[str, Any]] = []

        for zid in zone_ids:
            history = [
                next(z for z in f["zones"] if z["id"] == zid)
                for f in frames
            ]
            trend_info = calculate_zone_trend(history)
            latest = history[-1].copy()
            latest["trend"] = trend_info["trend"]
            latest["display_condition"] = trend_info["display_condition"]
            final_zones.append(latest)

        overall_pred = calculate_overall_prediction(final_zones)
        overall_trend = "Drying"  # mock always dries

        timeline = [
            {
                "timestamp_seconds": f["timestamp_seconds"],
                "overall_wetness": f["overall_wetness"],
            }
            for f in frames
        ]

        return {
            "frames": frames,
            "summary": {
                "zones": final_zones,
                "overall_wetness": overall_pred["overall_wetness"],
                "overall_trend": overall_trend,
                "highest_risk_zone": overall_pred["highest_risk_zone"],
                "tire_recommendation": overall_pred["tire_recommendation"],
                "radio_call": radio_call(
                    final_zones,
                    overall_pred["overall_wetness"],
                    overall_trend,
                ),
            },
            "timeline": timeline,
        }
