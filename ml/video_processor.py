"""
ml.video_processor — Video frame extraction and multi-frame analysis.

Uses OpenCV for frame sampling and delegates per-frame classification
to ``ml.inference.analyze_image`` (or ``MockAnalyzer`` in mock mode).

Public function
---------------
    analyze_video(video_path, frame_stride=5, max_frames=30)
        → dict   (JSON-serialisable video-level prediction)
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

import cv2
import numpy as np

_LOGGER = logging.getLogger("gripline.video")


def _is_mock() -> bool:
    return os.getenv("GRIPLINE_MOCK", "").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: str | pathlib.Path,
    frame_stride: int = 5,
    max_frames: int = 30,
) -> list[dict[str, Any]]:
    """
    Extract up to *max_frames* from *video_path*, taking every
    *frame_stride*-th frame.

    Returns
    -------
    list[dict]
        Each dict has keys ``frame`` (np.ndarray BGR) and
        ``timestamp_seconds`` (float).

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    ValueError
        If the file cannot be opened as a video.
    """
    path = pathlib.Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _LOGGER.info(
        "Video: %s — %.1f fps, %d total frames", path.name, fps, total_frames
    )

    frames: list[dict[str, Any]] = []
    frame_idx = 0

    while len(frames) < max_frames:
        target = frame_idx * frame_stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            break
        ts = round(target / fps, 3)
        frames.append({"frame": frame, "timestamp_seconds": ts})
        frame_idx += 1

    cap.release()
    _LOGGER.info("Extracted %d frames from %s.", len(frames), path.name)
    return frames


# ---------------------------------------------------------------------------
# Video-level analysis (public API)
# ---------------------------------------------------------------------------


def analyze_video(
    video_path: str | pathlib.Path,
    frame_stride: int = 5,
    max_frames: int = 30,
) -> dict[str, Any]:
    """
    End-to-end video analysis.

    1. Extract sampled frames.
    2. Classify each frame (per-zone).
    3. Compute trends and overall prediction.

    Parameters
    ----------
    video_path : str | Path
    frame_stride : int
        Take every N-th frame (default 5).
    max_frames : int
        Maximum number of frames to process (default 30).

    Returns
    -------
    dict
        JSON-serialisable result with keys:
        ``frames``, ``summary``, ``timeline``.
    """
    # --- Mock mode (no real video needed) ---
    if _is_mock():
        from ml.mock_analyzer import MockAnalyzer

        _LOGGER.info("Mock mode — returning simulated video analysis.")
        return MockAnalyzer().analyze_video_mock(
            num_frames=min(max_frames, 5),
        )

    # --- Real mode ---
    from ml.inference import analyze_image
    from ml.trend import (
        calculate_zone_trend,
        calculate_overall_prediction,
        display_condition as trend_display,
        radio_call,
        trend_from_delta,
    )

    extracted = extract_frames(video_path, frame_stride, max_frames)
    if not extracted:
        return {
            "frames": [],
            "summary": {
                "zones": [],
                "overall_wetness": 0.0,
                "overall_trend": "Stable",
                "highest_risk_zone": None,
                "tire_recommendation": "No data available.",
                "radio_call": "GripLine to pit wall. No frame data.",
            },
            "timeline": [],
        }

    # --- Per-frame classification ---
    frame_results: list[dict[str, Any]] = []
    prev_state: dict[str, Any] | None = None

    for item in extracted:
        result = analyze_image(
            item["frame"],
            timestamp_seconds=item["timestamp_seconds"],
            previous_state=prev_state,
        )
        frame_results.append(result)
        prev_state = result

    # --- Zone-level trend calculation ---
    # Collect zone IDs from the first frame
    zone_ids = [z["id"] for z in frame_results[0]["zones"]]
    final_zones: list[dict[str, Any]] = []

    for zid in zone_ids:
        history = []
        for fr in frame_results:
            match = [z for z in fr["zones"] if z["id"] == zid]
            if match:
                history.append(match[0])

        trend_info = calculate_zone_trend(history)
        latest = history[-1].copy() if history else {}
        latest["trend"] = trend_info["trend"]
        latest["display_condition"] = trend_info["display_condition"]
        final_zones.append(latest)

    # --- Overall aggregation ---
    overall_pred = calculate_overall_prediction(final_zones)

    # Overall trend from timeline wetness
    timeline_scores = [fr["overall_wetness"] for fr in frame_results]
    if len(timeline_scores) >= 2:
        from ml.trend import exponential_smooth

        smoothed = exponential_smooth(timeline_scores)
        overall_delta = smoothed[-1] - smoothed[0]
        overall_trend = trend_from_delta(overall_delta)
    else:
        overall_trend = "Stable"

    timeline = [
        {
            "timestamp_seconds": fr["timestamp_seconds"],
            "overall_wetness": fr["overall_wetness"],
        }
        for fr in frame_results
    ]

    return {
        "frames": frame_results,
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
