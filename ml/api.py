"""
ml.api — FastAPI REST server for GripLine ML inference.

Endpoints
---------
    GET  /api/v1/health      → service health check
    POST /api/v1/analyze      → video / image analysis (returns full prediction)

Run:
    GRIPLINE_MOCK=true uvicorn ml.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.inference import (
    base_condition_from_score,
    is_mock_mode,
    is_model_available,
    risk_level_from_score,
)
from ml.trend import (
    calculate_overall_prediction,
    tire_recommendation,
    radio_call,
    trend_from_delta,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s",
)
_LOGGER = logging.getLogger("gripline.api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_START_TIME = time.time()

app = FastAPI(
    title="GripLine ML API",
    description="AI co-pilot for track-condition analysis",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response models (match the target contract)
# ---------------------------------------------------------------------------


class ZoneResponse(BaseModel):
    id: str
    name: str
    base_condition: str
    display_condition: str
    wetness_score: float
    trend: str
    risk: str
    confidence: float


class OverallResponse(BaseModel):
    base_condition: str
    display_condition: str
    wetness_score: float
    trend: str
    confidence: float


class RecommendationResponse(BaseModel):
    text: str
    suggested_tire: str
    confidence: float


class RadioCallResponse(BaseModel):
    text: str
    severity: str
    should_play: bool


class AlertResponse(BaseModel):
    id: str
    timestamp_seconds: float
    severity: str
    zone_id: str
    message: str


class TimelineEntry(BaseModel):
    timestamp_seconds: float
    overall_wetness: float
    trend: str


class SourceInfo(BaseModel):
    filename: str
    type: str
    duration_seconds: float


class AnalyzeResponse(BaseModel):
    session_id: str
    source: SourceInfo
    processed_frames: int
    overall: OverallResponse
    zones: list[ZoneResponse]
    recommendation: RecommendationResponse
    radio_call: RadioCallResponse
    alerts: list[AlertResponse]
    timeline: list[TimelineEntry]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _severity_from_risk(risk: str) -> str:
    """Map risk level → alert severity."""
    return {"Low": "low", "Medium": "medium", "High": "high"}.get(risk, "low")


def _suggested_tire(overall_wetness: float) -> str:
    """Map overall wetness to a tire compound name."""
    if overall_wetness >= 0.75:
        return "Full Wet"
    if overall_wetness >= 0.45:
        return "Intermediate"
    if overall_wetness >= 0.25:
        return "Intermediate / Slick"
    return "Slick"


def _build_radio_call_response(
    zone_predictions: list[dict[str, Any]],
    overall_wetness: float,
    overall_trend: str,
) -> RadioCallResponse:
    """Build a structured radio-call response."""
    text = radio_call(zone_predictions, overall_wetness, overall_trend)
    severity = "high" if overall_wetness >= 0.60 else (
        "medium" if overall_wetness >= 0.40 else "low"
    )
    should_play = overall_trend != "Stable" or overall_wetness >= 0.60
    return RadioCallResponse(text=text, severity=severity, should_play=should_play)


def _build_alerts(
    zone_predictions: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
) -> list[AlertResponse]:
    """Generate alerts for high-risk zones."""
    alerts: list[AlertResponse] = []
    for zone in zone_predictions:
        risk = zone.get("risk_level", zone.get("risk", "Low"))
        if risk == "High":
            # Use the latest timestamp from the timeline
            ts = timeline[-1]["timestamp_seconds"] if timeline else 0.0
            alerts.append(
                AlertResponse(
                    id=f"alert_{uuid.uuid4().hex[:6]}",
                    timestamp_seconds=ts,
                    severity="high",
                    zone_id=zone["id"],
                    message=f"{zone['name']} remains {zone.get('display_condition', zone.get('base_condition', 'wet')).lower()} and is a high-risk zone.",
                )
            )
    return alerts


def _build_analyze_response(
    video_result: dict[str, Any],
    source_filename: str,
    source_type: str = "video",
    duration_seconds: float = 0.0,
) -> AnalyzeResponse:
    """
    Transform the raw ML output (from analyze_video / mock) into
    the target API response contract.
    """
    summary = video_result["summary"]
    frames = video_result.get("frames", [])
    raw_timeline = video_result.get("timeline", [])

    # --- Zones ---
    zone_responses: list[ZoneResponse] = []
    for z in summary["zones"]:
        zone_responses.append(
            ZoneResponse(
                id=z["id"],
                name=z["name"],
                base_condition=z["base_condition"],
                display_condition=z["display_condition"],
                wetness_score=z["wetness_score"],
                trend=z["trend"],
                risk=z.get("risk_level", z.get("risk", risk_level_from_score(z["wetness_score"]))),
                confidence=z.get("confidence", 0.0),
            )
        )

    # --- Overall ---
    overall_wetness = summary["overall_wetness"]
    overall_trend = summary.get("overall_trend", "Stable")
    overall_base = base_condition_from_score(overall_wetness)
    overall_display = overall_trend if overall_trend != "Stable" else overall_base

    avg_confidence = (
        round(sum(z.confidence for z in zone_responses) / len(zone_responses), 2)
        if zone_responses
        else 0.0
    )

    overall = OverallResponse(
        base_condition=overall_base,
        display_condition=overall_display,
        wetness_score=round(overall_wetness, 2),
        trend=overall_trend,
        confidence=avg_confidence,
    )

    # --- Recommendation ---
    rec_text = summary.get("tire_recommendation", tire_recommendation(overall_wetness))
    highest_risk = summary.get("highest_risk_zone")
    if highest_risk:
        worst_zone = next((z for z in zone_responses if z.id == highest_risk), None)
        if worst_zone and worst_zone.risk == "High":
            rec_text += f" {worst_zone.name} remains high risk."

    recommendation = RecommendationResponse(
        text=rec_text,
        suggested_tire=_suggested_tire(overall_wetness),
        confidence=avg_confidence,
    )

    # --- Radio call ---
    radio = _build_radio_call_response(
        summary["zones"], overall_wetness, overall_trend,
    )

    # --- Timeline (with trend) ---
    timeline_entries: list[TimelineEntry] = []
    for i, entry in enumerate(raw_timeline):
        if i == 0:
            t = "Stable"
        else:
            delta = entry["overall_wetness"] - raw_timeline[i - 1]["overall_wetness"]
            t = trend_from_delta(delta)
        timeline_entries.append(
            TimelineEntry(
                timestamp_seconds=entry["timestamp_seconds"],
                overall_wetness=round(entry["overall_wetness"], 2),
                trend=t,
            )
        )

    # --- Alerts ---
    alerts = _build_alerts(summary["zones"], raw_timeline)

    return AnalyzeResponse(
        session_id=f"session_{uuid.uuid4().hex[:8]}",
        source=SourceInfo(
            filename=source_filename,
            type=source_type,
            duration_seconds=round(duration_seconds, 1),
        ),
        processed_frames=len(frames),
        overall=overall,
        zones=zone_responses,
        recommendation=recommendation,
        radio_call=radio,
        alerts=alerts,
        timeline=timeline_entries,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------


@app.get("/api/v1/health")
def health_check() -> dict[str, Any]:
    """
    Returns service health, model readiness, and runtime metadata.
    """
    mock = is_mock_mode()
    model_ready = True if mock else is_model_available()
    uptime_seconds = round(time.time() - _START_TIME, 2)

    return {
        "status": "ok" if model_ready else "degraded",
        "version": app.version,
        "mock_mode": mock,
        "model_ready": model_ready,
        "uptime_seconds": uptime_seconds,
    }


# ---------------------------------------------------------------------------
# GET /api/v1/zones
# ---------------------------------------------------------------------------


@app.get("/api/v1/zones")
def get_zones() -> list[dict[str, Any]]:
    """
    Return zone definitions with layout coordinates for track-map rendering.
    """
    from ml.inference import load_zone_config

    cfg = load_zone_config()
    zones: list[dict[str, Any]] = []

    for z in cfg["zones"]:
        layout = z.get("map_layout", {})
        zones.append(
            {
                "id": z["id"],
                "name": z["name"],
                "x": layout.get("x", 0.0),
                "y": layout.get("y", 0.0),
                "width": layout.get("width", 0.25),
                "height": layout.get("height", 0.25),
                "is_corner": layout.get("is_corner", False),
                "description": z.get("description", ""),
            }
        )

    return zones


# ---------------------------------------------------------------------------
# POST /api/v1/analyze
# ---------------------------------------------------------------------------


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    frame_stride: int = Form(default=5),
    max_frames: int = Form(default=30),
) -> AnalyzeResponse:
    """
    Analyse an uploaded video file and return the full prediction contract.

    Accepts multipart/form-data with a video file and optional parameters.
    """
    import pathlib
    import tempfile

    from ml.video_processor import analyze_video

    # Validate file type
    content_type = file.content_type or ""
    filename = file.filename or "upload.mp4"

    if not (content_type.startswith("video/") or filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".webm")
    )):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Upload a video file.",
        )

    # Save to a temp file for OpenCV processing
    suffix = pathlib.Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        _LOGGER.info(
            "Analyzing video: %s (%d bytes, stride=%d, max_frames=%d)",
            filename, len(content), frame_stride, max_frames,
        )
        result = analyze_video(tmp_path, frame_stride=frame_stride, max_frames=max_frames)

        # Estimate duration from timeline
        timeline = result.get("timeline", [])
        duration = timeline[-1]["timestamp_seconds"] if timeline else 0.0

        return _build_analyze_response(
            video_result=result,
            source_filename=filename,
            source_type="video",
            duration_seconds=duration,
        )

    except Exception as exc:
        _LOGGER.exception("Analysis failed for %s", filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        import os
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# POST /api/v1/analyze/mock  (convenience — always uses mock mode)
# ---------------------------------------------------------------------------


@app.post("/api/v1/analyze/mock", response_model=AnalyzeResponse)
def analyze_mock_endpoint(
    num_frames: int = 5,
    filename: str = "mock_track.mp4",
    duration_seconds: float = 12.4,
) -> AnalyzeResponse:
    """
    Run a mock analysis without uploading a file.
    Useful for front-end development and demos.
    """
    from ml.mock_analyzer import MockAnalyzer

    mock = MockAnalyzer()
    result = mock.analyze_video_mock(num_frames=num_frames)

    return _build_analyze_response(
        video_result=result,
        source_filename=filename,
        source_type="video",
        duration_seconds=duration_seconds,
    )
