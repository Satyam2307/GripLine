"""
backend.app.routes.analysis — /api/v1/health and /api/v1/analyze endpoints.
"""

from __future__ import annotations

import logging
import os
import pathlib
import tempfile
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from backend.app import config
from backend.app.schemas import (
    Alert,
    AnalyzeResponse,
    ErrorResponse,
    GripLineError,
    HealthResponse,
    OverallCondition,
    RadioCall,
    Recommendation,
    SourceMetadata,
    TimelinePoint,
    ZonePrediction,
)
from backend.app.services import alert_engine, recommendation_engine
from backend.app.services.analyzer_adapter import (
    analyze_file,
    is_model_loaded,
    try_load_model,
)

_LOGGER = logging.getLogger("gripline.routes")

router = APIRouter(prefix="/api/v1")


# =========================================================================
# GET /health
# =========================================================================


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Report service health and model availability.

    Does NOT attempt to download the model — only checks cached state.
    """
    loaded = is_model_loaded()
    mock_env = config.GRIPLINE_MOCK
    status = "ok" if loaded else "degraded"
    return HealthResponse(
        status=status,
        service="gripline-backend",
        model_loaded=loaded,
        mock_available=True,
    )


# =========================================================================
# POST /analyze
# =========================================================================


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def analyze(
    file: UploadFile = File(...),
    frame_stride: int = Form(default=config.DEFAULT_FRAME_STRIDE),
    max_frames: int = Form(default=config.DEFAULT_MAX_FRAMES),
    demo_mode: bool = Form(default=False),
) -> AnalyzeResponse:
    """
    Analyse an uploaded image or video.

    Accepts multipart/form-data with:
        file          — .jpg/.jpeg/.png/.mp4/.mov
        frame_stride  — every N-th frame (default 5)
        max_frames    — max frames to process (default 30)
        demo_mode     — force mock analyzer (default false)
    """
    # --- Validate file ----------------------------------------------------
    _validate_upload(file)

    # --- Write to temp file -----------------------------------------------
    tmp_path: pathlib.Path | None = None
    try:
        suffix = pathlib.Path(file.filename or "upload").suffix.lower()
        tmp_fd, tmp_str = tempfile.mkstemp(suffix=suffix)
        tmp_path = pathlib.Path(tmp_str)

        content = await file.read()

        # Empty file check (after read)
        if len(content) == 0:
            raise GripLineError("EMPTY_FILE", "Uploaded file is empty.", 400)

        # Size check
        if len(content) > config.MAX_FILE_SIZE_BYTES:
            raise GripLineError(
                "FILE_TOO_LARGE",
                f"File exceeds the {config.MAX_FILE_SIZE_MB} MB limit.",
                413,
            )

        os.write(tmp_fd, content)
        os.close(tmp_fd)

        # --- Run analysis -------------------------------------------------
        use_demo = demo_mode or config.GRIPLINE_MOCK
        raw = analyze_file(
            tmp_path,
            frame_stride=frame_stride,
            max_frames=max_frames,
            demo_mode=use_demo,
        )

        # --- Build response -----------------------------------------------
        return _build_response(raw, file.filename or "upload", use_demo)

    finally:
        # Always clean up temp file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                _LOGGER.warning("Could not delete temp file: %s", tmp_path)


# =========================================================================
# File validation
# =========================================================================


def _validate_upload(file: UploadFile) -> None:
    """Raise GripLineError if the upload is invalid."""
    # Content-type
    ct = (file.content_type or "").lower()
    if ct not in config.ALLOWED_CONTENT_TYPES:
        raise GripLineError(
            "INVALID_FILE",
            "Please upload a valid image or video file.",
        )

    # Extension
    if file.filename:
        ext = pathlib.Path(file.filename).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            raise GripLineError(
                "INVALID_FILE",
                "Please upload a valid image or video file.",
            )


# =========================================================================
# Response builder
# =========================================================================


def _build_response(
    raw: dict[str, Any],
    filename: str,
    demo_mode: bool,
) -> AnalyzeResponse:
    """
    Convert raw ML predictions into the full GripLine response contract.
    """
    file_type = raw.get("type", "image")
    frames = raw.get("frames", [])
    processed_frames = raw.get("processed_frames", len(frames))
    duration = raw.get("duration_seconds", 0.0)

    # --- Session ID -------------------------------------------------------
    session_id = "demo_001" if demo_mode else f"session_{uuid.uuid4().hex[:8]}"

    # --- Source metadata --------------------------------------------------
    source = SourceMetadata(
        filename=filename,
        type=file_type,
        duration_seconds=round(duration, 2),
    )

    # --- Zone aggregation -------------------------------------------------
    # If the ML layer already computed a summary (video mock), use it.
    summary = raw.get("summary")
    if summary:
        zone_data = summary.get("zones", [])
        overall_wetness = summary.get("overall_wetness", 0.0)
        overall_trend = summary.get("overall_trend", "Stable")
    else:
        # Single-frame path: use zones from the first frame
        first = frames[0] if frames else {}
        zone_data = first.get("zones", [])
        overall_wetness = first.get("overall_wetness", 0.0)
        overall_trend = raw.get("overall_trend", "Stable")

    # Build ZonePrediction list
    zones = _build_zones(zone_data)

    # Compute overall confidence
    confidences = [z.confidence for z in zones]
    overall_confidence = (
        round(sum(confidences) / len(confidences), 4) if confidences else 0.8
    )

    # --- Overall condition ------------------------------------------------
    overall_base = _condition(overall_wetness)
    overall_display = _display_condition(overall_base, overall_trend)

    overall = OverallCondition(
        base_condition=overall_base,
        display_condition=overall_display,
        wetness_score=round(overall_wetness, 4),
        trend=overall_trend,
        confidence=round(overall_confidence, 2),
    )

    # --- Recommendation ---------------------------------------------------
    rec_data = recommendation_engine.generate_recommendation(
        overall_wetness,
        overall_trend,
        [z.model_dump() for z in zones],
        overall_confidence,
    )
    recommendation = Recommendation(**rec_data)

    # --- Radio call -------------------------------------------------------
    radio_data = recommendation_engine.generate_radio_call(
        [z.model_dump() for z in zones],
        overall_wetness,
        overall_trend,
    )
    radio = RadioCall(**radio_data)

    # --- Alerts -----------------------------------------------------------
    alert_list = alert_engine.generate_alerts(
        [z.model_dump() for z in zones],
        overall_trend=overall_trend,
        overall_confidence=overall_confidence,
    )
    alerts = [Alert(**a) for a in alert_list]

    # --- Timeline ---------------------------------------------------------
    raw_timeline = raw.get("timeline", [])
    if not raw_timeline and frames:
        raw_timeline = [
            {
                "timestamp_seconds": f.get("timestamp_seconds", 0.0),
                "overall_wetness": f.get("overall_wetness", 0.0),
            }
            for f in frames
        ]

    timeline = [
        TimelinePoint(
            timestamp_seconds=t["timestamp_seconds"],
            overall_wetness=t["overall_wetness"],
            trend=overall_trend,
        )
        for t in raw_timeline
    ]

    return AnalyzeResponse(
        session_id=session_id,
        source=source,
        processed_frames=processed_frames,
        overall=overall,
        zones=zones,
        recommendation=recommendation,
        radio_call=radio,
        alerts=alerts,
        timeline=timeline,
    )


# =========================================================================
# Helpers
# =========================================================================


def _build_zones(zone_data: list[dict[str, Any]]) -> list[ZonePrediction]:
    """Convert raw zone dicts into validated ZonePrediction models."""
    zones: list[ZonePrediction] = []
    for z in zone_data:
        wetness = z.get("wetness_score", 0.0)
        risk = z.get("risk", z.get("risk_level", _risk(wetness)))
        trend = z.get("trend", "Stable")
        base = z.get("base_condition", _condition(wetness))
        display = z.get("display_condition", _display_condition(base, trend))

        zones.append(ZonePrediction(
            id=z["id"],
            name=z.get("name", z["id"]),
            base_condition=base,
            display_condition=display,
            wetness_score=round(wetness, 4),
            trend=trend,
            risk=risk,
            confidence=round(z.get("confidence", 0.8), 4),
        ))
    return zones


def _condition(wetness: float) -> str:
    if wetness >= config.CONDITION_WET_MIN:
        return "Wet"
    if wetness >= config.CONDITION_DAMP_MIN:
        return "Damp"
    return "Dry"


def _risk(wetness: float) -> str:
    if wetness >= config.RISK_HIGH_MIN:
        return "High"
    if wetness >= config.RISK_MEDIUM_MIN:
        return "Medium"
    return "Low"


def _display_condition(base: str, trend: str) -> str:
    if trend in ("Drying", "Worsening"):
        return trend
    return base
