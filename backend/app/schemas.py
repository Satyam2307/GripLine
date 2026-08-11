"""
Pydantic models for the GripLine API response contract.

These schemas enforce the shared contract between backend and frontend.
All values must be JSON-serialisable (str, int, float, bool, list, dict, None).
"""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------


class SourceMetadata(BaseModel):
    """Information about the uploaded file."""

    filename: str
    type: str  # "image" or "video"
    duration_seconds: float = 0.0


class OverallCondition(BaseModel):
    """Aggregated track-level condition summary."""

    base_condition: str
    display_condition: str
    wetness_score: float
    trend: str
    confidence: float


class ZonePrediction(BaseModel):
    """Per-zone condition prediction."""

    id: str
    name: str
    base_condition: str
    display_condition: str
    wetness_score: float
    trend: str
    risk: str
    confidence: float


class Recommendation(BaseModel):
    """Tire strategy recommendation (decision-support only)."""

    text: str
    suggested_tire: str
    confidence: float


class RadioCall(BaseModel):
    """Race-engineer-style radio call."""

    text: str
    severity: str  # "low" | "medium" | "high"
    should_play: bool


class Alert(BaseModel):
    """Actionable alert for the pit wall."""

    id: str
    timestamp_seconds: float
    severity: str  # "low" | "medium" | "high"
    zone_id: str
    message: str


class TimelinePoint(BaseModel):
    """Single data point in the wetness timeline."""

    timestamp_seconds: float
    overall_wetness: float
    trend: str


# ---------------------------------------------------------------------------
# Top-level responses
# ---------------------------------------------------------------------------


class AnalyzeResponse(BaseModel):
    """Complete analysis response — the GripLine contract."""

    session_id: str
    source: SourceMetadata
    processed_frames: int
    overall: OverallCondition
    zones: list[ZonePrediction]
    recommendation: Recommendation
    radio_call: RadioCall
    alerts: list[Alert]
    timeline: list[TimelinePoint]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    service: str = "gripline-backend"
    model_loaded: bool
    mock_available: bool = True


# ---------------------------------------------------------------------------
# Custom exception (used by routes, handled by main.py)
# ---------------------------------------------------------------------------


class GripLineError(Exception):
    """Structured API error that maps to an ErrorResponse."""

    def __init__(
        self, code: str, message: str, status_code: int = 400
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)
