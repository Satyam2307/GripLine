"""
GripLine backend configuration.

All settings are read from environment variables with sensible defaults.
No side effects at import time — no model downloads, no network calls.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------

GRIPLINE_MOCK: bool = os.getenv("GRIPLINE_MOCK", "false").lower() in (
    "true",
    "1",
    "yes",
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

GRIPLINE_MODEL_ID: str = os.getenv(
    "GRIPLINE_MODEL_ID",
    "openai/clip-vit-base-patch32",
)

# CLIP candidate labels — used for zero-shot image classification.
# These describe *visual track conditions*, NOT tire compounds or strategy.
CLIP_LABELS: list[str] = [
    "dry racing track asphalt",
    "damp racing track asphalt",
    "wet racing track asphalt",
    "standing water on racing track asphalt",
]

# Wetness weight per label.  The final score is:
#     wetness = sum(P(label) * weight[label])
# and is clamped to [0, 1].
WETNESS_WEIGHTS: dict[str, float] = {
    "dry racing track asphalt": 0.10,
    "damp racing track asphalt": 0.40,
    "wet racing track asphalt": 0.75,
    "standing water on racing track asphalt": 1.00,
}

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_MB: int = int(os.getenv("GRIPLINE_MAX_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_CONTENT_TYPES: set[str] = {
    "image/jpeg",
    "image/png",
    "video/mp4",
    "video/quicktime",
}

ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".mp4", ".mov"}

# ---------------------------------------------------------------------------
# Analysis defaults
# ---------------------------------------------------------------------------

DEFAULT_FRAME_STRIDE: int = int(
    os.getenv("GRIPLINE_DEFAULT_FRAME_STRIDE", "5")
)
DEFAULT_MAX_FRAMES: int = int(
    os.getenv("GRIPLINE_DEFAULT_MAX_FRAMES", "30")
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://localhost:3000",
]

# ---------------------------------------------------------------------------
# Condition / risk thresholds  (backend decision layer)
# ---------------------------------------------------------------------------

# wetness_score → base_condition
CONDITION_WET_MIN: float = 0.75
CONDITION_DAMP_MIN: float = 0.45

# wetness_score → risk
RISK_HIGH_MIN: float = 0.75
RISK_MEDIUM_MIN: float = 0.45

# trend delta thresholds
TREND_DRYING_DELTA: float = -0.08
TREND_WORSENING_DELTA: float = 0.08
