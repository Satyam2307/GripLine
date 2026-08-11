"""
backend.app.services.analyzer_adapter — Unified interface to ML inference.

Supports three modes:
    1. Real CLIP (openai/clip-vit-base-patch32)
    2. Deterministic mock (GRIPLINE_MOCK=true or demo_mode=true)
    3. Graceful fallback (model fails → auto-switch to mock)
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from typing import Any

_LOGGER = logging.getLogger("gripline.adapter")

# Ensure the project root is importable so ``ml.*`` resolves.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_mock_env() -> bool:
    return os.getenv("GRIPLINE_MOCK", "false").lower() in ("true", "1", "yes")


def _try_import_ml() -> bool:
    """Return True if the ML package can be imported."""
    try:
        import ml.inference  # noqa: F401
        import ml.mock_analyzer  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

_model_loaded: bool = False
_model_error: str | None = None


def is_model_loaded() -> bool:
    """Return True if the real CLIP model has been loaded successfully."""
    return _model_loaded


def get_model_error() -> str | None:
    return _model_error


def try_load_model() -> bool:
    """
    Attempt to load the real CLIP model.  Returns True on success.

    This is intentionally **not** called at import time so that the module
    remains importable even without PyTorch / Transformers.
    """
    global _model_loaded, _model_error
    if _model_loaded:
        return True
    try:
        from ml.inference import is_model_available
        if is_model_available():
            _model_loaded = True
            _model_error = None
            _LOGGER.info("CLIP model loaded successfully.")
            return True
        else:
            _model_error = "Model reported unavailable."
            return False
    except Exception as exc:
        _model_error = str(exc)
        _LOGGER.warning("Could not load CLIP model: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------


def analyze_file(
    file_path: str | pathlib.Path,
    *,
    frame_stride: int = 5,
    max_frames: int = 30,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Analyse an image or video file and return raw ML predictions.

    Parameters
    ----------
    file_path : str | Path
        Path to the uploaded file on disk.
    frame_stride : int
        Take every N-th frame for video (default 5).
    max_frames : int
        Maximum frames to process (default 30).
    demo_mode : bool
        If True, always use the mock analyzer.

    Returns
    -------
    dict
        Raw predictions from ML layer (frames, zones, wetness, trends).
    """
    global _model_loaded, _model_error

    use_mock = demo_mode or _is_mock_env()

    path = pathlib.Path(file_path)
    is_video = path.suffix.lower() in (".mp4", ".mov")

    # ------------------------------------------------------------------
    # Mock mode (explicit)
    # ------------------------------------------------------------------
    if use_mock:
        _LOGGER.info("Using mock analyzer (demo_mode=%s, env=%s).",
                      demo_mode, _is_mock_env())
        return _mock_analyze(path, is_video, frame_stride, max_frames)

    # ------------------------------------------------------------------
    # Real CLIP mode — with fallback
    # ------------------------------------------------------------------
    if not _try_import_ml():
        _LOGGER.warning("ML package unavailable — falling back to mock.")
        _model_error = "ML package could not be imported."
        return _mock_analyze(path, is_video, frame_stride, max_frames)

    try:
        if is_video:
            return _real_video(path, frame_stride, max_frames)
        else:
            return _real_image(path)
    except Exception as exc:
        _model_error = str(exc)
        _LOGGER.error("Real inference failed — falling back to mock: %s", exc)
        return _mock_analyze(path, is_video, frame_stride, max_frames)


# ---------------------------------------------------------------------------
# Real inference helpers
# ---------------------------------------------------------------------------


def _real_image(path: pathlib.Path) -> dict[str, Any]:
    """Run CLIP inference on a single image."""
    import cv2
    from ml.inference import analyze_image

    global _model_loaded
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    result = analyze_image(image, timestamp_seconds=0.0)
    _model_loaded = True
    return {
        "type": "image",
        "frames": [result],
        "processed_frames": 1,
        "duration_seconds": 0.0,
    }


def _real_video(
    path: pathlib.Path, frame_stride: int, max_frames: int
) -> dict[str, Any]:
    """Run CLIP inference on a video (frame extraction + per-frame analysis)."""
    from ml.video_processor import analyze_video

    global _model_loaded
    result = analyze_video(str(path), frame_stride=frame_stride,
                           max_frames=max_frames)
    _model_loaded = True
    # Compute duration from last frame timestamp
    frames = result.get("frames", [])
    duration = frames[-1]["timestamp_seconds"] if frames else 0.0
    return {
        "type": "video",
        "frames": frames,
        "processed_frames": len(frames),
        "duration_seconds": round(duration, 2),
        "summary": result.get("summary"),
        "timeline": result.get("timeline"),
    }


# ---------------------------------------------------------------------------
# Mock analyzer helpers
# ---------------------------------------------------------------------------


def _mock_analyze(
    path: pathlib.Path,
    is_video: bool,
    frame_stride: int,
    max_frames: int,
) -> dict[str, Any]:
    """Return deterministic mock predictions."""
    from ml.mock_analyzer import MockAnalyzer

    mock = MockAnalyzer()

    if is_video:
        num_frames = min(max_frames, 5)
        result = mock.analyze_video_mock(num_frames=num_frames)
        frames = result.get("frames", [])
        duration = frames[-1]["timestamp_seconds"] if frames else 0.0
        return {
            "type": "video",
            "frames": frames,
            "processed_frames": len(frames),
            "duration_seconds": round(duration, 2),
            "summary": result.get("summary"),
            "timeline": result.get("timeline"),
        }
    else:
        filename_lower = path.name.lower()
        if "dry" in filename_lower:
            frame_idx = 4
            trend = "Drying"
            timeline = [
                {"timestamp_seconds": 0.0, "overall_wetness": 0.55},
                {"timestamp_seconds": 2.0, "overall_wetness": 0.38},
                {"timestamp_seconds": 4.0, "overall_wetness": 0.25},
                {"timestamp_seconds": 6.0, "overall_wetness": 0.16},
            ]
        elif "damp" in filename_lower:
            frame_idx = 2
            trend = "Drying"
            timeline = [
                {"timestamp_seconds": 0.0, "overall_wetness": 0.78},
                {"timestamp_seconds": 2.0, "overall_wetness": 0.65},
                {"timestamp_seconds": 4.0, "overall_wetness": 0.54},
                {"timestamp_seconds": 6.0, "overall_wetness": 0.45},
            ]
        elif "wet" in filename_lower:
            frame_idx = 0
            trend = "Worsening"
            timeline = [
                {"timestamp_seconds": 0.0, "overall_wetness": 0.68},
                {"timestamp_seconds": 2.0, "overall_wetness": 0.76},
                {"timestamp_seconds": 4.0, "overall_wetness": 0.82},
                {"timestamp_seconds": 6.0, "overall_wetness": 0.88},
            ]
        else:
            frame_idx = None
            trend = "Stable"
            timeline = []

        frame = mock.analyze_image_mock(timestamp_seconds=6.0, frame_index=frame_idx)
        return {
            "type": "image",
            "frames": [frame],
            "processed_frames": 1,
            "duration_seconds": 6.0,
            "overall_trend": trend,
            "timeline": timeline,
        }
