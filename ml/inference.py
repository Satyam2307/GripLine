"""
ml.inference — CLIP-based zero-shot track-condition classification.

Model: openai/clip-vit-base-patch32  (Hugging Face)
Prompts: "a dry racing track", "a damp racing track", "a wet racing track"

Public function
---------------
    analyze_image(image, timestamp_seconds=0.0, previous_state=None)
        → dict   (JSON-serialisable frame prediction)
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_LOGGER = logging.getLogger("gripline.inference")
_CONFIG_PATH = pathlib.Path(__file__).parent / "zone_config.json"

# Weights used to convert class probabilities to a scalar wetness score.
_WETNESS_WEIGHTS = {"dry": 0.0, "damp": 0.5, "wet": 1.0}

# ---------------------------------------------------------------------------
# Zone config loader
# ---------------------------------------------------------------------------

def load_zone_config() -> dict:
    """Load and return the zone configuration dictionary."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


_ZONE_CFG: dict | None = None


def _get_zone_cfg() -> dict:
    global _ZONE_CFG
    if _ZONE_CFG is None:
        _ZONE_CFG = load_zone_config()
    return _ZONE_CFG

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_MODEL_LOAD_ERROR: str | None = None


def _load_model() -> tuple:
    """
    Lazily load the CLIP model and processor.

    Returns (model, processor) or raises RuntimeError.
    """
    global _CLIP_MODEL, _CLIP_PROCESSOR, _MODEL_LOAD_ERROR

    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR

    if _MODEL_LOAD_ERROR is not None:
        raise RuntimeError(
            f"Model previously failed to load: {_MODEL_LOAD_ERROR}"
        )

    try:
        import torch  # noqa: F811 — intentional lazy import
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-base-patch32"
        _LOGGER.info("Loading CLIP model '%s' …", model_id)
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_id)
        _CLIP_MODEL = CLIPModel.from_pretrained(model_id)
        _CLIP_MODEL.eval()
        _LOGGER.info("CLIP model loaded successfully.")
        return _CLIP_MODEL, _CLIP_PROCESSOR

    except Exception as exc:
        _MODEL_LOAD_ERROR = str(exc)
        _LOGGER.error("Failed to load CLIP model: %s", exc)
        raise RuntimeError(
            f"Could not load Hugging Face CLIP model: {exc}"
        ) from exc


def is_model_available() -> bool:
    """Return True if the model can be loaded (or is already loaded)."""
    try:
        _load_model()
        return True
    except RuntimeError:
        return False


def is_mock_mode() -> bool:
    """Return True when GRIPLINE_MOCK env-var is set to a truthy value."""
    return os.getenv("GRIPLINE_MOCK", "").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def crop_zone_from_image(
    image: np.ndarray,
    crop_normalized: dict[str, float],
) -> np.ndarray:
    """
    Crop a zone from *image* (HWC, BGR or RGB) using normalised coordinates.

    Parameters
    ----------
    image : np.ndarray
        Full-resolution frame (H × W × C).
    crop_normalized : dict
        Keys: x_min, y_min, x_max, y_max — each in [0, 1].

    Returns
    -------
    np.ndarray
        Cropped region with the same channel ordering as the input.
    """
    h, w = image.shape[:2]
    x1 = int(crop_normalized["x_min"] * w)
    y1 = int(crop_normalized["y_min"] * h)
    x2 = int(crop_normalized["x_max"] * w)
    y2 = int(crop_normalized["y_max"] * h)

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return image[y1:y2, x1:x2]


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert a BGR (OpenCV) ndarray to an RGB PIL Image."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def classify_zone_crop(pil_image: Image.Image) -> dict[str, float]:
    """
    Classify a single zone crop with CLIP zero-shot.

    Returns
    -------
    dict with keys: dry, damp, wet — each a probability in [0, 1].
    """
    import torch

    model, processor = _load_model()
    cfg = _get_zone_cfg()
    text_prompts: list[str] = cfg["text_prompts"]

    inputs = processor(
        text=text_prompts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape (1, 3)
        probs = logits_per_image.softmax(dim=1).squeeze(0).tolist()

    return {
        "dry": round(probs[0], 4),
        "damp": round(probs[1], 4),
        "wet": round(probs[2], 4),
    }


def wetness_score_from_probs(probs: dict[str, float]) -> float:
    """
    Weighted combination:
        wetness = P(dry)*0 + P(damp)*0.5 + P(wet)*1.0
    """
    return round(
        probs["dry"] * _WETNESS_WEIGHTS["dry"]
        + probs["damp"] * _WETNESS_WEIGHTS["damp"]
        + probs["wet"] * _WETNESS_WEIGHTS["wet"],
        4,
    )


def base_condition_from_score(score: float) -> str:
    """Map a wetness score to Dry / Damp / Wet."""
    cfg = _get_zone_cfg()
    thresholds = cfg["condition_thresholds"]
    if score <= thresholds["dry_max"]:
        return "Dry"
    if score <= thresholds["damp_max"]:
        return "Damp"
    return "Wet"


def risk_level_from_score(score: float) -> str:
    """Map a wetness score to Low / Medium / High risk."""
    cfg = _get_zone_cfg()
    thresholds = cfg["risk_thresholds"]
    if score <= thresholds["low_max"]:
        return "Low"
    if score <= thresholds["medium_max"]:
        return "Medium"
    return "High"


def confidence_from_probs(probs: dict[str, float]) -> float:
    """
    Use the maximum class probability as the confidence value.

    A model that is very certain about one class will have a high max
    probability.  A confused model will spread mass evenly (≈ 0.33 each).
    """
    return round(max(probs.values()), 4)

# ---------------------------------------------------------------------------
# Single-image analysis (public API)
# ---------------------------------------------------------------------------

def analyze_image(
    image: np.ndarray | Image.Image,
    timestamp_seconds: float = 0.0,
    previous_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Analyse a single frame and return per-zone predictions.

    Parameters
    ----------
    image : np.ndarray (BGR) or PIL.Image.Image (RGB)
        The input frame.
    timestamp_seconds : float
        Position of this frame in the source video (default 0).
    previous_state : dict, optional
        The return value of a prior ``analyze_image`` call.  Reserved for
        future smoothing; currently unused.

    Returns
    -------
    dict
        JSON-serialisable prediction with keys:
        ``timestamp_seconds``, ``zones``, ``overall_wetness``.
    """
    # --- Mock mode ---
    if is_mock_mode():
        from ml.mock_analyzer import MockAnalyzer

        return MockAnalyzer().analyze_image_mock(
            timestamp_seconds=timestamp_seconds,
        )

    # --- Real inference ---
    # Convert PIL → OpenCV-BGR if needed
    if isinstance(image, Image.Image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image_cv = image

    cfg = _get_zone_cfg()
    zones_out: list[dict[str, Any]] = []

    for zone_def in cfg["zones"]:
        crop = crop_zone_from_image(image_cv, zone_def["crop_normalized"])

        # Guard against degenerate crops
        if crop.size == 0:
            _LOGGER.warning(
                "Zone '%s' produced an empty crop — skipping.", zone_def["id"]
            )
            continue

        pil_crop = bgr_to_pil(crop)
        probs = classify_zone_crop(pil_crop)
        w_score = wetness_score_from_probs(probs)
        b_cond = base_condition_from_score(w_score)
        risk = risk_level_from_score(w_score)

        zones_out.append(
            {
                "id": zone_def["id"],
                "name": zone_def["name"],
                "base_condition": b_cond,
                "display_condition": b_cond,  # trend applied later
                "wetness_score": w_score,
                "trend": "Stable",            # single frame → no trend
                "confidence": confidence_from_probs(probs),
                "risk_level": risk,
                "class_probabilities": probs,
            }
        )

    overall = (
        round(
            sum(z["wetness_score"] for z in zones_out) / len(zones_out), 4
        )
        if zones_out
        else 0.0
    )

    return {
        "timestamp_seconds": timestamp_seconds,
        "zones": zones_out,
        "overall_wetness": overall,
    }
