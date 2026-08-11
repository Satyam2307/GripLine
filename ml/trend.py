"""
ml.trend — Temporal trend calculation for GripLine.

Implements exponential smoothing and drying / stable / worsening detection.

Public functions
----------------
    calculate_zone_trend(history)
    calculate_overall_prediction(zone_predictions)
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

_LOGGER = logging.getLogger("gripline.trend")
_CONFIG_PATH = pathlib.Path(__file__).parent / "zone_config.json"


def _load_trend_thresholds() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)["trend_thresholds"]


# ---------------------------------------------------------------------------
# Exponential smoothing
# ---------------------------------------------------------------------------


def exponential_smooth(
    scores: list[float],
    alpha: float = 0.5,
) -> list[float]:
    """
    Apply exponential smoothing to a time-ordered sequence of scores.

    Parameters
    ----------
    scores : list[float]
        Raw wetness scores ordered oldest → newest.
    alpha : float
        Smoothing factor (weight on the current observation).

    Returns
    -------
    list[float]
        Smoothed scores, same length as *scores*.
    """
    if not scores:
        return []

    smoothed = [scores[0]]
    for s in scores[1:]:
        smoothed.append(alpha * s + (1 - alpha) * smoothed[-1])
    return [round(v, 4) for v in smoothed]


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

# Cache thresholds (they never change at runtime)
_TREND_THRESHOLDS: dict | None = None


def _get_trend_thresholds() -> dict:
    global _TREND_THRESHOLDS
    if _TREND_THRESHOLDS is None:
        _TREND_THRESHOLDS = _load_trend_thresholds()
    return _TREND_THRESHOLDS


def trend_from_delta(delta: float) -> str:
    """
    Classify a delta value into Drying / Stable / Worsening.

    delta = latest_smoothed − oldest_smoothed
    Negative delta ⇒ wetness is decreasing ⇒ Drying.
    """
    th = _get_trend_thresholds()
    if delta <= th["drying_delta"]:
        return "Drying"
    if delta >= th["worsening_delta"]:
        return "Worsening"
    return "Stable"


def display_condition(base_condition: str, trend: str) -> str:
    """
    Combine the base condition with the trend for display purposes.

    A zone that is Wet and Drying is displayed as "Drying" to convey
    the most actionable information, but the caller still has access
    to ``base_condition`` for context (e.g. "still Wet, but Drying").
    """
    if trend == "Drying":
        return "Drying"
    if trend == "Worsening":
        return "Worsening"
    return base_condition


# ---------------------------------------------------------------------------
# Per-zone trend calculation (public)
# ---------------------------------------------------------------------------


def calculate_zone_trend(
    history: list[dict[str, Any]],
    alpha: float = 0.5,
) -> dict[str, Any]:
    """
    Compute the trend for **one zone** given its time-ordered history.

    Parameters
    ----------
    history : list[dict]
        Each dict must contain at least ``wetness_score`` (float).
        Ordered oldest → newest.
    alpha : float
        Exponential-smoothing factor.

    Returns
    -------
    dict
        Keys: ``trend``, ``smoothed_scores``, ``delta``,
        ``base_condition``, ``display_condition``.
    """
    if not history:
        return {
            "trend": "Stable",
            "smoothed_scores": [],
            "delta": 0.0,
            "base_condition": "Dry",
            "display_condition": "Dry",
        }

    raw_scores = [h["wetness_score"] for h in history]
    smoothed = exponential_smooth(raw_scores, alpha=alpha)
    delta = round(smoothed[-1] - smoothed[0], 4)
    trend = trend_from_delta(delta)

    # Use the latest smoothed score for base condition
    from ml.inference import base_condition_from_score

    latest_base = base_condition_from_score(smoothed[-1])

    return {
        "trend": trend,
        "smoothed_scores": smoothed,
        "delta": delta,
        "base_condition": latest_base,
        "display_condition": display_condition(latest_base, trend),
    }


# ---------------------------------------------------------------------------
# Overall prediction from zone predictions (public)
# ---------------------------------------------------------------------------


def calculate_overall_prediction(
    zone_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate zone-level predictions into a single frame-level summary.

    Parameters
    ----------
    zone_predictions : list[dict]
        Each element is a zone prediction dict (with ``wetness_score``,
        ``risk_level``, etc.).

    Returns
    -------
    dict
        Keys: ``overall_wetness``, ``highest_risk_zone``,
        ``tire_recommendation``.
    """
    if not zone_predictions:
        return {
            "overall_wetness": 0.0,
            "highest_risk_zone": None,
            "tire_recommendation": "No data available.",
        }

    overall = round(
        sum(z["wetness_score"] for z in zone_predictions)
        / len(zone_predictions),
        4,
    )

    # Find highest-risk zone by wetness score (simple proxy)
    highest = max(zone_predictions, key=lambda z: z["wetness_score"])

    rec = tire_recommendation(overall)

    return {
        "overall_wetness": overall,
        "highest_risk_zone": highest["id"] if "id" in highest else None,
        "tire_recommendation": rec,
    }


def tire_recommendation(overall_wetness: float) -> str:
    """
    Generate a tire-strategy suggestion based on overall wetness.

    This is decision-support only, not a certified instruction.
    """
    if overall_wetness >= 0.75:
        return "Wet tires recommended."
    if overall_wetness >= 0.55:
        return "Intermediate tires may be suitable."
    if overall_wetness >= 0.40:
        return "Tire-change window approaching."
    if overall_wetness >= 0.25:
        return "Slick tires may be considered."
    return "No immediate tire change required."


def radio_call(
    zone_predictions: list[dict[str, Any]],
    overall_wetness: float,
    trend: str = "Stable",
) -> str:
    """
    Generate a concise race-engineer-style radio message.

    Parameters
    ----------
    zone_predictions : list[dict]
        Current zone predictions.
    overall_wetness : float
        Aggregated wetness score.
    trend : str
        Overall trend (Drying / Stable / Worsening).

    Returns
    -------
    str
        A short radio-call string.
    """
    if not zone_predictions:
        return "GripLine to pit wall. No zone data available."

    # Find highest-risk zone
    worst = max(zone_predictions, key=lambda z: z["wetness_score"])
    worst_name = worst.get("name", worst.get("id", "Unknown"))
    worst_cond = worst.get("display_condition", worst.get("base_condition", "Unknown"))

    parts = []

    if trend == "Drying":
        parts.append("Track drying rapidly.")
        parts.append("Tire-change window approaching.")
    elif trend == "Worsening":
        parts.append(f"{worst_name} worsening. High-risk braking zone.")
    else:
        # Stable
        if overall_wetness >= 0.70:
            parts.append(f"{worst_name} remains {worst_cond.lower()}.")
            parts.append("High-risk braking zone.")
        elif overall_wetness >= 0.40:
            parts.append(f"Track conditions mixed. {worst_name} {worst_cond.lower()}.")
        else:
            parts.append("Conditions stable. No immediate tire change required.")

    return " ".join(parts)
