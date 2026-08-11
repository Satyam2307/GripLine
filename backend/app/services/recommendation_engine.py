"""
backend.app.services.recommendation_engine — Tire recommendation + radio call.

All rules are deterministic and use cautious language.
These are decision-support suggestions, NOT guaranteed racing instructions.
"""

from __future__ import annotations

from typing import Any

from backend.app import config


# =========================================================================
# Tire recommendation
# =========================================================================


def generate_recommendation(
    overall_wetness: float,
    trend: str,
    zones: list[dict[str, Any]],
    overall_confidence: float = 0.8,
) -> dict[str, Any]:
    """
    Generate a tire recommendation based on transparent deterministic rules.

    Parameters
    ----------
    overall_wetness : float
        Aggregated wetness (0–1).
    trend : str
        Drying / Stable / Worsening.
    zones : list[dict]
        Zone predictions (need wetness_score, risk, name, id).
    overall_confidence : float
        Mean confidence.

    Returns
    -------
    dict with keys: text, suggested_tire, confidence.
    """
    text, tire = _base_recommendation(overall_wetness, trend)

    # High-risk corner modifier
    high_risk_corners = _high_risk_corners(zones)
    if high_risk_corners:
        names = ", ".join(z["name"] for z in high_risk_corners)
        text += f" However, {names} remains high risk."

    return {
        "text": text,
        "suggested_tire": tire,
        "confidence": round(overall_confidence, 2),
    }


def _base_recommendation(wetness: float, trend: str) -> tuple[str, str]:
    """Apply the 4 deterministic rules (spec §22)."""
    # Rule 1: very wet
    if wetness >= 0.75:
        return "Wet tires recommended.", "Wet"

    # Rule 3: drying zone (check before Rule 2 — overlapping range)
    if 0.25 <= wetness <= 0.55 and trend == "Drying":
        return (
            "Track drying overall. Tire-change window approaching.",
            "Intermediate to Slick",
        )

    # Rule 2: intermediate
    if 0.45 <= wetness < 0.75 and trend != "Drying":
        return "Intermediate tires may be suitable.", "Intermediate"

    # Rule 4: dry + stable
    if wetness < 0.25 and trend == "Stable":
        return "Slick tires may be considered.", "Slick"

    # Fallback
    if wetness < 0.25:
        return "No immediate tire change required.", "Slick"

    # Damp-ish but drying above 0.55
    if trend == "Drying":
        return (
            "Track drying overall. Tire-change window approaching.",
            "Intermediate to Slick",
        )

    return "Intermediate tires may be suitable.", "Intermediate"


def _high_risk_corners(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return zones that are corners AND high risk."""
    meta = _load_zone_meta()
    result = []
    for z in zones:
        risk = z.get("risk", _risk(z["wetness_score"]))
        is_corner = meta.get(z["id"], {}).get("is_corner", False)
        if risk == "High" and is_corner:
            result.append(z)
    return result


# =========================================================================
# Radio-call generation
# =========================================================================


def generate_radio_call(
    zones: list[dict[str, Any]],
    overall_wetness: float,
    trend: str,
) -> dict[str, Any]:
    """
    Generate a race-engineer-style radio call.

    Returns dict with: text, severity, should_play.
    """
    parts = []
    severity = "low"

    # Find highest-risk zone
    worst = max(zones, key=lambda z: z["wetness_score"]) if zones else None
    worst_name = worst["name"] if worst else "Unknown"
    worst_risk = worst.get("risk", _risk(worst["wetness_score"])) if worst else "Low"

    # High-risk zone call
    if worst and worst_risk == "High":
        parts.append(f"{worst_name} remains wet. High-risk braking zone.")
        severity = "high"

    # Trend calls
    if trend == "Drying":
        parts.append("Track drying overall. Tire-change window approaching.")
        if severity != "high":
            severity = "medium"
    elif trend == "Worsening":
        if worst:
            parts.append(f"Conditions worsening near {worst_name}. Exercise caution.")
        severity = "high"
    else:
        # Stable
        if overall_wetness < 0.45 and worst_risk != "High":
            parts.append("Conditions stable. No immediate tire change required.")

    # Combine intelligently — avoid near-duplicate info
    text = " ".join(parts)

    should_play = severity in ("medium", "high")

    return {
        "text": text,
        "severity": severity,
        "should_play": should_play,
    }


# =========================================================================
# Shared helpers
# =========================================================================


def _risk(wetness: float) -> str:
    if wetness >= config.RISK_HIGH_MIN:
        return "High"
    if wetness >= config.RISK_MEDIUM_MIN:
        return "Medium"
    return "Low"


def _condition(wetness: float) -> str:
    if wetness >= config.CONDITION_WET_MIN:
        return "Wet"
    if wetness >= config.CONDITION_DAMP_MIN:
        return "Damp"
    return "Dry"


def _display_condition(base: str, trend: str) -> str:
    if trend in ("Drying", "Worsening"):
        return trend
    return base


_ZONE_META_CACHE: dict | None = None


def _load_zone_meta() -> dict[str, dict]:
    global _ZONE_META_CACHE
    if _ZONE_META_CACHE is not None:
        return _ZONE_META_CACHE

    try:
        import json
        import pathlib
        cfg = pathlib.Path(__file__).resolve().parents[3] / "ml" / "zone_config.json"
        if cfg.exists():
            with open(cfg) as fh:
                data = json.load(fh)
            _ZONE_META_CACHE = {
                z["id"]: z.get("map_layout", {"is_corner": False})
                for z in data["zones"]
            }
            return _ZONE_META_CACHE
    except Exception:
        pass

    _ZONE_META_CACHE = {
        "main_straight": {"is_corner": False},
        "turn_2": {"is_corner": True},
        "turn_4": {"is_corner": True},
        "final_corner": {"is_corner": True},
    }
    return _ZONE_META_CACHE
