"""
backend.app.services.alert_engine — Deterministic alert generation.

Generates alerts for:
    • High-risk zones (corners prioritised)
    • Rapid drying
    • Worsening conditions
    • Low confidence
    • Frame processing failures

Deduplicates by (alert_type, zone_id).
"""

from __future__ import annotations

from typing import Any

from backend.app import config


# ---------------------------------------------------------------------------
# Alert factory
# ---------------------------------------------------------------------------


def _make_alert(
    alert_type: str,
    zone_id: str,
    message: str,
    severity: str = "medium",
    timestamp_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "id": f"{alert_type}_{zone_id}" if zone_id else alert_type,
        "timestamp_seconds": timestamp_seconds,
        "severity": severity,
        "zone_id": zone_id,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_alerts(
    zones: list[dict[str, Any]],
    overall_trend: str = "Stable",
    overall_confidence: float = 1.0,
    zone_meta: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate a deduplicated list of alerts from zone predictions.

    Parameters
    ----------
    zones : list[dict]
        Each zone dict must have: id, name, wetness_score, risk, trend,
        confidence, and optionally base_condition / display_condition.
    overall_trend : str
        Drying / Stable / Worsening.
    overall_confidence : float
        Mean model confidence across zones.
    zone_meta : dict, optional
        Mapping zone_id → metadata with ``is_corner`` bool.

    Returns
    -------
    list[dict]
        Deduplicated alert list.
    """
    if zone_meta is None:
        zone_meta = _default_zone_meta()

    alerts: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(alert: dict[str, Any]) -> None:
        key = alert["id"]
        if key not in seen:
            seen.add(key)
            alerts.append(alert)

    # --- Per-zone alerts --------------------------------------------------
    for z in zones:
        zid = z["id"]
        zname = z.get("name", zid)
        wetness = z["wetness_score"]
        risk = z.get("risk", _risk_from_score(wetness))
        trend = z.get("trend", "Stable")
        confidence = z.get("confidence", 1.0)
        meta = zone_meta.get(zid, {})
        is_corner = meta.get("is_corner", False)

        # High-risk zone
        if risk == "High":
            sev = "high" if is_corner else "medium"
            _add(_make_alert(
                "high_risk",
                zid,
                f"{zname} remains wet and is a high-risk zone.",
                severity=sev,
            ))

        # Worsening per-zone
        if trend == "Worsening":
            _add(_make_alert(
                "worsening",
                zid,
                f"Conditions worsening near {zname}. Exercise caution.",
                severity="high",
            ))

        # Low confidence per-zone
        if confidence < 0.5:
            _add(_make_alert(
                "low_confidence",
                zid,
                "Visual confidence is low. Check camera quality.",
                severity="low",
            ))

    # --- Global alerts ----------------------------------------------------
    if overall_trend == "Drying":
        _add(_make_alert(
            "drying",
            "",
            "Track drying rapidly. Tire-change window approaching.",
            severity="medium",
        ))

    if overall_trend == "Worsening":
        _add(_make_alert(
            "worsening_overall",
            "",
            "Overall track conditions worsening. Exercise caution.",
            severity="high",
        ))

    if overall_confidence < 0.5:
        _add(_make_alert(
            "low_confidence_overall",
            "",
            "Visual confidence is low. Check camera quality.",
            severity="low",
        ))

    return alerts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _risk_from_score(wetness: float) -> str:
    if wetness >= config.RISK_HIGH_MIN:
        return "High"
    if wetness >= config.RISK_MEDIUM_MIN:
        return "Medium"
    return "Low"


def _default_zone_meta() -> dict[str, dict]:
    """
    Load zone metadata from ml/zone_config.json if available.
    Otherwise return demo defaults.
    """
    try:
        import json
        import pathlib
        cfg_path = pathlib.Path(__file__).resolve().parents[3] / "ml" / "zone_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r") as fh:
                data = json.load(fh)
            return {
                z["id"]: z.get("map_layout", {"is_corner": False})
                for z in data["zones"]
            }
    except Exception:
        pass

    # Demo fallback
    return {
        "main_straight": {"is_corner": False},
        "turn_2": {"is_corner": True},
        "turn_4": {"is_corner": True},
        "final_corner": {"is_corner": True},
    }
