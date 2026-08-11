"""
ml.evaluate — Run CLIP inference against the labelled test set and report accuracy.

Usage
-----
    # Real inference (downloads model on first run):
    python -m ml.evaluate

    # Mock mode (no model needed):
    GRIPLINE_MOCK=true python -m ml.evaluate

Output
------
    Per-image predictions, confusion matrix, and overall accuracy.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = pathlib.Path(__file__).parent / "data"
_SAMPLES_DIR = _DATA_DIR / "samples"
_LABELS_PATH = _DATA_DIR / "labels.json"


def load_labels() -> dict:
    """Load the labelled test-set manifest."""
    with open(_LABELS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(verbose: bool = True) -> dict[str, Any]:
    """
    Run CLIP (or mock) inference on every labelled sample image.

    Returns a JSON-serialisable results dict with per-image predictions
    and aggregate accuracy.
    """
    from ml.inference import (
        analyze_image,
        is_mock_mode,
        classify_zone_crop,
        wetness_score_from_probs,
        base_condition_from_score,
        confidence_from_probs,
        bgr_to_pil,
    )

    labels = load_labels()
    samples = labels["samples"]
    mode = "mock" if is_mock_mode() else "clip"

    results: list[dict[str, Any]] = []
    correct = 0
    total = 0

    for sample in samples:
        filename = sample["filename"]
        gt_label = sample["ground_truth"]
        img_path = _SAMPLES_DIR / filename

        if not img_path.exists():
            if verbose:
                print(f"  ⚠  SKIP  {filename} — file not found")
            continue

        total += 1
        image_bgr = cv2.imread(str(img_path))

        if image_bgr is None:
            if verbose:
                print(f"  ⚠  SKIP  {filename} — could not read image")
            continue

        if is_mock_mode():
            # In mock mode, use the full analyze_image and pick the
            # overall wetness as a proxy (since the mock doesn't
            # classify individual crops).
            frame = analyze_image(image_bgr, timestamp_seconds=0.0)
            overall_w = frame["overall_wetness"]
            predicted_cond = base_condition_from_score(overall_w).lower()
            probs = frame["zones"][0]["class_probabilities"] if frame["zones"] else {}
            confidence = max(probs.values()) if probs else 0.0
        else:
            # Real CLIP: classify the full image (not zone crops)
            pil_img = bgr_to_pil(image_bgr)
            probs = classify_zone_crop(pil_img)
            w_score = wetness_score_from_probs(probs)
            predicted_cond = base_condition_from_score(w_score).lower()
            confidence = confidence_from_probs(probs)
            overall_w = w_score

        is_correct = predicted_cond == gt_label
        if is_correct:
            correct += 1

        result_entry = {
            "filename": filename,
            "ground_truth": gt_label,
            "predicted": predicted_cond,
            "correct": is_correct,
            "wetness_score": round(overall_w, 4),
            "confidence": round(confidence, 4),
            "class_probabilities": probs,
        }
        results.append(result_entry)

        if verbose:
            mark = "✅" if is_correct else "❌"
            print(
                f"  {mark}  {filename:<22s}  "
                f"GT={gt_label:<5s}  PRED={predicted_cond:<5s}  "
                f"wetness={overall_w:.3f}  conf={confidence:.3f}"
            )

    accuracy = round(correct / total, 4) if total > 0 else 0.0

    summary = {
        "mode": mode,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }

    if verbose:
        print()
        print(f"  Mode:     {mode}")
        print(f"  Samples:  {total}")
        print(f"  Correct:  {correct}")
        print(f"  Accuracy: {accuracy * 100:.1f}%")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  GripLine — Labelled Test-Set Evaluation")
    print("=" * 60)
    print()

    result = evaluate(verbose=True)

    # Save results to JSON
    out_path = _DATA_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n  Results saved to {out_path}")
