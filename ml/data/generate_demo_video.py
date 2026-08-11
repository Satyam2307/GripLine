"""
Generate a demo video from the sample track images.

The video simulates a drying track:
  - Frames 1-10:  wet track images
  - Frames 11-20: damp track images
  - Frames 21-30: dry track images

Usage:
    python -m ml.data.generate_demo_video
"""

import cv2
import pathlib
import numpy as np

_SAMPLES_DIR = pathlib.Path(__file__).parent / "samples"
_OUTPUT_DIR = pathlib.Path(__file__).parent / "demo_video"


def generate_demo_video(
    output_path: str | pathlib.Path | None = None,
    fps: float = 10.0,
    frames_per_condition: int = 10,
) -> pathlib.Path:
    """
    Create a demo MP4 from the labelled sample images.

    Sequences: wet → damp → dry to simulate a drying track.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = _OUTPUT_DIR / "wet_track.mp4"
    output_path = pathlib.Path(output_path)

    # Load images in drying order: wet → damp → dry
    image_sequence = [
        ("track_wet_01.png", frames_per_condition),
        ("track_wet_02.png", frames_per_condition),
        ("track_damp_01.png", frames_per_condition),
        ("track_damp_02.png", frames_per_condition),
        ("track_dry_01.png", frames_per_condition),
        ("track_dry_02.png", frames_per_condition),
    ]

    # Determine video size from first image
    first_img = cv2.imread(str(_SAMPLES_DIR / image_sequence[0][0]))
    if first_img is None:
        raise FileNotFoundError(
            f"Sample image not found: {_SAMPLES_DIR / image_sequence[0][0]}"
        )
    h, w = first_img.shape[:2]

    # Target size for reasonable video
    target_w, target_h = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_w, target_h))

    total_frames = 0
    for filename, count in image_sequence:
        img_path = _SAMPLES_DIR / filename
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠  Skipping {filename} — not found")
            continue

        # Resize to target
        img_resized = cv2.resize(img, (target_w, target_h))

        for _ in range(count):
            writer.write(img_resized)
            total_frames += 1

    writer.release()
    print(f"  ✅ Demo video created: {output_path}")
    print(f"     {total_frames} frames @ {fps} fps = {total_frames / fps:.1f}s")
    return output_path


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  GripLine — Demo Video Generator")
    print("=" * 50)
    print()
    generate_demo_video()
