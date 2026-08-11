# GripLine — ML Inference Layer

> **Prototype — not a certified racing-safety system.**

## Model

| Property        | Value                                                                                 |
| --------------- | ------------------------------------------------------------------------------------- |
| Model           | `openai/clip-vit-base-patch32`                                                        |
| Hugging Face    | [huggingface.co/openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |
| Method          | Zero-shot image classification                                                        |
| Text prompts    | `"a dry racing track"`, `"a damp racing track"`, `"a wet racing track"`               |

## Architecture overview

```
ml/
├── __init__.py            # Package exports
├── zone_config.json       # Zone coordinates, prompts, thresholds
├── inference.py           # CLIP zero-shot classification + analyze_image()
├── video_processor.py     # OpenCV frame extraction + analyze_video()
├── trend.py               # Exponential smoothing, trend & tire logic
├── mock_analyzer.py       # Deterministic mock for demos / integration
├── README.md              # This file
└── tests/
    ├── __init__.py
    ├── test_trend.py
    └── test_mock_analyzer.py
```

## Installation

```bash
# From the project root
pip install torch torchvision transformers Pillow opencv-python-headless numpy pytest
```

> On Apple Silicon, `torch` with MPS acceleration is automatically available.
> On machines without a GPU, CPU inference works out of the box.

## Quick-start inference

```python
import cv2
from ml.inference import analyze_image

image = cv2.imread("track_photo.jpg")
result = analyze_image(image, timestamp_seconds=0.0)
print(result)
```

### Video analysis

```python
from ml.video_processor import analyze_video

result = analyze_video("race_clip.mp4", frame_stride=5, max_frames=30)
print(result["summary"])
```

### Using trends explicitly

```python
from ml.trend import calculate_zone_trend

history = [
    {"wetness_score": 0.85},
    {"wetness_score": 0.70},
    {"wetness_score": 0.50},
]
trend = calculate_zone_trend(history)
print(trend)
# → {'trend': 'Drying', 'delta': -0.2375, ...}
```

## Mock mode

Set the environment variable to bypass model loading:

```bash
export GRIPLINE_MOCK=true
```

In mock mode:

- No Hugging Face model is downloaded.
- `analyze_image()` and `analyze_video()` return deterministic data.
- The simulated track **starts wet and dries over time**.
- **Turn 4 remains wetter** than the other zones.
- Overall wetness progression: ≈ 0.82 → 0.65 → 0.48 → …

### Using mock mode from Python

```python
import os
os.environ["GRIPLINE_MOCK"] = "true"

from ml.inference import analyze_image
result = analyze_image(None, timestamp_seconds=0.0)
```

Or use the `MockAnalyzer` class directly:

```python
from ml.mock_analyzer import MockAnalyzer

mock = MockAnalyzer()
frame0 = mock.analyze_image_mock(timestamp_seconds=0.0)
video  = mock.analyze_video_mock(num_frames=5)
```

## Running tests

```bash
pytest ml/tests/ -v
```

Tests do **not** download the model. They cover:

- Wetness-score calculation
- Threshold mapping (Dry / Damp / Wet)
- Exponential smoothing
- Trend detection (Drying, Stable, Worsening)
- Risk-level mapping
- Normalised-crop conversion
- Mock-mode determinism
- JSON serialisability

## Zone configuration

Zones are defined in `ml/zone_config.json` with normalised crop coordinates:

| Zone           | x_min | y_min | x_max | y_max |
| -------------- | ----- | ----- | ----- | ----- |
| Main Straight  | 0.10  | 0.60  | 0.90  | 0.95  |
| Turn 2         | 0.00  | 0.30  | 0.35  | 0.60  |
| Turn 4         | 0.60  | 0.05  | 0.95  | 0.35  |
| Final Corner   | 0.55  | 0.55  | 0.95  | 0.90  |

## Wetness scoring

```
wetness = P(dry) × 0.0  +  P(damp) × 0.5  +  P(wet) × 1.0
```

| Score range | Condition |
| ----------- | --------- |
| 0.00 – 0.39 | Dry       |
| 0.40 – 0.69 | Damp      |
| 0.70 – 1.00 | Wet       |

## Trend calculation

Exponential smoothing (α = 0.5):

```
smoothed[t] = 0.5 × raw[t]  +  0.5 × smoothed[t-1]
```

Trend delta = `smoothed[latest] − smoothed[oldest]`:

| Delta       | Trend      |
| ----------- | ---------- |
| ≤ −0.08     | Drying     |
| −0.08 … 0.08 | Stable   |
| ≥ 0.08      | Worsening  |

## Limitations

1. **Zero-shot CLIP** was not trained specifically on racing tracks. It may confuse reflective tarmac with wetness, or miss subtle dampness.
2. **Fixed zones** — the crop coordinates assume a specific camera angle. Different tracks or camera positions require new coordinates in `zone_config.json`.
3. **No temporal memory** — the model classifies each crop independently. Drying/worsening is calculated purely from the wetness-score time series.
4. **No weather API** — the system relies solely on visual analysis.
5. **Prototype quality** — this is a hackathon MVP, not a safety-critical system. Do not use for real race-control decisions.

## API contract

### `analyze_image(image, timestamp_seconds=0.0, previous_state=None) → dict`

### `analyze_video(video_path, frame_stride=5, max_frames=30) → dict`

### `calculate_zone_trend(history) → dict`

### `calculate_overall_prediction(zone_predictions) → dict`

All return values are JSON-serialisable `dict` objects.
