# GripLine — Backend

> FastAPI backend for the GripLine AI track-condition co-pilot.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Run

```bash
# From the project root:
uvicorn backend.app.main:app --reload --port 8000
```

## Endpoints

### Health

```
GET http://localhost:8000/api/v1/health
```

Response:

```json
{
  "status": "ok",
  "service": "gripline-backend",
  "model_loaded": true,
  "mock_available": true
}
```

### Analyze

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@../ml/data/samples/track_wet_01.png" \
  -F "frame_stride=5" \
  -F "max_frames=30" \
  -F "demo_mode=true"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRIPLINE_MOCK` | `false` | Force mock analyzer (no model needed) |
| `GRIPLINE_MODEL_ID` | `openai/clip-vit-base-patch32` | Hugging Face model ID |
| `GRIPLINE_MAX_FILE_SIZE_MB` | `100` | Upload size limit |
| `GRIPLINE_DEFAULT_FRAME_STRIDE` | `5` | Default frame stride for video |
| `GRIPLINE_DEFAULT_MAX_FRAMES` | `30` | Max frames to process |

## Mock / Demo Mode

Set `GRIPLINE_MOCK=true` or pass `demo_mode=true` in the request.

Mock mode:
- Returns deterministic drying simulation (Wet → Damp → Dry)
- Turn 4 remains the highest-risk zone
- Works **offline** — no model download, no PyTorch required
- Ideal for frontend integration and hackathon demos

## CLIP Model

- **Model**: `openai/clip-vit-base-patch32`
- **Method**: Zero-shot image classification
- **Labels**: `dry racing track asphalt`, `damp racing track asphalt`, `wet racing track asphalt`, `standing water on racing track asphalt`
- **Loading**: Lazy — downloaded on first inference, cached by Hugging Face

## Reference Dataset

- **Dataset**: `rezzzq/RSCD-1million` ([Hugging Face](https://huggingface.co/datasets/rezzzq/RSCD-1million))
- **Purpose**: Reference for road-surface moisture categories and potential future fine-tuning
- **NOT required** for runtime — the backend works without it
- **NOT downloaded** at startup

## Fallback Behavior

```
Request → demo_mode/MOCK? → YES → Mock Analyzer
                          → NO  → Try CLIP → Success → CLIP
                                            → Failure → Mock Analyzer
```

The API **never crashes** due to model unavailability.

## Testing

```bash
# From the project root:
GRIPLINE_MOCK=true python3 -m pytest backend/tests/ -v
```

Tests run fully offline — no model download, no network access.

## Response Contract

See `contracts/gripline-response.example.json` for a full example.

Key structure:

```json
{
  "session_id": "demo_001",
  "source": { "filename": "...", "type": "image", "duration_seconds": 0 },
  "processed_frames": 1,
  "overall": { "base_condition": "Wet", "display_condition": "Wet", ... },
  "zones": [{ "id": "turn_4", "name": "Turn 4", ... }],
  "recommendation": { "text": "...", "suggested_tire": "Wet", ... },
  "radio_call": { "text": "GripLine to pit wall. ...", ... },
  "alerts": [],
  "timeline": []
}
```

## Architecture

```
FastAPI Backend
    │
    ├── routes/analysis.py      ← /health, /analyze
    │
    ├── services/
    │   ├── analyzer_adapter.py ← CLIP / Mock / Fallback
    │   ├── alert_engine.py     ← Deterministic alerts
    │   └── recommendation_engine.py ← Tire rec + radio call
    │
    ├── schemas.py              ← Pydantic response models
    └── config.py               ← Environment + thresholds
```
