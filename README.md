# Video Annotation 

Annotates laptop-camera videos frame-by-frame for eye state (Open/Closed) and posture (Straight/Hunched). Built with FastAPI, MediaPipe, and OpenCV.

## Features

- `POST /annotate` endpoint that accepts `.mp4` or `.avi` files.
- MediaPipe Face Mesh + Pose models for lightweight on-device inference.
- Eye Aspect Ratio (EAR) heuristic for eye openness.
- Spine inclination analysis for posture classification.
- Deterministic JSON schema with per-frame labels.

## Prerequisites

- Python 3.10 or newer
- [MediaPipe prerequisites](https://developers.google.com/mediapipe/solutions/setup) (OpenCV will try to use CPU acceleration where available)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the API

```powershell
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI. Upload a video to `/annotate` and download the JSON response.

### Evaluate with F1 metrics

Use `POST /annotate-with-metrics` to compare predictions against labeled data. Upload a video file under the `file` field and include a `ground_truth_json` form field containing a JSON object shaped like:

```json
{
  "labels_per_frame": {
    "0": {"eye_state": "Open", "posture": "Straight"},
    "1": {"eye_state": "Closed", "posture": "Hunched"}
  }
}
```

The response bundles the standard annotation plus macro-averaged F1 scores for eye state and posture, along with the number of frames evaluated.

## Response Format

```json
{
  "video_filename": "test_video_1.mp4",
  "total_frames": 240,
  "labels_per_frame": {
    "0": {
      "eye_state": "Open",
      "posture": "Hunched"
    }
  }
}
```

- **Eye state**: EAR threshold defaults to `0.23`; adjust inside `VideoAnnotator` if you find too many false positives/negatives.
- **Posture**: Uses the angle between the hip-to-shoulder vector and the vertical axis (threshold `15°`).
- Frames missing detections reuse the previous label (otherwise fall back to `Closed`/`Hunched`) to maintain consistent output length.
- Override thresholds or plug alternate models by subclassing `VideoAnnotator` or overriding the FastAPI dependency.

# F1 Score

-Eye_state_F1 score = 93.03%
-Posture_State_F1_score = 84.31% 