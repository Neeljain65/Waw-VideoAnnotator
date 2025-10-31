from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field

EyeState = Literal["Open", "Closed"]
PostureState = Literal["Straight", "Hunched", "Calibrating"]


class FrameLabel(BaseModel):
    eye_state: EyeState = Field(examples=["Open"])
    posture: PostureState = Field(examples=["Hunched"])


class VideoAnnotationResponse(BaseModel):
    video_filename: str
    total_frames: int = Field(ge=0)
    labels_per_frame: Dict[str, FrameLabel]

    model_config = {
        "json_schema_extra": {
            "example": {
                "video_filename": "test_video_1.mp4",
                "total_frames": 240,
                "labels_per_frame": {
                    "0": {"eye_state": "Open", "posture": "Hunched"},
                    "1": {"eye_state": "Open", "posture": "Hunched"},
                    "2": {"eye_state": "Closed", "posture": "Straight"},
                },
            }
        }
    }


class GroundTruthPayload(BaseModel):
    labels_per_frame: Dict[str, FrameLabel]


class AnnotationMetrics(BaseModel):
    eye_state_f1: float = Field(ge=0.0, le=1.0)
    posture_f1: float = Field(ge=0.0, le=1.0)
    eye_state_Precision: float = Field(ge=0.0, le=1.0)
    posture_Precision: float = Field(ge=0.0, le=1.0)
    frames_evaluated: int = Field(ge=0)


class AnnotationWithMetricsResponse(BaseModel):
    annotation: VideoAnnotationResponse
    metrics: AnnotationMetrics
