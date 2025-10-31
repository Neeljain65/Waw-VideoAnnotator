from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from .analyzer import VideoAnnotator
from .dependencies import get_video_annotator
from .metrics import compute_f1_scores
from .schemas import (
    AnnotationMetrics,
    AnnotationWithMetricsResponse,
    GroundTruthPayload,
    VideoAnnotationResponse,
)

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp4", ".avi"}

app = FastAPI(
    title="AI Video Annotation Service",
    description="Annotate eye state and posture for each frame in a video feed.",
    version="0.1.0",
)


@app.post("/annotate", response_model=VideoAnnotationResponse)
async def annotate_video(
    file: UploadFile = File(..., description="Video file (.mp4 or .avi)"),
    annotator: VideoAnnotator = Depends(get_video_annotator),
) -> VideoAnnotationResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Please upload an .mp4 or .avi file.",
        )

    tmp_path = None
    try:
        await file.seek(0)
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info("Annotating video '%s' stored at '%s'", file.filename, tmp_path)
        result = annotator.annotate(tmp_path, original_filename=file.filename)
        return result
    except ValueError as exc:
        logger.exception("Validation error while processing video: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safeguard
        logger.exception("Unexpected error while processing video")
        raise HTTPException(status_code=500, detail="Failed to process video.") from exc
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning("Temporary file cleanup failed for %s", tmp_path)


@app.post("/annotate-with-metrics", response_model=AnnotationWithMetricsResponse)
async def annotate_with_metrics(
    file: UploadFile = File(..., description="Video file (.mp4 or .avi)"),
    ground_truth_json: str = Form(..., description="JSON payload with labels_per_frame"),
    annotator: VideoAnnotator = Depends(get_video_annotator),
) -> AnnotationWithMetricsResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Please upload an .mp4 or .avi file.",
        )

    try:
        ground_truth = GroundTruthPayload.model_validate_json(ground_truth_json)
    except Exception as exc:
        logger.exception("Invalid ground truth payload: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid ground truth JSON provided.") from exc

    tmp_path = None
    try:
        await file.seek(0)
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info("Annotating video '%s' for evaluation", file.filename)
        annotation = annotator.annotate(tmp_path, original_filename=file.filename)

        try:
            metrics_dict = compute_f1_scores(annotation.labels_per_frame, ground_truth.labels_per_frame)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        metrics = AnnotationMetrics(**metrics_dict)
        return AnnotationWithMetricsResponse(annotation=annotation, metrics=metrics)
    except ValueError as exc:
        logger.exception("Validation error while processing video: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - safeguard
        logger.exception("Unexpected error while processing video with metrics")
        raise HTTPException(status_code=500, detail="Failed to process video.") from exc
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning("Temporary file cleanup failed for %s", tmp_path)
