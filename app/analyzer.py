from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

from .schemas import FrameLabel, VideoAnnotationResponse

logger = logging.getLogger(__name__)

_LEFT_EYE_IDX = (362, 385, 387, 263, 373, 380)
_RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)


@dataclass
class FrameAnalysis:
    eye_state: Optional[str]
    posture: Optional[str]


class VideoAnnotator:
    def __init__(
        self,
        eye_open_threshold: float = 0.23,
        posture_angle_threshold: float = 10.0,
        face_detection_confidence: float = 0.5,
        pose_detection_confidence: float = 0.5,
        calibration_frames: int = 10,
    ) -> None:
        self.eye_open_threshold = eye_open_threshold
        self.posture_angle_threshold = posture_angle_threshold
        self.calibration_frames = calibration_frames
        self.calibration_data = []
        self.calibrated = False
        self.avg_ratio = None

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=face_detection_confidence,
            min_tracking_confidence=face_detection_confidence,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=pose_detection_confidence,
            min_tracking_confidence=pose_detection_confidence,
        )

    def annotate(self, video_path: str, original_filename: Optional[str] = None) -> VideoAnnotationResponse:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        labels: Dict[str, FrameLabel] = OrderedDict()
        previous_label = FrameLabel(eye_state="Open", posture="Straight")
        frame_index = 0

        calibration_distances = []
        CALIBRATION_FRAMES = 30

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                posture_results = self.pose.process(frame)


                if frame_index < CALIBRATION_FRAMES:
                    ratio = self._compute_ear_shoulder_ratio(frame)

                    if ratio:
                        calibration_distances.append(ratio)

                    label = FrameLabel(
                        eye_state="Open",
                        posture="Straight"  # Mark explicitly
                    )
                    labels[str(frame_index)] = label
                    frame_index += 1
                    continue

                # --- AFTER CALIBRATION ---
                if frame_index == CALIBRATION_FRAMES:
                    
                    if calibration_distances:
                        self.avg_ratio = np.mean(calibration_distances)
                    print(f"Calibration complete. Neutral ratio = {self.avg_ratio:.3f}")

                # Now normal inference
                analysis = self._analyze_frame(frame)
                posture = analysis.posture or previous_label.posture

                label = FrameLabel(
                    eye_state=analysis.eye_state or previous_label.eye_state,
                    posture=posture
                )

                labels[str(frame_index)] = label
                previous_label = label
                frame_index += 1

        finally:
            cap.release()

        video_filename = original_filename or Path(video_path).name
        return VideoAnnotationResponse(
            video_filename=video_filename,
            total_frames=frame_index,
            labels_per_frame=labels,
        )


    def _compute_ear_shoulder_ratio(self, frame) -> Optional[float]:
        """
        Computes a stable 3D ear–shoulder metric for calibration.
        Includes 2D alignment (ratio), Z-axis depth, and Y-axis relative difference.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        posture_results = self.pose.process(rgb_frame)
        if not posture_results or not posture_results.pose_landmarks:
            return None

        landmarks = posture_results.pose_landmarks.landmark
        try:
            left_shoulder = self._landmark_to_np(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value])
            right_shoulder = self._landmark_to_np(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value])
            left_ear = self._landmark_to_np(landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value])
            right_ear = self._landmark_to_np(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value])
        except (IndexError, ValueError):
            return None

        
        shoulders_center = (left_shoulder + right_shoulder) / 2.0
        ears_center = (left_ear + right_ear) / 2.0

        x_diff = abs(ears_center[0] - shoulders_center[0])
        y_diff = abs(ears_center[1] - shoulders_center[1])
        ratio_2d = y_diff / (x_diff + 1e-6)

        z_diff_left = left_ear[2] - left_shoulder[2]
        z_diff_right = right_ear[2] - right_shoulder[2]
        avg_z_diff = (z_diff_left + z_diff_right) / 2.0

        if not hasattr(self, "neutral_y_diff"):
            self.neutral_y_diff = y_diff
            self.neutral_y_ear_cordinate = ears_center[1]
            self.neutral_y_shoulder_cordinate = shoulders_center[1]

        
        combined_ratio = ratio_2d * (1 - abs(avg_z_diff))
        print(f"Frame : ratio_2d={ratio_2d}, avg_z_diff={avg_z_diff:.2f}, combined={combined_ratio:.2f}")
        return combined_ratio


    def _analyze_frame(self, frame: np.ndarray) -> FrameAnalysis:
        if frame is None:
            return FrameAnalysis(eye_state=None, posture=None)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        face_results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        eye_state = self._infer_eye_state(face_results)

        posture_results = self.pose.process(rgb_frame)
        posture_state = self._infer_posture_state(posture_results)

        return FrameAnalysis(eye_state=eye_state, posture=posture_state)

    def _infer_eye_state(self, face_results) -> Optional[str]:
        if not face_results or not face_results.multi_face_landmarks:
            return None

        face_landmarks = face_results.multi_face_landmarks[0]
        left_ratio = self._eye_aspect_ratio(face_landmarks, _LEFT_EYE_IDX)
        right_ratio = self._eye_aspect_ratio(face_landmarks, _RIGHT_EYE_IDX)

        if left_ratio is None or right_ratio is None:
            return None

        ear = (left_ratio + right_ratio) / 2.0
        return "Open" if ear >= self.eye_open_threshold else "Closed"

    def _infer_posture_state(self, posture_results) -> Optional[str]:
        if not posture_results or not posture_results.pose_landmarks:
            return None

        landmarks = posture_results.pose_landmarks.landmark
        try:
            left_shoulder = self._landmark_to_np(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value])
            right_shoulder = self._landmark_to_np(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value])
            left_ear = self._landmark_to_np(landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value])
            right_ear = self._landmark_to_np(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value])
        except (IndexError, ValueError):
            return None
        shoulders_center = (left_shoulder + right_shoulder) / 2.0
        ears_center = (left_ear + right_ear) / 2.0
        x_diff = abs(ears_center[0] - shoulders_center[0])
        y_diff = abs(ears_center[1] - shoulders_center[1])
        ratio_2d = y_diff / (x_diff + 1e-6)
        z_diff_left = left_ear[2] - left_shoulder[2]
        z_diff_right = right_ear[2] - right_shoulder[2]
        avg_z_diff = (z_diff_left + z_diff_right) / 2.0
        combined_ratio = ratio_2d * (1 - abs(avg_z_diff))
        if self.avg_ratio is None or not hasattr(self, "neutral_y_diff"):
            return "Straight"
        relative_y_drop = (ears_center[1] - self.neutral_y_ear_cordinate) / (self.neutral_y_ear_cordinate + 1e-6)
        print(f"Combined Ratio: {combined_ratio}, Avg Ratio: {self.avg_ratio}, Relative Y Drop: {relative_y_drop}")
        if ((combined_ratio > self.avg_ratio * 1.3) and (relative_y_drop > 0.03)) or relative_y_drop > 0.1:
            return "Hunched"
        else:
            return "Straight"


    def _eye_aspect_ratio(self, face_landmarks, indices) -> Optional[float]:
        try:
            points = [self._landmark_to_np(face_landmarks.landmark[i]) for i in indices]
        except (IndexError, ValueError):
            return None

        p1, p2, p3, p4, p5, p6 = points
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal < 1e-6:
            return None
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    @staticmethod
    def _landmark_to_np(landmark) -> np.ndarray:
        return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)
