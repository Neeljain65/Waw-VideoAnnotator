import cv2
import mediapipe as mp
import re
import numpy as np
import json
import subprocess

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# ---- Configuration ----
VIDEO_PATH = "WIN_20251030_13_02_49_Pro.mp4"  # Replace with your file
MAX_FRAMES = 120  # 30 for context + 30 for test
Y_DROP_THRESHOLD = 0.1
COMBINED_RATIO_THRESHOLD = 1.5


def calculate_posture_metrics(left_shoulder, right_shoulder, left_ear, right_ear):
    """Compute math-based ratios for posture classification."""
    shoulder_mid = np.mean([left_shoulder, right_shoulder], axis=0)
    ear_mid = np.mean([left_ear, right_ear], axis=0)

    vertical_dist = abs(ear_mid[1] - shoulder_mid[1])  # Y difference
    z_diff = abs(ear_mid[2] - shoulder_mid[2])  # Forward lean
    combined_ratio = vertical_dist / (1 - z_diff if (1 - z_diff) != 0 else 1)
    return vertical_dist, combined_ratio


def classify_math(vertical_dist, avg_v, combined_ratio, avg_c):
    """Classify based on simple math thresholds."""
    y_drop = vertical_dist - avg_v
    if (combined_ratio > avg_c * 1.08) or (y_drop < -Y_DROP_THRESHOLD):
        return "Hunched"
    return "Straight"


def call_moondream(prompt_text):
    """Run MoonDream via Ollama and cleanly parse JSON output."""
    try:
        result = subprocess.run(
            ["ollama", "run", "moondream", prompt_text],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        raw_output = result.stdout.strip()

        # Remove irrelevant console lines
        raw_output = re.sub(r"failed to get console mode.*\n", "", raw_output, flags=re.IGNORECASE)
        raw_output = re.sub(r"(?i)warning:.*\n", "", raw_output)
        raw_output = raw_output.strip()

        # Try to find JSON block
        match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                parsed = json.loads(json_str)
                return parsed
            except Exception as e:
                return {"error": f"JSON parse failed: {e}", "raw": json_str}
        else:
            return {"error": "No JSON found", "raw": raw_output}

    except Exception as e:
        return {"error": str(e), "raw": ""}


# ---- Step 1: Extract landmarks ----
cap = cv2.VideoCapture(VIDEO_PATH)
frames_data = []
frame_count = 0

while cap.isOpened() and frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = np.array([
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
        ])
        right_shoulder = np.array([
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,
        ])
        left_ear = np.array([
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y,
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].z,
        ])
        right_ear = np.array([
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].z,
        ])

        frames_data.append({
            "frame": frame_count,
            "left_shoulder": left_shoulder.tolist(),
            "right_shoulder": right_shoulder.tolist(),
            "left_ear": left_ear.tolist(),
            "right_ear": right_ear.tolist()
        })
        frame_count += 1

cap.release()
print(f"[INFO] Processed {len(frames_data)} frames from MediaPipe.")


# ---- Step 2: Pure Math Classification ----
verticals = []
ratios = []
math_results = []

for f in frames_data:
    v, r = calculate_posture_metrics(
        np.array(f["left_shoulder"]),
        np.array(f["right_shoulder"]),
        np.array(f["left_ear"]),
        np.array(f["right_ear"])
    )
    verticals.append(v)
    ratios.append(r)

avg_v = np.mean(verticals)
avg_c = np.mean(ratios)

for i, f in enumerate(frames_data):
    label = classify_math(verticals[i], avg_v, ratios[i], avg_c)
    math_results.append({"frame": i, "label": label})

print("\n[RESULT] MediaPipe + Math Classification:")
for r in math_results:
    print(f"Frame {r['frame']:02d}: {r['label']}")


# ---- Step 3: MoonDream (LLM) Classification ----
context_frames = frames_data[:30]
test_frames = frames_data[:30]

moon_prompt = f"""
You are an AI posture analyzer. The following is JSON of 3D coordinates for a human's left/right shoulders and ears.

The first 30 frames are examples of correct "Straight" posture:
{json.dumps(context_frames, indent=2)}

Now, analyze these subsequent frames and classify each as either "Hunched" or "Straight" based on relative ear and shoulder position:
{json.dumps(test_frames, indent=2)}

Return ONLY a pure JSON array like:
[
  {{"frame": 31, "label": "Hunched"}},
  {{"frame": 32, "label": "Straight"}}
]
Do NOT include ellipsis (...), markdown, or explanations.
If you are unsure, label as "Unknown".
"""

print("\n[INFO] Sending contextual posture data to MoonDream via Ollama...")

result = subprocess.run(
    ["ollama", "run", "moondream", moon_prompt],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="ignore"
)

raw_output = result.stdout.strip()

# ---- Clean noisy console logs ----
raw_output = re.sub(r"failed to get console mode.*\n", "", raw_output, flags=re.IGNORECASE)
raw_output = re.sub(r"(?i)warning:.*\n", "", raw_output)
raw_output = raw_output.replace("...", "")  # remove ellipses
raw_output = raw_output.strip()

# ---- Extract JSON ----
match = re.search(r"\[.*\]", raw_output, re.DOTALL)
if match:
    json_str = match.group(0)
    try:
        # fix any trailing commas
        json_str = re.sub(r",\s*]", "]", json_str)
        llm_output = json.loads(json_str)
    except Exception as e:
        llm_output = {"error": f"JSON parse failed: {e}", "raw": json_str}
else:
    llm_output = {"error": "No JSON found", "raw": raw_output}

print("\n[RESULT] MediaPipe + MoonDream (Context-Aware) Classification:")
print(json.dumps(llm_output, indent=2))