from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .schemas import FrameLabel

_EYE_CLASSES = ("Open", "Closed")
_POSTURE_CLASSES = ("Straight", "Hunched")


def compute_macro_f1(predictions: Sequence[str], references: Sequence[str], classes: Iterable[str]) -> float:
    f1_scores: List[float] = []
    for cls in classes:
        tp = sum(1 for p, r in zip(predictions, references) if p == cls and r == cls)
        fp = sum(1 for p, r in zip(predictions, references) if p == cls and r != cls)
        fn = sum(1 for p, r in zip(predictions, references) if p != cls and r == cls)

        if tp == 0 and fp == 0 and fn == 0:
            f1_scores.append(0.0)
            continue
          
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    if not f1_scores:
        return 0.0
    return sum(f1_scores) / len(f1_scores)


def compute_macro_precision(predictions: Sequence[str], references: Sequence[str], classes: Iterable[str]) -> float:
    precision_scores: List[float] = []
    for cls in classes:
        tp = sum(1 for p, r in zip(predictions, references) if p == cls and r == cls)
        fp = sum(1 for p, r in zip(predictions, references) if p == cls and r != cls)

        if tp == 0 and fp == 0:
            precision_scores.append(0.0)
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision_scores.append(precision)

    if not precision_scores:
        return 0.0
    return sum(precision_scores) / len(precision_scores)

def compute_f1_scores(
    predicted: Dict[str, FrameLabel],
    ground_truth: Dict[str, FrameLabel],
) -> Dict[str, float]:
    common_keys = sorted(set(predicted.keys()) & set(ground_truth.keys()), key=lambda k: int(k))
    if not common_keys:
        raise ValueError("No overlapping frames between predictions and ground truth.")

    pred_eye = [predicted[key].eye_state for key in common_keys]
    pred_posture = [predicted[key].posture for key in common_keys]

    gt_eye = [ground_truth[key].eye_state for key in common_keys]
    gt_posture = [ground_truth[key].posture for key in common_keys]

    eye_f1 = compute_macro_f1(pred_eye, gt_eye, _EYE_CLASSES)
    eye_precision = compute_macro_precision(pred_eye, gt_eye, _EYE_CLASSES)
    posture_f1 = compute_macro_f1(pred_posture, gt_posture, _POSTURE_CLASSES)
    posture_precision = compute_macro_precision(pred_posture, gt_posture, _POSTURE_CLASSES)

    return {
        "eye_state_f1": eye_f1,
        "eye_state_Precision": eye_precision,
        "posture_f1": posture_f1,
        "posture_Precision": posture_precision,
        "frames_evaluated": len(common_keys),
    }
