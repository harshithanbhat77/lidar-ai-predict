from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from src.postprocess import axis_aligned_iou_2d
from src.schemas import Detection


@dataclass
class MetricsSummary:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float


def compute_precision_recall(tp: int, fp: int, fn: int) -> MetricsSummary:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return MetricsSummary(tp, fp, fn, precision, recall)


def _xyxy_from_detection(detection: Detection) -> list[float]:
    cx, cy, _cz = detection.center
    length, width, _height = detection.size
    half_l = length / 2.0
    half_w = width / 2.0
    return [cx - half_l, cy - half_w, cx + half_l, cy + half_w]


def simplified_detection_metrics(
    predictions: Iterable[Detection],
    ground_truths: Iterable[Detection],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    preds = list(predictions)
    gts = list(ground_truths)
    gt_used = [False] * len(gts)
    per_class_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
    )
    tp = 0
    fp = 0

    for pred in preds:
        match_index = None
        best_iou = 0.0
        for index, gt in enumerate(gts):
            if gt_used[index] or gt.class_name != pred.class_name:
                continue
            iou = axis_aligned_iou_2d(_xyxy_from_detection(pred), _xyxy_from_detection(gt))
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                match_index = index
        if match_index is not None:
            gt_used[match_index] = True
            tp += 1
            per_class_counts[pred.class_name]["true_positives"] += 1
        else:
            fp += 1
            per_class_counts[pred.class_name]["false_positives"] += 1

    fn = 0
    for used, gt in zip(gt_used, gts):
        if not used:
            fn += 1
            per_class_counts[gt.class_name]["false_negatives"] += 1

    summary = compute_precision_recall(tp, fp, fn)
    per_class = {}
    for class_name, counts in per_class_counts.items():
        class_summary = compute_precision_recall(
            counts["true_positives"], counts["false_positives"], counts["false_negatives"]
        )
        per_class[class_name] = {
            "true_positives": class_summary.true_positives,
            "false_positives": class_summary.false_positives,
            "false_negatives": class_summary.false_negatives,
            "precision": class_summary.precision,
            "recall": class_summary.recall,
        }

    return {
        "mode": "simplified_educational_metrics",
        "true_positives": summary.true_positives,
        "false_positives": summary.false_positives,
        "false_negatives": summary.false_negatives,
        "precision": summary.precision,
        "recall": summary.recall,
        "per_class_summary": per_class,
        "notes": (
            "This is a simplified educational metric based on axis-aligned 2D overlap derived "
            "from 3D boxes. It is not official KITTI or OpenPCDet AP evaluation."
        ),
    }


def run_official_openpcdet_evaluation() -> dict[str, str]:
    return {
        "mode": "official_openpcdet_wrapper",
        "status": "not_executed",
        "message": (
            "Use the OpenPCDet dataset-specific evaluation scripts with matching labels and "
            "predictions. This repository does not fabricate AP values."
        ),
    }
