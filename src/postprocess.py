from __future__ import annotations

from typing import Iterable, Sequence

from src.schemas import Detection


def filter_detections_by_score(
    detections: Iterable[Detection], score_threshold: float
) -> list[Detection]:
    return [d for d in detections if d.score >= score_threshold]


def filter_detections_by_class(
    detections: Iterable[Detection], class_filter: Sequence[str] | None
) -> list[Detection]:
    if not class_filter:
        return list(detections)
    allowed = {name.lower() for name in class_filter}
    return [d for d in detections if d.class_name.lower() in allowed]


def apply_class_mapping(
    raw_labels: Iterable[int | str], class_names: Sequence[str] | None
) -> list[str]:
    mapped: list[str] = []
    for label in raw_labels:
        if isinstance(label, str):
            mapped.append(label)
            continue
        if class_names and 0 <= label - 1 < len(class_names):
            mapped.append(class_names[label - 1])
        else:
            mapped.append(f"class_{label}")
    return mapped


def nms_already_applied_note() -> str:
    return (
        "OpenPCDet typically applies score filtering and NMS during model post-processing. "
        "This project exposes those post-processed results instead of reimplementing rotated 3D NMS."
    )


def axis_aligned_iou_2d(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union
