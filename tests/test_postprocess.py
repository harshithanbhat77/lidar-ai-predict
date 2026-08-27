from src.evaluate import simplified_detection_metrics
from src.postprocess import axis_aligned_iou_2d, filter_detections_by_score
from src.schemas import Detection


def test_score_threshold_filtering():
    detections = [
        Detection(class_name="Car", score=0.9, center=[0, 0, 0], size=[4, 2, 1], yaw=0),
        Detection(class_name="Car", score=0.2, center=[1, 1, 0], size=[4, 2, 1], yaw=0),
    ]

    filtered = filter_detections_by_score(detections, 0.5)

    assert len(filtered) == 1
    assert filtered[0].score == 0.9


def test_axis_aligned_iou_2d():
    assert axis_aligned_iou_2d([0, 0, 2, 2], [1, 1, 3, 3]) == 1 / 7


def test_empty_detections_are_valid():
    metrics = simplified_detection_metrics([], [])

    assert metrics["true_positives"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
