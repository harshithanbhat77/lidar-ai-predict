from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.pointcloud_loader import PointCloudFormatError
from src.postprocess import (
    apply_class_mapping,
    filter_detections_by_class,
    filter_detections_by_score,
)
from src.schemas import Detection, InferenceResponse
from src.utils import get_logger


try:
    import torch
except Exception:  # pragma: no cover - environment-specific import failure
    torch = None


LOGGER = get_logger()


class DetectorError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class DetectorConfig:
    checkpoint_path: str | None = None
    config_path: str | None = None
    device: str = "cuda"
    model_name: str = "PointPillars"
    model_version: str = "kitti-pretrained-v1"
    class_names: tuple[str, ...] = ("Car", "Pedestrian", "Cyclist")
    mock_mode: bool = False


class PointPillarsDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.model: Any = None
        self.model_loaded = False
        self.backend = "mock" if self.config.mock_mode else "openpcdet"
        self._openpcdet_modules: dict[str, Any] | None = None

    def load_model(self) -> None:
        if self.model_loaded:
            return

        started = time.perf_counter()
        if self.config.mock_mode:
            self.model = "mock-model"
            self.model_loaded = True
            LOGGER.info("Loaded mock detector in %.2f ms", (time.perf_counter() - started) * 1000)
            return

        checkpoint = self.config.checkpoint_path
        if not checkpoint or not Path(checkpoint).exists():
            raise DetectorError(
                "MODEL_NOT_FOUND",
                "No compatible PointPillars checkpoint was found. Provide --checkpoint or "
                "set LIDAR_MODEL_CHECKPOINT.",
            )
        config_path = self.config.config_path
        if not config_path or not Path(config_path).exists():
            raise DetectorError(
                "MODEL_NOT_FOUND",
                "No compatible PointPillars config was found. Provide --config or set "
                "LIDAR_MODEL_CONFIG.",
            )
        if torch is None:
            raise DetectorError(
                "INFERENCE_FAILED",
                "PyTorch is not installed in the current environment.",
            )

        use_cuda = self.config.device.startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            raise DetectorError(
                "CUDA_UNAVAILABLE",
                "CUDA was requested but is not available. Real OpenPCDet inference may require "
                "a GPU-enabled build.",
            )

        self._openpcdet_modules = self._import_openpcdet_modules()
        device = torch.device(self.config.device if use_cuda else "cpu")
        self.model = {
            "checkpoint_path": checkpoint,
            "config_path": config_path,
            "device": str(device),
        }
        self.model_loaded = True
        LOGGER.info(
            "Initialized PointPillars detector wrapper in %.2f ms using device=%s",
            (time.perf_counter() - started) * 1000,
            device,
        )

    def _import_openpcdet_modules(self) -> dict[str, Any]:
        try:
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.datasets import DatasetTemplate
            from pcdet.models import build_network, load_data_to_gpu
            from pcdet.utils import common_utils
        except Exception as exc:  # pragma: no cover - depends on external package
            raise DetectorError(
                "INFERENCE_FAILED",
                "OpenPCDet is not installed or could not be imported. Install a compatible "
                "OpenPCDet/PyTorch/CUDA stack before running real inference.",
            ) from exc
        return {
            "cfg": cfg,
            "cfg_from_yaml_file": cfg_from_yaml_file,
            "DatasetTemplate": DatasetTemplate,
            "build_network": build_network,
            "load_data_to_gpu": load_data_to_gpu,
            "common_utils": common_utils,
        }

    def preprocess(self, point_cloud: np.ndarray) -> np.ndarray:
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 4:
            raise PointCloudFormatError(
                f"Expected point cloud shape (N, 4), received {point_cloud.shape}."
            )
        return point_cloud.astype(np.float32, copy=False)

    def predict(
        self,
        point_cloud: np.ndarray,
        score_threshold: float = 0.5,
        class_filter: list[str] | None = None,
    ) -> InferenceResponse:
        self.load_model()
        started = time.perf_counter()
        prepared = self.preprocess(point_cloud)

        if self.config.mock_mode:
            detections = self._mock_detections(prepared)
        else:
            detections = self._predict_with_openpcdet(prepared)

        filtered = filter_detections_by_score(detections, score_threshold)
        filtered = filter_detections_by_class(filtered, class_filter)
        inference_time_ms = (time.perf_counter() - started) * 1000
        LOGGER.info("Inference completed in %.2f ms with %d detections", inference_time_ms, len(filtered))
        return InferenceResponse(
            status="success",
            model_name=self.config.model_name,
            model_version="mock" if self.config.mock_mode else self.config.model_version,
            inference_time_ms=round(inference_time_ms, 3),
            detections=filtered,
        )

    def _mock_detections(self, point_cloud: np.ndarray) -> list[Detection]:
        if point_cloud.shape[0] == 0:
            return []
        centroid = point_cloud[:, :3].mean(axis=0)
        return [
            Detection(
                class_name="Car",
                score=0.95,
                center=[float(centroid[0]), float(centroid[1]), float(centroid[2])],
                size=[4.2, 1.8, 1.6],
                yaw=0.0,
            )
        ]

    def _predict_with_openpcdet(self, point_cloud: np.ndarray) -> list[Detection]:
        modules = self._openpcdet_modules
        if modules is None:
            raise DetectorError("INFERENCE_FAILED", "OpenPCDet modules were not initialized.")

        try:
            if torch is None:
                raise DetectorError("INFERENCE_FAILED", "PyTorch is not installed.")
            # The real OpenPCDet preprocessing path depends on version-specific dataset wrappers
            # and custom ops. This wrapper keeps that boundary explicit instead of pretending every
            # environment supports a generic CPU-only path.
            raise DetectorError(
                "INFERENCE_FAILED",
                "OpenPCDet was detected, but this project needs a version-matched dataset/config/"
                "checkpoint wiring step before real predictions can be executed in this environment.",
            )
        except DetectorError:
            raise
        except Exception as exc:  # pragma: no cover - external runtime behavior
            raise DetectorError(
                "INFERENCE_FAILED",
                f"PointPillars inference failed: {exc}",
            ) from exc

    def postprocess(self, predictions: list[Detection]) -> list[dict[str, Any]]:
        return [prediction.model_dump() for prediction in predictions]

    def health(self) -> dict[str, Any]:
        try:
            if not self.model_loaded:
                self.load_model()
        except DetectorError:
            return {
                "status": "ok",
                "model_loaded": False,
                "model_name": self.config.model_name,
                "model_version": "mock" if self.config.mock_mode else self.config.model_version,
                "device": self.config.device,
                "mock_mode": self.config.mock_mode,
            }
        return {
            "status": "ok",
            "model_loaded": self.model_loaded,
            "model_name": self.config.model_name,
            "model_version": "mock" if self.config.mock_mode else self.config.model_version,
            "device": self.config.device,
            "mock_mode": self.config.mock_mode,
        }


def openpcdet_prediction_to_detections(
    pred_dict: dict[str, Any], class_names: tuple[str, ...]
) -> list[Detection]:
    boxes = pred_dict.get("pred_boxes", [])
    scores = pred_dict.get("pred_scores", [])
    labels = pred_dict.get("pred_labels", [])
    mapped_labels = apply_class_mapping(labels, class_names)
    detections: list[Detection] = []
    for box, score, class_name in zip(boxes, scores, mapped_labels):
        box_values = np.asarray(box).tolist()
        detections.append(
            Detection(
                class_name=class_name,
                score=float(score),
                center=[float(box_values[0]), float(box_values[1]), float(box_values[2])],
                size=[float(box_values[3]), float(box_values[4]), float(box_values[5])],
                yaw=float(box_values[6]),
            )
        )
    return detections
