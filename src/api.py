from __future__ import annotations

import tempfile
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.detector import DetectorConfig, DetectorError, PointPillarsDetector
from src.pointcloud_loader import PointCloudFormatError, load_kitti_bin
from src.schemas import ErrorResponse, HealthResponse, InferenceResponse
from src.utils import env_flag, normalize_path


def create_detector() -> PointPillarsDetector:
    checkpoint = os.getenv("LIDAR_MODEL_CHECKPOINT")
    config = os.getenv("LIDAR_MODEL_CONFIG")
    return PointPillarsDetector(
        DetectorConfig(
            checkpoint_path=str(Path(checkpoint) if checkpoint else Path("models/pointpillars_kitti.pth")),
            config_path=str(Path(config) if config else Path("configs/pointpillars_kitti.yaml")),
            device=os.getenv("LIDAR_DEVICE", "cuda"),
            mock_mode=env_flag("LIDAR_ENABLE_MOCK", default=False),
        )
    )


app = FastAPI(title="LiDAR PointPillars Detection API", version="1.0.0")
app.state.detector = create_detector()


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(status="error", code=code, message=message).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**app.state.detector.health())


def _validate_threshold(score_threshold: float) -> None:
    if not 0.0 <= score_threshold <= 1.0:
        raise _http_error(400, "INVALID_THRESHOLD", "score_threshold must be between 0.0 and 1.0.")


def _resolve_development_path(file_path: str) -> Path:
    allowed_root = normalize_path(Path.cwd())
    resolved = normalize_path(file_path)
    if not str(resolved).startswith(str(allowed_root)):
        raise _http_error(
            400,
            "INVALID_POINT_CLOUD",
            "file_path must resolve inside the project workspace in development mode.",
        )
    if not resolved.exists():
        raise _http_error(404, "INVALID_POINT_CLOUD", f"Point cloud path does not exist: {resolved}")
    return resolved


@app.post(
    "/detect",
    response_model=InferenceResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def detect(
    file: Annotated[UploadFile | None, File()] = None,
    file_path: Annotated[str | None, Form()] = None,
    score_threshold: Annotated[float, Form()] = 0.5,
    class_filter: Annotated[str | None, Form()] = None,
) -> InferenceResponse:
    _validate_threshold(score_threshold)
    if file is None and not file_path:
        raise _http_error(400, "INVALID_POINT_CLOUD", "Provide either an uploaded .bin file or file_path.")

    try:
        if file is not None:
            payload = await file.read()
            with tempfile.NamedTemporaryFile(suffix=".bin") as temp_file:
                temp_file.write(payload)
                temp_file.flush()
                point_cloud = load_kitti_bin(temp_file.name)
        else:
            resolved = _resolve_development_path(file_path or "")
            point_cloud = load_kitti_bin(resolved)

        class_names = [item.strip() for item in class_filter.split(",")] if class_filter else None
        return app.state.detector.predict(
            point_cloud, score_threshold=score_threshold, class_filter=class_names
        )
    except PointCloudFormatError as exc:
        raise _http_error(400, "INVALID_POINT_CLOUD", str(exc)) from exc
    except DetectorError as exc:
        status_code = 404 if exc.code == "MODEL_NOT_FOUND" else 500
        raise _http_error(status_code, exc.code, exc.message) from exc
    except TimeoutError as exc:
        raise _http_error(504, "TIMEOUT", str(exc)) from exc
