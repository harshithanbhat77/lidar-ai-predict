import asyncio

import numpy as np
import pytest
from fastapi import HTTPException

from src import api
from src.detector import DetectorConfig, PointPillarsDetector


class FakeUploadFile:
    def __init__(self, filename: str, payload: bytes) -> None:
        self.filename = filename
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload


def _set_mock_detector():
    api.app.state.detector = PointPillarsDetector(DetectorConfig(mock_mode=True, device="cpu"))


def test_health_endpoint():
    _set_mock_detector()

    response = api.health()

    assert response.status == "ok"
    assert response.mock_mode is True
    assert response.model_loaded is True


def test_detect_validation_requires_input():
    _set_mock_detector()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api.detect(score_threshold=0.5))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "INVALID_POINT_CLOUD"


def test_detect_invalid_threshold():
    _set_mock_detector()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api.detect(file_path="data/sample/000123.bin", score_threshold=1.5))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "INVALID_THRESHOLD"


def test_detect_uploaded_file_success():
    _set_mock_detector()
    sample = np.array([[1.0, 2.0, 3.0, 0.2]], dtype=np.float32).tobytes()
    upload = FakeUploadFile(filename="sample.bin", payload=sample)

    response = asyncio.run(api.detect(file=upload, score_threshold=0.5))

    assert response.status == "success"
    assert len(response.detections) == 1


def test_detect_empty_file_produces_empty_detections():
    _set_mock_detector()
    upload = FakeUploadFile(filename="empty.bin", payload=b"")

    response = asyncio.run(api.detect(file=upload, score_threshold=0.5))

    assert response.status == "success"
    assert response.detections == []
