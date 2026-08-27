from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_name: str
    score: float
    center: List[float] = Field(min_length=3, max_length=3)
    size: List[float] = Field(min_length=3, max_length=3)
    yaw: float


class InferenceResponse(BaseModel):
    status: Literal["success"]
    model_name: str
    model_version: str
    inference_time_ms: float
    detections: List[Detection]


class ErrorResponse(BaseModel):
    status: Literal["error"]
    code: str
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_name: str
    model_version: str
    device: str
    mock_mode: bool


class DetectionRequestOptions(BaseModel):
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    class_filter: Optional[List[str]] = None
