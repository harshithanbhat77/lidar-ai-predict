from __future__ import annotations

from pathlib import Path


def describe_onnx_export(checkpoint_path: str | None, config_path: str | None) -> dict[str, object]:
    checkpoint_exists = bool(checkpoint_path and Path(checkpoint_path).exists())
    config_exists = bool(config_path and Path(config_path).exists())
    export_supported = False
    reasons = [
        "PointPillars/OpenPCDet may depend on custom CUDA ops for voxelization and NMS.",
        "A generic ONNX export path is not guaranteed across OpenPCDet versions.",
        "Validation must compare PyTorch and ONNX Runtime outputs before claiming parity.",
    ]
    if not checkpoint_exists:
        reasons.append("Checkpoint file is missing.")
    if not config_exists:
        reasons.append("Model config file is missing.")
    return {
        "export_supported": export_supported,
        "checkpoint_exists": checkpoint_exists,
        "config_exists": config_exists,
        "reasons": reasons,
        "validation_plan": {
            "compare": ["class IDs", "scores", "box coordinates", "latency"],
            "runtimes": ["PyTorch", "ONNX Runtime"],
        },
    }
