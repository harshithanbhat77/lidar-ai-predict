from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class PointCloudFormatError(ValueError):
    """Raised when a KITTI-style point cloud does not contain x, y, z, reflectance."""


@dataclass
class PointCloudSummary:
    num_points: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    reflectance_min: float
    reflectance_max: float


def load_kitti_bin(path: str | Path) -> np.ndarray:
    point_cloud_path = Path(path)
    points = np.fromfile(point_cloud_path, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if points.size % 4 != 0:
        raise PointCloudFormatError(
            f"Malformed point cloud '{point_cloud_path}': expected float32 values in groups of 4 "
            f"(x, y, z, reflectance), found {points.size} values."
        )
    reshaped = points.reshape(-1, 4)
    return reshaped


def summarize_point_cloud(point_cloud: np.ndarray) -> PointCloudSummary:
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 4:
        raise PointCloudFormatError(
            f"Expected point cloud shape (N, 4), received {point_cloud.shape}."
        )
    if point_cloud.shape[0] == 0:
        return PointCloudSummary(
            num_points=0,
            min_x=0.0,
            max_x=0.0,
            min_y=0.0,
            max_y=0.0,
            min_z=0.0,
            max_z=0.0,
            reflectance_min=0.0,
            reflectance_max=0.0,
        )

    return PointCloudSummary(
        num_points=int(point_cloud.shape[0]),
        min_x=float(np.min(point_cloud[:, 0])),
        max_x=float(np.max(point_cloud[:, 0])),
        min_y=float(np.min(point_cloud[:, 1])),
        max_y=float(np.max(point_cloud[:, 1])),
        min_z=float(np.min(point_cloud[:, 2])),
        max_z=float(np.max(point_cloud[:, 2])),
        reflectance_min=float(np.min(point_cloud[:, 3])),
        reflectance_max=float(np.max(point_cloud[:, 3])),
    )


def inspect_point_cloud(path: str | Path) -> dict[str, float | int]:
    points = load_kitti_bin(path)
    summary = summarize_point_cloud(points)
    return {
        "num_points": summary.num_points,
        "min_x": summary.min_x,
        "max_x": summary.max_x,
        "min_y": summary.min_y,
        "max_y": summary.max_y,
        "min_z": summary.min_z,
        "max_z": summary.max_z,
        "reflectance_min": summary.reflectance_min,
        "reflectance_max": summary.reflectance_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a KITTI point cloud .bin file.")
    parser.add_argument("path", help="Path to KITTI .bin file")
    args = parser.parse_args()
    print(inspect_point_cloud(args.path))


if __name__ == "__main__":
    main()
