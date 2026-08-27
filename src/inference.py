from __future__ import annotations

import argparse
import sys

from src.detector import DetectorConfig, DetectorError, PointPillarsDetector
from src.pointcloud_loader import PointCloudFormatError, load_kitti_bin
from src.utils import write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PointPillars inference on a KITTI point cloud.")
    parser.add_argument("--input", required=True, help="Path to KITTI .bin point cloud")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", default=None, help="Path to PointPillars checkpoint")
    parser.add_argument("--config", default=None, help="Path to model config")
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cpu")
    parser.add_argument("--mock", action="store_true", help="Use mock detections for integration testing")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        print("score_threshold must be between 0.0 and 1.0", file=sys.stderr)
        return 2

    config = DetectorConfig(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        mock_mode=args.mock,
    )
    detector = PointPillarsDetector(config=config)

    try:
        point_cloud = load_kitti_bin(args.input)
        response = detector.predict(point_cloud, score_threshold=args.score_threshold)
        write_json(args.output, response.model_dump())
        return 0
    except PointCloudFormatError as exc:
        print(f"INVALID_POINT_CLOUD: {exc}", file=sys.stderr)
        return 3
    except DetectorError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
