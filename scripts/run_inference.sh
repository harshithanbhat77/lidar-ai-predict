#!/usr/bin/env bash
set -euo pipefail

python -m src.inference \
  --input "${1:-data/sample/000123.bin}" \
  --output "${2:-outputs/000123.json}" \
  --score-threshold "${3:-0.5}" \
  --checkpoint "${LIDAR_MODEL_CHECKPOINT:-models/pointpillars_kitti.pth}" \
  --config "${LIDAR_MODEL_CONFIG:-configs/pointpillars_kitti.yaml}"
