# LiDAR 3D Object Detection & Desktop AI Predict Integration

This project is an interview-defendable prototype for 3D object detection on KITTI-style LiDAR point clouds using the PointPillars architecture through OpenPCDet and PyTorch. It exposes a stable JSON inference contract through both a CLI and a FastAPI service so a desktop C++/Qt application can consume detections without embedding ML runtime logic.

## Project Overview

- Dataset: KITTI-style LiDAR point clouds stored as `.bin` files.
- ML framework: PyTorch.
- 3D detection framework: OpenPCDet.
- Model architecture: PointPillars.
- Integration boundary: JSON over CLI output or HTTP.
- Desktop integration target: a C++/Qt app can call the API and render returned detections.

## Architecture

```text
KITTI .bin file
    -> pointcloud_loader.py
    -> detector.py
       -> OpenPCDet PointPillars model
       -> OpenPCDet post-processing / NMS
    -> postprocess.py
    -> schemas.py
    -> inference.py or api.py
    -> JSON response consumed by desktop app
```

## What PointPillars Is

PointPillars is a LiDAR 3D object detection architecture that converts sparse 3D point clouds into a pseudo-image representation built from vertical columns called pillars. A backbone network then predicts 3D bounding boxes, classes, and confidence scores.

This project does not reimplement PointPillars from scratch. Instead, it wraps a maintained open-source implementation from OpenPCDet.

## KITTI `.bin` Format

Each KITTI LiDAR file is a flat sequence of `float32` values:

```text
x, y, z, reflectance, x, y, z, reflectance, ...
```

After loading, the array is reshaped to `(-1, 4)`.

## Point Fields: x, y, z, Reflectance

- `x`: forward distance
- `y`: lateral distance
- `z`: vertical position
- `reflectance`: laser return intensity

## 3D Box Representation

Each detection box includes:

- `center`: `[x, y, z]`
- `size`: `[length, width, height]`
- `yaw`: rotation around the vertical axis
- `class_name`
- `score`

## Confidence Filtering and NMS

- Confidence filtering removes low-score predictions below a configured threshold.
- NMS, or Non-Maximum Suppression, removes overlapping duplicate detections.
- In the real OpenPCDet path, post-processing and NMS are expected to happen inside the framework configuration and model post-processing pipeline.
- This repo does not reimplement rotated 3D NMS. It exposes the post-processed predictions from OpenPCDet and documents that boundary clearly.
- A simple axis-aligned 2D IoU helper is included only for teaching and testing purposes.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Important Compatibility Note

OpenPCDet, PyTorch, CUDA, and GPU driver versions must be compatible with each other. This repository pins the API/test/tooling dependencies, but OpenPCDet installation may vary by your CUDA environment.

## OpenPCDet Setup

Install OpenPCDet into the same environment. A common pattern is:

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git third_party/OpenPCDet
cd third_party/OpenPCDet
pip install -r requirements.txt
pip install -e .
cd ../..
```

Depending on your PyTorch and CUDA versions, some OpenPCDet ops may require a GPU-capable build to function.

## Sample KITTI Data

To obtain KITTI-style LiDAR `.bin` files:

1. Download the KITTI 3D object detection dataset from the official KITTI site.
2. Extract Velodyne point cloud files.
3. Copy one or more `.bin` files into `data/sample/`.

Example:

```bash
mkdir -p data/sample
cp /path/to/kitti/training/velodyne/000123.bin data/sample/
```

## Compatible Pretrained PointPillars Checkpoint

Use a PointPillars checkpoint that matches:

- the OpenPCDet version you installed
- the dataset configuration
- the class ordering
- the model config file

Place the checkpoint under `models/`, for example:

```text
models/pointpillars_kitti.pth
```

Also place or reference the matching config, for example:

```text
configs/pointpillars_kitti.yaml
```

The default project configuration expects a KITTI PointPillars setup, but you should verify the exact paths and class names for your installed OpenPCDet release.

## Running Inference

Real inference requires a valid OpenPCDet installation and a compatible checkpoint.

```bash
python -m src.inference \
  --input data/sample/000123.bin \
  --output outputs/000123.json \
  --score-threshold 0.5 \
  --checkpoint models/pointpillars_kitti.pth \
  --config configs/pointpillars_kitti.yaml
```

If the checkpoint is missing, the CLI fails with `MODEL_NOT_FOUND` instead of generating fake detections.

### Mock Mode

Mock mode exists for API wiring and desktop integration tests only:

```bash
python -m src.inference \
  --input data/sample/000123.bin \
  --output outputs/mock.json \
  --mock
```

Mock responses are clearly labeled with `model_version: "mock"` so they cannot be confused with real ML inference.

## Running the FastAPI Service

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

You can also use:

```bash
bash scripts/run_api.sh
```

Environment variables:

- `LIDAR_MODEL_CHECKPOINT`: checkpoint path
- `LIDAR_MODEL_CONFIG`: config path
- `LIDAR_DEVICE`: `cuda` or `cpu`
- `LIDAR_ENABLE_MOCK`: `true` or `false`
- `LIDAR_ALLOWED_DATA_ROOT`: optional base directory for development file-path requests

## Example `curl` Request

Using multipart upload:

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@data/sample/000123.bin" \
  -F "score_threshold=0.5"
```

Using a local file path in development mode:

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file_path=data/sample/000123.bin" \
  -F "score_threshold=0.5"
```

## Example JSON Response

```json
{
  "status": "success",
  "model_name": "PointPillars",
  "model_version": "kitti-pretrained-v1",
  "inference_time_ms": 45.2,
  "detections": [
    {
      "class_name": "Car",
      "score": 0.91,
      "center": [12.4, -3.1, 0.8],
      "size": [4.2, 1.8, 1.6],
      "yaw": 1.42
    }
  ]
}
```

No detections is still a successful response:

```json
{
  "status": "success",
  "model_name": "PointPillars",
  "model_version": "kitti-pretrained-v1",
  "inference_time_ms": 42.7,
  "detections": []
}
```

## CPU and GPU Limitations

- The point-cloud loader, schema validation, post-processing helpers, evaluation helpers, CLI argument parsing, API validation, and tests work without a GPU.
- Real OpenPCDet PointPillars inference may require CUDA depending on the installed ops and build configuration.
- This project checks for missing dependencies and CUDA-related limitations explicitly and returns readable errors such as `CUDA_UNAVAILABLE` or `INFERENCE_FAILED`.
- CPU fallback is attempted only where the installed backend supports it. The code does not silently claim CPU inference is always supported.

## ONNX Limitations

- PointPillars in OpenPCDet may rely on custom CUDA ops for voxelization, NMS, or other preprocessing stages.
- Because of that, full ONNX export is not guaranteed.
- This repository includes a conservative ONNX export stub and validation design.
- It does not claim successful ONNX support unless export and runtime validation actually succeed in your environment.

## Interview Explanation: What Exactly Did You Build?

You built a production-style integration layer around a real 3D detection stack:

- A loader for KITTI LiDAR point clouds.
- A reusable detector wrapper that loads PointPillars once and reuses it.
- Structured JSON schemas for detections and errors.
- A CLI for offline inference.
- A FastAPI service that a desktop application can call.
- Simple educational post-processing and evaluation helpers.
- Tests for input validation, filtering, empty outputs, and API behavior.

The ML model itself comes from OpenPCDet, the framework itself uses PyTorch, the dataset format is KITTI-style LiDAR, and the application contract is JSON.

## Known Failure Cases

- Missing checkpoint or config path.
- OpenPCDet not installed in the environment.
- CUDA-only ops when running on a CPU-only machine.
- Incompatible checkpoint/config/class mapping combinations.
- Malformed `.bin` files whose float count is not divisible by 4.
- Extremely large uploads that exceed server resource limits.

## Future Improvements

- Add authenticated desktop-to-service communication.
- Add asynchronous batch inference.
- Add official KITTI evaluation automation against full labels.
- Add request tracing and structured observability.
- Add proper OpenPCDet config discovery and validation.
- Add Docker packaging for reproducible deployment.

## Exact Commands

### Project Tree

```bash
find . -maxdepth 3 | sort
```

### Setup Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Tests

```bash
pytest -q
```

### Inspect One KITTI `.bin` File

```bash
python -m src.pointcloud_loader data/sample/000123.bin
```

### Run Real Inference

```bash
python -m src.inference \
  --input data/sample/000123.bin \
  --output outputs/000123.json \
  --score-threshold 0.5 \
  --checkpoint models/pointpillars_kitti.pth \
  --config configs/pointpillars_kitti.yaml
```

### Run FastAPI

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Example `curl`

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@data/sample/000123.bin" \
  -F "score_threshold=0.5"
```

### Which Parts Require GPU?

- Usually real OpenPCDet PointPillars inference.
- Possibly ONNX parity validation if export depends on custom CUDA ops.

### Which Parts Work Without GPU?

- Loader
- Schemas
- CLI validation
- API validation
- Mock mode
- Post-processing helpers
- Evaluation helpers
- Tests
- README/documentation
