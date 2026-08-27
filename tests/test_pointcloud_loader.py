import numpy as np
import pytest

from src.pointcloud_loader import PointCloudFormatError, load_kitti_bin, summarize_point_cloud


def test_load_valid_point_cloud(tmp_path):
    sample = np.array(
        [
            [1.0, 2.0, 3.0, 0.5],
            [4.0, 5.0, 6.0, 0.8],
        ],
        dtype=np.float32,
    )
    target = tmp_path / "sample.bin"
    sample.tofile(target)

    loaded = load_kitti_bin(target)

    assert loaded.shape == (2, 4)
    assert np.allclose(loaded, sample)


def test_load_malformed_point_cloud(tmp_path):
    bad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target = tmp_path / "bad.bin"
    bad.tofile(target)

    with pytest.raises(PointCloudFormatError):
        load_kitti_bin(target)


def test_summarize_point_cloud():
    point_cloud = np.array(
        [
            [1.0, -2.0, 0.5, 0.1],
            [3.0, 4.0, 2.5, 0.9],
        ],
        dtype=np.float32,
    )

    summary = summarize_point_cloud(point_cloud)

    assert summary.num_points == 2
    assert summary.min_x == 1.0
    assert summary.max_y == 4.0
    assert summary.reflectance_max == pytest.approx(0.9)
