from dataclasses import dataclass

import numpy as np
import pytest

from realsense_nav.geometry import direction_from_xz, localize_detection, robust_depth_sample
from realsense_nav.models import Detection


@dataclass
class Intrinsics:
    fx: float = 100.0
    fy: float = 100.0
    ppx: float = 50.0
    ppy: float = 40.0


def test_direction_convention() -> None:
    signed, direction = direction_from_xz(0.5, 2.0)
    assert signed == pytest.approx(14.036, abs=0.01)
    assert direction == pytest.approx(104.036, abs=0.01)
    left_signed, left_direction = direction_from_xz(-0.5, 2.0)
    assert left_signed < 0
    assert left_direction < 90


def test_robust_depth_rejects_zero_and_outlier() -> None:
    depth = np.full((80, 100), 2.0, dtype=np.float32)
    depth[30:35, 40:45] = 0.0
    depth[35:38, 45:48] = 7.0
    sample = robust_depth_sample(depth, (20, 10, 80, 70), central_ratio=0.5)
    assert sample is not None
    assert sample.depth_m == pytest.approx(2.0)
    assert sample.count > 100
    assert sample.mad_m == pytest.approx(0.0)


def test_depth_prefers_near_credible_foreground_cluster() -> None:
    depth = np.full((100, 100), 4.0, dtype=np.float32)
    depth[40:60, 40:60] = 2.0
    sample = robust_depth_sample(depth, (0, 0, 100, 100), central_ratio=1.0)
    assert sample is not None
    assert sample.depth_m == pytest.approx(2.0)


def test_localize_detection_uses_depth_and_intrinsics() -> None:
    depth = np.full((80, 100), 2.0, dtype=np.float32)
    detection = Detection(track_id=7, class_name="chair", confidence=0.9, bbox=(60, 20, 80, 60))
    target = localize_detection(detection, depth, Intrinsics())
    assert target is not None
    assert target.id == 7
    assert target.x_m > 0
    assert target.z_m == pytest.approx(2.0)
    assert target.direction_deg > 90


def test_invalid_z_rejected() -> None:
    with pytest.raises(ValueError):
        direction_from_xz(0.0, 0.0)
