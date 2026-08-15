from types import SimpleNamespace

import pytest

from realsense_nav.camera import _RotatedIntrinsics


def source():
    return SimpleNamespace(width=640, height=480, fx=600.0, fy=610.0, ppx=319.0, ppy=239.0)


def test_intrinsics_rotate_clockwise_90() -> None:
    rotated = _RotatedIntrinsics.from_source(source(), 90)
    assert rotated.fx == 610.0
    assert rotated.fy == 600.0
    assert rotated.ppx == pytest.approx(240.0)
    assert rotated.ppy == pytest.approx(319.0)


def test_intrinsics_rotate_180() -> None:
    rotated = _RotatedIntrinsics.from_source(source(), 180)
    assert rotated.ppx == pytest.approx(320.0)
    assert rotated.ppy == pytest.approx(240.0)
