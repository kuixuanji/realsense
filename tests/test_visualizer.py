import numpy as np
import pytest

from realsense_nav.visualizer import (
    colorize_depth,
    draw_depth_scene,
    draw_live_view,
    next_depth_color_mode,
    next_view_mode,
)


def test_colorize_depth_marks_invalid_and_out_of_range_black() -> None:
    depth = np.array([[0.0, np.nan, 0.2, 2.0, 8.0, 8.1]], dtype=np.float32)

    colored = colorize_depth(depth, min_depth_m=0.2, max_depth_m=8.0)

    assert colored.shape == (1, 6, 3)
    assert np.all(colored[0, 0] == 0)
    assert np.all(colored[0, 1] == 0)
    assert np.all(colored[0, 5] == 0)
    assert np.any(colored[0, 2] != 0)
    assert np.any(colored[0, 4] != 0)
    assert not np.array_equal(colored[0, 2], colored[0, 4])


@pytest.mark.parametrize(
    ("view_mode", "expected_width"),
    (("color", 100), ("depth", 100), ("split", 200)),
)
def test_live_view_dimensions(view_mode: str, expected_width: int) -> None:
    color = np.zeros((80, 100, 3), dtype=np.uint8)
    depth = np.full((80, 100), 2.0, dtype=np.float32)

    rendered = draw_live_view(color, depth, [], {}, view_mode=view_mode)

    assert rendered.shape == (80, expected_width, 3)


def test_live_view_requires_aligned_depth() -> None:
    color = np.zeros((80, 100, 3), dtype=np.uint8)
    depth = np.ones((40, 100), dtype=np.float32)

    with pytest.raises(ValueError, match="尺寸不一致"):
        draw_live_view(color, depth, [], {})


def test_depth_view_contains_multicolor_metric_legend() -> None:
    depth = np.linspace(0.2, 8.0, 480 * 640, dtype=np.float32).reshape(480, 640)

    rendered = draw_depth_scene(
        depth,
        [],
        {},
        depth_color_mode="metric",
        min_depth_m=0.2,
        max_depth_m=8.0,
    )
    legend_region = rendered[60:260, 600:620]

    # A uniform input has one base color; many colors here come from the metric scale.
    assert np.unique(legend_region.reshape(-1, 3), axis=0).shape[0] > 20


def test_view_mode_cycles() -> None:
    assert next_view_mode("split") == "color"
    assert next_view_mode("color") == "depth"
    assert next_view_mode("depth") == "split"
    assert next_depth_color_mode("official") == "metric"
    assert next_depth_color_mode("metric") == "official"


def test_official_depth_color_is_used_and_invalid_pixels_stay_black() -> None:
    depth = np.full((200, 240), 2.0, dtype=np.float32)
    depth[100, 100] = 0.0
    sdk_color = np.full((200, 240, 3), (17, 83, 191), dtype=np.uint8)

    rendered = draw_depth_scene(depth, [], {}, depth_color_bgr=sdk_color)

    # Choose pixels away from labels, the crosshair, and the right-side legend.
    assert np.array_equal(rendered[150, 40], sdk_color[150, 40])
    assert np.all(rendered[100, 100] == 0)
