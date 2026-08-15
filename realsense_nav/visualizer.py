from __future__ import annotations

import cv2
import numpy as np

from .models import Detection, LocalizedObject


VIEW_MODES = ("split", "color", "depth")
DEPTH_COLOR_MODES = ("official", "metric")


def next_view_mode(current: str) -> str:
    """Return the next live-view mode used by the D hotkey."""
    try:
        index = VIEW_MODES.index(current)
    except ValueError as exc:
        raise ValueError(f"未知显示模式：{current}") from exc
    return VIEW_MODES[(index + 1) % len(VIEW_MODES)]


def next_depth_color_mode(current: str) -> str:
    """Toggle between the SDK histogram view and a fixed metric view."""
    try:
        index = DEPTH_COLOR_MODES.index(current)
    except ValueError as exc:
        raise ValueError(f"未知深度着色模式：{current}") from exc
    return DEPTH_COLOR_MODES[(index + 1) % len(DEPTH_COLOR_MODES)]


def colorize_depth(
    depth_m: np.ndarray,
    *,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> np.ndarray:
    """Convert metric depth to a stable BGR colormap.

    Near pixels use warm colors, far pixels use cool colors, and invalid/out-of-range
    pixels are black.  The fixed metric range prevents the same distance changing color
    when a new near/far object enters the frame.
    """
    if depth_m.ndim != 2:
        raise ValueError("depth_m 必须是二维数组")
    if min_depth_m < 0 or max_depth_m <= min_depth_m:
        raise ValueError("深度显示范围无效")

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        normalized[valid] = np.uint8(
            np.clip(
                (max_depth_m - depth[valid]) / (max_depth_m - min_depth_m) * 255.0,
                0.0,
                255.0,
            )
        )
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def colorize_depth_histogram(
    depth_m: np.ndarray,
    *,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> np.ndarray:
    """Histogram-equalized fallback for tests or non-RealSense frame sources.

    Real D435 frames use ``rs.colorizer`` in ``camera.py``.  This NumPy version keeps
    the visualizer usable when callers only have a metric depth array.
    """
    if depth_m.ndim != 2:
        raise ValueError("depth_m 必须是二维数组")
    if min_depth_m < 0 or max_depth_m <= min_depth_m:
        raise ValueError("深度显示范围无效")

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    values = depth[valid]
    if values.size:
        unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        del unique
        cdf = np.cumsum(counts, dtype=np.float64)
        ranks = cdf / cdf[-1]
        normalized[valid] = np.uint8(np.clip(ranks[inverse] * 255.0, 0.0, 255.0))
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    return (
        80 + (track_id * 67) % 176,
        80 + (track_id * 131) % 176,
        80 + (track_id * 37) % 176,
    )


def _dashed_rectangle(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = bbox
    dash = 10
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_scene(
    image: np.ndarray,
    detections: list[Detection],
    localized: dict[int, LocalizedObject],
    *,
    selected_id: int | None = None,
) -> np.ndarray:
    canvas = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        is_selected = detection.track_id == selected_id
        color = (0, 255, 0) if is_selected else _color_for_id(detection.track_id)
        thickness = 4 if is_selected else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        target = localized.get(detection.track_id)
        if target is None:
            details = "depth=N/A"
        else:
            depth_mark = f" depth-age={target.depth_age_s:.1f}s" if target.depth_age_s > 0.25 else ""
            details = f"{target.distance_m:.2f}m dir={target.direction_deg:.1f}deg{depth_mark}"
        label = f"#{detection.track_id} {detection.class_name} {detection.confidence:.2f} {details}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        top = max(0, y1 - text_h - baseline - 6)
        cv2.rectangle(canvas, (x1, top), (min(canvas.shape[1] - 1, x1 + text_w + 6), y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, max(text_h + 1, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    for target in localized.values():
        if target.is_visible:
            continue
        color = (0, 255, 255) if target.id == selected_id else (140, 140, 140)
        _dashed_rectangle(canvas, target.detection.bbox, color, 3 if target.id == selected_id else 2)
        x1, y1, _, _ = target.detection.bbox
        cv2.putText(
            canvas,
            f"#{target.id} MEMORY {target.last_seen_age_s:.1f}s",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "Type target | D: view | H: depth palette | Q / ESC: quit",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_depth_scale(
    canvas: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    depth_m: np.ndarray | None = None,
    histogram_palette_bgr: np.ndarray | None = None,
) -> None:
    height, width = canvas.shape[:2]
    if height < 180 or width < 180:
        return
    panel_width = 126
    panel_x = width - panel_width - 8
    panel_y = 38
    bar_height = min(210, height - 118)
    panel_height = bar_height + 68
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (width - 8, panel_y + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)

    cv2.putText(
        canvas,
        "DEPTH (m) AUTO" if histogram_palette_bgr is not None else "DEPTH (m) FIXED",
        (panel_x + 7, panel_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    bar_width = 18
    x1 = width - 38
    y1 = panel_y + 28
    tick_ratios = np.linspace(0.0, 1.0, 5)
    if depth_m is not None and histogram_palette_bgr is not None:
        # A small regular sample is enough for the on-screen legend and avoids sorting
        # all ~300k pixels on every video frame.
        stride = max(1, int(np.sqrt(depth_m.size / 12000.0)))
        sampled_depth = depth_m[::stride, ::stride]
        sampled_palette = histogram_palette_bgr[::stride, ::stride]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= min_depth_m)
            & (sampled_depth <= max_depth_m)
        )
        values = np.asarray(sampled_depth[valid], dtype=np.float32)
        colors = np.asarray(sampled_palette[valid], dtype=np.uint8)
    else:
        values = np.empty(0, dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)

    if values.size:
        order = np.argsort(values, kind="stable")
        sample_indices = np.linspace(0, order.size - 1, bar_height).astype(np.int64)
        bar_line = colors[order[sample_indices]]
        bar = np.repeat(bar_line[:, None, :], bar_width, axis=1)
        tick_values = np.quantile(values, tick_ratios)
    else:
        gradient = np.linspace(255, 0, bar_height, dtype=np.uint8).reshape(-1, 1)
        gradient = np.repeat(gradient, bar_width, axis=1)
        bar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        tick_values = min_depth_m + tick_ratios * (max_depth_m - min_depth_m)
    canvas[y1 : y1 + bar_height, x1 : x1 + bar_width] = bar
    cv2.rectangle(canvas, (x1 - 1, y1 - 1), (x1 + bar_width, y1 + bar_height), (255, 255, 255), 1)
    label_x = panel_x + 7
    tick_count = len(tick_ratios)
    for index, ratio in enumerate(tick_ratios):
        y = y1 + round(ratio * (bar_height - 1))
        value = float(tick_values[index])
        suffix = " near" if index == 0 else " far" if index == tick_count - 1 else ""
        cv2.line(canvas, (x1 - 4, y), (x1 + bar_width + 3, y), (255, 255, 255), 1)
        cv2.putText(
            canvas,
            f"{value:.1f}{suffix}",
            (label_x, min(y1 + bar_height - 2, max(y1 + 10, y + 4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    invalid_y = y1 + bar_height + 13
    cv2.rectangle(canvas, (label_x, invalid_y), (label_x + 13, invalid_y + 10), (0, 0, 0), -1)
    cv2.rectangle(canvas, (label_x, invalid_y), (label_x + 13, invalid_y + 10), (255, 255, 255), 1)
    cv2.putText(
        canvas,
        "invalid",
        (label_x + 20, invalid_y + 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_depth_scene(
    depth_m: np.ndarray,
    detections: list[Detection],
    localized: dict[int, LocalizedObject],
    *,
    depth_color_bgr: np.ndarray | None = None,
    depth_color_mode: str = "official",
    selected_id: int | None = None,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> np.ndarray:
    """Draw the aligned live depth image with current-frame detection boxes."""
    if depth_color_mode not in DEPTH_COLOR_MODES:
        raise ValueError(f"未知深度着色模式：{depth_color_mode}")
    if depth_color_bgr is not None and (
        depth_color_bgr.ndim != 3
        or depth_color_bgr.shape[2] != 3
        or depth_color_bgr.shape[:2] != depth_m.shape
    ):
        raise ValueError("SDK 深度彩图尺寸与米制深度图不一致")

    valid = (
        np.isfinite(depth_m)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
    )
    histogram_palette: np.ndarray | None
    if depth_color_mode == "official":
        if depth_color_bgr is None:
            canvas = colorize_depth_histogram(
                depth_m,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
        else:
            canvas = depth_color_bgr.copy()
            canvas[~valid] = 0
        histogram_palette = canvas.copy()
        mode_label = "RS HIST"
    else:
        canvas = colorize_depth(
            depth_m,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        histogram_palette = None
        mode_label = "METRIC"
    height, width = depth_m.shape
    center_x, center_y = width // 2, height // 2
    center_depth = float(depth_m[center_y, center_x])
    center_valid = np.isfinite(center_depth) and min_depth_m <= center_depth <= max_depth_m
    center_label = f"center={center_depth:.2f}m" if center_valid else "center=N/A"

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        is_selected = detection.track_id == selected_id
        color = (0, 255, 0) if is_selected else _color_for_id(detection.track_id)
        target = localized.get(detection.track_id)
        if target is None:
            depth_label = "z=N/A"
        elif target.depth_age_s > 0.25:
            depth_label = f"z={target.z_m:.2f}m stale"
        else:
            depth_label = f"z={target.z_m:.2f}m"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4 if is_selected else 2)
        cv2.putText(
            canvas,
            f"#{detection.track_id} {depth_label}",
            (x1, max(44, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    crosshair_color = (255, 255, 255)
    cv2.line(canvas, (center_x - 6, center_y), (center_x + 6, center_y), crosshair_color, 1)
    cv2.line(canvas, (center_x, center_y - 6), (center_x, center_y + 6), crosshair_color, 1)
    cv2.putText(
        canvas,
        f"{mode_label} DEPTH | {center_label} | H: palette | black=invalid",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    _draw_depth_scale(
        canvas,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        depth_m=depth_m if histogram_palette is not None else None,
        histogram_palette_bgr=histogram_palette,
    )
    return canvas


def draw_live_view(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    detections: list[Detection],
    localized: dict[int, LocalizedObject],
    *,
    depth_color_bgr: np.ndarray | None = None,
    depth_color_mode: str = "official",
    view_mode: str = "split",
    selected_id: int | None = None,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> np.ndarray:
    """Render color, aligned depth, or a side-by-side live view."""
    if view_mode not in VIEW_MODES:
        raise ValueError(f"未知显示模式：{view_mode}")
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr 必须是 HxWx3 图像")
    if color_bgr.shape[:2] != depth_m.shape:
        raise ValueError("彩色图和对齐深度图尺寸不一致")

    if view_mode == "color":
        return draw_scene(color_bgr, detections, localized, selected_id=selected_id)
    if view_mode == "depth":
        return draw_depth_scene(
            depth_m,
            detections,
            localized,
            depth_color_bgr=depth_color_bgr,
            depth_color_mode=depth_color_mode,
            selected_id=selected_id,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
    color_canvas = draw_scene(color_bgr, detections, localized, selected_id=selected_id)
    depth_canvas = draw_depth_scene(
        depth_m,
        detections,
        localized,
        depth_color_bgr=depth_color_bgr,
        depth_color_mode=depth_color_mode,
        selected_id=selected_id,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    return np.hstack((color_canvas, depth_canvas))
