from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from .models import BBox, Detection, LocalizedObject


class IntrinsicsLike(Protocol):
    fx: float
    fy: float
    ppx: float
    ppy: float


DeprojectFn = Callable[[object, Sequence[float], float], Sequence[float]]


@dataclass(frozen=True, slots=True)
class DepthSample:
    pixel_x: float
    pixel_y: float
    depth_m: float
    count: int
    mad_m: float
    valid_fraction: float


def direction_from_xz(x_m: float, z_m: float) -> tuple[float, float]:
    """返回相对前方有符号角和 0°左/90°前/180°右方向角。"""
    if not math.isfinite(x_m) or not math.isfinite(z_m) or z_m <= 0:
        raise ValueError("目标必须位于相机前方，且 X/Z 必须为有限数值")
    signed = math.degrees(math.atan2(x_m, z_m))
    return signed, min(180.0, max(0.0, 90.0 + signed))


def pinhole_deproject(
    intrinsics: IntrinsicsLike,
    pixel: Sequence[float],
    depth_m: float,
) -> tuple[float, float, float]:
    """无畸变针孔模型反投影，主要供测试或 SDK 不可用时使用。"""
    u, v = pixel
    x = (float(u) - float(intrinsics.ppx)) / float(intrinsics.fx) * depth_m
    y = (float(v) - float(intrinsics.ppy)) / float(intrinsics.fy) * depth_m
    return x, y, depth_m


def _central_roi(bbox: BBox, shape: tuple[int, int], ratio: float) -> tuple[int, int, int, int]:
    if not 0.05 <= ratio <= 1.0:
        raise ValueError("central_ratio 必须位于 [0.05, 1.0]")
    height, width = shape
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = max(1.0, (x2 - x1) * ratio / 2.0)
    half_h = max(1.0, (y2 - y1) * ratio / 2.0)
    rx1 = max(0, min(width - 1, int(math.floor(cx - half_w))))
    ry1 = max(0, min(height - 1, int(math.floor(cy - half_h))))
    rx2 = max(rx1 + 1, min(width, int(math.ceil(cx + half_w))))
    ry2 = max(ry1 + 1, min(height, int(math.ceil(cy + half_h))))
    return rx1, ry1, rx2, ry2


def robust_depth_sample(
    depth_m: np.ndarray,
    bbox: BBox,
    *,
    central_ratio: float = 0.45,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    min_samples: int = 12,
) -> DepthSample | None:
    """从框中央区域选择最近的可信深度簇，再用 MAD 排除飞点。

    单纯对检测框取中位数会在椅子、桌子等镂空物体上偏向背景。这里先按
    深度断层分簇，从像素数量足够的簇中取最近一簇，再做稳健统计。
    """
    if depth_m.ndim != 2:
        raise ValueError("depth_m 必须是二维数组")
    rx1, ry1, rx2, ry2 = _central_roi(bbox, depth_m.shape, central_ratio)
    roi = np.asarray(depth_m[ry1:ry2, rx1:rx2], dtype=np.float32)
    valid = np.isfinite(roi) & (roi >= min_depth_m) & (roi <= max_depth_m)
    if int(valid.sum()) < min_samples:
        return None

    all_values = roi[valid]
    sorted_values = np.sort(all_values)
    if sorted_values.size > 1:
        adaptive_gap = np.maximum(0.04, sorted_values[:-1] * 0.025)
        boundaries = np.flatnonzero(np.diff(sorted_values) > adaptive_gap) + 1
        clusters = np.split(sorted_values, boundaries)
    else:
        clusters = [sorted_values]
    credible_count = max(min_samples, int(math.ceil(sorted_values.size * 0.03)))
    credible = [cluster for cluster in clusters if cluster.size >= credible_count]
    chosen = min(credible or clusters, key=lambda cluster: float(np.median(cluster)))
    cluster_center = float(np.median(chosen))
    cluster_radius = max(0.025, 3.0 * float(np.median(np.abs(chosen - cluster_center))))
    cluster_radius = max(cluster_radius, cluster_center * 0.015)

    ys, xs = np.nonzero(valid)
    cluster_keep = np.abs(all_values - cluster_center) <= cluster_radius
    values = all_values[cluster_keep]
    xs = xs[cluster_keep]
    ys = ys[cluster_keep]
    if values.size < min_samples:
        return None
    median = float(np.median(values))
    abs_dev = np.abs(values - median)
    mad = float(np.median(abs_dev))
    if mad > 1e-6:
        threshold = max(0.015, 3.5 * 1.4826 * mad)
        keep_values = abs_dev <= threshold
    else:
        keep_values = abs_dev <= max(0.015, median * 0.02)

    xs = xs[keep_values]
    ys = ys[keep_values]
    values = values[keep_values]
    if values.size < min_samples:
        return None
    return DepthSample(
        pixel_x=float(rx1 + np.median(xs)),
        pixel_y=float(ry1 + np.median(ys)),
        depth_m=float(np.median(values)),
        count=int(values.size),
        mad_m=float(np.median(np.abs(values - np.median(values)))),
        valid_fraction=float(all_values.size / roi.size),
    )


def localize_detection(
    detection: Detection,
    depth_m: np.ndarray,
    intrinsics: object,
    *,
    deproject: DeprojectFn | None = None,
    central_ratio: float = 0.45,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    min_samples: int = 12,
    max_depth_mad_m: float = 0.15,
) -> LocalizedObject | None:
    sample = robust_depth_sample(
        depth_m,
        detection.bbox,
        central_ratio=central_ratio,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        min_samples=min_samples,
    )
    if sample is None:
        return None
    if sample.mad_m > max_depth_mad_m:
        return None
    project = deproject or pinhole_deproject
    x_m, y_m, z_m = (
        float(value)
        for value in project(
            intrinsics,
            (sample.pixel_x, sample.pixel_y),
            sample.depth_m,
        )
    )
    signed, direction = direction_from_xz(x_m, z_m)
    return LocalizedObject(
        detection=detection,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        distance_m=math.hypot(x_m, z_m),
        signed_angle_deg=signed,
        direction_deg=direction,
        depth_sample_count=sample.count,
        depth_mad_m=sample.mad_m,
    )
