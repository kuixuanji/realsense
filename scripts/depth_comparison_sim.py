#!/usr/bin/env python3
# 运行命令：
#   cd /home/kui/realsense
#   /home/kui/miniconda3/envs/realsense/bin/python scripts/depth_comparison_sim.py

"""复现论文表4-2中的三种深度提取方法带噪对照结果。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realsense_nav.geometry import robust_depth_sample


RNG = np.random.default_rng(435)
TRIALS = 300
HEIGHT = 100
WIDTH = 100
ROI_START = 27
ROI_END = 73
TARGET_DEPTH_M = 2.0


def build_base_scene(kind: str) -> np.ndarray:
    """构造无噪声基准场景；中央46×46区域对应检测框中央45%。"""
    depth = np.full((HEIGHT, WIDTH), 4.0, dtype=np.float32)
    roi = depth[ROI_START:ROI_END, ROI_START:ROI_END]

    if kind == "solid":
        roi[:] = 2.0
    elif kind == "hollow":
        roi[:] = 4.0
        roi[:, :8] = 2.0
        roi[:, -8:] = 2.0
    elif kind == "center_outliers":
        roi[:] = 2.0
        for y, x in ((23, 23), (5, 5), (9, 31), (34, 12), (40, 40), (14, 37)):
            roi[y, x] = 0.30
    elif kind == "occluder":
        roi[:] = 2.0
        roi[17:29, 17:29] = 1.20
    else:
        raise ValueError(f"未知场景：{kind}")

    return depth


def add_noise(base: np.ndarray) -> np.ndarray:
    """加入深度相关像素噪声、帧间偏差、无效像素和随机飞点。"""
    depth = base.astype(np.float64).copy()

    for base_depth in np.unique(base):
        mask = base == base_depth
        frame_bias = RNG.normal(0.0, 0.005 * float(base_depth))
        pixel_sigma = 0.003 * float(base_depth) ** 2
        depth[mask] += frame_bias + RNG.normal(0.0, pixel_sigma, int(mask.sum()))

    roi = depth[ROI_START:ROI_END, ROI_START:ROI_END]
    invalid_mask = RNG.random(roi.shape) < 0.01
    roi[invalid_mask] = 0.0

    outlier_mask = RNG.random(roi.shape) < 0.005
    roi[outlier_mask] = RNG.uniform(0.2, 8.0, int(outlier_mask.sum()))
    return depth.astype(np.float32)


def measure(depth: np.ndarray, method: str) -> float | None:
    """使用指定方法计算中央区域的代表深度。"""
    roi = depth[ROI_START:ROI_END, ROI_START:ROI_END]

    if method == "center":
        value = float(roi[roi.shape[0] // 2, roi.shape[1] // 2])
        if np.isfinite(value) and 0.2 <= value <= 8.0:
            return value
        return None

    if method == "median":
        valid = roi[np.isfinite(roi) & (roi >= 0.2) & (roi <= 8.0)]
        return float(np.median(valid)) if valid.size else None

    if method == "robust":
        sample = robust_depth_sample(
            depth,
            (0, 0, WIDTH, HEIGHT),
            central_ratio=0.45,
            min_depth_m=0.2,
            max_depth_m=8.0,
            min_samples=12,
        )
        return None if sample is None else sample.depth_m

    raise ValueError(f"未知方法：{method}")


def evaluate_scene(kind: str) -> dict[str, tuple[float, float, float]]:
    """重复仿真并返回各方法的MAE、P95和有效输出率。"""
    values: dict[str, list[float | None]] = {
        "center": [],
        "median": [],
        "robust": [],
    }

    for _ in range(TRIALS):
        depth = add_noise(build_base_scene(kind))
        for method in values:
            values[method].append(measure(depth, method))

    result: dict[str, tuple[float, float, float]] = {}
    for method, samples in values.items():
        valid = np.asarray([value for value in samples if value is not None])
        absolute_error = np.abs(valid - TARGET_DEPTH_M)
        mae = float(np.mean(absolute_error))
        p95 = float(np.percentile(absolute_error, 95))
        valid_rate = 100.0 * len(valid) / TRIALS
        result[method] = (mae, p95, valid_rate)
    return result


SCENES = {
    "solid": "完整目标表面",
    "hollow": "镂空目标、背景占多数",
    "center_outliers": "中心附近6个飞点",
    "occluder": "前景遮挡约7%",
}


def main() -> None:
    for scene_key, scene_name in SCENES.items():
        result = evaluate_scene(scene_key)
        print(scene_name)
        for method in ("center", "median", "robust"):
            mae, p95, valid_rate = result[method]
            print(
                f"  {method:7s} "
                f"MAE={mae:.3f} m, P95={p95:.3f} m, valid={valid_rate:.1f}%"
            )


if __name__ == "__main__":
    main()
