from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraFrame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    depth_color_bgr: np.ndarray | None
    intrinsics: object
    timestamp_ms: float


class RealSenseCamera:
    def __init__(
        self,
        *,
        color_width: int = 640,
        color_height: int = 480,
        depth_width: int = 848,
        depth_height: int = 480,
        fps: int = 30,
        use_filters: bool = True,
        create_depth_color: bool = True,
        serial: str = "",
        rotate: int = 0,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("缺少 pyrealsense2，请先激活 realsense 环境") from exc
        self.rs = rs
        self.color_width = color_width
        self.color_height = color_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps
        self.use_filters = use_filters
        self.create_depth_color = create_depth_color
        self.serial = serial.strip()
        if rotate not in (0, 90, 180, 270):
            raise ValueError("rotate 必须是 0、90、180 或 270")
        self.rotate = rotate
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        # Match the RealSense Viewer-style depth rendering: the SDK colorizer uses
        # histogram equalization by default and reveals much more local depth detail
        # than a fixed 0-8 m linear colormap.
        self.colorizer = rs.colorizer() if create_depth_color else None
        if self.colorizer is not None:
            self.colorizer.set_option(rs.option.color_scheme, 0.0)
            self.colorizer.set_option(rs.option.histogram_equalization_enabled, 1.0)
        self.spatial = rs.spatial_filter() if use_filters else None
        self.temporal = rs.temporal_filter() if use_filters else None
        self.hole_filling = rs.hole_filling_filter() if use_filters else None
        self.depth_scale = 0.001
        self.profile = None

    @staticmethod
    def list_devices() -> list[dict[str, str]]:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("缺少 pyrealsense2，请先激活 realsense 环境") from exc
        devices: list[dict[str, str]] = []
        for device in rs.context().query_devices():
            info: dict[str, str] = {}
            for key, label in (
                (rs.camera_info.name, "name"),
                (rs.camera_info.serial_number, "serial"),
                (rs.camera_info.firmware_version, "firmware"),
                (rs.camera_info.product_line, "product_line"),
                (rs.camera_info.product_id, "product_id"),
                (rs.camera_info.usb_type_descriptor, "usb_type"),
            ):
                if device.supports(key):
                    info[label] = device.get_info(key)
            devices.append(info)
        return devices

    def start(self, warmup_frames: int = 15) -> None:
        rs = self.rs
        if self.serial:
            self.config.enable_device(self.serial)
        self.config.enable_stream(
            rs.stream.depth,
            self.depth_width,
            self.depth_height,
            rs.format.z16,
            self.fps,
        )
        self.config.enable_stream(
            rs.stream.color,
            self.color_width,
            self.color_height,
            rs.format.bgr8,
            self.fps,
        )
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as exc:
            raise RuntimeError(
                "无法启动 RealSense。请确认 D435 已连接、未被 RealSense Viewer 占用，并尝试 USB 3.x 接口。"
            ) from exc
        sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(sensor.get_depth_scale())
        for _ in range(max(0, warmup_frames)):
            self.pipeline.wait_for_frames(5000)

    def stop(self) -> None:
        if self.profile is not None:
            self.pipeline.stop()
            self.profile = None

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def get_frame(self, timeout_ms: int = 5000) -> CameraFrame:
        frames = self.pipeline.wait_for_frames(timeout_ms)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("未同时取得彩色帧和对齐深度帧")
        if self.use_filters:
            assert self.spatial is not None
            assert self.temporal is not None
            assert self.hole_filling is not None
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        depth_color_bgr: np.ndarray | None = None
        if self.colorizer is not None:
            depth_color_rgb = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            depth_color_bgr = depth_color_rgb[:, :, ::-1].copy()
        intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        if self.rotate:
            k = {90: 3, 180: 2, 270: 1}[self.rotate]
            color_bgr = np.rot90(color_bgr, k=k).copy()
            depth_m = np.rot90(depth_m, k=k).copy()
            if depth_color_bgr is not None:
                depth_color_bgr = np.rot90(depth_color_bgr, k=k).copy()
            intrinsics = _RotatedIntrinsics.from_source(intrinsics, self.rotate)
        return CameraFrame(
            color_bgr=color_bgr,
            depth_m=depth_m,
            depth_color_bgr=depth_color_bgr,
            intrinsics=intrinsics,
            timestamp_ms=float(color_frame.get_timestamp()),
        )

    def deproject(
        self,
        intrinsics: object,
        pixel: Sequence[float],
        depth_m: float,
    ) -> Sequence[float]:
        if isinstance(intrinsics, _RotatedIntrinsics):
            return (
                (float(pixel[0]) - intrinsics.ppx) / intrinsics.fx * float(depth_m),
                (float(pixel[1]) - intrinsics.ppy) / intrinsics.fy * float(depth_m),
                float(depth_m),
            )
        return self.rs.rs2_deproject_pixel_to_point(
            intrinsics, [float(pixel[0]), float(pixel[1])], float(depth_m)
        )


@dataclass(frozen=True, slots=True)
class _RotatedIntrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float

    @classmethod
    def from_source(cls, source: object, rotate: int) -> "_RotatedIntrinsics":
        width = float(getattr(source, "width"))
        height = float(getattr(source, "height"))
        fx = float(getattr(source, "fx"))
        fy = float(getattr(source, "fy"))
        ppx = float(getattr(source, "ppx"))
        ppy = float(getattr(source, "ppy"))
        if rotate == 90:
            return cls(fx=fy, fy=fx, ppx=height - 1.0 - ppy, ppy=ppx)
        if rotate == 180:
            return cls(fx=fx, fy=fy, ppx=width - 1.0 - ppx, ppy=height - 1.0 - ppy)
        return cls(fx=fy, fy=fx, ppx=ppy, ppy=width - 1.0 - ppx)
