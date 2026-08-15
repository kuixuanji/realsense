from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import Detection
from .tracking import IouTracker


class YoloDetector:
    def __init__(
        self,
        model_path: str | Path = "yolo26m.pt",
        *,
        confidence: float = 0.15,
        image_size: int = 640,
        device: str = "auto",
        max_detections: int = 50,
        tracker: str = "botsort.yaml",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("缺少 ultralytics，请先激活 realsense 环境") from exc
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.max_detections = max_detections
        self.tracker_config = tracker
        self.fallback_tracker = IouTracker(iou_threshold=0.20, max_missed=30)

        try:
            import torch
        except ImportError:
            self.runtime_device = "unknown"
        else:
            if device == "auto":
                self.runtime_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                self.runtime_device = str(device)

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        kwargs: dict[str, object] = {
            "source": bgr_image,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "max_det": self.max_detections,
            "verbose": False,
        }
        if self.device != "auto":
            kwargs["device"] = self.device
        result = self.model.track(
            **kwargs,
            persist=True,
            tracker=self.tracker_config,
        )[0]
        raw: list[Detection] = []
        if result.boxes is None:
            return self.fallback_tracker.update(raw)
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        ids = result.boxes.id
        track_ids = ids.detach().cpu().numpy().astype(int) if ids is not None else None
        names = result.names
        height, width = bgr_image.shape[:2]
        for index, (box, confidence, class_id) in enumerate(
            zip(xyxy, confidences, classes, strict=True)
        ):
            x1, y1, x2, y2 = (int(round(float(value))) for value in box)
            bbox = (
                max(0, min(width - 1, x1)),
                max(0, min(height - 1, y1)),
                max(1, min(width, x2)),
                max(1, min(height, y2)),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            raw.append(
                Detection(
                    track_id=int(track_ids[index]) if track_ids is not None else -1,
                    class_name=str(names[int(class_id)]),
                    confidence=float(confidence),
                    bbox=bbox,
                )
            )
        if track_ids is None:
            return self.fallback_tracker.update(raw)
        return raw
