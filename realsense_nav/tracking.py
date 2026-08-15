from __future__ import annotations

from dataclasses import replace

from .models import BBox, Detection


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class IouTracker:
    """无需额外依赖的轻量跟踪器，使画面中的目标编号跨帧稳定。"""

    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 12) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 0
        self._tracks: dict[int, Detection] = {}
        self._missed: dict[int, int] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        pairs: list[tuple[float, int, int]] = []
        for track_id, previous in self._tracks.items():
            for index, current in enumerate(detections):
                if previous.class_name != current.class_name:
                    continue
                score = bbox_iou(previous.bbox, current.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, track_id, index))
        pairs.sort(reverse=True)

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        output: list[Detection | None] = [None] * len(detections)
        for _, track_id, index in pairs:
            if track_id in assigned_tracks or index in assigned_detections:
                continue
            tracked = replace(detections[index], track_id=track_id)
            output[index] = tracked
            self._tracks[track_id] = tracked
            self._missed[track_id] = 0
            assigned_tracks.add(track_id)
            assigned_detections.add(index)

        for index, detection in enumerate(detections):
            if index in assigned_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            tracked = replace(detection, track_id=track_id)
            output[index] = tracked
            self._tracks[track_id] = tracked
            self._missed[track_id] = 0
            assigned_tracks.add(track_id)

        for track_id in list(self._tracks):
            if track_id in assigned_tracks:
                continue
            self._missed[track_id] = self._missed.get(track_id, 0) + 1
            if self._missed[track_id] > self.max_missed:
                del self._tracks[track_id]
                del self._missed[track_id]
        return [item for item in output if item is not None]

