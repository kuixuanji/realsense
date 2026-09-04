from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Detection:
    track_id: int
    class_name: str
    confidence: float
    bbox: BBox

    @property
    def center_x(self) -> float:
        x1, _, x2, _ = self.bbox
        return (x1 + x2) / 2.0

    @property
    def center_y(self) -> float:
        _, y1, _, y2 = self.bbox
        return (y1 + y2) / 2.0


@dataclass(frozen=True, slots=True)
class LocalizedObject:
    detection: Detection
    x_m: float
    y_m: float
    z_m: float
    distance_m: float
    signed_angle_deg: float
    direction_deg: float
    depth_sample_count: int
    is_visible: bool = True
    last_seen_age_s: float = 0.0
    depth_age_s: float = 0.0
    observation_count: int = 1
    depth_mad_m: float = 0.0

    @property
    def id(self) -> int:
        return self.detection.track_id

    @property
    def class_name(self) -> str:
        return self.detection.class_name

    @property
    def horizontal_position(self) -> str:
        if self.signed_angle_deg < -8.0:
            return "left"
        if self.signed_angle_deg > 8.0:
            return "right"
        return "center"

    def as_selector_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "class": self.class_name,
            "confidence": round(self.detection.confidence, 3),
            "horizontal_position": self.horizontal_position,
            "distance_m": round(self.distance_m, 3),
            "signed_angle_deg": round(self.signed_angle_deg, 2),
            "direction_0_180_deg": round(self.direction_deg, 2),
            "bbox": list(self.detection.bbox),
            "visible": self.is_visible,
            "last_seen_age_s": round(self.last_seen_age_s, 2),
            "depth_age_s": round(self.depth_age_s, 2),
            "observation_count": self.observation_count,
            "depth_mad_m": round(self.depth_mad_m, 3),
        }


SelectionStatus = Literal["matched", "ambiguous", "not_found"]


@dataclass(frozen=True, slots=True)
class SelectionResult:
    status: SelectionStatus
    target_id: int | None = None
    candidate_ids: tuple[int, ...] = ()
    reason: str = ""
    source: str = "local"
