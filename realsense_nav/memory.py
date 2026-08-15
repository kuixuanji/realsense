from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

from .geometry import direction_from_xz
from .models import Detection, LocalizedObject
from .tracking import bbox_iou


@dataclass(slots=True)
class _MemoryEntry:
    obj: LocalizedObject
    last_seen_s: float
    observation_count: int
    last_depth_s: float


class TargetMemory:
    """在检测短暂漏帧时保留目标，并平滑三维定位结果。

    普通候选保留 ``memory_seconds``；用户已经选中的目标保留更久，并可在
    BoT-SORT 分配了新 ID 后依据类别、框位置和三维位置重新绑定原 ID。
    """

    def __init__(
        self,
        *,
        memory_seconds: float = 4.0,
        selected_memory_seconds: float = 15.0,
        smoothing: float = 0.35,
        min_observations: int = 2,
        candidate_confidence: float = 0.30,
    ) -> None:
        if memory_seconds <= 0:
            raise ValueError("memory_seconds 必须大于 0")
        if selected_memory_seconds < memory_seconds:
            raise ValueError("selected_memory_seconds 不能短于 memory_seconds")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing 必须位于 (0, 1]")
        if min_observations < 1:
            raise ValueError("min_observations 必须至少为 1")
        if not 0.0 < candidate_confidence <= 1.0:
            raise ValueError("candidate_confidence 必须位于 (0, 1]")
        self.memory_seconds = memory_seconds
        self.selected_memory_seconds = selected_memory_seconds
        self.smoothing = smoothing
        self.min_observations = min_observations
        self.candidate_confidence = candidate_confidence
        self.selected_id: int | None = None
        self._entries: dict[int, _MemoryEntry] = {}
        # BoT-SORT may assign a new ID after a short occlusion.  Once the new ID has
        # been matched to a remembered target, keep using the original, user-facing
        # ID for observations, depth-hole detections and rendering.
        self._id_aliases: dict[int, int] = {}
        self.reacquired_on_last_update = False

    def select(self, track_id: int | None) -> None:
        canonical_id = self._canonical_id(track_id) if track_id is not None else None
        self.selected_id = canonical_id if canonical_id in self._entries else None

    def canonicalize_detections(self, detections: list[Detection]) -> list[Detection]:
        """将跟踪器的新 ID 归一为稳定 ID，并删除同一目标的重复框。"""
        result: list[Detection] = []
        positions: dict[int, int] = {}
        for detection in detections:
            canonical_id = self._canonical_id(detection.track_id)
            current = (
                detection
                if canonical_id == detection.track_id
                else replace(detection, track_id=canonical_id)
            )
            position = positions.get(canonical_id)
            if position is None:
                positions[canonical_id] = len(result)
                result.append(current)
            elif current.confidence > result[position].confidence:
                result[position] = current
        return result

    def update(
        self,
        observations: list[LocalizedObject],
        *,
        detections: list[Detection] | None = None,
        now_s: float | None = None,
    ) -> list[LocalizedObject]:
        now = time.monotonic() if now_s is None else float(now_s)
        self.reacquired_on_last_update = False
        observations = self._canonicalize_observations(observations)
        observations = self._rebind_selected(observations, now)
        # _rebind_selected may have added a new alias.  Apply it to every input so
        # an old/new-ID pair in the same frame becomes exactly one memory entry.
        observations = self._canonicalize_observations(observations)
        canonical_detections = self.canonicalize_detections(detections or [])
        seen: set[int] = set()

        for current in observations:
            seen.add(current.id)
            previous = self._entries.get(current.id)
            if previous is None or previous.obj.class_name != current.class_name:
                count = 1
                smoothed = replace(
                    current,
                    is_visible=True,
                    last_seen_age_s=0.0,
                    observation_count=count,
                )
            else:
                count = previous.observation_count + 1
                smoothed = self._smooth(previous.obj, current, count)
            self._entries[current.id] = _MemoryEntry(smoothed, now, count, now)

        # 深度偶尔产生空洞时，目标仍属于“视觉可见”；沿用最近三维位置而不是失焦。
        for detection in canonical_detections:
            if detection.track_id in seen:
                continue
            entry = self._entries.get(detection.track_id)
            if entry is None or entry.obj.class_name != detection.class_name:
                continue
            seen.add(detection.track_id)
            entry.observation_count += 1
            entry.last_seen_s = now
            entry.obj = replace(
                entry.obj,
                detection=detection,
                is_visible=True,
                last_seen_age_s=0.0,
                depth_age_s=max(0.0, now - entry.last_depth_s),
                observation_count=entry.observation_count,
            )

        expired: list[int] = []
        for track_id, entry in self._entries.items():
            if track_id in seen:
                continue
            age = max(0.0, now - entry.last_seen_s)
            ttl = (
                self.selected_memory_seconds
                if track_id == self.selected_id
                else self.memory_seconds
            )
            if age > ttl:
                expired.append(track_id)
                continue
            entry.obj = replace(
                entry.obj,
                is_visible=False,
                last_seen_age_s=age,
                depth_age_s=max(0.0, now - entry.last_depth_s),
            )

        for track_id in expired:
            del self._entries[track_id]
            self._remove_aliases_for(track_id)
        if self.selected_id is not None and self.selected_id not in self._entries:
            self.selected_id = None
        return self.objects()

    def objects(self, *, confirmed_only: bool = False) -> list[LocalizedObject]:
        values = [entry.obj for entry in self._entries.values()]
        if confirmed_only:
            values = [
                obj
                for obj in values
                if obj.observation_count >= self.min_observations
                and (
                    obj.detection.confidence >= self.candidate_confidence
                    or obj.id == self.selected_id
                )
            ]
        return sorted(values, key=lambda obj: obj.id)

    def visible_objects(self, *, confirmed_only: bool = False) -> list[LocalizedObject]:
        return [obj for obj in self.objects(confirmed_only=confirmed_only) if obj.is_visible]

    def get(self, track_id: int | None) -> LocalizedObject | None:
        if track_id is None:
            return None
        entry = self._entries.get(self._canonical_id(track_id))
        return entry.obj if entry is not None else None

    def _canonical_id(self, track_id: int) -> int:
        current = track_id
        visited: set[int] = set()
        while current in self._id_aliases and current not in visited:
            visited.add(current)
            current = self._id_aliases[current]
        return current

    def _register_alias(self, source_id: int, target_id: int) -> None:
        canonical_target = self._canonical_id(target_id)
        if source_id == canonical_target:
            return
        self._id_aliases[source_id] = canonical_target
        # A fragmented ID may already have slipped into memory while both tracker
        # IDs were visible.  The original selected entry is authoritative.
        self._entries.pop(source_id, None)

    def _remove_aliases_for(self, target_id: int) -> None:
        stale = [
            source_id
            for source_id in self._id_aliases
            if source_id == target_id or self._canonical_id(source_id) == target_id
        ]
        for source_id in stale:
            self._id_aliases.pop(source_id, None)

    def _canonicalize_observations(
        self,
        observations: list[LocalizedObject],
    ) -> list[LocalizedObject]:
        result: list[LocalizedObject] = []
        positions: dict[int, int] = {}
        for observation in observations:
            canonical_id = self._canonical_id(observation.id)
            current = (
                observation
                if canonical_id == observation.id
                else replace(
                    observation,
                    detection=replace(observation.detection, track_id=canonical_id),
                )
            )
            position = positions.get(canonical_id)
            if position is None:
                positions[canonical_id] = len(result)
                result.append(current)
            elif current.detection.confidence > result[position].detection.confidence:
                result[position] = current
        return result

    def _smooth(
        self,
        previous: LocalizedObject,
        current: LocalizedObject,
        count: int,
    ) -> LocalizedObject:
        alpha = self.smoothing
        # 检测框跳到完全不同位置时相信新观测，避免 EMA 把两个物体拉在一起。
        jump = math.hypot(current.x_m - previous.x_m, current.z_m - previous.z_m)
        limit = max(0.60, 0.30 * previous.distance_m)
        if jump > limit:
            alpha = 1.0
        x_m = alpha * current.x_m + (1.0 - alpha) * previous.x_m
        y_m = alpha * current.y_m + (1.0 - alpha) * previous.y_m
        z_m = alpha * current.z_m + (1.0 - alpha) * previous.z_m
        signed, direction = direction_from_xz(x_m, z_m)
        confidence = alpha * current.detection.confidence + (1.0 - alpha) * previous.detection.confidence
        detection = replace(current.detection, confidence=confidence)
        return replace(
            current,
            detection=detection,
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            distance_m=math.hypot(x_m, z_m),
            signed_angle_deg=signed,
            direction_deg=direction,
            is_visible=True,
            last_seen_age_s=0.0,
            depth_age_s=0.0,
            observation_count=count,
            depth_mad_m=alpha * current.depth_mad_m + (1.0 - alpha) * previous.depth_mad_m,
        )

    def _rebind_selected(
        self,
        observations: list[LocalizedObject],
        now: float,
    ) -> list[LocalizedObject]:
        selected = (
            self._entries.get(self.selected_id)
            if self.selected_id is not None
            else None
        )
        if selected is None:
            return observations
        if now - selected.last_seen_s > self.selected_memory_seconds:
            return observations

        selected_observation = next(
            (obj for obj in observations if obj.id == self.selected_id),
            None,
        )
        if selected_observation is not None:
            # Trackers can briefly emit the old and new ID together.  Merge only
            # strongly overlapping duplicates; nearby objects of the same class
            # must remain independent targets.
            rebound = False
            for current in observations:
                if (
                    current.id == self.selected_id
                    or current.class_name != selected_observation.class_name
                ):
                    continue
                iou = bbox_iou(
                    selected_observation.detection.bbox,
                    current.detection.bbox,
                )
                spatial = math.hypot(
                    current.x_m - selected_observation.x_m,
                    current.z_m - selected_observation.z_m,
                )
                angle_delta = abs(
                    current.signed_angle_deg - selected_observation.signed_angle_deg
                )
                close_limit = max(
                    0.20,
                    min(0.60, selected_observation.distance_m * 0.15),
                )
                if iou >= 0.55 or (
                    iou >= 0.25 and spatial <= close_limit and angle_delta <= 6.0
                ):
                    self._register_alias(current.id, selected_observation.id)
                    rebound = True
            if rebound:
                self.reacquired_on_last_update = True
            return observations

        best: tuple[float, int] | None = None
        old = selected.obj
        for index, current in enumerate(observations):
            if current.class_name != old.class_name:
                continue
            iou = bbox_iou(old.detection.bbox, current.detection.bbox)
            spatial = math.hypot(current.x_m - old.x_m, current.z_m - old.z_m)
            angle_delta = abs(current.signed_angle_deg - old.signed_angle_deg)
            if iou < 0.12 and (
                spatial > max(0.45, old.distance_m * 0.25)
                or angle_delta > 12.0
            ):
                continue
            # Merging an ID that already owns a memory entry is riskier than
            # rebinding a brand-new tracker ID, so require a close visual/spatial
            # match before deleting that duplicate entry.
            if current.id in self._entries:
                close_limit = max(0.20, min(0.60, old.distance_m * 0.15))
                if iou < 0.55 and not (
                    iou >= 0.25 and spatial <= close_limit and angle_delta <= 6.0
                ):
                    continue
            score = iou * 2.0 - spatial * 0.5 - angle_delta * 0.02
            if best is None or score > best[0]:
                best = (score, index)
        if best is None:
            return observations

        index = best[1]
        source_id = observations[index].id
        self._register_alias(source_id, old.id)
        rebound = replace(
            observations[index],
            detection=replace(observations[index].detection, track_id=old.id),
        )
        result = list(observations)
        result[index] = rebound
        self.reacquired_on_last_update = True
        return result
