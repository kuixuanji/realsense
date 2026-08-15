from realsense_nav.memory import TargetMemory
from realsense_nav.models import Detection, LocalizedObject


def obj(track_id: int, *, x: float = 0.0, z: float = 2.0, bbox=(10, 10, 50, 50)) -> LocalizedObject:
    return LocalizedObject(
        detection=Detection(track_id, "chair", 0.8, bbox),
        x_m=x,
        y_m=0.0,
        z_m=z,
        distance_m=(x * x + z * z) ** 0.5,
        signed_angle_deg=0.0,
        direction_deg=90.0,
        depth_sample_count=100,
    )


def test_memory_keeps_short_missing_detection() -> None:
    memory = TargetMemory(memory_seconds=4.0, selected_memory_seconds=10.0, min_observations=1)
    memory.update([obj(2)], now_s=1.0)
    remembered = memory.update([], now_s=3.0)
    assert len(remembered) == 1
    assert not remembered[0].is_visible
    assert remembered[0].last_seen_age_s == 2.0


def test_normal_target_expires_but_selected_target_lives_longer() -> None:
    memory = TargetMemory(memory_seconds=2.0, selected_memory_seconds=8.0, min_observations=1)
    memory.update([obj(2), obj(3, x=0.5)], now_s=1.0)
    memory.select(2)
    remaining = memory.update([], now_s=4.0)
    assert [item.id for item in remaining] == [2]
    assert memory.selected_id == 2


def test_position_is_smoothed_across_frames() -> None:
    memory = TargetMemory(smoothing=0.25, min_observations=1)
    memory.update([obj(2, x=0.0)], now_s=1.0)
    current = memory.update([obj(2, x=0.4)], now_s=2.0)[0]
    assert current.x_m == 0.1
    assert current.observation_count == 2


def test_unconfirmed_target_is_hidden_from_candidates() -> None:
    memory = TargetMemory(min_observations=2)
    memory.update([obj(2)], now_s=1.0)
    assert memory.objects(confirmed_only=True) == []
    memory.update([obj(2)], now_s=2.0)
    assert [item.id for item in memory.objects(confirmed_only=True)] == [2]


def test_low_confidence_track_is_remembered_but_not_a_candidate() -> None:
    memory = TargetMemory(min_observations=1, candidate_confidence=0.5)
    low = obj(2)
    low = LocalizedObject(
        detection=Detection(2, "chair", 0.2, low.detection.bbox),
        x_m=low.x_m,
        y_m=low.y_m,
        z_m=low.z_m,
        distance_m=low.distance_m,
        signed_angle_deg=low.signed_angle_deg,
        direction_deg=low.direction_deg,
        depth_sample_count=low.depth_sample_count,
    )
    assert memory.update([low], now_s=1.0)
    assert memory.objects(confirmed_only=True) == []


def test_selected_target_rebinds_new_tracker_id() -> None:
    memory = TargetMemory(min_observations=1)
    memory.update([obj(2)], now_s=1.0)
    memory.select(2)
    new_detection = Detection(99, "chair", 0.8, (12, 10, 52, 50))
    current = memory.update(
        [obj(99, x=0.05, bbox=new_detection.bbox)],
        detections=[new_detection],
        now_s=2.0,
    )
    assert [item.id for item in current] == [2]
    assert memory.reacquired_on_last_update
    assert [item.track_id for item in memory.canonicalize_detections([new_detection])] == [2]


def test_rebound_alias_survives_a_depth_hole() -> None:
    memory = TargetMemory(min_observations=1)
    memory.update([obj(2)], now_s=1.0)
    memory.select(2)
    new_detection = Detection(99, "chair", 0.75, (12, 10, 52, 50))
    memory.update(
        [obj(99, x=0.05, bbox=new_detection.bbox)],
        detections=[new_detection],
        now_s=2.0,
    )

    current = memory.update([], detections=[new_detection], now_s=3.0)

    assert [item.id for item in current] == [2]
    assert current[0].is_visible
    assert current[0].depth_age_s == 1.0
    assert current[0].detection.track_id == 2
    assert memory.reacquired_on_last_update is False


def test_old_and_rebound_ids_in_same_frame_are_merged() -> None:
    memory = TargetMemory(min_observations=1)
    memory.update([obj(2)], now_s=1.0)
    memory.select(2)
    old_detection = Detection(2, "chair", 0.70, (10, 10, 50, 50))
    new_detection = Detection(99, "chair", 0.85, (12, 10, 52, 50))

    current = memory.update(
        [obj(2), obj(99, x=0.05, bbox=new_detection.bbox)],
        detections=[old_detection, new_detection],
        now_s=2.0,
    )
    display = memory.canonicalize_detections([old_detection, new_detection])

    assert [item.id for item in current] == [2]
    assert [item.track_id for item in display] == [2]
    assert display[0].confidence == 0.85
    assert memory.reacquired_on_last_update


def test_rebind_deletes_new_id_already_present_in_memory() -> None:
    memory = TargetMemory(min_observations=1)
    memory.update(
        [obj(2), obj(99, x=1.0, bbox=(100, 10, 140, 50))],
        now_s=1.0,
    )
    memory.select(2)
    assert [item.id for item in memory.objects()] == [2, 99]

    current = memory.update(
        [obj(99, x=0.05, bbox=(12, 10, 52, 50))],
        now_s=2.0,
    )

    assert [item.id for item in current] == [2]
    assert memory.reacquired_on_last_update


def test_nearby_same_class_target_is_not_merged_without_overlap() -> None:
    memory = TargetMemory(min_observations=1)
    memory.update(
        [obj(2), obj(3, x=0.4, bbox=(70, 10, 110, 50))],
        now_s=1.0,
    )
    memory.select(2)

    current = memory.update(
        [obj(2), obj(3, x=0.4, bbox=(70, 10, 110, 50))],
        now_s=2.0,
    )

    assert [item.id for item in current] == [2, 3]


def test_rebound_alias_is_removed_when_canonical_target_expires() -> None:
    memory = TargetMemory(
        memory_seconds=2.0,
        selected_memory_seconds=5.0,
        min_observations=1,
    )
    memory.update([obj(2)], now_s=1.0)
    memory.select(2)
    new_detection = Detection(99, "chair", 0.8, (12, 10, 52, 50))
    memory.update([obj(99, bbox=new_detection.bbox)], now_s=2.0)

    assert memory.update([], now_s=8.0) == []
    assert memory.selected_id is None
    assert memory.canonicalize_detections([new_detection])[0].track_id == 99


def test_visual_detection_keeps_target_visible_when_depth_temporarily_fails() -> None:
    memory = TargetMemory(min_observations=1)
    memory.update([obj(2)], now_s=1.0)
    detection = Detection(2, "chair", 0.7, (12, 10, 52, 50))
    current = memory.update([], detections=[detection], now_s=2.5)[0]
    assert current.is_visible
    assert current.depth_age_s == 1.5
    assert current.detection.bbox == detection.bbox
