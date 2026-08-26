from realsense_nav.app import _refresh_selection_for_output, build_parser
from realsense_nav.memory import TargetMemory
from realsense_nav.models import Detection, LocalizedObject, SelectionResult


def obj(track_id: int, distance_m: float) -> LocalizedObject:
    return LocalizedObject(
        detection=Detection(track_id, "chair", 0.8, (10, 10, 50, 50)),
        x_m=0.0,
        y_m=0.0,
        z_m=distance_m,
        distance_m=distance_m,
        signed_angle_deg=0.0,
        direction_deg=90.0,
        depth_sample_count=100,
    )


def test_tracking_and_candidate_thresholds_are_separate() -> None:
    args = build_parser().parse_args([])
    assert args.confidence < args.candidate_confidence
    assert args.command_wait_frames > args.min_observations
    assert args.view_mode == "split"
    assert args.depth_color_mode == "official"


def test_cli_exposes_only_supported_selector_modes() -> None:
    parser = build_parser()
    assert parser.parse_args([]).selector == "local"
    selector_action = next(action for action in parser._actions if action.dest == "selector")
    assert set(selector_action.choices or ()) == {"auto", "local", "deepseek"}


def test_selection_output_uses_latest_localization() -> None:
    memory = TargetMemory(smoothing=1.0, min_observations=1)
    snapshot = memory.update([obj(3, 2.0)], now_s=1.0)
    memory.update([obj(3, 1.2)], now_s=2.0)
    result = SelectionResult(status="matched", target_id=3, reason="匹配")

    refreshed, output_objects = _refresh_selection_for_output(result, snapshot, memory)

    assert refreshed.target_id == 3
    assert output_objects[0].distance_m == 1.2


def test_selection_output_rejects_expired_target() -> None:
    memory = TargetMemory(memory_seconds=1.0, selected_memory_seconds=2.0, min_observations=1)
    snapshot = memory.update([obj(3, 2.0)], now_s=1.0)
    memory.update([], now_s=3.0)
    result = SelectionResult(status="matched", target_id=3, reason="匹配")

    refreshed, output_objects = _refresh_selection_for_output(result, snapshot, memory)

    assert refreshed.status == "not_found"
    assert output_objects == []
