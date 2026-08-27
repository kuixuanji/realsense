from realsense_nav.app import build_parser


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
