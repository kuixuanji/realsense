from realsense_nav.models import Detection
from realsense_nav.tracking import IouTracker, bbox_iou


def test_iou() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_tracker_preserves_id() -> None:
    tracker = IouTracker()
    first = tracker.update([Detection(-1, "chair", 0.9, (10, 10, 50, 50))])[0]
    second = tracker.update([Detection(-1, "chair", 0.8, (12, 11, 52, 51))])[0]
    assert first.track_id == second.track_id


def test_tracker_does_not_cross_classes() -> None:
    tracker = IouTracker()
    chair = tracker.update([Detection(-1, "chair", 0.9, (10, 10, 50, 50))])[0]
    bottle = tracker.update([Detection(-1, "bottle", 0.9, (10, 10, 50, 50))])[0]
    assert chair.track_id != bottle.track_id

