import json
from realsense_nav.models import Detection, LocalizedObject
from types import SimpleNamespace

import pytest

from realsense_nav.selector import DeepSeekSelector, LocalRuleSelector, build_selector


def obj(track_id: int, class_name: str, angle: float, distance: float) -> LocalizedObject:
    return LocalizedObject(
        detection=Detection(track_id, class_name, 0.9, (0, 0, 10, 10)),
        x_m=0.0,
        y_m=0.0,
        z_m=distance,
        distance_m=distance,
        signed_angle_deg=angle,
        direction_deg=90.0 + angle,
        depth_sample_count=100,
    )


def test_chinese_right_chair() -> None:
    objects = [obj(0, "chair", -20, 2.0), obj(3, "chair", 18, 2.5), obj(4, "bottle", 25, 1.0)]
    result = LocalRuleSelector().select("前往右边的椅子", objects)
    assert result.status == "matched"
    assert result.target_id == 3


def test_nearest_bottle() -> None:
    objects = [obj(1, "bottle", -10, 3.0), obj(2, "bottle", 12, 1.2)]
    result = LocalRuleSelector().select("去最近的瓶子", objects)
    assert result.target_id == 2


def test_ambiguous_same_class() -> None:
    objects = [obj(1, "chair", -10, 2.0), obj(2, "chair", 12, 2.2)]
    result = LocalRuleSelector().select("去椅子那里", objects)
    assert result.status == "ambiguous"
    assert set(result.candidate_ids) == {1, 2}


def test_direct_id_is_validated() -> None:
    objects = [obj(5, "chair", 0, 2.0)]
    assert LocalRuleSelector().select("去编号5", objects).target_id == 5
    assert LocalRuleSelector().select("去编号9", objects).status == "not_found"


def test_relation_uses_last_object_noun() -> None:
    objects = [obj(1, "person", -5, 2.0), obj(2, "chair", 5, 2.1)]
    result = LocalRuleSelector().select("去人旁边的椅子", objects)
    assert result.target_id == 2


class FakeResponses:
    def __init__(self, contents: list[str | None]) -> None:
        self.contents = contents
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(output_text=content)


def deepseek_with_fake(contents: list[str | None]) -> tuple[DeepSeekSelector, FakeResponses]:
    responses = FakeResponses(contents)
    selector = object.__new__(DeepSeekSelector)
    selector.model = "deepseek-v4-flash"
    selector.client = SimpleNamespace(responses=responses)
    return selector, responses


def test_deepseek_responses_selection_and_request_shape() -> None:
    selector, responses = deepseek_with_fake(
        ['{"status":"matched","target_id":2,"candidate_ids":[],"reason":"最近的瓶子"}']
    )
    objects = [obj(1, "bottle", -10, 3.0), obj(2, "bottle", 12, 1.2)]
    result = selector.select("去最近的瓶子", objects)
    assert result.status == "matched"
    assert result.target_id == 2
    assert result.source == "deepseek"
    call = responses.calls[0]
    assert call["model"] == "deepseek-v4-flash"
    assert "只输出一个 JSON" in str(call["instructions"])
    payload = json.loads(str(call["input"]))
    assert payload["command"] == "去最近的瓶子"
    assert {item["id"] for item in payload["objects"]} == {1, 2}
    assert "messages" not in call


def test_deepseek_retries_empty_response_once() -> None:
    selector, responses = deepseek_with_fake(
        [None, '{"status":"not_found","target_id":null,"candidate_ids":[],"reason":"没有箱子"}']
    )
    result = selector.select("去箱子那里", [obj(1, "chair", 0, 2.0)])
    assert result.status == "not_found"
    assert len(responses.calls) == 2


def test_deepseek_rejects_hallucinated_id() -> None:
    selector, _ = deepseek_with_fake(
        ['{"status":"matched","target_id":99,"candidate_ids":[],"reason":"匹配"}']
    )
    result = selector.select("去椅子", [obj(1, "chair", 0, 2.0)])
    assert result.status == "not_found"


def test_unknown_selector_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="未知选择器模式"):
        build_selector("unsupported")
