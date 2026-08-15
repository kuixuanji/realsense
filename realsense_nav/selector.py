from __future__ import annotations

import json
import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .models import LocalizedObject, SelectionResult
from .settings import get_setting, has_setting


class TargetSelector(Protocol):
    def select(self, command: str, objects: list[LocalizedObject]) -> SelectionResult: ...


CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "person": ("person", "people", "human", "人", "行人", "那个人"),
    "bicycle": ("bicycle", "bike", "自行车", "单车"),
    "car": ("car", "automobile", "汽车", "小汽车", "轿车"),
    "motorcycle": ("motorcycle", "motorbike", "摩托车"),
    "airplane": ("airplane", "plane", "飞机"),
    "bus": ("bus", "公交车", "巴士"),
    "train": ("train", "火车", "列车"),
    "truck": ("truck", "卡车", "货车"),
    "boat": ("boat", "ship", "船"),
    "traffic light": ("traffic light", "红绿灯", "交通灯"),
    "fire hydrant": ("fire hydrant", "消防栓", "消火栓"),
    "stop sign": ("stop sign", "停止标志", "停车标志"),
    "parking meter": ("parking meter", "停车计时器"),
    "bench": ("bench", "长椅", "长凳"),
    "bird": ("bird", "鸟"),
    "cat": ("cat", "猫"),
    "dog": ("dog", "狗", "犬"),
    "horse": ("horse", "马"),
    "sheep": ("sheep", "羊"),
    "cow": ("cow", "牛"),
    "elephant": ("elephant", "大象"),
    "bear": ("bear", "熊"),
    "zebra": ("zebra", "斑马"),
    "giraffe": ("giraffe", "长颈鹿"),
    "backpack": ("backpack", "背包", "书包"),
    "umbrella": ("umbrella", "伞", "雨伞"),
    "handbag": ("handbag", "手提包", "包包"),
    "tie": ("tie", "领带"),
    "suitcase": ("suitcase", "行李箱", "箱子"),
    "frisbee": ("frisbee", "飞盘"),
    "skis": ("skis", "滑雪板"),
    "snowboard": ("snowboard", "单板滑雪板"),
    "sports ball": ("sports ball", "ball", "球"),
    "kite": ("kite", "风筝"),
    "baseball bat": ("baseball bat", "棒球棒"),
    "baseball glove": ("baseball glove", "棒球手套"),
    "skateboard": ("skateboard", "滑板"),
    "surfboard": ("surfboard", "冲浪板"),
    "tennis racket": ("tennis racket", "网球拍"),
    "bottle": ("bottle", "瓶子", "水瓶", "瓶"),
    "wine glass": ("wine glass", "酒杯", "高脚杯"),
    "cup": ("cup", "杯子", "水杯", "杯"),
    "fork": ("fork", "叉子", "餐叉"),
    "knife": ("knife", "刀", "餐刀"),
    "spoon": ("spoon", "勺子", "汤匙"),
    "bowl": ("bowl", "碗"),
    "banana": ("banana", "香蕉"),
    "apple": ("apple", "苹果"),
    "sandwich": ("sandwich", "三明治"),
    "orange": ("orange", "橙子", "橘子"),
    "broccoli": ("broccoli", "西兰花"),
    "carrot": ("carrot", "胡萝卜"),
    "hot dog": ("hot dog", "热狗"),
    "pizza": ("pizza", "披萨", "比萨"),
    "donut": ("donut", "甜甜圈"),
    "cake": ("cake", "蛋糕"),
    "chair": ("chair", "椅子", "座椅", "椅"),
    "couch": ("couch", "sofa", "沙发"),
    "potted plant": ("potted plant", "plant", "盆栽", "植物"),
    "bed": ("bed", "床"),
    "dining table": ("dining table", "table", "餐桌", "桌子", "桌"),
    "toilet": ("toilet", "马桶", "坐便器"),
    "tv": ("tv", "television", "电视", "显示器"),
    "laptop": ("laptop", "笔记本电脑", "笔记本"),
    "mouse": ("mouse", "鼠标"),
    "remote": ("remote", "遥控器"),
    "keyboard": ("keyboard", "键盘"),
    "cell phone": ("cell phone", "phone", "手机", "电话"),
    "microwave": ("microwave", "微波炉"),
    "oven": ("oven", "烤箱"),
    "toaster": ("toaster", "烤面包机"),
    "sink": ("sink", "水槽", "洗手池"),
    "refrigerator": ("refrigerator", "fridge", "冰箱"),
    "book": ("book", "书", "书本"),
    "clock": ("clock", "钟", "时钟"),
    "vase": ("vase", "花瓶"),
    "scissors": ("scissors", "剪刀"),
    "teddy bear": ("teddy bear", "泰迪熊", "玩具熊"),
    "hair drier": ("hair drier", "hair dryer", "吹风机"),
    "toothbrush": ("toothbrush", "牙刷"),
}


def _contains(command: str, tokens: tuple[str, ...]) -> bool:
    return any(token.casefold() in command for token in tokens)


class LocalRuleSelector:
    """离线解析常见中文/英文物体名和方位限定。"""

    def select(self, command: str, objects: list[LocalizedObject]) -> SelectionResult:
        normalized = command.strip().casefold()
        if not normalized or not objects:
            return SelectionResult(status="not_found", reason="命令为空或当前没有可定位目标")

        ids = {obj.id for obj in objects}
        direct = re.search(r"(?:#|编号\s*|id\s*)(\d+)", normalized)
        if direct is None:
            direct = re.search(r"(\d+)\s*号", normalized)
        if direct is not None:
            target_id = int(direct.group(1))
            if target_id in ids:
                return SelectionResult(status="matched", target_id=target_id, reason="按目标编号匹配")
            return SelectionResult(status="not_found", reason=f"当前画面没有编号 {target_id}")

        class_positions: dict[str, int] = {}
        present_classes = {obj.class_name for obj in objects}
        for class_name in present_classes:
            aliases = CLASS_ALIASES.get(class_name, (class_name,))
            positions = [normalized.rfind(alias.casefold()) for alias in aliases]
            best = max(positions, default=-1)
            if best >= 0:
                class_positions[class_name] = best

        if class_positions:
            # 中文关系表达中，目标通常是最后出现的名词，例如“人旁边的椅子”。
            target_class = max(class_positions, key=class_positions.get)
            candidates = [obj for obj in objects if obj.class_name == target_class]
        elif _contains(normalized, ("物体", "东西", "object", "thing", "目标")):
            candidates = list(objects)
        else:
            return SelectionResult(status="not_found", reason="命令中的物体类别未出现在当前候选列表")

        # 有同类当前可见目标时，不让旧记忆无故制造歧义；按编号仍可直接选择记忆目标。
        visible_candidates = [obj for obj in candidates if obj.is_visible]
        if visible_candidates:
            candidates = visible_candidates

        wants_left = _contains(normalized, ("最左", "左边", "左侧", "左前", "left"))
        wants_right = _contains(normalized, ("最右", "右边", "右侧", "右前", "right"))
        wants_center = _contains(normalized, ("中间", "中央", "正前", "center", "centre", "front"))
        wants_nearest = _contains(normalized, ("最近", "近一点", "nearest", "closest"))
        wants_farthest = _contains(normalized, ("最远", "farther", "farthest"))

        if wants_left and not wants_right:
            left_side = [obj for obj in candidates if obj.signed_angle_deg < -2.0]
            candidates = left_side or candidates
        elif wants_right and not wants_left:
            right_side = [obj for obj in candidates if obj.signed_angle_deg > 2.0]
            candidates = right_side or candidates
        elif wants_center:
            return self._matched(min(candidates, key=lambda obj: abs(obj.signed_angle_deg)), "按画面中央匹配")

        if wants_nearest:
            return self._matched(min(candidates, key=lambda obj: obj.distance_m), "按最近距离匹配")
        if wants_farthest:
            return self._matched(max(candidates, key=lambda obj: obj.distance_m), "按最远距离匹配")
        if wants_left and not wants_right:
            return self._matched(min(candidates, key=lambda obj: obj.signed_angle_deg), "按左侧方位匹配")
        if wants_right and not wants_left:
            return self._matched(max(candidates, key=lambda obj: obj.signed_angle_deg), "按右侧方位匹配")

        ordinal = self._parse_ordinal(normalized)
        if ordinal is not None:
            ordered = sorted(candidates, key=lambda obj: obj.signed_angle_deg)
            if 0 <= ordinal < len(ordered):
                return self._matched(ordered[ordinal], "按从左到右的序号匹配")
            return SelectionResult(status="not_found", reason="指定序号超出候选数量")

        if len(candidates) == 1:
            return self._matched(candidates[0], "类别唯一匹配")
        return SelectionResult(
            status="ambiguous",
            candidate_ids=tuple(obj.id for obj in candidates),
            reason="有多个同类目标，请补充左右、远近或编号",
        )

    @staticmethod
    def _matched(obj: LocalizedObject, reason: str) -> SelectionResult:
        return SelectionResult(status="matched", target_id=obj.id, reason=reason)

    @staticmethod
    def _parse_ordinal(command: str) -> int | None:
        chinese = {"一": 0, "二": 1, "两": 1, "三": 2, "四": 3, "五": 4, "六": 5, "七": 6, "八": 7, "九": 8}
        match = re.search(r"第\s*([一二两三四五六七八九]|\d+)\s*(?:个|把|张|只|台|辆|件)?", command)
        if match:
            token = match.group(1)
            return chinese[token] if token in chinese else int(token) - 1
        match = re.search(r"(?:first|second|third|fourth|fifth)", command)
        if match:
            return {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}[match.group(0)]
        return None


class _LLMSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["matched", "ambiguous", "not_found"]
    target_id: int | None = None
    candidate_ids: list[int] = Field(default_factory=list)
    reason: str = ""


class DeepSeekSelector:
    """通过 DeepSeek 的 OpenAI 兼容 Responses API 选择目标。"""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_s: float = 15.0,
    ) -> None:
        api_key = get_setting("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY")
        try:
            from openai import DefaultHttpxClient, OpenAI as DeepSeekClient
        except ImportError as exc:
            raise RuntimeError("缺少 openai Python SDK") from exc
        self.model = model
        self.client = DeepSeekClient(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            max_retries=1,
            http_client=DefaultHttpxClient(trust_env=False),
        )

    def select(self, command: str, objects: list[LocalizedObject]) -> SelectionResult:
        if not objects:
            return SelectionResult(status="not_found", reason="当前没有可定位目标", source="deepseek")
        payload = {
            "command": command,
            "objects": [obj.as_selector_dict() for obj in objects],
        }
        system_prompt = (
            "你只负责从候选物体中解析用户要前往的目标。理解中英文类别、左右、远近、编号和简单空间关系。"
            "不能创造不存在的编号；证据不足时返回 ambiguous；画面没有目标时返回 not_found。"
            "目标通常是关系描述中最后的中心名词，例如‘人旁边的椅子’的目标是椅子。"
            "visible=false 表示短暂漏检的记忆目标；除非用户明确说编号，否则优先 visible=true。"
            "你必须只输出一个 JSON 对象，不要 Markdown。"
            "JSON 格式示例："
            '{"status":"matched","target_id":3,"candidate_ids":[],"reason":"右侧椅子"}。'
            "status 只能是 matched、ambiguous 或 not_found。"
        )
        last_error: Exception | None = None
        # 网络正常但模型返回空文本或非法 JSON 时补试一次。
        for _ in range(2):
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=json.dumps(payload, ensure_ascii=False),
            )
            content = response.output_text
            if not content or not content.strip():
                last_error = ValueError("DeepSeek 返回空内容")
                continue
            try:
                parsed = _LLMSelection.model_validate_json(content)
            except Exception as exc:
                last_error = exc
                continue
            return _validated_model_selection(parsed, objects, source="deepseek")
        raise RuntimeError("DeepSeek 未返回符合约定的 JSON") from last_error


def _validated_model_selection(
    parsed: _LLMSelection,
    objects: list[LocalizedObject],
    *,
    source: str,
) -> SelectionResult:
    valid_ids = {obj.id for obj in objects}
    if parsed.status == "matched":
        if parsed.target_id not in valid_ids:
            return SelectionResult(status="not_found", reason="模型返回了不存在的目标编号", source=source)
        return SelectionResult(
            status="matched",
            target_id=parsed.target_id,
            reason=parsed.reason,
            source=source,
        )
    if parsed.status == "ambiguous":
        candidates = tuple(item for item in parsed.candidate_ids if item in valid_ids)
        if not candidates:
            candidates = tuple(sorted(valid_ids))
        return SelectionResult(
            status="ambiguous",
            candidate_ids=candidates,
            reason=parsed.reason,
            source=source,
        )
    return SelectionResult(status="not_found", reason=parsed.reason, source=source)


class FallbackSelector:
    def __init__(self, primary: TargetSelector, fallback: TargetSelector, provider_name: str) -> None:
        self.primary = primary
        self.fallback = fallback
        self.provider_name = provider_name

    def select(self, command: str, objects: list[LocalizedObject]) -> SelectionResult:
        try:
            return self.primary.select(command, objects)
        except Exception as exc:  # API、网络或模型权限错误均安全回退
            result = self.fallback.select(command, objects)
            return SelectionResult(
                status=result.status,
                target_id=result.target_id,
                candidate_ids=result.candidate_ids,
                reason=f"{self.provider_name} 选择失败（{type(exc).__name__}），已回退本地解析；{result.reason}",
                source="local-fallback",
            )


def build_selector(
    mode: str,
    *,
    deepseek_model: str = "deepseek-v4-flash",
    deepseek_base_url: str = "https://api.deepseek.com",
) -> TargetSelector:
    if mode not in {"auto", "local", "deepseek"}:
        raise ValueError(f"未知选择器模式：{mode}")
    local = LocalRuleSelector()
    if mode == "local":
        return local
    if mode == "deepseek":
        return DeepSeekSelector(model=deepseek_model, base_url=deepseek_base_url)
    if has_setting("DEEPSEEK_API_KEY"):
        return FallbackSelector(
            DeepSeekSelector(model=deepseek_model, base_url=deepseek_base_url),
            local,
            "DeepSeek",
        )
    return local
