from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import ModuleType


@lru_cache(maxsize=1)
def _local_config() -> ModuleType | None:
    try:
        return import_module("local_config")
    except ModuleNotFoundError as exc:
        if exc.name != "local_config":
            raise
        return None


def get_setting(name: str, default: str = "") -> str:
    """从本地 Python 配置读取字符串设置，未配置时返回默认值。"""
    module = _local_config()
    if module is not None:
        local_value = getattr(module, name, None)
        if local_value is not None and str(local_value).strip():
            return str(local_value).strip()
    return default


def has_setting(name: str) -> bool:
    return bool(get_setting(name))
