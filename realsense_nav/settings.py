from __future__ import annotations

import os
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
    """按环境变量、本地 Python 配置、默认值的顺序读取字符串设置。"""
    environment_value = os.getenv(name)
    if environment_value is not None and environment_value.strip():
        return environment_value.strip()
    module = _local_config()
    if module is not None:
        local_value = getattr(module, name, None)
        if local_value is not None and str(local_value).strip():
            return str(local_value).strip()
    return default


def has_setting(name: str) -> bool:
    return bool(get_setting(name))

