"""Moneta 物理单位与量纲工具模块 (_units)。

定义资产数量与标价中使用的非货币物理单位规范、通配符常量以及单位兼容性判定。
"""

from __future__ import annotations

from typing import Any

WILDCARD_UNIT: str = "Any"  # 通配单位：可与任意标的资产的数量单位兼容匹配


def check_unit(unit: Any) -> str:
    """单位必须是非空 str，去掉首尾空白后返回。"""
    if not isinstance(unit, str):
        raise TypeError(f"单位必须是 str，收到 {type(unit).__name__}")
    unit = unit.strip()
    if not unit:
        raise ValueError("单位不能为空")
    return unit


def units_compatible(u1: str, u2: str) -> bool:
    """两个单位是否兼容：相同，或至少一方为通配单位 'Any'。"""
    return u1 == u2 or u1 == WILDCARD_UNIT or u2 == WILDCARD_UNIT


__all__ = ["WILDCARD_UNIT", "check_unit", "units_compatible"]
