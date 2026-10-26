"""Moneta 内部通用技术工具模块 (_utils)。

提供纯数学计算、日期解析、调用栈追踪与日历判定等无状态工具函数。
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from typing import Any, Literal, Optional, overload

import numpy as np


def max_abs(x: np.ndarray) -> int:
    """计算整数数组绝对值的最大值（Python int，-2**63 也不溢出）；空数组安全返回 0。"""
    return max(-int(x.min()), int(x.max())) if x.size else 0


@overload
def to_date(value: Any, what: str = ..., optional: Literal[False] = ...) -> date: ...


@overload
def to_date(value: Any, what: str = ..., optional: bool = ...) -> Optional[date]: ...


def to_date(value: Any, what: str = "日期", optional: bool = False) -> Optional[date]:
    """将 date、datetime 或 ISO 格式日期字符串解析为标准 date 对象。

    optional=True 时 None 返回 None；optional=False 时 None 抛出 TypeError。
    """
    if value is None and optional:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            raise ValueError(f"{what} 不是合法的 ISO 日期: {value!r}") from None
    raise TypeError(f"{what} 必须是 date、datetime 或 ISO 字符串，收到 {type(value).__name__}")


def is_leap_year(year: int) -> bool:
    """判断年份是否为公历闰年。"""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def user_stacklevel() -> int:
    """告警指向的栈层级：跳过 moneta 包内部的调用帧，精准指向用户业务代码。"""
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_globals.get("__name__", "").split(".")[0] == "moneta":
        frame, level = frame.f_back, level + 1
    return level


__all__ = ["is_leap_year", "max_abs", "to_date", "user_stacklevel"]
