"""金融与会计舍入规则枚举与核心裁决引擎。

提供统一的 RoundingMode 枚举、舍入规则校验以及全坐标系的舍入进位裁决函数。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Union

import numpy as np


class RoundingMode(str, Enum):
    """金融与会计舍入规则枚举（继承自 str，完全兼容裸字符串与强类型枚举）。"""

    HALF_UP = "half_up"  # 四舍五入（逢五远离零，金融零售常用）
    HALF_EVEN = "half_even"  # 银行家舍入（四舍六入五成双，GAAP / IFRS 推荐，消除系统性统计偏差）
    DOWN = "down"  # 向零截断（直接舍去尾数）
    UP = "up"  # 远离零进位
    FLOOR = "floor"  # 向负无穷截断
    CEIL = "ceil"  # 向正无穷进位


_ROUNDINGS = tuple(m.value for m in RoundingMode)


def check_rounding(mode: Union[RoundingMode, str]) -> str:
    """校验并规范化舍入模式名称。"""
    if isinstance(mode, RoundingMode):
        return mode.value
    if mode not in _ROUNDINGS:
        raise ValueError(f"rounding 只能是 {_ROUNDINGS} 之一，收到 {mode!r}")
    return str(mode)


def decide_bump(mode: str, *, is_negative: Any, odd_q: Any, has_fraction: Any, over_half: Any, exact_half: Any,
                floor_base: bool = False) -> Any:
    """统一收口的舍入进位裁决函数。

    根据截断状态和半数关系，为 6 大舍入模式（half_up, half_even, down, up, floor, ceil）
    统一裁决是否需要对商基准值进位（bump + 1）。

    Parameters
    ----------
    mode : str
        舍入模式名称，必须已通过 check_rounding 校验。
    is_negative : bool or array_like of bool
        数值是否为负。
    odd_q : bool or array_like of bool
        商（基准整数）是否为奇数。
    has_fraction : bool or array_like of bool
        是否存在非零小数部分。
    over_half : bool or array_like of bool
        小数部分是否严格大于 0.5。
    exact_half : bool or array_like of bool
        小数部分是否恰好等于 0.5。
    floor_base : bool, default False
        基准坐标系：
        - False (Truncation Base): 商 q 是向零截断的绝对值 (|q| >= 0)，bump 作用于 |q|。
          返回 uint64(0) 或 uint64(1)。
        - True (Floor Base): 商 q 是向下取整的带符号数 (q = floor(m/d))，bump 作用于 q。
          返回 bool 或 bool 数组。
    """
    if floor_base:
        if mode == "half_up":
            return np.where(is_negative, over_half, over_half | exact_half)
        elif mode == "half_even":
            return np.where(over_half, True, np.where(exact_half, odd_q, False))
        elif mode == "down":
            return has_fraction & is_negative
        elif mode == "up":
            return has_fraction & np.logical_not(is_negative)
        elif mode == "floor":
            return False
        elif mode == "ceil":
            return has_fraction
        else:
            raise ValueError(f"rounding 只能是 {_ROUNDINGS} 之一，收到 {mode!r}")
    else:
        if mode == "half_up":
            return np.where(over_half | exact_half, np.uint64(1), np.uint64(0))
        elif mode == "half_even":
            return np.where(over_half, np.uint64(1), np.where(exact_half & odd_q, np.uint64(1), np.uint64(0)))
        elif mode == "down":
            return np.uint64(0)
        elif mode == "up":
            return np.where(has_fraction, np.uint64(1), np.uint64(0))
        elif mode == "floor":
            return np.where(is_negative & has_fraction, np.uint64(1), np.uint64(0))
        elif mode == "ceil":
            return np.where(np.logical_not(is_negative) & has_fraction, np.uint64(1), np.uint64(0))
        else:
            raise ValueError(f"rounding 只能是 {_ROUNDINGS} 之一，收到 {mode!r}")


def round_quotient(q: Any, r: Any, d: Any, mode: str) -> Any:
    """由向下取整的商 q 与余数 r（m = q·d + r，r 与 d 同号，|r| < |d|）按 mode 取整。

    真实值为 q + r/d，r/d ∈ [0, 1)。判断 r/d 与 1/2 的大小时比较 r 与 d - r：二者同号
    且 |d - r| ≤ |d|，不会溢出。调用方保证 q + 1 不溢出。

    Parameters
    ----------
    q : int, np.integer 或 np.ndarray
        向下取整的商（带符号数）。
    r : int, np.integer 或 np.ndarray
        余数，与 d 同号，绝对值严格小于 |d|。
    d : int, np.integer 或 np.ndarray
        除数，不能为 0。
    mode : str
        舍入模式，取值为 "half_up", "half_even", "down", "up", "floor", "ceil" 之一。

    Returns
    -------
    int, np.integer 或 np.ndarray
        舍入后的商。
    """
    if mode == "floor":
        return q
    has_fraction = (r != 0)
    is_negative = (q < 0)
    if mode in ("half_even", "half_up"):
        over_half = np.where(d > 0, r > d - r, r < d - r)
        exact_half = (r == d - r)
        odd_q = (q % 2 != 0)
    else:
        over_half = exact_half = odd_q = False
    bump = decide_bump(
        mode,
        is_negative=is_negative,
        odd_q=odd_q,
        has_fraction=has_fraction,
        over_half=over_half,
        exact_half=exact_half,
        floor_base=True,
    )
    if isinstance(q, int) and np.ndim(bump) == 0:
        return int(q) + int(bump)
    return q + bump


__all__ = ["RoundingMode", "_ROUNDINGS", "check_rounding", "decide_bump", "round_quotient"]
