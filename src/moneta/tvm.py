"""Moneta 货币时间价值 (Time Value of Money)。

包含现值 (pv)、终值 (fv)、净现值 (npv) 与贴现因子 (discount_factors)。

约定：
- 利率与期数按 float 计算增长因子：annual 为 (1 + r)^t，continuous 为 e^(r·t)，simple 为 1 + r·t；
- 作用到金额上时，因子按其中绝对值最大者保留 18 位有效数字转成定点数，再与金额精确相乘、只舍入一次；
- npv 各笔现值以币种精度 + 4 位累加，最后只舍入一次。
"""

from __future__ import annotations

import math
import numbers
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable, Union

import numpy as np

from ._utils import to_date
from .cashflow import CashFlow
from .conventions import DayCountConvention, _parse_convention, year_fraction
from .fixed_array import FixedArray
from .money import Money

_COMPOUNDINGS = ("annual", "continuous", "simple")
_NPV_GUARD_DIGITS = 4


def _real_array(value: Any, name: str) -> np.ndarray:
    """利率、期数 → float64 数组（标量为 0 维）：接受实数、数值字符串、FixedArray 或实数数组，必须有限。"""
    if isinstance(value, FixedArray):
        if value.mask is not None and value.mask.any():
            raise ValueError(f"{name} 不能包含缺失值")
        arr = np.asarray(value.to_float(), dtype=np.float64)
    elif isinstance(value, str):
        try:
            arr = np.asarray(float(value))
        except ValueError:
            raise ValueError(f"{name} 不是合法的数值: {value!r}") from None
    elif isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} 必须是实数，收到 {type(value).__name__}")
    elif isinstance(value, (numbers.Real, Decimal)):
        arr = np.asarray(float(value))
    else:
        arr = np.asarray(value)
        if arr.dtype.kind not in "iuf":
            raise TypeError(f"{name} 必须是实数或实数数组，收到 {type(value).__name__} (dtype={arr.dtype})")
        arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 必须是有限数")
    return arr


def _check_compounding(compounding: Any) -> str:
    if not isinstance(compounding, str) or compounding not in _COMPOUNDINGS:
        raise ValueError(f"不支持的复利方式 {compounding!r}：可选 {list(_COMPOUNDINGS)}")
    return compounding


def _growth(r: np.ndarray, t: np.ndarray, compounding: str) -> np.ndarray:
    """增长因子：annual (1 + r)^t，continuous e^(r·t)，simple 1 + r·t；结果必须是有限正数。"""
    with np.errstate(over="ignore", under="ignore"):
        if compounding == "annual":
            if np.any(r <= -1):
                raise ValueError("annual 复利要求 rate > -1")
            g = np.power(1.0 + r, t)
        elif compounding == "continuous":
            g = np.exp(r * t)
        else:
            g = 1.0 + r * t
            if np.any(g <= 0):
                raise ValueError("simple 计息要求 1 + rate × periods > 0")
    if not np.all(np.isfinite(g) & (g > 0)):
        raise ValueError("增长因子超出浮点数范围（利率或期数过大）")
    return g


def _factor(f: np.ndarray) -> FixedArray:
    """浮点因子 → FixedArray：按绝对值最大者保留 18 位有效数字（int64 尾数放得下），较小的因子绝对误差不超过它。"""
    peak = float(np.max(np.abs(f))) if f.size else 0.0
    if peak >= 1e18:
        raise OverflowError(f"增长因子 {peak:.6g} 超出定点数范围（利率或期数过大）")
    scale = max(0, 18 - (math.floor(math.log10(peak)) + 1)) if peak > 0 else 0
    return FixedArray(f, scale=scale)


def pv(future_value: Money, rate: Any, periods: Any, *, compounding: str = "annual") -> Money:
    """现值：future_value / 增长因子，舍入到币种精度（half_up）。

    Parameters
    ----------
    future_value : Money
        未来金额，可以是数组。
    rate : 实数、数值字符串、FixedArray 或实数数组
        每期利率（如 0.05 表示 5%）。
    periods : 实数或实数数组
        期数，可为小数或负数；与 future_value、rate 按广播规则对齐。
    compounding : {"annual", "continuous", "simple"}, optional
        复利方式，默认 "annual"（按期复利）。
    """
    if not isinstance(future_value, Money):
        raise TypeError(f"future_value 必须是 Money，收到 {type(future_value).__name__}")
    g = _growth(_real_array(rate, "rate"), _real_array(periods, "periods"), _check_compounding(compounding))
    return future_value.multiply(_factor(1.0 / g))


def fv(present_value: Money, rate: Any, periods: Any, *, compounding: str = "annual") -> Money:
    """终值：present_value × 增长因子，舍入到币种精度（half_up）；参数含义同 pv。"""
    if not isinstance(present_value, Money):
        raise TypeError(f"present_value 必须是 Money，收到 {type(present_value).__name__}")
    g = _growth(_real_array(rate, "rate"), _real_array(periods, "periods"), _check_compounding(compounding))
    return present_value.multiply(_factor(g))


def npv(rate: Any, cash_flows: Iterable[Union[Money, CashFlow]],
        as_of_date: Union[str, date, datetime, None] = None, *,
        convention: Union[DayCountConvention, str] = DayCountConvention.ACT_365,
        compounding: str = "annual") -> Money:
    """净现值：各笔现金流现值之和，最后只舍入一次到币种精度（half_up）。

    Parameters
    ----------
    rate : 实数、数值字符串、FixedArray 或实数数组
        每期（给出 as_of_date 时为每年）利率。
    cash_flows : Money 或 CashFlow 的序列
        - 不给 as_of_date 时元素必须是 Money，第 i 笔（从 0 开始）贴现 i 期；
        - 给出 as_of_date 时元素必须是 CashFlow，按 convention 计算 as_of_date 至 pay_date 的年数贴现。
        所有现金流须为同一币种。
    as_of_date : date、datetime 或 ISO 字符串, optional
        估值日。
    convention : DayCountConvention 或 str, optional
        计日惯例，默认 ACT/365 Fixed；仅在给出 as_of_date 时使用。
    compounding : {"annual", "continuous", "simple"}, optional
        复利方式，默认 "annual"。
    """
    if isinstance(cash_flows, (Money, CashFlow, str)) or not isinstance(cash_flows, Iterable):
        raise TypeError(f"cash_flows 必须是 Money 或 CashFlow 的序列，收到 {type(cash_flows).__name__}")
    flows = list(cash_flows)
    if not flows:
        raise ValueError("cash_flows 不能为空")
    r = _real_array(rate, "rate")
    compounding = _check_compounding(compounding)
    conv = _parse_convention(convention)

    if as_of_date is None:
        for cf in flows:
            if isinstance(cf, CashFlow):
                raise ValueError("cash_flows 含 CashFlow 时必须给出 as_of_date；不带日期的现金流请用 Money")
            if not isinstance(cf, Money):
                raise TypeError(f"cash_flows 元素必须是 Money，收到 {type(cf).__name__}")
        amounts, times = flows, [float(i) for i in range(len(flows))]
    else:
        day = to_date(as_of_date, "as_of_date")
        for cf in flows:
            if isinstance(cf, Money):
                raise ValueError("给出 as_of_date 时 cash_flows 元素必须是带日期的 CashFlow")
            if not isinstance(cf, CashFlow):
                raise TypeError(f"cash_flows 元素必须是 CashFlow，收到 {type(cf).__name__}")
        amounts = [cf.amount for cf in flows]
        times = [year_fraction(day, cf.pay_date, conv) for cf in flows]

    currency = amounts[0].currency
    for amount in amounts[1:]:
        if amount.currency != currency:
            raise TypeError(f"币种不匹配: {currency.code} 与 {amount.currency.code}")

    guard = currency.digits + _NPV_GUARD_DIGITS
    total = None
    for amount, t in zip(amounts, times):
        term = amount.data.mul_round(_factor(1.0 / _growth(r, np.asarray(t), compounding)), scale=guard)
        total = term if total is None else total + term
    return Money._wrap(total.rescale(currency.digits), currency)


def discount_factors(rates: Any, times: Any, *, compounding: str = "annual") -> Union[np.ndarray, np.float64]:
    """贴现因子 1 / 增长因子（float64），rates 与 times 按广播规则对齐；两者都是标量时返回 np.float64。

    Parameters
    ----------
    rates : 实数、数值字符串、FixedArray 或实数数组
        零息利率。
    times : 实数或实数数组
        期限（期数或年数）。
    compounding : {"annual", "continuous", "simple"}, optional
        复利方式，默认 "annual"。
    """
    return 1.0 / _growth(_real_array(rates, "rates"), _real_array(times, "times"), _check_compounding(compounding))


__all__ = ["pv", "fv", "npv", "discount_factors"]
