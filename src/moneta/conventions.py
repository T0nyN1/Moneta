"""Moneta 金融市场惯例模块 (Market Conventions)。

包含专业计日惯例 (Day Count Conventions) 及年化时间跨度 (Year Fraction) 计算。
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from fractions import Fraction
from typing import Union

from ._utils import is_leap_year as _is_leap_year, to_date as _to_date


class DayCountConvention(Enum):
    """金融计日惯例（用于利率、贴现与持有期利息折算）。"""

    ACT_360 = "ACT/360"  # 实际天数 / 360：货币市场、短期利率（美元 SOFR 等）常用
    ACT_365 = "ACT/365"  # ACT/365 Fixed：实际天数 / 365（不论闰年），英镑、人民币货币市场常用
    THIRTY_360 = "30/360"  # 30/360 Bond Basis（ISDA 2006 4.16(f)）：每月按 30 天，无 2 月月末规则
    ACT_ACT = "ACT/ACT"  # ACT/ACT ISDA：按自然年分段，各段除以当年实际天数（365 或 366）


_CONVENTION_KEYS = {key: m for m in DayCountConvention for key in (m.value, m.name)}


def _parse_convention(convention: Union[DayCountConvention, str]) -> DayCountConvention:
    """接受枚举成员，或其值/成员名字符串（不区分大小写，忽略首尾空白）。"""
    if isinstance(convention, DayCountConvention):
        return convention
    if not isinstance(convention, str):
        raise TypeError(f"convention 必须是 DayCountConvention 或 str，收到 {type(convention).__name__}")
    key = convention.strip().upper()
    if key not in _CONVENTION_KEYS:
        raise ValueError(f"不支持的计日惯例 {convention!r}：可选 {[m.value for m in DayCountConvention]}"
                         f" 或 {[m.name for m in DayCountConvention]}")
    return _CONVENTION_KEYS[key]


def year_fraction(start_date: Union[str, date, datetime], end_date: Union[str, date, datetime],
                  convention: Union[DayCountConvention, str] = DayCountConvention.ACT_365) -> float:
    """计算两个日期之间在指定计日惯例下的年化时间跨度 (Year Fraction tau)。

    Parameters
    ----------
    start_date : Union[str, date, datetime]
        起始日期。
    end_date : Union[str, date, datetime]
        终止日期；早于 start_date 时返回负值。
    convention : Union[DayCountConvention, str], optional
        计日惯例，默认为 DayCountConvention.ACT_365；字符串可用值（如 "ACT/360"）或成员名（如 "ACT_360"），
        不区分大小写。

    Returns
    -------
    float
        年化时间跨度 tau = Delta t。
    """
    d1 = _to_date(start_date, "start_date")
    d2 = _to_date(end_date, "end_date")
    conv = _parse_convention(convention)

    sign = 1
    if d1 > d2:
        d1, d2 = d2, d1
        sign = -1

    days_diff = (d2 - d1).days

    if conv is DayCountConvention.ACT_360:
        return sign * days_diff / 360

    if conv is DayCountConvention.ACT_365:
        return sign * days_diff / 365

    if conv is DayCountConvention.THIRTY_360:
        day1 = min(d1.day, 30)
        day2 = min(d2.day, 30) if day1 == 30 else d2.day
        days_30_360 = 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (day2 - day1)
        return sign * days_30_360 / 360

    # ACT/ACT ISDA：按自然年分段精确求和，最后只转换一次 float
    def days_in(year: int) -> int:
        return 366 if _is_leap_year(year) else 365

    if d1.year == d2.year:
        tau = Fraction(days_diff, days_in(d1.year))
    else:
        rem_y1 = (date(d1.year + 1, 1, 1) - d1).days
        cur_y2 = (d2 - date(d2.year, 1, 1)).days
        tau = Fraction(rem_y1, days_in(d1.year)) + (d2.year - d1.year - 1) + Fraction(cur_y2, days_in(d2.year))
    return float(sign * tau)


__all__ = ["DayCountConvention", "year_fraction"]
