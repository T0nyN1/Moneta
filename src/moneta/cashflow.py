"""Moneta 现金流与付息排程 (CashFlow & Schedule)。

包含：
1. CashFlow：单笔带日期的现金流（不可变）；
2. generate_cash_flow_schedule：按付息频率生成定期票息与到期本金现金流。

约定：
- 付息日由起始日直接推算（起始日 + k 个周期），不逐期累加，不会产生月末漂移；起始日是月末时，所有付息日都取月末。
- 到期日必须正好落在付息日序列上，不支持不足一期的付息。
- 票息由 principal × coupon_rate / 每年付息次数 精确计算，只舍入一次。
"""

from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any, List, Optional, Union

from ._utils import to_date
from .currency import Currency
from .fixed_array import FixedArray
from .money import Money

DateLike = Union[str, date, datetime]

_FREQ_NAMES = {"monthly": 12, "quarterly": 4, "semi-annual": 2, "semiannual": 2, "annual": 1, "annually": 1}


class CashFlow:
    """单笔带日期的现金流（不可变）。

    Parameters
    ----------
    amount : Money
        现金流金额（正数表示流入，负数表示流出），可以是数组。
    pay_date : date、datetime 或 ISO 字符串
        支付日期。
    label : str, optional
        标签，如 "Coupon"、"Principal"、"Fee"。
    """

    __slots__ = ("_amount", "_pay_date", "_label")

    def __init__(self, amount: Money, pay_date: DateLike, label: str = "") -> None:
        if not isinstance(amount, Money):
            raise TypeError(f"amount 必须是 Money，收到 {type(amount).__name__}")
        if not isinstance(label, str):
            raise TypeError(f"label 必须是 str，收到 {type(label).__name__}")
        for name, value in (("_amount", amount), ("_pay_date", to_date(pay_date, "pay_date")), ("_label", label)):
            object.__setattr__(self, name, value)

    @property
    def amount(self) -> Money:
        """现金流金额。"""
        return self._amount

    @property
    def currency(self) -> Currency:
        """现金流币种。"""
        return self._amount.currency

    @property
    def pay_date(self) -> date:
        """支付日期。"""
        return self._pay_date

    @property
    def label(self) -> str:
        """标签。"""
        return self._label

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(f"CashFlow 是不可变对象，禁止设置属性 '{key}'")

    def __delattr__(self, key: str) -> None:
        raise AttributeError(f"CashFlow 是不可变对象，禁止删除属性 '{key}'")

    def __repr__(self) -> str:
        return f"CashFlow(amount={self._amount!r}, pay_date='{self._pay_date}', label='{self._label}')"

    def __str__(self) -> str:
        return f"CashFlow({self._amount} on {self._pay_date} [{self._label}])"


def _parse_frequency(freq: Any) -> int:
    """付息频率 → 每年付息次数：接受 'monthly'、'quarterly'、'semi-annual'、'annual'（不区分大小写）或 1、2、4、12。"""
    if isinstance(freq, str) and freq.strip().lower() in _FREQ_NAMES:
        return _FREQ_NAMES[freq.strip().lower()]
    if isinstance(freq, int) and not isinstance(freq, bool) and freq in _FREQ_NAMES.values():
        return freq
    raise ValueError(f"不支持的付息频率 {freq!r}：可选 {list(_FREQ_NAMES)} 或 1、2、4、12")


def _add_months(d: date, months: int, end_of_month: bool) -> date:
    """d 加 months 个月；日超出当月天数时取月末，end_of_month 为 True 时总取月末。"""
    year, month = divmod(d.month - 1 + months, 12)
    year, month = d.year + year, month + 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, last if end_of_month else min(d.day, last))


def _coupon_amount(principal: Money, rate: FixedArray, n: int) -> Money:
    """principal × rate / n，half_up 精确舍入一次到币种精度。

    先把乘积按 digits + 1 位向零截断（避免精确乘积超出 int64），再除以 n 舍入：舍入的平局点乘以 n 恰在
    digits + 1 位的网格上，截断不会越过它，因此结果与对精确值舍入相同。
    """
    digits = principal.currency.digits
    product = principal.data.mul_round(rate, scale=digits + 1, rounding="down")
    return Money._wrap(product.divide(n, scale=digits), principal.currency)


def generate_cash_flow_schedule(start_date: DateLike, end_date: DateLike, freq: Union[str, int], *,
                                coupon: Optional[Money] = None, principal: Optional[Money] = None,
                                coupon_rate: Any = None) -> List[CashFlow]:
    """生成定期付息排程：每个付息日一笔票息（label 'Coupon'），到期日另加一笔本金（label 'Principal'）。

    Parameters
    ----------
    start_date, end_date : date、datetime 或 ISO 字符串
        起息日与到期日；到期日须严格晚于起息日，且正好落在付息日序列上。
    freq : str 或 int
        付息频率：'monthly'、'quarterly'、'semi-annual'、'annual' 或每年付息次数 12、4、2、1。
    coupon : Money, optional
        每期票息金额。
    principal : Money, optional
        到期归还的本金。
    coupon_rate : 数值或字符串, optional
        年票息率；须与 principal 同时给出且不能与 coupon 同时给出，每期票息 = principal × coupon_rate / 每年付息次数。
    """
    start, end = to_date(start_date, "start_date"), to_date(end_date, "end_date")
    if start >= end:
        raise ValueError(f"end_date ({end}) 必须严格晚于 start_date ({start})")
    n = _parse_frequency(freq)
    for name, value in (("coupon", coupon), ("principal", principal)):
        if value is not None and not isinstance(value, Money):
            raise TypeError(f"{name} 必须是 Money，收到 {type(value).__name__}")
    if coupon_rate is not None:
        if principal is None:
            raise ValueError("给出 coupon_rate 时必须同时给出 principal")
        if coupon is not None:
            raise ValueError("coupon 与 coupon_rate 只能给出一个")
        rate = FixedArray(coupon_rate)
        if rate.mask is not None:
            raise ValueError("coupon_rate 不能是缺失值")
        coupon = _coupon_amount(principal, rate, n)
    if coupon is None and principal is None:
        raise ValueError("coupon、principal、coupon_rate 至少需要给出一种现金流")
    if coupon is not None and principal is not None and coupon.currency != principal.currency:
        raise ValueError(f"票息币种 {coupon.currency.code} 与本金币种 {principal.currency.code} 不一致")

    step, end_of_month = 12 // n, start.day == calendar.monthrange(start.year, start.month)[1]
    dates, k = [], 1
    while not dates or dates[-1] < end:
        dates.append(_add_months(start, k * step, end_of_month))
        k += 1
    if dates[-1] != end:
        before = f"付息日 {dates[-2]}" if len(dates) > 1 else f"起息日 {start}"
        raise ValueError(f"end_date ({end}) 不在付息日序列上（位于{before} 与付息日 {dates[-1]} 之间），不支持不足一期的付息")

    flows = [CashFlow(coupon, d, "Coupon") for d in dates] if coupon is not None else []
    if principal is not None:
        flows.append(CashFlow(principal, end, "Principal"))
    return flows


__all__ = ["CashFlow", "generate_cash_flow_schedule"]
