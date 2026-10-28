"""Moneta 金额展示格式化工具 (Monetary Formatter)。

约定：
- 千分位与小数点分隔符由调用方显式指定（默认 "," 与 "."），不按地区名推断；
- 默认同时显示币种符号与代码（"¥" 可能是 CNY 也可能是 JPY，代码用于消除歧义）；符号与代码相同时只显示代码
  （如 "1.00 CHF"），以字母结尾的符号与数字之间加一个空格（如 "CHF 1.00"）；
- 缺失值显示为 "<NA>"。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .money import Money

_NA_TEXT = "<NA>"


class MonetaryFormatter:
    """金额展示格式化器：千分位分隔、币种符号与代码、会计负数括号 `($1,234.50 USD)`。"""

    @classmethod
    def format(cls, money: "Money", accounting: bool = False, show_symbol: bool = True, show_code: bool = True,
               thousands_sep: str = ",", decimal_sep: str = ".") -> str:
        """格式化单个金额标量。

        Parameters
        ----------
        money : Money
            标量金额（ndim == 0）；数组请用 format_array。
        accounting : bool, optional
            为 True 时负数用会计括号表示，如 "($1,234.50 USD)"。
        show_symbol, show_code : bool, optional
            是否显示币种符号（前置）与币种代码（后置）。
        thousands_sep, decimal_sep : str, optional
            千分位与小数点分隔符；千分位可为空串（不分组），小数点不能为空，两者不能相同。
        """
        cls._check(money, thousands_sep, decimal_sep)
        if money.ndim != 0:
            raise ValueError("format 只支持标量 Money (ndim == 0)；数组请用 format_array")
        return cls._format_one(money, int(money.mantissa), bool(money.isna()), accounting, show_symbol, show_code,
                               thousands_sep, decimal_sep)

    @classmethod
    def format_array(cls, money: "Money", accounting: bool = False, show_symbol: bool = True,
                     show_code: bool = True, thousands_sep: str = ",", decimal_sep: str = ".") -> np.ndarray:
        """逐个格式化金额数组，返回与输入同形状的 object 数组（元素为 str），参数含义同 format。"""
        cls._check(money, thousands_sep, decimal_sep)
        mantissa, na = money.mantissa, money.isna()
        out = np.empty(money.shape, dtype=object)
        for idx in np.ndindex(money.shape):
            out[idx] = cls._format_one(money, int(mantissa[idx]), bool(na[idx]), accounting, show_symbol,
                                       show_code, thousands_sep, decimal_sep)
        return out

    @staticmethod
    def _check(money: Any, thousands_sep: Any, decimal_sep: Any) -> None:
        from .money import Money

        if not isinstance(money, Money):
            raise TypeError(f"money 必须是 Money，收到 {type(money).__name__}")
        for name, sep in (("thousands_sep", thousands_sep), ("decimal_sep", decimal_sep)):
            if not isinstance(sep, str):
                raise TypeError(f"{name} 必须是 str，收到 {type(sep).__name__}")
        if not decimal_sep:
            raise ValueError("decimal_sep 不能为空")
        if thousands_sep == decimal_sep:
            raise ValueError(f"thousands_sep 与 decimal_sep 不能相同: {decimal_sep!r}")

    @staticmethod
    def _format_one(money: "Money", mantissa: int, na: bool, accounting: bool, show_symbol: bool, show_code: bool,
                    thousands_sep: str, decimal_sep: str) -> str:
        if na:
            return _NA_TEXT
        currency = money.currency
        digits = currency.digits
        int_val, frac_val = divmod(abs(mantissa), 10 ** digits)
        num_str = f"{int_val:,}".replace(",", thousands_sep)
        if digits > 0:
            num_str += decimal_sep + str(frac_val).zfill(digits)

        symbol = currency.symbol if show_symbol else ""
        if show_code and symbol == currency.code:
            symbol = ""
        if symbol[-1:].isalpha():
            symbol += " "
        body = f"{symbol}{num_str}" + (f" {currency.code}" if show_code else "")

        if mantissa >= 0:
            return body
        return f"({body})" if accounting else f"-{body}"


__all__ = ["MonetaryFormatter"]
