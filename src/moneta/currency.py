"""Moneta 币种与度量衡定义系统 (Currency & Currency Registry)。

定义法定货币与金融资产的量纲、原子精度（digits）及享元单例注册表。

约定：
- Currency 是不可变值对象，构造时校验代码与精度；同一代码在注册表中只有一个定义。
- 注册表是唯一事实来源：重复注册完全相同的定义无副作用，任何字段不同即报错，内置币种不可覆盖。
- 相等比较代码与 digits；copy / deepcopy 返回自身，pickle 还原时回到注册表中的同一单例。
"""

from __future__ import annotations

import operator
import re
from typing import Any, Dict, Tuple, Union

import numpy as np

_CODE_RE = re.compile(r"^[A-Z]{3,5}$")
_MAX_DIGITS = 18  # int64 尾数在 18 位小数下仍能表示 9.22


class Currency:
    """法定货币或资产度量衡标准定义（不可变，享元单例）。

    定义特定币种的代码及其最小原子不可分割单位（严格遵循 ISO 4217 标准法定结算精度，如 CNY/USD/EUR 为 2 位对应分/Cent，JPY 为 0 位，BHD 为 3 位）。

    Parameters
    ----------
    code : str
        ISO 4217 货币代码，如 "CNY"、"USD"、"EUR"、"JPY"；也允许 3 到 5 个字母的自定义代码（如 "USDT"）。
        首尾空白会被去除，并统一转为大写。
    digits : int
        最小原子结算单位对应的小数位数（0 <= digits <= 18），不接受 bool 与 float。
    symbol : str, optional
        货币符号，如 "¥"、"$"、"€"。默认为空字符串。
    name : str, optional
        货币全称，如 "Chinese Yuan"、"US Dollar"。默认与代码相同。

    Raises
    ------
    TypeError
        参数类型不符。
    ValueError
        代码不是 3 到 5 个字母，或 digits 超出 [0, 18]。
    """

    __slots__ = ("_code", "_digits", "_symbol", "_name")

    def __init__(self, code: str, digits: int, symbol: str = "", name: str = "") -> None:
        if not isinstance(code, str):
            raise TypeError(f"货币代码必须是 str，收到 {type(code).__name__}")
        normalized = code.strip().upper()
        if not _CODE_RE.match(normalized):
            raise ValueError(f"货币代码必须是 3 到 5 个英文字母，收到 {code!r}")
        if isinstance(digits, (bool, np.bool_)):  # NumPy < 2 的 np.bool_ 仍能被 operator.index 接受
            raise TypeError("digits 不能是 bool")
        try:
            digits = operator.index(digits)  # 接受 int 与 NumPy 整数，拒绝 float
        except TypeError:
            raise TypeError(f"digits 必须是整数，收到 {type(digits).__name__}") from None
        if not 0 <= digits <= _MAX_DIGITS:
            raise ValueError(f"digits 必须在 0 到 {_MAX_DIGITS} 之间，收到 {digits}")
        if not isinstance(symbol, str):
            raise TypeError(f"symbol 必须是 str，收到 {type(symbol).__name__}")
        if not isinstance(name, str):
            raise TypeError(f"name 必须是 str，收到 {type(name).__name__}")

        object.__setattr__(self, "_code", normalized)
        object.__setattr__(self, "_digits", digits)
        object.__setattr__(self, "_symbol", symbol)
        object.__setattr__(self, "_name", name or normalized)

    @property
    def code(self) -> str:
        """货币代码（大写），如 'USD'。"""
        return self._code

    @property
    def digits(self) -> int:
        """最小原子单位的小数位数。"""
        return self._digits

    @property
    def symbol(self) -> str:
        """货币显示符号。"""
        return self._symbol

    @property
    def name(self) -> str:
        """货币全称。"""
        return self._name

    def _definition(self) -> Tuple[str, int, str, str]:
        """完整定义，用于判断注册冲突。"""
        return self._code, self._digits, self._symbol, self._name

    # ---------- 不可变 ----------

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(f"Currency 是不可变对象，禁止设置属性 '{key}'")

    def __delattr__(self, key: str) -> None:
        raise AttributeError(f"Currency 是不可变对象，禁止设置属性 '{key}'")

    # ---------- 单例语义：copy / deepcopy / pickle ----------

    def __copy__(self) -> "Currency":
        return self

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Currency":
        return self

    def __reduce__(self) -> Tuple[Any, Tuple[str, int, str, str]]:
        return _restore_currency, self._definition()

    # ---------- 相等与展示 ----------

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Currency):
            return NotImplemented
        return self._code == other._code and self._digits == other._digits

    def __hash__(self) -> int:
        return hash(self._code)

    def __repr__(self) -> str:
        return (f"Currency(code='{self._code}', digits={self._digits}, "
                f"symbol='{self._symbol}', name='{self._name}')")

    def __str__(self) -> str:
        return self._code


class CurrencyRegistry:
    """货币注册表与享元工厂（Currency Registry）。

    维护全局 ISO 4217 标准币种单例，是币种定义的唯一事实来源：
    同一代码只能有一个定义，不能覆盖已注册的币种（包括内置币种）。
    """

    _registry: Dict[str, Currency] = {}

    @classmethod
    def register(cls, currency: Currency) -> None:
        """注册一个货币定义到全局注册表中。

        重复注册完全相同的定义无副作用；同一代码的 digits、symbol 或 name 不同时报 ValueError。

        Raises
        ------
        TypeError
            currency 不是 Currency 实例。
        ValueError
            代码已注册为不同的定义。
        """
        if not isinstance(currency, Currency):
            raise TypeError(f"只能注册 Currency 实例，收到 {type(currency).__name__}")
        existing = cls._registry.get(currency.code)
        if existing is None:
            cls._registry[currency.code] = currency
        elif existing._definition() != currency._definition():
            raise ValueError(f"货币代码 {currency.code} 已注册为 {existing!r}，不能改注册为 {currency!r}")

    @classmethod
    def get(cls, code: Union[str, Currency]) -> Currency:
        """根据代码获取货币单例。

        传入 Currency 实例时：已注册且定义一致则返回注册表中的单例；定义不一致报 ValueError；
        未注册则原样返回。

        Raises
        ------
        TypeError
            code 既不是 str 也不是 Currency。
        KeyError
            代码未注册。
        ValueError
            Currency 实例与注册表中同代码的定义不一致。
        """
        if isinstance(code, Currency):
            registered = cls._registry.get(code.code)
            if registered is None:
                return code
            if registered._definition() != code._definition():
                raise ValueError(f"{code!r} 与注册表中的定义 {registered!r} 不一致")
            return registered
        if not isinstance(code, str):
            raise TypeError(f"货币代码必须是 str 或 Currency，收到 {type(code).__name__}")
        registered = cls._registry.get(code.strip().upper())
        if registered is None:
            raise KeyError(f"未注册的货币代码: {code!r}")
        return registered

    @classmethod
    def has(cls, code: Union[str, Currency]) -> bool:
        """检查货币代码（或与注册定义一致的 Currency 实例）是否已注册。"""
        if isinstance(code, Currency):
            registered = cls._registry.get(code.code)
            return registered is not None and registered._definition() == code._definition()
        if not isinstance(code, str):
            return False
        return code.strip().upper() in cls._registry

    @classmethod
    def clear(cls) -> None:
        """清空除内置预设外的注册表（用于测试隔离）。"""
        cls._registry.clear()
        _init_defaults()


def get_currency(cur: Union[str, Currency]) -> Currency:
    """通用类型提升助手：将字符串或 Currency 统一转换为注册表中的 Currency 单例。"""
    return CurrencyRegistry.get(cur)


def _restore_currency(code: str, digits: int, symbol: str, name: str) -> Currency:
    """pickle 还原：已注册则回到注册表单例（定义不一致报错），未注册则还原为独立实例。"""
    restored = Currency(code, digits, symbol, name)
    registered = CurrencyRegistry._registry.get(restored.code)
    if registered is None:
        return restored
    if registered._definition() != restored._definition():
        raise ValueError(f"pickle 中的 {restored!r} 与注册表中的定义 {registered!r} 不一致")
    return registered


# 全局预定义标准货币（常用单例）
USD = Currency("USD", 2, "$", "US Dollar")
CNY = Currency("CNY", 2, "¥", "Chinese Yuan")
EUR = Currency("EUR", 2, "€", "Euro")
JPY = Currency("JPY", 0, "¥", "Japanese Yen")
GBP = Currency("GBP", 2, "£", "British Pound")
HKD = Currency("HKD", 2, "HK$", "Hong Kong Dollar")
AUD = Currency("AUD", 2, "A$", "Australian Dollar")
CAD = Currency("CAD", 2, "C$", "Canadian Dollar")
CHF = Currency("CHF", 2, "CHF", "Swiss Franc")
SGD = Currency("SGD", 2, "S$", "Singapore Dollar")


def _init_defaults() -> None:
    for c in (USD, CNY, EUR, JPY, GBP, HKD, AUD, CAD, CHF, SGD):
        CurrencyRegistry.register(c)


_init_defaults()

__all__ = [
    "Currency",
    "CurrencyRegistry",
    "get_currency",
    "CNY",
    "USD",
    "EUR",
    "JPY",
    "GBP",
    "HKD",
    "AUD",
    "CAD",
    "CHF",
    "SGD",
]
