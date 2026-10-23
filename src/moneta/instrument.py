"""Moneta 金融标的物资产参考数据实体 (Instrument)。

表示除法定现金货币之外的独立金融标的不可变参考合约（如股票、大宗商品、加密资产、期货合约）。
保存代码、计价货币、交易基准单位、合约乘数、最小变动价位及资产大类。

约定：
- Instrument 是不可变值对象，在组合中作为持仓的身份标识：相等与哈希只看 (symbol, quote_currency)，
  名称、乘数等参考数据不参与比较。
- 构造时校验全部参数：代码为非空 str 并转大写；基准单位必须是具体单位（不能是通配单位 "Any"）；
  合约乘数与 tick 必须是正的数值标量。
- quantity / price / notional 按本标的的单位、计价货币与合约乘数构造数量、价格与名义金额；
  裸数字价格按本标的的计价货币解释。持仓份额由 Quantity / PositionLot 维护。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

from .currency import Currency, get_currency
from .fixed_array import FixedArray, RoundingMode
from .money import Money
from .price import Price
from .quantity import WILDCARD_UNIT, Quantity
from .unit_array import UnitArray

Rounding = Union[RoundingMode, str]


class Instrument:
    """金融标的物参考数据实体（不可变合约定义）。

    Parameters
    ----------
    symbol : str
        资产代码，如 "AAPL"、"BTC"、"CL2412"。首尾空白会被去除并统一转为大写，不能为空。
    quote_currency : Union[Currency, str]
        计价/结算货币，如 "USD" 或 "CNY"。
    base_unit : str, optional
        报价与持仓的基准单位，如 "share"、"barrel"、"contract"。默认为 "share"；不能是通配单位 "Any"。
    multiplier : int, float, Decimal, Fraction 或 FixedArray, optional
        合约乘数：每 1 个基准单位对应的标的数量（如原油期货每手 1000 桶）。必须为正的标量，默认为 1。
    tick_size : 数值, optional
        最小变动价位（如 Decimal("0.01")），必须为正的标量；默认为 None（不限制）。
    name : str, optional
        资产全称，如 "Apple Inc."。默认与代码相同。
    asset_class : str, optional
        资产大类，如 "Equity"、"Commodity"、"Crypto"、"Option"。默认为 "Equity"。

    Raises
    ------
    TypeError
        参数类型不符（如代码、名称不是 str，乘数或 tick 不是数值）。
    ValueError
        代码为空、基准单位为空或为通配单位、乘数或 tick 不是正的标量（含缺失值）。
    """

    __slots__ = ("_symbol", "_quote_currency", "_base_unit", "_multiplier", "_tick_size", "_name", "_asset_class")

    def __init__(self, symbol: str, quote_currency: Union[Currency, str], base_unit: str = "share",
                 multiplier: Any = 1, tick_size: Optional[Any] = None, name: str = "",
                 asset_class: str = "Equity") -> None:
        if not isinstance(symbol, str):
            raise TypeError(f"标的代码必须是 str，收到 {type(symbol).__name__}")
        code = symbol.strip().upper()
        if not code:
            raise ValueError("标的代码不能为空")
        unit = UnitArray._check_unit(base_unit)
        if unit == WILDCARD_UNIT:
            raise ValueError(f"标的的基准单位必须是具体单位，不能是通配单位 '{WILDCARD_UNIT}'")
        for label, value in (("name", name), ("asset_class", asset_class)):
            if not isinstance(value, str):
                raise TypeError(f"{label} 必须是 str，收到 {type(value).__name__}")

        object.__setattr__(self, "_symbol", code)
        object.__setattr__(self, "_quote_currency", get_currency(quote_currency))
        object.__setattr__(self, "_base_unit", unit)
        object.__setattr__(self, "_multiplier", self._positive_scalar(multiplier, "multiplier"))
        object.__setattr__(self, "_tick_size",
                           None if tick_size is None else self._positive_scalar(tick_size, "tick_size"))
        object.__setattr__(self, "_name", name.strip() or code)
        object.__setattr__(self, "_asset_class", asset_class.strip() or "Equity")

    @staticmethod
    def _positive_scalar(value: Any, what: str) -> FixedArray:
        """数值白名单内、正的、非缺失的标量（精确保留输入精度，float 按最短写法）。"""
        fa = UnitArray._as_grid(value, f"{what} ")
        if fa.ndim != 0:
            raise ValueError(f"{what} 必须是标量")
        if fa.mantissa <= 0:
            raise ValueError(f"{what} 必须为正数")
        return fa

    # ---------- 属性 ----------

    @property
    def symbol(self) -> str:
        """资产代码（大写），如 'AAPL'。"""
        return self._symbol

    @property
    def quote_currency(self) -> Currency:
        """计价货币。"""
        return self._quote_currency

    @property
    def base_unit(self) -> str:
        """基准单位（如 'share'）。"""
        return self._base_unit

    @property
    def multiplier(self) -> FixedArray:
        """合约乘数（0 维 FixedArray）。"""
        return self._multiplier

    @property
    def tick_size(self) -> Optional[FixedArray]:
        """最小变动价位（0 维 FixedArray；未设置为 None）。"""
        return self._tick_size

    @property
    def name(self) -> str:
        """资产全称。"""
        return self._name

    @property
    def asset_class(self) -> str:
        """资产大类（如 'Equity'）。"""
        return self._asset_class

    # ---------- 按本标的构造数量、价格与名义金额 ----------

    def as_quantity(self, values: Any, scale: Optional[int] = None) -> Quantity:
        """以本标的的基准单位构造持仓数量。"""
        return Quantity(values, unit=self._base_unit, scale=scale)

    def as_price(self, values: Any, scale: Optional[int] = None) -> Price:
        """以本标的的计价货币、基准单位与 tick 构造价格；已是 Price 时校验量纲（scale 不为 None 时调整精度）。"""
        if isinstance(values, Price):
            if values.quote_currency != self._quote_currency:
                raise TypeError(f"价格计价货币 {values.quote_currency.code} 与标的 {self._symbol} 的计价货币 "
                                f"{self._quote_currency.code} 不一致")
            if values.base_unit not in (self._base_unit, WILDCARD_UNIT):
                raise TypeError(
                    f"价格单位 '{values.base_unit}' 与标的 {self._symbol} 的基准单位 '{self._base_unit}' 不一致")
            return values if scale is None else Price(values, self._quote_currency, values.base_unit, scale=scale)
        return Price(values, self._quote_currency, base_unit=self._base_unit, tick_size=self._tick_size, scale=scale)

    def calc_notional(self, price: Any, quantity: Any, rounding: Rounding = "half_up") -> Money:
        """名义金额 = 价格 × 数量 × 合约乘数，按 rounding 只舍入一次到计价货币精度。

        price 可以是 Price（计价货币须一致，单位须一致或为通配）或裸数字（按计价货币解释）；
        quantity 可以是 Quantity（单位须一致或为通配）或裸数字（按基准单位解释）。
        """
        p = self.as_price(price)
        q = quantity if isinstance(quantity, Quantity) else self.as_quantity(quantity)
        if q.unit not in (self._base_unit, WILDCARD_UNIT):
            raise TypeError(f"数量单位 '{q.unit}' 与标的 {self._symbol} 的基准单位 '{self._base_unit}' 不一致")
        exact = p.data.mul_round(q.data, rounding=None)  # 价格 × 数量精确相乘，只在乘以合约乘数时舍入一次
        total = exact.mul_round(self._multiplier, scale=self._quote_currency.digits, rounding=rounding)
        return Money._wrap(total, self._quote_currency)

    # ---------- 不可变与值语义 ----------

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(f"Instrument 是不可变对象，禁止设置属性 '{key}'")

    def __delattr__(self, key: str) -> None:
        raise AttributeError(f"Instrument 是不可变对象，禁止删除属性 '{key}'")

    def __copy__(self) -> "Instrument":
        return self

    def __deepcopy__(self, memo: Any) -> "Instrument":
        return self

    def __reduce__(self) -> Tuple[Any, Tuple[Any, ...]]:
        return Instrument, (self._symbol, self._quote_currency, self._base_unit, self._multiplier,
                            self._tick_size, self._name, self._asset_class)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Instrument):
            return NotImplemented
        return self._symbol == other._symbol and self._quote_currency == other._quote_currency

    def __hash__(self) -> int:
        return hash((self._symbol, self._quote_currency.code))

    def __repr__(self) -> str:
        tick = "None" if self._tick_size is None else f"'{self._tick_size}'"
        return (f"Instrument(symbol='{self._symbol}', quote_currency='{self._quote_currency.code}', "
                f"base_unit='{self._base_unit}', multiplier='{self._multiplier}', tick_size={tick}, "
                f"name='{self._name}', asset_class='{self._asset_class}')")

    def __str__(self) -> str:
        return f"{self._symbol} ({self._name}, {self._quote_currency.code}/{self._base_unit})"


__all__ = ["Instrument"]
