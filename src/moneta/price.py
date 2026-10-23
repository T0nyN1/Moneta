"""Moneta 强度比率量与金融标价体系 (Price & Pricing Grid)。

定义单位标的报价，支持 Tick 报价网格约束与点差。

约定：
- 量纲为 [计价货币 / 标的单位]；两个 Price 兼容当且仅当计价货币相同、单位相同或任一方为通配单位 "Any"，
  Price 与 Quantity 相乘时同样双向通配。运算结果取具体的那个单位。
- 运算精确：乘以数值时 scale 为两侧之和；除法默认保留 max(两侧 scale) + 8 位小数；
  需要对齐报价网格时显式调用 round_to_step（不传 step 时使用自带的 tick）。Price × Quantity → Money（multiply 的 rounding 决定舍入），
  结果为 Money 时精度固定为币种精度。
- tick_size 是元数据：构造时不要求价格在网格上（由 round_to_step / is_on_step 显式处理），不参与相等比较；两个 Price 运算时 tick 相同则保留，否则结果没有 tick。
- 构造指定 scale 时与 Money 一致：float 按最短写法 half_up 静默舍入；精确输入超出 scale 时由 FixedArray 发出 PrecisionLossWarning 后舍入。
- 运算只接受数值白名单内的操作数，拒绝字符串；运算符遇到不认识的类型返回 NotImplemented。
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from ._rounding import check_rounding
from ._units import WILDCARD_UNIT, check_unit, units_compatible
from .currency import Currency, get_currency
from .fixed_array import FixedArray, RoundingMode
from .money import Money
from .quantity import Quantity
from .unit_array import UnitArray

_RATIO_EXTRA_DIGITS = 8  # 除法结果相对两侧 scale 额外保留的小数位数
_KEEP = object()  # _create_like 的哨兵：沿用原来的 tick

Rounding = Union[RoundingMode, str]


class Price(UnitArray):
    """金融标价（Intensive Quantity / 强度量）。

    表示单位标的物的价格（报价），量纲为 [计价货币 / 报价单位]。
    继承自 UnitArray，自动获得切片、形状变形、容器协议、取负、绝对值、求和及比较转换能力。
    支持报价网格（Tick Size / Pip）约束与点差（Spread）。

    Parameters
    ----------
    values : int, float, str, Decimal, Fraction, FixedArray, Price, Sequence 或 np.ndarray
        价格数值或高维矩阵。
    quote_currency : Union[Currency, str]
        计价货币（如 USD）。
    base_unit : str, optional
        计价标的单位，如 "share"、"barrel"、"EUR"；"Any" 为通配单位。默认为 "share"。
    tick_size : 数值或数值数组, optional
        最小变动价位（如 Decimal("0.01")），必须是正的标量。
    scale : int, optional
        小数位数。默认按输入本身的精度；指定时 float 按最短写法 half_up 舍入，
        精确输入超出 scale 时发出 PrecisionLossWarning 后按 half_up 舍入，不足时补 0。
    """

    __slots__ = ("_quote_currency", "_base_unit", "_tick_size")

    def __init__(self, values: Any, quote_currency: Union[Currency, str], base_unit: str = "share",
                 tick_size: Optional[Any] = None, scale: Optional[int] = None, ) -> None:
        currency = get_currency(quote_currency)
        unit = check_unit(base_unit)
        tick = None if tick_size is None else self._as_tick(tick_size)
        if tick is not None and tick.ndim:
            raise ValueError("tick_size 必须是标量；逐元素的价格网格请用 round_to_step(step=...) 显式传入")
        if isinstance(values, Price):
            if values._quote_currency != currency or not units_compatible(values._base_unit, unit):
                raise TypeError(f"价格量纲不匹配: {values._quote_currency.code}/{values._base_unit} "
                                f"vs {currency.code}/{unit}")
            data = values._data
            if tick is None:
                tick = values._tick_size
        else:
            data = values
        # 指定 scale 时 float 静默舍入；精确输入丢精度时由 FixedArray 告警后舍入
        self._data = FixedArray(data, scale=scale, on_excess="warn")
        self._quote_currency = currency
        self._base_unit = unit
        self._tick_size = tick

    @classmethod
    def _as_tick(cls, tick_size: Any, what: str = "tick_size") -> FixedArray:
        tick = cls._as_grid(tick_size, f"{what} ")
        if np.any(tick.mantissa <= 0):
            raise ValueError(f"{what} 必须为正数")
        return tick

    # ---------- 核心抽象钩子实现 ----------

    def _create_like(self, new_data: FixedArray, base_unit: Optional[str] = None, tick_size: Any = _KEEP) -> "Price":
        """构建同计价货币的新 Price（默认沿用单位与 tick）。"""
        obj = self.__class__.__new__(self.__class__)
        obj._data = new_data
        obj._quote_currency = self._quote_currency
        obj._base_unit = self._base_unit if base_unit is None else base_unit
        obj._tick_size = self._tick_size if tick_size is _KEEP else tick_size
        return obj

    def _check_compatible(self, other: Any) -> None:
        """检查另一个对象是否是同计价货币、同单位（或通配单位）的 Price。"""
        if not isinstance(other, Price):
            raise TypeError(f"操作数必须是 Price，得到 {type(other).__name__}")
        if self._quote_currency != other._quote_currency or not units_compatible(self._base_unit, other._base_unit):
            raise TypeError(
                f"价格量纲不匹配: {self._quote_currency.code}/{self._base_unit} "
                f"vs {other._quote_currency.code}/{other._base_unit}"
            )

    def _resolve_unit(self, other_unit: str) -> str:
        return other_unit if self._base_unit == WILDCARD_UNIT else self._base_unit

    def _resolve_tick(self, other: "Price") -> Optional[FixedArray]:
        """两个 Price 运算后的 tick：相同则保留，否则为 None。"""
        a, b = self._tick_size, other._tick_size
        if a is None or b is None:
            return None
        if a is b or bool(a == b):
            return a
        return None

    # ---------- 属性 ----------

    @property
    def quote_currency(self) -> Currency:
        """计价货币。"""
        return self._quote_currency

    @property
    def base_unit(self) -> str:
        """计价基准单位。"""
        return self._base_unit

    @property
    def tick_size(self) -> Optional[FixedArray]:
        """最小变动价位 Tick Size（未设置为 None）。"""
        return self._tick_size

    # ---------- 报价网格 ----------
    def round_to_step(self, step: Optional[Any] = None, rounding: Rounding = "half_up") -> "Price":
        """对齐到步长网格：step 为 None 时使用自带的 tick（未设置 tick 时原样返回），否则使用给定的 step。"""
        mode = check_rounding(rounding)
        grid = self._tick_size if step is None else self._as_tick(step, "step")
        if grid is None:
            return self
        return self._create_like(self._data.round_to_step(grid, rounding=mode))

    def is_on_step(self, step: Optional[Any] = None) -> Union[bool, np.ndarray]:
        """检查价格是否严格在步长网格上：step 为 None 时使用自带的 tick（未设置 tick 时除缺失值外恒为 True）。"""
        grid = self._tick_size if step is None else self._as_tick(step, "step")
        if grid is None:
            valid = self._data.notna()
            return bool(valid) if self.ndim == 0 else valid
        return self._data.is_on_step(grid)

    # ---------- 加减 ----------

    def __add__(self, other: Any) -> "Price":
        """同计价货币、兼容单位的价格相加。量纲不匹配报 TypeError；非 Price 返回 NotImplemented。"""
        if not isinstance(other, Price):
            return NotImplemented
        self._check_compatible(other)
        return self._create_like(self._data + other._data, self._resolve_unit(other._base_unit),
                                 self._resolve_tick(other))

    def __sub__(self, other: Any) -> "Price":
        """同计价货币、兼容单位的价格相减（价差）。量纲不匹配报 TypeError；非 Price 返回 NotImplemented。"""
        if not isinstance(other, Price):
            return NotImplemented
        self._check_compatible(other)
        return self._create_like(self._data - other._data, self._resolve_unit(other._base_unit),
                                 self._resolve_tick(other))

    # ---------- 乘法 ----------

    def multiply(self, other: Any, scale: Optional[int] = None, rounding: Rounding = "half_up") -> Union[
        "Price", Money]:
        """价格乘法，返回类型由乘数决定：
        - Price × Quantity → Money（成交金额），按 rounding 舍入到币种精度；精度固定，不接受 scale。
          数量单位须与 base_unit 相同，或任一方为通配单位 "Any"。
        - Price × 无量纲数值 → Price：不指定 scale 时精确相乘（scale 为两侧之和），否则按 rounding 舍入到 scale。
        """
        mode = check_rounding(rounding)
        if isinstance(other, Quantity):
            if scale is not None:
                raise ValueError("结果为 Money 时精度固定为币种精度，不能指定 scale")
            if not units_compatible(other.unit, self._base_unit):
                raise TypeError(f"标的单位不匹配: 数量单位 '{other.unit}' 与标价基准单位 '{self._base_unit}' 不一致")
            total = self._data.mul_round(other.data, scale=self._quote_currency.digits, rounding=mode)
            return Money._wrap(total, self._quote_currency)
        if isinstance(other, UnitArray):
            raise TypeError(f"价格只能乘以 Quantity 或数值，收到 {type(other).__name__}")
        factor = self._as_factor(other, "乘数")
        if scale is None:
            return self._create_like(self._data.mul_round(factor, rounding=None))
        return self._create_like(self._data.mul_round(factor, scale=scale, rounding=mode))

    def __mul__(self, other: Any) -> Any:
        """Price * Quantity → Money（half_up 舍入到币种精度）；Price * 数值 → 精确的 Price；其他返回 NotImplemented。"""
        if not isinstance(other, Quantity) and not FixedArray._is_numeric_operand(other):
            return NotImplemented
        return self.multiply(other)

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)

    # ---------- 除法 ----------

    def divide(self, other: Any, scale: Optional[int] = None, rounding: Rounding = "half_up") -> Union[
        "Price", FixedArray]:
        """价格除法：
        - Price / 同计价货币、兼容单位的 Price → 无量纲比率 FixedArray；
        - Price / 无量纲数值 → Price。
        默认保留 max(两侧 scale) + 8 位小数，可用 scale 指定；按 rounding 舍入。
        """
        mode = check_rounding(rounding)
        if isinstance(other, Price):
            self._check_compatible(other)
            target = max(self.scale, other.scale) + _RATIO_EXTRA_DIGITS if scale is None else scale
            return self._data.divide(other._data, scale=target, rounding=mode)
        if isinstance(other, UnitArray):
            raise TypeError(f"价格只能除以数值或同类 Price，收到 {type(other).__name__}")
        divisor = self._as_factor(other, "除数")
        target = max(self.scale, divisor.scale) + _RATIO_EXTRA_DIGITS if scale is None else scale
        return self._create_like(self._data.divide(divisor, scale=target, rounding=mode))

    def __truediv__(self, other: Any) -> Union["Price", FixedArray]:
        if not isinstance(other, Price) and not FixedArray._is_numeric_operand(other):
            return NotImplemented
        return self.divide(other)

    # ---------- 展示与转换 ----------

    def to_pandas(self, index: Optional[Any] = None, name: Optional[str] = None, as_float: bool = True) -> Any:
        """导出为 pandas.Series 或 pandas.DataFrame（自动注入 quote_currency 与 base_unit 元数据）。"""
        res = self._data.to_pandas(index=index, name=name, as_float=as_float)
        res.attrs["quote_currency"] = self._quote_currency.code
        res.attrs["base_unit"] = self._base_unit
        return res

    def __repr__(self) -> str:
        return f"Price({self._data!r}, quote_currency='{self._quote_currency.code}', base_unit='{self._base_unit}')"

    def __str__(self) -> str:
        return f"{self._data} {self._quote_currency.code}/{self._base_unit}"


__all__ = ["Price"]
