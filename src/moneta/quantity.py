"""Moneta 标的物份额与物理数量张量体系 (Quantity)。

表示股票份额、期货合约数、大宗商品度量或加密货币持仓。
底层由定点数张量 FixedArray 承载，默认单位为 "share"；"Any" 为通配单位，可与任意单位运算。

约定：
- 数量没有固定精度：运算精确，乘法结果的 scale 为两侧之和；需要取整到整股、整手时显式调用
  multiply(scale=...)、rescale 或 round_to_step。
- 除以数值或同单位数量时，默认保留 max(两侧 scale) + 8 位小数，可用 scale 指定。
- 构造时指定 scale：float 按最短写法 half_up 舍入；精确输入超出 scale 时由 FixedArray 发出 PrecisionLossWarning 后舍入。
- 运算只接受数值白名单内的操作数，拒绝字符串；运算符遇到不认识的类型返回 NotImplemented。
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from ._rounding import check_rounding
from .fixed_array import FixedArray, RoundingMode
from .unit_array import UnitArray

_RATIO_EXTRA_DIGITS = 8  # 除法结果相对两侧 scale 额外保留的小数位数
WILDCARD_UNIT = "Any"

Rounding = Union[RoundingMode, str]


class Quantity(UnitArray):
    """标的物数量/份额张量（Quantity）。

    表示非货币标的物的物理份额或数量，量纲为资产单位（如 share、barrel、BTC）。单位区分大小写；
    通配单位 "Any" 可与任意单位相加、比较，结果取具体的那个单位。

    Parameters
    ----------
    values : int, float, str, Decimal, Fraction, FixedArray, Quantity, Sequence 或 np.ndarray
        数量数值或张量矩阵。
    unit : str, optional
        标的物单位，如 "share"、"barrel"、"contract"。默认为 "share"。
    scale : int, optional
        小数位数。默认按输入本身的精度；指定时 float 按最短写法 half_up 舍入，
        精确输入超出 scale 时发出 PrecisionLossWarning 后按 half_up 舍入，不足时补 0。
    """

    __slots__ = ("_unit",)

    def __init__(self, values: Any, unit: str = "share", scale: Optional[int] = None) -> None:
        unit = self._check_unit(unit)
        if isinstance(values, Quantity):
            if WILDCARD_UNIT not in (values._unit, unit) and values._unit != unit:
                raise TypeError(f"单位不匹配: {values._unit} 与 {unit}")
            data = values._data
        else:
            data = values
        # 指定 scale 时 float 静默舍入；精确输入丢精度时由 FixedArray 告警后舍入
        self._data = FixedArray(data, scale=scale, on_excess="warn")
        self._unit = unit

    # ---------- 核心抽象钩子实现 ----------

    def _create_like(self, new_data: FixedArray, unit: Optional[str] = None) -> "Quantity":
        """构建同资产单位（或指定单位）的新 Quantity 实例。"""
        obj = self.__class__.__new__(self.__class__)
        obj._data = new_data
        obj._unit = self._unit if unit is None else unit
        return obj

    def _check_compatible(self, other: Any) -> None:
        """检查另一个对象是否是同单位或通配单位的 Quantity。"""
        if not isinstance(other, Quantity):
            raise TypeError(f"操作数必须是 Quantity，得到 {type(other).__name__}")
        if WILDCARD_UNIT not in (self._unit, other._unit) and self._unit != other._unit:
            raise TypeError(f"单位不匹配: {self._unit} 与 {other._unit}")

    def _resolve_unit(self, other: "Quantity") -> str:
        """两个兼容的数量运算后的单位：优先取具体单位。"""
        return other._unit if self._unit == WILDCARD_UNIT else self._unit

    @property
    def unit(self) -> str:
        """标的物单位（默认为 'share'）。"""
        return self._unit

    # ---------- 加减 ----------

    def __add__(self, other: Any) -> "Quantity":
        """同单位或通配单位相加。单位不匹配报 TypeError；非 Quantity 返回 NotImplemented。"""
        if not isinstance(other, Quantity):
            return NotImplemented
        self._check_compatible(other)
        return self._create_like(self._data + other._data, self._resolve_unit(other))

    def __sub__(self, other: Any) -> "Quantity":
        """同单位或通配单位相减。单位不匹配报 TypeError；非 Quantity 返回 NotImplemented。"""
        if not isinstance(other, Quantity):
            return NotImplemented
        self._check_compatible(other)
        return self._create_like(self._data - other._data, self._resolve_unit(other))

    # ---------- 乘除 ----------

    def multiply(self, other: Any, scale: Optional[int] = None, rounding: Rounding = "half_up") -> "Quantity":
        """数量乘以无量纲数值：不指定 scale 时精确相乘（scale 为两侧之和），否则按 rounding 舍入到 scale。"""
        factor = self._as_factor(other, "乘数")
        mode = check_rounding(rounding)
        if scale is None:
            return self._create_like(self._data.mul_round(factor, rounding=None))
        return self._create_like(self._data.mul_round(factor, scale=scale, rounding=mode))

    def divide(self, other: Any, scale: Optional[int] = None, rounding: Rounding = "half_up") -> Union[
        "Quantity", FixedArray]:
        """数量除法：
        - Quantity / 无量纲数值 → Quantity；
        - Quantity / 同单位（或通配单位）Quantity → 无量纲比率 FixedArray。
        默认保留 max(两侧 scale) + 8 位小数，可用 scale 指定；按 rounding 舍入。
        """
        mode = check_rounding(rounding)
        if isinstance(other, Quantity):
            self._check_compatible(other)
            target = max(self.scale, other.scale) + _RATIO_EXTRA_DIGITS if scale is None else scale
            return self._data.divide(other._data, scale=target, rounding=mode)
        if isinstance(other, UnitArray):
            raise TypeError(f"数量只能除以数值或同单位的 Quantity，收到 {type(other).__name__}")
        divisor = self._as_factor(other, "除数")
        target = max(self.scale, divisor.scale) + _RATIO_EXTRA_DIGITS if scale is None else scale
        return self._create_like(self._data.divide(divisor, scale=target, rounding=mode))

    def __mul__(self, other: Any) -> "Quantity":
        """Quantity * 数值 → 精确的 Quantity；Quantity * Price 交给 Price 计算（结果为 Money）；其他返回 NotImplemented。"""
        if not FixedArray._is_numeric_operand(other):
            return NotImplemented
        return self.multiply(other)

    def __rmul__(self, other: Any) -> "Quantity":
        if not FixedArray._is_numeric_operand(other):
            return NotImplemented
        return self.multiply(other)

    def __truediv__(self, other: Any) -> Union["Quantity", FixedArray]:
        if not isinstance(other, Quantity) and not FixedArray._is_numeric_operand(other):
            return NotImplemented
        return self.divide(other)

    # ---------- 显式取整 ----------

    def rescale(self, scale: int, rounding: Rounding = "half_up") -> "Quantity":
        """调整小数位数：变小时按 rounding 舍入（如取整到整股），变大时补 0。"""
        return self._create_like(self._data.rescale(scale, rounding))

    def round_to_step(self, step: Any, rounding: Rounding = "half_up") -> "Quantity":
        """对齐到步长网格，例如按每手 100 股取整：round_to_step(100, "down")。"""
        return self._create_like(self._data.round_to_step(self._as_grid(step, "步长"), rounding=rounding))

    def is_on_step(self, step: Any) -> Union[bool, np.ndarray]:
        """检查数量是否严格位于步长网格上（如是否为整手；缺失值不在任何格点上）。"""
        return self._data.is_on_step(self._as_grid(step, "步长"))

    # ---------- 展示与转换 ----------

    def to_pandas(self, index: Optional[Any] = None, name: Optional[str] = None, as_float: bool = True) -> Any:
        """导出为 pandas.Series 或 pandas.DataFrame（自动注入 unit 元数据）。"""
        res = self._data.to_pandas(index=index, name=name, as_float=as_float)
        res.attrs["unit"] = self._unit
        return res

    def __repr__(self) -> str:
        return f"Quantity({self._data!r}, unit='{self._unit}')"

    def __str__(self) -> str:
        return f"{self._data} {self._unit}"


__all__ = ["Quantity", "WILDCARD_UNIT"]
