"""Moneta 标称/量纲张量统一抽象基类 (UnitArray)。

为具备单位/量纲标签的定点数张量（Money, Price, Quantity）提供统一的
底层 FixedArray 委托管理、NumPy 切片索引、高维张量变形、格式转换以及单位校验钩子。

约定：
- 不改变单位的运算（取负、绝对值、求和、切片、变形）由基类统一提供，结果保持原单位。
- ``==`` / ``!=`` 永不因单位不兼容而报错：不兼容即不相等；``< <= > >=`` 遇到单位不兼容的
  UnitArray 报 TypeError，遇到其他类型返回 NotImplemented（最终由 Python 报 TypeError）。
- 与 FixedArray 一样禁止 NumPy ufunc 与隐式转数组，避免被静默降级为 object 数组。
- 缺失值（NA）由底层 FixedArray 的掩码承载：运算按掩码传播；isna / notna / fillna / dropna 与
  sum(skipna=...) 由基类统一提供。步长、tick 等网格参数不能是缺失值。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import numpy as np

from .fixed_array import FixedArray


class UnitArray(ABC):
    """带单位/量纲的定点数张量抽象基类。

    所有子类均封装一个底层的 FixedArray 数据实例，并在外部附加金融量纲或业务标签（如 Currency、Unit）。
    统一提供维度检查、切片、张量变形、格式转换等纯数学操作，并暴露单位兼容性检查钩子。
    """

    __slots__ = ("_data",)
    __array_ufunc__ = None  # 阻止 NumPy ufunc 把 UnitArray 当作 object 逐元素运算
    _data: FixedArray

    # ---------- 核心抽象钩子 ----------

    @abstractmethod
    def _create_like(self, new_data: FixedArray) -> "UnitArray":
        """使用同单位/同配置与给定的新底层 FixedArray 创建同类型实例。"""
        pass

    @abstractmethod
    def _check_compatible(self, other: Any) -> None:
        """检查另一个对象是否与当前实例在单位/量纲上兼容。不兼容时抛出 TypeError 或 ValueError。"""
        pass

    @staticmethod
    def _as_factor(other: Any, what: str) -> FixedArray:
        """把无量纲操作数转为 FixedArray；只接受数值白名单内的类型（拒绝字符串、布尔与带单位对象）。"""
        factor = FixedArray._coerce(other, strict=False)
        if factor is None:
            raise TypeError(f"{what}只接受数值（int、float、Decimal、Fraction、FixedArray 或数值数组），"
                            f"收到 {type(other).__name__}")
        return factor

    @classmethod
    def _as_grid(cls, step: Any, what: str) -> FixedArray:
        """步长、tick 等网格参数：数值白名单内且不含缺失值。"""
        grid = cls._as_factor(step, what)
        if grid.mask is not None:
            raise ValueError(f"{what}不能是缺失值")
        return grid

    def _fill_value(self, value: Any) -> FixedArray:
        """fillna 的数值填充值 → FixedArray；Money 等有固定精度的子类覆盖它，按自己的构造规则换算。"""
        return self._as_factor(value, "填充值")


    # ---------- 底层数据与数组属性 ----------

    @property
    def data(self) -> FixedArray:
        """底层定点数张量数据 FixedArray。"""
        return self._data

    @property
    def mantissa(self) -> np.ndarray:
        """底层纯整数尾数矩阵（只读视图）。缺失位置为 0，请配合 isna() 使用。"""
        return self._data.mantissa

    @property
    def scale(self) -> int:
        """全局小数位数。"""
        return self._data.scale

    @property
    def shape(self) -> Tuple[int, ...]:
        """张量形状。"""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """张量维度数。"""
        return self._data.ndim

    @property
    def size(self) -> int:
        """元素总个数。"""
        return self._data.size

    @property
    def dtype(self) -> np.dtype:
        """底层整数数据类型（int64）。"""
        return self._data.dtype

    # ---------- 缺失值（NA） ----------

    def isna(self) -> np.ndarray:
        """标记缺失位置的布尔数组（0 维时为 0 维数组）。"""
        return self._data.isna()

    isnull = isna

    def notna(self) -> np.ndarray:
        """标记有效数值位置的布尔数组。"""
        return self._data.notna()

    notnull = notna

    def fillna(self, value: Any) -> "UnitArray":
        """用同类对象（单位须兼容）或数值填充缺失位置；数值按本类的构造规则换算，结果保持原单位。"""
        if isinstance(value, UnitArray):
            self._check_compatible(value)
            fill = value._data
        else:
            fill = self._fill_value(value)
        return self._create_like(self._data.fillna(fill))

    def dropna(self) -> "UnitArray":
        """丢弃缺失值（仅限 1 维），结果保持原单位。"""
        return self._create_like(self._data.dropna())

    # ---------- 索引、切片与容器协议 ----------

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: Any) -> "UnitArray":
        sliced_data = self._data[key]
        return self._create_like(sliced_data)

    def __iter__(self) -> Iterator["UnitArray"]:
        """按首维迭代，每个元素保持原单位；0 维禁止迭代（TypeError）。"""
        if self.ndim == 0:
            raise TypeError(f"iteration over a 0-d {type(self).__name__}")
        return (self._create_like(item) for item in self._data)

    def __bool__(self) -> bool:
        """真值测试：与 NumPy 一致，只有恰好 1 个元素时按值判断，其余情况报 ValueError。"""
        return bool(self._data)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """禁止隐式转成 NumPy 数组（np.asarray 等），避免带单位对象被降级为 object 数组。"""
        raise TypeError(f"{type(self).__name__} 不能隐式转成数组：要 float 请用 to_float()，要整数尾数请用 .mantissa")

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        """NumPy 函数白名单，规则同 FixedArray：其余函数报 TypeError，防止 np.array_equal 等静默出错。"""
        return FixedArray._array_function(func, args, kwargs, type(self).__name__)

    def reshape(self, *shape: Any) -> "UnitArray":
        """改变张量形状。"""
        return self._create_like(self._data.reshape(*shape))

    def transpose(self, *axes: int) -> "UnitArray":
        """矩阵/张量转置。"""
        return self._create_like(self._data.transpose(*axes))

    @property
    def T(self) -> "UnitArray":
        """转置快捷属性。"""
        return self.transpose()

    def swapaxes(self, axis1: int, axis2: int) -> "UnitArray":
        """交换两个轴。"""
        return self._create_like(self._data.swapaxes(axis1, axis2))

    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> "UnitArray":
        """移除长度为 1 的轴。"""
        return self._create_like(self._data.squeeze(axis=axis))

    # ---------- 不改变单位的运算 ----------

    def __pos__(self) -> "UnitArray":
        return self  # 不可变，直接返回自身

    def __neg__(self) -> "UnitArray":
        """取负，int64 最小值取负报 OverflowError。"""
        return self._create_like(-self._data)

    def __abs__(self) -> "UnitArray":
        """绝对值，int64 最小值报 OverflowError。"""
        return self._create_like(abs(self._data))

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False,
            skipna: bool = True, out: None = None, dtype: None = None) -> "UnitArray":
        """沿轴精确求和，结果保持原单位。``np.sum(x)`` 也会转到这里；skipna、out、dtype 规则同 FixedArray.sum。"""
        return self._create_like(self._data.sum(axis=axis, keepdims=keepdims, skipna=skipna, out=out, dtype=dtype))

    def is_zero(self) -> Union[np.bool_, np.ndarray]:
        """是否为 0：0 维返回 np.bool_，数组返回逐元素的 bool 数组；缺失值不是 0。"""
        return (self._data.mantissa == 0) & self._data.notna()

    # ---------- 比较操作符 (经由单位兼容性安检) ----------

    def _compatible_or_none(self, other: Any) -> bool:
        """单位兼容返回 True；不兼容（含非 UnitArray）返回 False。"""
        try:
            self._check_compatible(other)
        except (TypeError, ValueError):
            return False
        return True

    def __eq__(self, other: Any) -> Any:
        if not self._compatible_or_none(other):
            return NotImplemented  # 不兼容即不相等（Python 最终返回 False），并允许对方反向比较
        return self._data == other._data

    def __ne__(self, other: Any) -> Any:
        if not self._compatible_or_none(other):
            return NotImplemented
        return self._data != other._data

    def _ordered(self, other: Any, op: Callable[[FixedArray, FixedArray], Any]) -> Any:
        if not isinstance(other, UnitArray):
            return NotImplemented
        self._check_compatible(other)  # 单位不兼容的 UnitArray：由子类给出明确的 TypeError
        return op(self._data, other._data)

    def __lt__(self, other: Any) -> Any:
        return self._ordered(other, lambda a, b: a < b)

    def __le__(self, other: Any) -> Any:
        return self._ordered(other, lambda a, b: a <= b)

    def __gt__(self, other: Any) -> Any:
        return self._ordered(other, lambda a, b: a > b)

    def __ge__(self, other: Any) -> Any:
        return self._ordered(other, lambda a, b: a >= b)

    # ---------- 格式转换与数据分析桥梁 ----------

    def to_float(self) -> Union[float, np.ndarray]:
        """转换为 float64 标量或数组。"""
        return self._data.to_float()

    def to_decimal(self) -> Union[Decimal, np.ndarray]:
        """转换为 Python Decimal 对象。"""
        return self._data.to_decimal()

    def to_pandas(
            self,
            index: Optional[Any] = None,
            name: Optional[str] = None,
            as_float: bool = True,
    ) -> Any:
        """导出为 pandas.Series 或 pandas.DataFrame。"""
        return self._data.to_pandas(index=index, name=name, as_float=as_float)

    def to_polars(self, name: str = "value", as_float: bool = True) -> Any:
        """导出为 polars.Series。"""
        return self._data.to_polars(name=name, as_float=as_float)


__all__ = ["UnitArray"]
