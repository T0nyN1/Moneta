"""定点数矩阵计算核心引擎（FixedArray）。

纯数学定点数矩阵系统：数学形式为 v = mantissa * 10**(-scale)。
保持 100% 纯整数底层运算，绝对消除 IEEE 754 浮点累积误差与精度漂移。
"""

from __future__ import annotations

import sys
import warnings
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from ._limbs import limbs_matmul, limbs_mul, limbs_round_div_pow10, limbs_sum, limbs_to_int64
from ._rounding import RoundingMode, check_rounding, decide_bump, round_quotient

# int64 边界常量
_MAX = (1 << 63) - 1
_MIN = -(1 << 63)
_ON_EXCESS = ("raise", "warn")


class PrecisionLossWarning(UserWarning):
    """精确输入（str、Decimal、Fraction、FixedArray 等）超出目标精度、被舍入时发出的警告。"""


def _user_stacklevel() -> int:
    """告警指向的栈层级：跳过 moneta 包内部的调用帧，指向用户代码。"""
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_globals.get("__name__", "").split(".")[0] == "moneta":
        frame, level = frame.f_back, level + 1
    return level


class FixedArray:
    """定点数高维矩阵核心引擎。

    底层由纯 int64 数组与整型 scale 支撑，支持标量与高维张量。
    完全遵循定点代数：加减法自动对齐 scale，乘法两端 scale 相加，除法保留额外商精度。

    Parameters
    ----------
    values : int, float, str, Decimal, Fraction, Sequence 或 np.ndarray
        输入数值或矩阵。
    scale : int, optional
        显式指定的小数位数。若为 None，标量按最短写法解析，float 数组逐位做往返检验。
    """

    __slots__ = ("_mantissa", "_scale", "_mask")
    __array_ufunc__ = None  # 阻止 NumPy ufunc 静默运算，确保精度与类型安全
    __hash__ = None  # 可变/数组对象不可哈希

    # ---------- 构造与核心属性 ----------

    def __init__(self, values: Any, scale: Optional[int] = None, mask: Optional[Any] = None,
                 on_excess: str = "raise") -> None:
        """构造 FixedArray。支持显式 scale、未知 scale 自动探测、字符串/对象解析及可选 mask 缺失值。

        指定 scale 时：float 是近似值，按最短写法 half_up 静默舍入；精确输入（str、Decimal、Fraction、
        FixedArray 等）在多出的小数位上有非零数字时按 on_excess 处理——"raise" 报 ValueError，
        "warn" 发出 PrecisionLossWarning 后按 half_up 舍入。多出的只是末尾的 0 时直接对齐，不报错也不告警。
        """
        if on_excess not in _ON_EXCESS:
            raise ValueError(f"on_excess 只能是 {_ON_EXCESS} 之一，收到 {on_excess!r}")
        if values is None or (isinstance(values, (float, np.floating)) and np.isnan(values)):
            s = 0 if scale is None else self._check_scale(scale)
            self._set(np.int64(0), s, mask=np.array(True, dtype=bool))
            return
        if scale is not None:
            scale = self._check_scale(scale)
        if isinstance(values, FixedArray):
            m, s = values._mantissa, values._scale
            if scale is not None:
                m, s = self._quantize_precision(m, s, scale, values._mask, on_excess), scale
            eff_mask = values._mask if mask is None else self._combine_masks(values._mask, np.asarray(mask, dtype=bool))
            self._set(m, s, mask=eff_mask)
            return

        try:
            arr = np.asarray(values)
        except TypeError:
            if not isinstance(values, (list, tuple)):
                raise
            # 序列中含 FixedArray 元素：NumPy 会调用元素的 __array__ 而报错，先换成等值的 Decimal
            arr = np.asarray(self._expand_nested(values))
        kind = arr.dtype.kind
        if kind in "iuf" and isinstance(values, (list, tuple)) and self._contains_bool(values):
            raise TypeError("FixedArray 不支持布尔类型")  # np.asarray 会把序列里的 bool 悄悄转成 0/1

        if kind in "iu":
            # 路径 1：整数
            s = 0 if scale is None else scale
            m = self._as_int64(arr)
            if s == 0 and isinstance(values, np.ndarray) and np.may_share_memory(m, values):
                m = m.copy()  # 不与调用方的数组共享内存，保证构造后不可变
            self._set(self._mul_pow10(m, s), s, mask=mask)
        elif kind == "f":
            if np.any(np.isinf(arr)):
                raise ValueError("FixedArray 不能包含无穷大 (inf)")
            nan_mask = np.isnan(arr)
            has_nan = bool(np.any(nan_mask))
            finite_arr = np.where(nan_mask, 0.0, arr) if has_nan else arr
            top = float(np.abs(finite_arr).max()) if finite_arr.size else 0.0
            if scale is not None and top and (scale > 400 or self._mul_pow10_float(top, scale) >= 2.0 ** 63):
                raise OverflowError(f"浮点数乘以 10^{scale} 后超出 int64 范围")  # 非零 float 至少 5e-324
            if finite_arr.ndim == 0:
                if has_nan:
                    m, s = np.int64(0), (0 if scale is None else scale)
                else:
                    m, s = self._parse_float_scalar(finite_arr[()])
                    m = self._as_int64(m)
            else:
                m, s = self._parse_float_array(finite_arr, target_scale=scale)
            if scale is not None:
                if s < scale:
                    m = self._mul_pow10(m, scale - s)
                elif s > scale:
                    m = self._round_div_pow10(m, 0, 1, s - scale, mode="half_up")
                s = scale
            eff_mask = nan_mask if mask is None else self._combine_masks(nan_mask, np.asarray(mask, dtype=bool))
            self._set(m, s, mask=eff_mask)
        elif kind in "UO":
            # 路径 5：字符串、Decimal、Fraction 或混合对象（标量与数组同一条路径）
            m, s, text_mask = self._parse_text_array(arr)
            if scale is not None:
                m, s = self._quantize_precision(m, s, scale, text_mask, on_excess), scale
            eff_mask = text_mask if mask is None else self._combine_masks(text_mask, np.asarray(mask, dtype=bool))
            self._set(m, s, mask=eff_mask)
        else:
            raise TypeError(f"不支持从 {type(values).__name__} (dtype={arr.dtype}) 构造 FixedArray")

    # ---------- 公共核心属性 ----------

    @property
    def mantissa(self) -> np.ndarray:
        """底层纯整数尾数矩阵（只读视图）。"""
        return self._mantissa

    @property
    def scale(self) -> int:
        """全局十进制小数位数。"""
        return self._scale

    @property
    def shape(self) -> Tuple[int, ...]:
        """矩阵维度。"""
        return self._mantissa.shape

    @property
    def ndim(self) -> int:
        """矩阵维度数。"""
        return self._mantissa.ndim

    @property
    def size(self) -> int:
        """元素总个数。"""
        return self._mantissa.size

    @property
    def dtype(self) -> np.dtype:
        """底层整数数据类型（int64）。"""
        return self._mantissa.dtype

    @property
    def mask(self) -> Optional[np.ndarray]:
        """缺失值掩码矩阵（只读布尔视图，True 表示缺失值）。无缺失值时为 None。"""
        return self._mask

    def isna(self) -> np.ndarray:
        """返回布尔矩阵，标记哪些位置是缺失值（Null / NA）。"""
        if self._mask is None:
            return np.zeros(self.shape, dtype=bool)
        return self._mask.copy()

    isnull = isna

    def notna(self) -> np.ndarray:
        """返回布尔矩阵，标记哪些位置是有效数值。"""
        return ~self.isna()

    notnull = notna

    def fillna(self, value: Any) -> "FixedArray":
        """用指定数值填充缺失位置，返回新的 FixedArray。填充值在需要填充的位置上不能是缺失值。"""
        if self._mask is None or not np.any(self._mask):
            return self
        val_fa = self._coerce(value)
        if val_fa._mask is not None and np.any(np.broadcast_to(val_fa._mask, self.shape) & self._mask):
            raise ValueError("fillna 的填充值不能是缺失值")
        val_aligned, self_aligned = self._align_scales(val_fa, self)
        new_m = np.where(self._mask, val_aligned._mantissa, self_aligned._mantissa)
        return self._from_mantissa(new_m, self_aligned._scale, mask=None)

    def dropna(self) -> "FixedArray":
        """丢弃缺失值（仅限 1 维向量）。"""
        if self.ndim != 1:
            raise ValueError("dropna() 仅支持 1 维 FixedArray")
        if self._mask is None:
            return self
        valid = ~self._mask
        return self._from_mantissa(self._mantissa[valid], self._scale, mask=None)

    # ---------- 构造收口与类型转换（私有） ----------

    def _set(self, mantissa: Any, scale: int, mask: Optional[Any] = None) -> None:
        """收口：所有构造路径最后都经过这里，是唯一给 _mantissa、_scale、_mask 赋值的地方。

        尾数统一转成 int64 数组，取视图后设为只读。
        若 mask 存在且包含 True，存储只读布尔掩码，并将缺失位置的尾数强制重置为 0；
        若无缺失值，_mask 保持为 None。
        """
        view = self._as_int64(mantissa).view()
        view.flags.writeable = False
        self._mantissa = view
        self._scale = int(scale)
        if mask is not None:
            m_arr = np.asarray(mask, dtype=bool)
            if np.any(m_arr):
                if m_arr.shape != view.shape:
                    m_arr = np.broadcast_to(m_arr, view.shape)
                m_view = m_arr.copy().view()
                m_view.flags.writeable = False
                self._mask = m_view
                if np.any(view[m_view] != 0):
                    unlocked = view.copy()
                    unlocked[m_view] = 0
                    unlocked.flags.writeable = False
                    self._mantissa = unlocked
                return
        self._mask = None

    @classmethod
    def _from_mantissa(cls, mantissa: Any, scale: int, mask: Optional[Any] = None) -> "FixedArray":
        """直接使用已有的整数尾数矩阵、scale 与可选 mask 构造（内部零拷贝快速路径）。

        int64 数组不复制：调用方之后不应再修改它。
        """
        obj = cls.__new__(cls)
        obj._set(mantissa, scale, mask=mask)
        return obj

    @staticmethod
    def _combine_masks(m1: Optional[np.ndarray], m2: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """合并两个掩码（True 表示缺失）：只要任一方缺失，结果即缺失。"""
        if m1 is None and m2 is None:
            return None
        if m1 is None:
            return m2.copy() if hasattr(m2, "ndim") and m2.ndim else np.asarray(m2).copy()
        if m2 is None:
            return m1.copy() if hasattr(m1, "ndim") and m1.ndim else np.asarray(m1).copy()
        res = m1 | m2
        if not np.any(res):
            return None
        return res

    @staticmethod
    def _matmul_mask(m1: Optional[np.ndarray], m2: Optional[np.ndarray],
                     s1: Tuple[int, ...], s2: Tuple[int, ...]) -> Optional[np.ndarray]:
        """计算矩阵乘法结果的掩码：只要参与内积的点乘中存在缺失值，结果对应单元格即为缺失值。"""
        if m1 is None and m2 is None:
            return None
        # bool 矩阵乘法即「内积中任一项为 True」，不存在计数回绕
        u1 = np.zeros(s1, dtype=bool) if m1 is None else np.broadcast_to(m1, s1)
        u2 = np.zeros(s2, dtype=bool) if m2 is None else np.broadcast_to(m2, s2)
        res = np.matmul(u1, np.ones(s2, dtype=bool)) | np.matmul(np.ones(s1, dtype=bool), u2)
        return res if np.any(res) else None

    @staticmethod
    def _is_numeric_operand(val: Any) -> bool:
        """运算操作数白名单：数值标量、数值 ndarray，或只含这些元素的 list / tuple / object 数组。

        FixedArray 不认识任何上层类：白名单之外的一切（字符串、布尔、带单位对象等）都不是合法操作数。
        """
        if isinstance(val, (bool, np.bool_)):
            return False
        if val is None:
            return True
        if isinstance(val, (FixedArray, int, np.integer, float, np.floating, Decimal, Fraction)):
            return True
        if isinstance(val, np.ndarray):
            if val.dtype.kind in "iuf":
                return True
            if val.dtype.kind == "O":
                return all(FixedArray._is_numeric_operand(x) for x in val.ravel())
            return False
        if isinstance(val, (list, tuple)):
            return all(FixedArray._is_numeric_operand(x) for x in val)
        return False

    @classmethod
    def _coerce(cls, other: Any, strict: bool = True) -> Optional["FixedArray"]:
        """运算的另一侧：转换为 FixedArray。

        只接受 ``_is_numeric_operand`` 白名单内的数值（FixedArray、int、float、Decimal、Fraction、
        数值 ndarray 及只含数值的序列）。字符串、布尔、带单位对象等一律拒绝，防止弱类型隐式转换。
        白名单内的值若本身非法（NaN、无限小数的 Fraction、超出 int64），照常抛出构造时的异常。

        Parameters
        ----------
        other : Any
            待转换的无量纲数值或矩阵。
        strict : bool, default True
            类型不在白名单时：True 抛出 TypeError；False 返回 None（供运算符返回 NotImplemented）。
        """
        if isinstance(other, FixedArray):
            return other
        if other is None:
            return cls(None)
        if not cls._is_numeric_operand(other):
            if strict:
                raise TypeError(f"FixedArray 只能与数值类型运算、比较或对齐，收到 {type(other).__name__}")
            return None
        return cls(other)

    @staticmethod
    def _quantize_precision(m: np.ndarray, s: int, scale: int, mask: Optional[np.ndarray],
                            on_excess: str) -> np.ndarray:
        """精确输入的尾数对齐到 scale：不足时补 0；多出的位上有非零数字（缺失位置除外）时按 on_excess
        报 ValueError，或发出 PrecisionLossWarning 后 half_up 舍入；多出的只是末尾的 0 时直接对齐。"""
        if s <= scale:
            return FixedArray._mul_pow10(m, scale - s)
        k = s - scale
        lost = (m != 0) if k > 18 else (m % (10 ** k) != 0)  # k > 18 时 |尾数| < 10^k，只有 0 能被整除
        if mask is not None:
            lost = lost & ~mask
        if np.any(lost):
            message = f"输入的小数位数 ({s}) 超出指定 scale ({scale})"
            if on_excess == "raise":
                raise ValueError(message)
            warnings.warn(f"{message}，已按 half_up 舍入", PrecisionLossWarning, stacklevel=_user_stacklevel())
        return FixedArray._round_div_pow10(m, 0, 1, k, "half_up")

    @staticmethod
    def _check_scale(scale: Any) -> int:
        """scale 必须是非负整数（int 或 NumPy 整数，不接受 bool、float）。"""
        if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, (int, np.integer)) or scale < 0:
            raise ValueError(f"scale 必须是非负整数，收到 {scale!r}")
        return int(scale)

    @staticmethod
    def _as_int64(x: Any) -> np.ndarray:
        """整数标量或数组 → int64 数组（0 维或多维），不复制已是 int64 的数组。

        非整数（含 bool、float、object）报 TypeError；超出 int64 报 OverflowError。
        """
        if isinstance(x, int) and not isinstance(x, bool):  # bool 是 int 的子类，交给下面报错
            if not _MIN <= x <= _MAX:
                raise OverflowError(f"{x} 超出 int64 范围")
            return np.asarray(x, dtype=np.int64)
        arr = np.asarray(x)
        if arr.dtype.kind not in "iu":
            raise TypeError(f"只接受整数，收到 {arr.dtype}")
        if arr.dtype == np.uint64 and arr.size and int(arr.max()) > _MAX:
            raise OverflowError("整数超出 int64 范围")
        return arr.astype(np.int64, copy=False)

    @classmethod
    def _align_scales(cls, a: "FixedArray", b: "FixedArray") -> Tuple["FixedArray", "FixedArray"]:
        """将两个 FixedArray 的 scale 自动对齐到 max(a.scale, b.scale)。"""
        if a._scale == b._scale:
            return a, b
        target = max(a._scale, b._scale)
        return a.rescale(target), b.rescale(target)

    # ---------- 输入解析（私有） ----------

    @staticmethod
    def _parse_decimal_str(s: str) -> Tuple[int, int]:
        """解析十进制字符串（支持科学计数法），返回 (mantissa, scale)。"""
        try:
            d = Decimal(s.strip())
        except Exception as e:
            raise ValueError(f"无法解析数值: {s!r}") from e
        if not d.is_finite():
            raise ValueError(f"FixedArray 不能接受非有限数值: {s!r}")

        sign, digits, exp = d.as_tuple()
        m = 0
        for digit in digits:
            m = m * 10 + digit
        if sign:
            m = -m
        if exp < 0:
            return m, -exp  # 声明的小数位数原样保留，"0.00" 也是 scale 2
        if m == 0:
            return 0, 0
        if exp > 19 or not (_MIN <= m * (10 ** exp) <= _MAX):  # 先判断位数，避免算出巨大的 10**exp
            raise OverflowError(f"{s!r} 超出 int64 范围")
        return m * (10 ** exp), 0

    @staticmethod
    def _parse_fraction(f: Fraction) -> Tuple[int, int]:
        """解析 Fraction 为有限十进制数（分母只含因子 2、5），否则报 ValueError。"""
        rest, twos, fives = f.denominator, 0, 0  # Fraction 的分母总是正数
        while rest % 2 == 0:
            rest //= 2
            twos += 1
        while rest % 5 == 0:
            rest //= 5
            fives += 1
        if rest != 1:
            raise ValueError(f"{f} 不是有限小数")
        scale = max(twos, fives)
        return f.numerator * 2 ** (scale - twos) * 5 ** (scale - fives), scale

    @classmethod
    def _parse_float_scalar(cls, val: Any) -> Tuple[int, int]:
        """按最短十进制写法（Dragonbox）解析单个浮点数，返回 (mantissa, scale)。

        NaN、无穷写成 "nan"、"inf"，由 _parse_decimal_str 报错。
        """
        # trim="-" 去掉末尾的 ".0"：1.0 → "1"，scale 为 0，与数组路径的往返检验一致
        return cls._parse_decimal_str(np.format_float_positional(val, unique=True, trim="-"))

    @staticmethod
    def _is_missing(val: Any) -> bool:
        """单个元素是否为缺失值：None、float NaN、Decimal NaN、文本 NaN（如 "nan"）、pandas.NA。"""
        if val is None:
            return True
        if isinstance(val, (float, np.floating)):
            return bool(np.isnan(val))
        if isinstance(val, Decimal):
            return val.is_nan()
        if isinstance(val, (str, np.str_)):
            if "n" not in val and "N" not in val:  # 绝大多数数字文本不含 n，省去一次 Decimal 解析
                return False
            try:
                return Decimal(val.strip()).is_nan()
            except Exception:
                return False  # 无法解析的文本交给解析函数报错
        return type(val).__name__ == "NAType"

    @staticmethod
    def _contains_bool(values: Any) -> bool:
        """list / tuple（可嵌套）中是否混有 bool 或 bool 数组。"""
        for v in values:
            if isinstance(v, (bool, np.bool_)) or (isinstance(v, np.ndarray) and v.dtype.kind == "b"):
                return True
            if isinstance(v, (list, tuple)) and FixedArray._contains_bool(v):
                return True
        return False

    @classmethod
    def _expand_nested(cls, values: Any) -> Any:
        """把 list / tuple（可嵌套）中的 FixedArray 换成等值的 Decimal（数组则为 Decimal 对象数组），保留小数位数。"""
        if isinstance(values, FixedArray):
            return values.to_decimal()
        if isinstance(values, (list, tuple)):
            return [cls._expand_nested(v) for v in values]
        return values

    @classmethod
    def _parse_object_element(cls, val: Any) -> Tuple[int, int]:
        """将单个标量（int、float、Fraction、str、Decimal、0 维 FixedArray 等）解析为 (mantissa, scale)。"""
        if isinstance(val, (bool, np.bool_)):
            raise TypeError("FixedArray 不支持布尔类型")
        if isinstance(val, FixedArray):
            if val.ndim:
                raise ValueError(f"对象数组的元素必须是标量，收到形状为 {val.shape} 的 FixedArray")
            return int(val.mantissa), val.scale
        if isinstance(val, (int, np.integer)):
            return int(val), 0
        if isinstance(val, (float, np.floating)):
            return cls._parse_float_scalar(val)
        if isinstance(val, Fraction):
            return cls._parse_fraction(val)
        return cls._parse_decimal_str(str(val))

    @classmethod
    def _parse_float_array(cls, arr: np.ndarray, target_scale: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """float 数组 → (尾数, scale)，与逐个元素取最短写法的结果一致。

        逐位试探 s = 0, 1, 2, …：m = rint(x·10^s)，若 m·10^-s 转回 x 的 float 类型后与 x
        完全相等（往返检验），x 的最短写法就有 s 位小数，必须在**首次通过**时记下 (m, s)——
        大数在更多小数位上也能往返，但那已不是最短写法。不设容差，大数的零头不会被当成噪声吞掉。

        |x·10^s| 超过 2^(nmant+1)（float64 为 2**53，float32 为 2**24）或 s 超过 22 时，
        rint(x·10^s) 不再可靠或已不是最短写法，这些元素另行处理：
        - target_scale 为 None：逐个解析最短写法，全列对齐到最大的小数位数；
        - 指定 target_scale：只试探到 target_scale + 1 位，其余元素由 ``_round_float_to_scale``
          按 half_up 舍入到 target_scale，不需要把全列对齐到全精度。
        """
        flat = arr.ravel()
        n = flat.size
        flat64 = flat.astype(np.float64, copy=False)
        max_abs = float(np.abs(flat64).max()) if n else 0.0
        mantissas = np.zeros(n, dtype=np.int64)
        scales = np.zeros(n, dtype=np.int64)
        recorded = np.zeros(n, dtype=bool)
        rest = []  # 无法用 float 试探确定的元素下标
        pending = None  # None 表示整列试探，此时不做下标筛选，省去复制
        max_probe = 22 if target_scale is None else min(22, target_scale + 1)
        exact_limit = 2.0 ** (np.finfo(arr.dtype).nmant + 1)  # 超过后相邻 float 间距 ≥ 1，最短写法可能省略整数末位
        s = 0
        while s <= max_probe:
            if target_scale is None and max_abs * 10.0 ** s > _MAX:  # 还有元素没通过，scale 至少是 s，最大的数必然溢出
                raise OverflowError(f"浮点数需要超过 {s - 1} 位小数，对齐后超出 int64 范围")
            values = flat if pending is None else flat[pending]
            values64 = flat64 if pending is None else flat64[pending]
            m = np.rint(values64 * 10.0 ** s)
            exact = np.abs(m) <= exact_limit
            ok = exact & ((m / 10.0 ** s).astype(arr.dtype, copy=False) == values)
            if pending is None:
                new = ok & ~recorded
                mantissas[new] = m[new]
                scales[new] = s
                recorded |= new
                remaining = ~recorded
                n_remaining = int(np.count_nonzero(remaining))
                if n_remaining == 0:
                    break
                if n_remaining > n // 10 and exact[remaining].all() and s < max_probe:
                    s += 1
                    continue
                # 剩余不多或出现无法精确试探的大数：改为只试探没通过的元素
                rest.append(np.flatnonzero(remaining & ~exact))
                pending = np.flatnonzero(remaining & exact)
            else:
                done = pending[ok]
                mantissas[done] = m[ok]
                scales[done] = s
                rest.append(pending[~exact])
                pending = pending[exact & ~ok]
            s += 1
            if not pending.size:
                break
        if pending is not None:
            rest.append(pending)
        rest = np.concatenate(rest) if rest else np.zeros(0, dtype=np.intp)

        if target_scale is not None:
            t = target_scale
            low = scales <= t  # 未记录的元素尾数为 0，结果随后被覆盖
            result = cls._align_mantissas(np.where(low, mantissas, 0), np.where(low, scales, t), t)
            high = scales == t + 1
            if np.any(high):
                result[high] = cls._round_div_pow10(mantissas[high], 0, 1, 1, "half_up")
            if rest.size:
                result[rest] = cls._round_float_to_scale(flat[rest], t)
            return result.reshape(arr.shape), t

        # float 运算无法精确判断的元素：按最短写法逐个解析（同值只解析一次）
        rest_parsed = []
        if rest.size:
            distinct, inverse = np.unique(flat[rest], return_inverse=True)
            parsed = [cls._parse_float_scalar(v) for v in distinct]
            rest_parsed = [parsed[i] for i in inverse]
        target = max(int(scales.max()) if n else 0, max((sc for _, sc in rest_parsed), default=0))
        result = cls._align_mantissas(mantissas, scales, target)
        for i, (m_val, sc) in zip(rest, rest_parsed):
            aligned = m_val * 10 ** (target - sc)
            if not _MIN <= aligned <= _MAX:
                raise OverflowError(f"浮点数对齐到 {target} 位小数后超出 int64 范围")
            result[i] = aligned
        return result.reshape(arr.shape), target

    @staticmethod
    def _align_mantissas(mantissas: np.ndarray, scales: np.ndarray, target: int) -> np.ndarray:
        """把各自 scale 不同的尾数对齐到 target（scales ≤ target），先整列判断溢出再整列相乘。"""
        shift = target - scales
        if np.any((mantissas != 0) & (shift > 18)):
            raise OverflowError(f"浮点数对齐到 {target} 位小数后超出 int64 范围")
        factor = np.power(10, np.minimum(shift, 18), dtype=np.int64)
        if np.any(np.abs(mantissas) > _MAX // factor):
            raise OverflowError(f"浮点数对齐到 {target} 位小数后超出 int64 范围")
        return mantissas * factor

    @classmethod
    def _round_float_to_scale(cls, values: np.ndarray, scale: int) -> np.ndarray:
        """把 float 的最短写法按 half_up 舍入到 scale 位，返回 int64 尾数。

        先用 float 算 z = x·10^scale。x 与最短写法之差不超过 x 的半个 ulp，放大后不超过
        |z|·eps/2；z 自身的计算误差不超过 |z|·2^-53（10^scale > 10^22 时再加同量级的一项）。
        只要 z 离最近的 k + 1/2 超过 |z|·(eps + 2^-50)（两倍余量），x 与最短写法必在 k + 1/2
        同侧，二者舍入结果相同，取 floor(z) 或 floor(z) + 1 即可。
        贴近 k + 1/2、|z| ≥ 2^51 或 x 为次正规数（ulp 不再受 |x|·eps 约束）的少数元素，
        逐个解析最短写法后用整数精确舍入。
        """
        z = cls._mul_pow10_float(values.astype(np.float64, copy=False), scale)
        floor = np.floor(z)
        frac = z - floor  # |z| < 2^52 时精确
        info = np.finfo(values.dtype)
        margin = np.abs(z) * (float(info.eps) + 2.0 ** -50)
        normal = (values == 0) | (np.abs(values) >= info.tiny)
        safe = normal & (np.abs(z) < 2.0 ** 51) & (np.abs(frac - 0.5) > margin)
        out = np.where(safe, floor + (frac > 0.5), 0.0).astype(np.int64)

        unsafe = np.flatnonzero(~safe)
        if unsafe.size:
            distinct, inverse = np.unique(values[unsafe], return_inverse=True)
            exact = []
            for v in distinct:
                m, s = cls._parse_float_scalar(v)
                if s > scale:
                    m = int(cls._round_div_pow10(np.int64(m), 0, 1, s - scale, "half_up"))
                else:
                    m = m * 10 ** (scale - s)
                    if not _MIN <= m <= _MAX:
                        raise OverflowError(f"浮点数乘以 10^{scale} 后超出 int64 范围")
                exact.append(m)
            out[unsafe] = np.array(exact, dtype=np.int64)[inverse]
        return out

    @classmethod
    def _parse_text_array(cls, arr: np.ndarray) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
        """文本、Decimal、Fraction 或混合对象数组（含 0 维）→ (对齐后的尾数, scale, mask)。"""
        if arr.ndim == 0:  # 标量直接解析，省去去重的开销
            val = arr[()]
            if cls._is_missing(val):
                return cls._as_int64(0), 0, np.array(True, dtype=bool)
            m, scale = cls._parse_object_element(val)
            return cls._as_int64(m), scale, None
        flat = arr.ravel()
        null_mask = None
        if arr.dtype.kind == "U":
            # 字符串可排序：先去重，只解析不同的取值；文本 NaN 是缺失值
            distinct, inverse = np.unique(flat, return_inverse=True)
            missing = np.array([cls._is_missing(str(v)) for v in distinct], dtype=bool)
            parsed = [(0, 0) if miss else cls._parse_decimal_str(str(v)) for v, miss in zip(distinct, missing)]
            if missing.any():
                null_mask = missing[inverse].reshape(arr.shape)
        else:
            is_null = np.array([cls._is_missing(v) for v in flat], dtype=bool)
            if np.any(is_null):
                null_mask = is_null.reshape(arr.shape)
            index = {}
            parsed = []
            inverse = np.empty(flat.size, dtype=np.intp)
            for i, val in enumerate(flat):
                if is_null[i]:
                    key = ("NoneType", "None")
                    j = index.get(key)
                    if j is None:
                        j = index[key] = len(parsed)
                        parsed.append((0, 0))
                    inverse[i] = j
                else:
                    key = (type(val).__name__, str(val))
                    j = index.get(key)
                    if j is None:
                        j = index[key] = len(parsed)
                        parsed.append(cls._parse_object_element(val))
                    inverse[i] = j
        scale = max((s for _, s in parsed), default=0)
        aligned = []
        for m, s in parsed:
            if m and (scale - s > 18 or not (_MIN <= m * 10 ** (scale - s) <= _MAX)):  # 先判断位数，避免巨大的幂
                raise OverflowError(f"输入对齐到 {scale} 位小数后超出 int64 范围")
            aligned.append(m * 10 ** (scale - s) if m else 0)
        res_m = np.array(aligned, dtype=np.int64)[inverse].reshape(arr.shape)
        if null_mask is not None:
            res_m[null_mask] = 0
        return res_m, scale, null_mask

    # ---------- 底层数学与防溢出舍入引擎（私有） ----------

    @staticmethod
    def _check_division(m: np.ndarray, d: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """除数不能含 0；-2**63 除以 -1 的商超出 int64。跳过被 mask 标记为缺失的位置。"""
        valid_zero = (d == 0) if mask is None else ((d == 0) & (~mask))
        if np.any(valid_zero):
            raise ZeroDivisionError("除数包含 0")
        valid_overflow = ((d == -1) & (m == _MIN)) if mask is None else (((d == -1) & (m == _MIN)) & (~mask))
        if np.any(valid_overflow):
            raise OverflowError("-2**63 除以 -1 超出 int64 范围")

    @staticmethod
    def _max_abs(x: np.ndarray) -> int:
        """绝对值的最大值（Python int，-2**63 也不溢出）；空数组安全返回 0。"""
        return max(-int(x.min()), int(x.max())) if x.size else 0

    @staticmethod
    def _mul_pow10(x: np.ndarray, k: int) -> np.ndarray:
        """x·10^k，先用上下界判断，可能越界就报 OverflowError。"""
        if k == 0:
            return x
        top = FixedArray._max_abs(x)
        if top == 0:
            return np.zeros_like(x)
        if k > 18 or top > _MAX // 10 ** k:
            raise OverflowError(f"乘以 10^{k} 后超出 int64 范围")
        return x * 10 ** k

    @staticmethod
    def _mul_pow10_float(x: Any, scale: int) -> Any:
        """x·10^scale（float），scale 超过 300 时分两步乘，避免 10.0 ** scale 本身溢出。"""
        return x * 10.0 ** min(scale, 300) * 10.0 ** max(scale - 300, 0)

    @staticmethod
    def _divmod_round(m: Any, d: Any, mode: str = "half_up") -> Any:
        """整数带余除法舍入：round(m / d)，全程 int64，结果精确。

        先做向下取整的带余除法 ``m = q·d + r``，再由 ``round_quotient`` 决定 q 是否加 1。

        Parameters
        ----------
        m, d : int 或整数数组
            被除数与除数，必须在 int64 范围内，可广播；d 可正可负，不能为 0。
        mode : {"half_up", "half_even", "down", "up", "floor", "ceil"}
            ``half_up`` 四舍五入、恰好一半时远离零；``half_even`` 银行家舍入；
            ``down`` 向零；``up`` 远离零；``floor`` 向负无穷；``ceil`` 向正无穷。

        Returns
        -------
        int 或 numpy.ndarray
            两个参数都是整数标量时返回 Python int，否则返回 int64 数组。

        Raises
        ------
        ValueError
            mode 无效。
        ZeroDivisionError
            除数含 0。
        OverflowError
            输入超出 int64 范围，或 -2**63 除以 -1。
        """
        check_rounding(mode)
        scalar = isinstance(m, (int, np.integer)) and isinstance(d, (int, np.integer))
        m, d = FixedArray._as_int64(m), FixedArray._as_int64(d)
        FixedArray._check_division(m, d)
        q, r = np.divmod(m, d)
        q = round_quotient(q, r, d, mode)
        return int(q) if scalar else np.asarray(q)

    @staticmethod
    def _doubling_step_10r(r: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """无溢出单步倍增：计算 (r * 10) // d 与 (r * 10) % d。
        利用 10r = 8r + 2r 与 2r < 2d <= 2^64（放得进 uint64）消除中间值溢出。
        """
        d_neg = d < 0
        with np.errstate(over="ignore"):
            # 补码转换防 -2**63 在有符号取负时溢出
            d_u = np.where(d_neg, (~d.astype(np.uint64)) + np.uint64(1), d.astype(np.uint64))
            r_u = np.where(d_neg, (~r.astype(np.uint64)) + np.uint64(1), r.astype(np.uint64))

            def _step2(cur_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                r2 = cur_r << np.uint64(1)
                q = (r2 >= d_u).astype(np.uint64)
                return q, r2 - q * d_u

            q1, r1 = _step2(r_u)  # 2r
            q2, r2 = _step2(r1)  # 4r
            q3, r3 = _step2(r2)  # 8r

            q_8 = (q1 << np.uint64(2)) | (q2 << np.uint64(1)) | q3
            q_10 = q_8 + q1
            r_sum = r3 + r1
            adj = (r_sum >= d_u).astype(np.uint64)

            digits = (q_10 + adj).astype(np.int64)
            new_r_u = r_sum - adj * d_u
            new_r_neg = ((~new_r_u) + np.uint64(1)).view(np.int64)
            new_r = np.where(d_neg, new_r_neg, new_r_u.view(np.int64))
        return digits, new_r

    @staticmethod
    def _chunked_divmod_round(m: np.ndarray, d: np.ndarray, k: int, mode: str) -> np.ndarray:
        """round(m·10^k / d)，全程 int64 的分段长除法；只有结果本身超出 int64 时才报错。

        统一应对两类极端情况：
        1. 被除数/放大倍数极大 (m·10^k 溢出)：通过外层长除法循环逐段推进 k。
        2. 除数极大 (|d| > MAX // 10)：普通 r * 10 会溢出，自动切换至无溢出二进制倍增内核 (10r = 8r + 2r)。
        """
        FixedArray._check_division(m, d)
        d_top = FixedArray._max_abs(d)
        is_huge = d_top > _MAX // 10

        if not is_huge:
            chunk = 1
            while chunk < 18 and d_top * 10 ** (chunk + 1) <= _MAX:
                chunk += 1
        else:
            chunk = 1

        q, r = np.divmod(m, d)
        while k:
            c = 1 if is_huge else min(chunk, k)
            p = 10 ** c

            if is_huge:
                digits, r = FixedArray._doubling_step_10r(r, d)
            else:
                digits, r = np.divmod(r * p, d)

            # 统一的精确溢出边界检查 (解决 B.3)
            up, lo = _MAX // p, _MIN // p
            ok = np.where(q >= 0, (q < up) | ((q == up) & (digits <= _MAX - up * p)),
                          (q > lo) | ((q == lo) & (digits >= _MIN - lo * p)))
            if not np.all(ok):
                raise OverflowError("除法结果超出 int64 范围")

            with np.errstate(over="ignore"):
                q = q * p + digits
            k -= c

        # 终局舍入
        if np.any(q == _MAX):
            rounded = round_quotient(q.astype(object), r, d, mode)
            if np.any(rounded > _MAX):
                raise OverflowError("除法舍入后超出 int64 范围")
            return np.asarray(rounded, dtype=np.int64)
        return round_quotient(q, r, d, mode)

    @staticmethod
    def _round_div_pow10(q: Any, r: Any, d: Any, k: int, mode: str) -> Any:
        """round((q + r/d) / 10^k)。q 是整数部分，r/d in [0, 1) 是额外的小数（r 与 d 同号）。
        支持标量与数组，全程原生 int64/uint64 整数运算。
        """
        if k == 0:
            return round_quotient(q, r, d, mode)

        q = np.asarray(q, dtype=np.int64)
        r = np.asarray(r, dtype=np.int64)
        d = np.asarray(d, dtype=np.int64)

        if k <= 18:
            p = 10 ** k
            q_k, r_k = np.divmod(q, p)
            delta = 2 * r_k - p
            r_gt_half = np.where(d > 0, r > d - r, r < d - r)
            r_eq_half = (r == d - r)
            has_fraction = (r_k != 0) | (r != 0)
            over_half = (delta > 0) | ((delta == 0) & (r != 0)) | ((delta == -1) & r_gt_half)
            exact_half = ((delta == 0) & (r == 0)) | ((delta == -1) & r_eq_half)
        else:
            # |q| <= 2**63 < 10**19 <= 10**k
            has_fraction = (q != 0) | (r != 0)
            q_neg = (q < 0)
            q_k = np.where(q_neg, -1, 0)
            if k == 19:
                half_19 = np.int64(5_000_000_000_000_000_000)
                over_pos = (q > half_19) | ((q == half_19) & (r != 0))
                exact_pos = (q == half_19) & (r == 0)
                over_neg = (q > -half_19) | ((q == -half_19) & (r != 0))
                exact_neg = (q == -half_19) & (r == 0)
                over_half = np.where(q_neg, over_neg, over_pos)
                exact_half = np.where(q_neg, exact_neg, exact_pos)
            else:
                over_half = q_neg
                exact_half = np.zeros_like(q_neg, dtype=bool)

        bump = decide_bump(
            mode,
            is_negative=q_k < 0,
            odd_q=q_k % 2 != 0,
            has_fraction=has_fraction,
            over_half=over_half,
            exact_half=exact_half,
            floor_base=True,
        )
        return q_k + bump

    @staticmethod
    def _limbs_to_mantissa(limbs: Any, shift: int, mode: str) -> np.ndarray:
        """精确分块结果 → 目标 scale 的 int64 尾数：shift > 0 时在分块上除以 10^shift 并只舍入一次，
        shift ≤ 0 时先收口为 int64 再放大（溢出判断复用 _mul_pow10）。只有真实结果超出 int64 才报错。"""
        if shift > 0:
            return limbs_to_int64(limbs_round_div_pow10(limbs, shift, mode))
        return FixedArray._mul_pow10(limbs_to_int64(limbs), -shift)

    # ---------- 代数运算与操作符重载（公开 API） ----------

    def __add__(self, other: Any) -> "FixedArray":
        """加法：自动对齐 scale，纯整数相加，溢出抛出 OverflowError。"""
        b = self._coerce(other, strict=False)
        if b is None:
            return NotImplemented
        a, b_aligned = self._align_scales(self, b)
        out_mask = self._combine_masks(a._mask, b_aligned._mask)
        res = a._mantissa + b_aligned._mantissa
        overflow = ((a._mantissa ^ res) & (b_aligned._mantissa ^ res)) < 0
        if out_mask is not None:
            overflow = overflow & (~out_mask)
        if np.any(overflow):
            raise OverflowError("加法结果超出 int64 范围")
        return self._from_mantissa(res, a._scale, mask=out_mask)

    def __radd__(self, other: Any) -> "FixedArray":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "FixedArray":
        """减法：自动对齐 scale，纯整数相减，溢出抛出 OverflowError。"""
        b = self._coerce(other, strict=False)
        if b is None:
            return NotImplemented
        a, b_aligned = self._align_scales(self, b)
        out_mask = self._combine_masks(a._mask, b_aligned._mask)
        res = a._mantissa - b_aligned._mantissa
        overflow = ((a._mantissa ^ b_aligned._mantissa) & (a._mantissa ^ res)) < 0
        if out_mask is not None:
            overflow = overflow & (~out_mask)
        if np.any(overflow):
            raise OverflowError("减法结果超出 int64 范围")
        return self._from_mantissa(res, a._scale, mask=out_mask)

    def __rsub__(self, other: Any) -> "FixedArray":
        b = self._coerce(other, strict=False)
        if b is None:
            return NotImplemented
        return b.__sub__(self)

    def __pos__(self) -> "FixedArray":
        """一元正号 (+x)，FixedArray 为不可变对象，直接返回自身。"""
        return self

    def __neg__(self) -> "FixedArray":
        """取负，溢出抛出 OverflowError。"""
        overflow = (self._mantissa == _MIN)
        if self._mask is not None:
            overflow = overflow & (~self._mask)
        if np.any(overflow):
            raise OverflowError("取负导致数值超出 int64 范围")
        return self._from_mantissa(-self._mantissa, self._scale, mask=self._mask)

    def __abs__(self) -> "FixedArray":
        """绝对值，溢出抛出 OverflowError。"""
        overflow = (self._mantissa == _MIN)
        if self._mask is not None:
            overflow = overflow & (~self._mask)
        if np.any(overflow):
            raise OverflowError("绝对值导致数值超出 int64 范围")
        return self._from_mantissa(np.abs(self._mantissa), self._scale, mask=self._mask)

    def mul_round(self, other: Any, scale: Optional[int] = None,
                  rounding: Optional[Union[RoundingMode, str]] = "half_up") -> "FixedArray":
        """融合乘法与可选舍入 (Fused Multiply-Round)，全程纯 int64。

        - 当 ``rounding=None`` 时：不做舍入（精确相乘），默认 ``scale = self.scale + other.scale``；
        - 当 ``rounding`` 非 None 时（默认 ``"half_up"``）：按 ``rounding`` 舍入到目标 ``scale``
          （若 ``scale=None``，默认保留 ``max(self.scale, other.scale)`` 位小数）。

        内部在快速通道不溢出时直接单指令相乘，否则采用二进制 Base 2^21 分块长乘法，
        中间运算完全维持在 int64 原生整数内，仅当最终结果超出 int64 范围时才抛出 OverflowError。

        Parameters
        ----------
        other : int, float, Decimal, Sequence 或 FixedArray
            乘数。
        scale : Optional[int], optional
            目标输出的 scale。若为 None：``rounding=None`` 时为 ``self.scale + other.scale``，
            否则为 ``max(self.scale, other.scale)``。
        rounding : {"half_up", "half_even", "down", "up", "floor", "ceil"} 或 None，默认 "half_up"
            取整方式；传入 None 表示不做舍入。

        Returns
        -------
        FixedArray
            目标 scale 的定点数矩阵。
        """
        other_fa = self._coerce(other)
        natural_scale = self._scale + other_fa._scale
        if rounding is None:
            target_scale = natural_scale if scale is None else self._check_scale(scale)
            if target_scale < natural_scale:
                raise ValueError("rounding=None 时不支持缩小 scale（需要指定舍入模式）")
            mode = "half_up"
        else:
            mode = check_rounding(rounding)
            target_scale = max(self._scale, other_fa._scale) if scale is None else self._check_scale(scale)

        out_mask = self._combine_masks(self._mask, other_fa._mask)
        shift = natural_scale - target_scale

        # 快速通道：若两者全局极值之积在安全范围内且无需额外放大，直接单指令乘法与除法舍入
        if 0 <= shift <= 18 and self._max_abs(self._mantissa) * self._max_abs(other_fa._mantissa) <= _MAX:
            prod = np.multiply(self._mantissa, other_fa._mantissa)
            res_m = prod if shift == 0 else self._divmod_round(prod, 10 ** shift, mode=mode)
            return self._from_mantissa(res_m, target_scale, mask=out_mask)

        # 纯 int64 分块长乘法通道
        res_m = self._limbs_to_mantissa(limbs_mul(self._mantissa, other_fa._mantissa), shift, mode)
        return self._from_mantissa(res_m, target_scale, mask=out_mask)

    def __mul__(self, other: Any) -> "FixedArray":
        """乘法：默认保留 a.scale + b.scale 位精确小数，溢出抛出 OverflowError。"""
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return self.mul_round(other_fa, rounding=None)

    def __rmul__(self, other: Any) -> "FixedArray":
        return self.__mul__(other)

    def matmul(self, other: Any, scale: Optional[int] = None,
               rounding: Optional[Union[RoundingMode, str]] = "half_up") -> "FixedArray":
        """矩阵乘法与可选舍入 (Matrix Multiplication / Fused Matmul-Round)，全程纯 int64。

        - 当 ``rounding=None`` 时：不做舍入（精确内积），默认 ``scale = self.scale + other.scale``；
        - 当 ``rounding`` 非 None 时（默认 ``"half_up"``）：按 ``rounding`` 舍入到目标 ``scale``
          （若 ``scale=None``，默认保留 ``max(self.scale, other.scale)`` 位小数）。

        内部在最坏上界不溢出时直接走原生矩阵乘法，否则采用二进制 Base 2^21 分块矩阵乘法，
        中间运算完全维持在 int64 原生整数内，仅当最终结果超出 int64 范围时才抛出 OverflowError。

        Parameters
        ----------
        other : Sequence, ndarray 或 FixedArray
            乘数矩阵/向量。
        scale : Optional[int], optional
            目标输出的 scale。若为 None：``rounding=None`` 时为 ``self.scale + other.scale``，
            否则为 ``max(self.scale, other.scale)``。
        rounding : {"half_up", "half_even", "down", "up", "floor", "ceil"} 或 None，默认 "half_up"
            取整方式；传入 None 表示不做舍入。

        Returns
        -------
        FixedArray
            目标 scale 的矩阵相乘结果。

        Raises
        ------
        ValueError
            rounding=None 时传入了比自然 scale 更小的 scale，或 rounding 不合法。
        OverflowError
            结果真实超出 int64 范围。
        """
        other_fa = self._coerce(other)
        natural_scale = self._scale + other_fa._scale
        if rounding is None:
            target_scale = natural_scale if scale is None else self._check_scale(scale)
            if target_scale < natural_scale:
                raise ValueError("rounding=None 时不支持缩小 scale（需要指定舍入模式）")
            mode = "half_up"
        else:
            mode = check_rounding(rounding)
            target_scale = max(self._scale, other_fa._scale) if scale is None else self._check_scale(scale)

        a, b = self._mantissa, other_fa._mantissa
        out_mask = self._matmul_mask(self._mask, other_fa._mask, a.shape, b.shape)
        terms = a.shape[-1] if a.ndim else 0
        if self._max_abs(a) * self._max_abs(b) * terms <= _MAX:
            result = np.matmul(a, b)  # 形状不合法时由 NumPy 报错
            raw_fa = self._from_mantissa(result, natural_scale, mask=out_mask)
            if target_scale == natural_scale:
                return raw_fa
            return raw_fa.rescale(target_scale, rounding=mode)

        # 慢速保底路径：基于 _limbs 分段计算精确内积；降精度时直接在分块上除以 10^k 并舍入，中间值不受 int64 限制
        shift = natural_scale - target_scale
        res_m = self._limbs_to_mantissa(limbs_matmul(a, b), shift, mode)
        return self._from_mantissa(res_m, target_scale, mask=out_mask)

    def __matmul__(self, other: Any) -> "FixedArray":
        """矩阵乘法（@ 运算符），scale = 两者之和，全程 int64。"""
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return self.matmul(other_fa, rounding=None)

    def __rmatmul__(self, other: Any) -> "FixedArray":
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return other_fa.matmul(self, rounding=None)

    def divide(self, other: Any, scale: Optional[int] = None, rounding: str = "half_up", ) -> "FixedArray":
        """定点数除法：商保留 scale 位小数（默认 ``max(self.scale, other.scale)``），按 rounding 取整。

        全程 int64：商的小数位数比自然精度多时走分段长除法（大除数自动切换至二进制倍增内核）；
        比自然精度少时对整数商与余数直接除以 10 的幂舍入，只有最终商超出 int64 才报错。

        Parameters
        ----------
        other : FixedArray 或可构造 FixedArray 的值
            除数，可广播。
        scale : int，可选
            商的小数位数，默认 ``max(self.scale, other.scale)``。
        rounding : {"half_up", "half_even", "down", "up", "floor", "ceil"} 或 RoundingMode，默认 "half_up"
            取整方式。

        Raises
        ------
        ValueError
            rounding 无效或 scale 不是非负整数。
        ZeroDivisionError
            除数含 0。
        OverflowError
            结果超出 int64 范围。
        """
        mode = check_rounding(rounding)
        other = self._coerce(other)
        target = max(self._scale, other._scale) if scale is None else self._check_scale(scale)
        out_mask = self._combine_masks(self._mask, other._mask)
        num, den = self._mantissa, other._mantissa
        if out_mask is not None:
            num = np.where(out_mask, np.int64(0), num)
            den = np.where(out_mask, np.int64(1), den)
        # 商的尾数 = round(a·10^(target + other.scale - self.scale) / b)
        shift = target + other._scale - self._scale
        if shift > 0:
            quotient = self._chunked_divmod_round(num, den, shift, mode)  # 内部检查除零与 MIN / -1
        elif shift == 0:
            self._check_division(num, den)
            q, r = np.divmod(num, den)
            quotient = round_quotient(q, r, den, mode)
        else:  # 结果还要再除以 10^k：MIN / -1 放得下，只检查除零
            if np.any(den == 0):
                raise ZeroDivisionError("除数包含 0")
            k = -shift
            special = (num == _MIN) & (den == -1)
            q, r = np.divmod(num, np.where(special, 1, den))
            quotient = self._round_div_pow10(q, r, np.where(special, 1, den), k, mode)
            if np.any(special):
                # round(x / -1) = -round'(x)，round' 为 floor 与 ceil 互换后的模式（其余模式关于 0 对称）
                mirrored = {"floor": "ceil", "ceil": "floor"}.get(mode, mode)
                quotient = np.where(special, -self._round_div_pow10(num, 0, 1, k, mirrored), quotient)
        return self._from_mantissa(quotient, target, mask=out_mask)

    def __truediv__(self, other: Any) -> "FixedArray":
        """真除法：默认保留 max(self.scale, other.scale) 位商精度，half_up 舍入。"""
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return self.divide(other_fa)

    def __rtruediv__(self, other: Any) -> "FixedArray":
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return other_fa.divide(self)

    def __floordiv__(self, other: Any) -> "FixedArray":
        """向下取整的商（无量纲整数）：返回 scale 为 0 的 FixedArray，缺失值按掩码传播。

        Raises
        ------
        ZeroDivisionError
            除数含 0。
        OverflowError
            商超出 int64 范围（-2**63 除以 -1）。
        """
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        a, b = self._align_scales(self, other_fa)
        out_mask = self._combine_masks(a._mask, b._mask)
        self._check_division(a._mantissa, b._mantissa, mask=out_mask)
        num = np.where(out_mask, np.int64(0), a._mantissa) if out_mask is not None else a._mantissa
        den = np.where(out_mask, np.int64(1), b._mantissa) if out_mask is not None else b._mantissa
        return self._from_mantissa(num // den, 0, mask=out_mask)

    def __rfloordiv__(self, other: Any) -> Any:
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return other_fa.__floordiv__(self)

    def __mod__(self, other: Any) -> "FixedArray":
        """余数，与除数同号，满足 ``a == (a // b) * b + (a % b)``。"""
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        a, b = self._align_scales(self, other_fa)
        out_mask = self._combine_masks(a._mask, b._mask)
        valid_zero = (b._mantissa == 0) if out_mask is None else ((b._mantissa == 0) & (~out_mask))
        if np.any(valid_zero):
            raise ZeroDivisionError("除数包含 0")
        den = np.where(out_mask, np.int64(1), b._mantissa) if out_mask is not None else b._mantissa
        res = a._mantissa % den
        return self._from_mantissa(res, a._scale, mask=out_mask)

    def __rmod__(self, other: Any) -> "FixedArray":
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return other_fa.__mod__(self)

    def __divmod__(self, other: Any) -> Tuple["FixedArray", "FixedArray"]:
        """(商, 余数)：商同 ``//``，余数同 ``%``。"""
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return self.__floordiv__(other_fa), self.__mod__(other_fa)

    def __rdivmod__(self, other: Any) -> Tuple["FixedArray", "FixedArray"]:
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return other_fa.__divmod__(self)

    # ---------- 张量规约统计（公开 API） ----------

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False,
            skipna: bool = True, out: None = None, dtype: None = None) -> "FixedArray":
        """沿轴求和，全程 int64。``np.sum(fa)`` 也会转到这里。

        精确溢出判定：快速路径在最坏情况不超过 int64 时直接计算；超出时使用 Base 2^21
        分块累加判定真实结果，只有最终结果真实超出 int64 时才报错。
        全程维持在 int64 原生整数内，不升级为 Python 任意精度大整数。

        Parameters
        ----------
        axis : int 或 int 元组，可选
            默认对全部元素求和。
        keepdims : bool，可选
            保留被求和的轴。
        skipna : bool, default True
            是否忽略缺失值 (NA)。为 True 时缺失元素视为 0；为 False 时若切片包含缺失值则结果为 NA。
        out, dtype : None
            只为兼容 ``np.sum`` 的调用约定；传入非 None 报 TypeError。要 float 请先
            ``to_float()``。

        Raises
        ------
        OverflowError
            结果真实超出 int64 范围。
        """
        if out is not None or dtype is not None:
            raise TypeError("FixedArray.sum 不支持 out、dtype；要 float 请先 to_float()")

        if self._mask is None:
            out_mask = None
            eff_m = self._mantissa
        else:
            if skipna:
                out_mask = None
                eff_m = self._mantissa  # 缺失值在 _set 中已严格恒为 0
            else:
                any_na = self._mask.any(axis=axis, keepdims=keepdims)
                out_mask = any_na if np.any(any_na) else None
                eff_m = self._mantissa

        total = eff_m.sum(axis=axis, keepdims=keepdims)
        count = eff_m.size // max(np.size(total), 1)  # 每个结果单元格累加的元素个数
        if self._max_abs(eff_m) * count <= _MAX:
            res_m = total
        else:
            # 慢速保底路径：纯 int64 Base 2^21 分块累加判定真实求和，消除正负对冲假阳性误报
            res_m = limbs_sum(eff_m, axis=axis, keepdims=keepdims)

        if out_mask is not None:
            res_m = np.where(out_mask, np.int64(0), res_m)
        return self._from_mantissa(res_m, self._scale, mask=out_mask)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False,
             scale: Optional[int] = None, rounding: Union[RoundingMode, str] = "half_up",
             skipna: bool = True, out: None = None, dtype: None = None) -> "FixedArray":
        """沿轴均值，商余分别累加法防溢出，四舍五入到 scale（默认保留到 self.scale）。

        利用商余分别累加消除求和中间溢出假阳性，仅在提升 scale 放大数值时按最大值检验溢出。
        ``np.mean(fa)`` 也会转到这里。

        Parameters
        ----------
        axis : int 或 int 元组，可选
            默认对全部元素求均值。
        keepdims : bool，可选
            保留被求均值的轴。
        scale : int, optional
            目标小数位数，默认保持当前 scale。
        rounding : {"half_up", "half_even", "down", "up", "floor", "ceil"}，默认 "half_up"
            舍入模式。
        skipna : bool, default True
            是否忽略缺失值 (NA)。为 True 时只对有效值计算均值（全为 NA 时返回 NA）；为 False 时若切片包含缺失值则结果为 NA。
        out, dtype : None
            只为兼容 ``np.mean`` 的调用约定；传入非 None 报 TypeError。要 float 请先
            ``to_float()``。

        Raises
        ------
        ZeroDivisionError
            参与求均值的元素个数为 0。
        OverflowError
            均值计算结果（提升 scale 后）超出 int64 范围。
        """
        if out is not None or dtype is not None:
            raise TypeError("FixedArray.mean 不支持 out、dtype；要 float 请先 to_float()")

        mode = check_rounding(rounding)
        target_scale = self._scale if scale is None else self._check_scale(scale)
        shift = target_scale - self._scale

        ndim = self._mantissa.ndim
        AxisError = getattr(getattr(np, "exceptions", np), "AxisError", IndexError)
        if axis is None:
            total_count = self._mantissa.size
        elif isinstance(axis, (int, np.integer)):
            if axis < -ndim or axis >= ndim:
                raise AxisError(f"axis {axis} is out of bounds for array of dimension {ndim}")
            total_count = self._mantissa.shape[axis]
        elif isinstance(axis, tuple):
            axes = set()
            total_count = 1
            for a in axis:
                if not isinstance(a, (int, np.integer)):
                    raise TypeError(f"tuple axis must contain integers, got {type(a).__name__}")
                norm_a = int(a if a >= 0 else a + ndim)
                if norm_a < 0 or norm_a >= ndim:
                    raise AxisError(f"axis {a} is out of bounds for array of dimension {ndim}")
                if norm_a in axes:
                    raise ValueError("duplicate value in 'axis'")
                axes.add(norm_a)
                total_count *= self._mantissa.shape[norm_a]
        else:
            raise TypeError(f"axis must be None, int, or tuple of ints, got {type(axis).__name__}")

        if total_count == 0:
            raise ZeroDivisionError("空数组没有均值")

        if self._mask is None:
            out_mask = None
            count = total_count
            count_keep = total_count
        else:
            if skipna:
                valid_count = (~self._mask).sum(axis=axis, keepdims=keepdims)
                out_mask = (valid_count == 0)
                if not np.any(out_mask):
                    out_mask = None
                count = np.where(valid_count == 0, 1, valid_count)
                if not keepdims and axis is not None:
                    count_keep = (~self._mask).sum(axis=axis, keepdims=True)
                    count_keep = np.where(count_keep == 0, 1, count_keep)
                else:
                    count_keep = count
            else:
                any_na = self._mask.any(axis=axis, keepdims=keepdims)
                out_mask = any_na if np.any(any_na) else None
                count = total_count
                count_keep = total_count

        with np.errstate(over="ignore"):
            # q_sum 依赖 int64 回绕模运算，与 q_r 相加后得到正确的商总和
            q, r = np.divmod(self._mantissa, count_keep)
            q_sum = q.sum(axis=axis, keepdims=keepdims)
            r_sum = r.sum(axis=axis, keepdims=keepdims)
            q_r, r_tot = np.divmod(r_sum, count)
            q_tot = q_sum + q_r

        if shift == 0:
            res = round_quotient(q_tot, r_tot, count, mode)
        elif shift < 0:
            res = self._round_div_pow10(q_tot, r_tot, count, -shift, mode)
        else:  # shift > 0
            top = self._max_abs(q_tot)
            if shift > 18 or (top and top > _MAX // 10 ** shift):
                raise OverflowError(f"均值乘以 10^{shift} 后超出 int64 范围")
            q_base = q_tot * 10 ** shift
            max_c = int(np.max(count)) if hasattr(count, "__iter__") or isinstance(count, np.ndarray) else count
            if max_c <= _MAX // 10 ** shift:
                q_frac, r_frac = np.divmod(r_tot * 10 ** shift, count)
            else:
                q_frac = np.zeros_like(r_tot)
                cur_r = r_tot.copy()
                k = shift
                chunk = 1
                while chunk < 18 and max_c * 10 ** (chunk + 1) <= _MAX:
                    chunk += 1
                while k:
                    c = min(chunk, k)
                    p = 10 ** c
                    d, cur_r = np.divmod(cur_r * p, count)
                    q_frac = q_frac * p + d
                    k -= c
                r_frac = cur_r

            if np.any((q_base > 0) & (q_base > _MAX - q_frac)):
                raise OverflowError(f"均值乘以 10^{shift} 后超出 int64 范围")
            Q = q_base + q_frac
            res = round_quotient(Q, r_frac, count, mode)
            if np.any((Q == _MAX) & (res < 0)):
                raise OverflowError(f"均值乘以 10^{shift} 后超出 int64 范围")

        if out_mask is not None:
            res = np.where(out_mask, np.int64(0), res)
        return self._from_mantissa(res, target_scale, mask=out_mask)

    # ---------- 形状变换与转置（公开 API） ----------

    def reshape(self, *shape: Any) -> "FixedArray":
        """矩阵变形。"""
        new_m = self._mantissa.reshape(*shape)
        new_mask = self._mask.reshape(*shape) if self._mask is not None else None
        return self._from_mantissa(new_m, self._scale, mask=new_mask)

    def transpose(self, *axes: int) -> "FixedArray":
        """矩阵转置。"""
        new_m = self._mantissa.transpose(*axes)
        new_mask = self._mask.transpose(*axes) if self._mask is not None else None
        return self._from_mantissa(new_m, self._scale, mask=new_mask)

    @property
    def T(self) -> "FixedArray":
        return self.transpose()

    def swapaxes(self, axis1: int, axis2: int) -> "FixedArray":
        """交换两个轴。"""
        new_m = self._mantissa.swapaxes(axis1, axis2)
        new_mask = self._mask.swapaxes(axis1, axis2) if self._mask is not None else None
        return self._from_mantissa(new_m, self._scale, mask=new_mask)

    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> "FixedArray":
        """移除长度为 1 的轴。"""
        new_m = self._mantissa.squeeze(axis=axis)
        new_mask = self._mask.squeeze(axis=axis) if self._mask is not None else None
        return self._from_mantissa(new_m, self._scale, mask=new_mask)

    # ---------- 精度调整与网格对齐（公开 API） ----------

    def rescale(self, new_scale: int, rounding: Union[RoundingMode, str] = "half_up") -> "FixedArray":
        """调整小数位数到 new_scale；变小时按 rounding 取整，变大时越界报 OverflowError。"""
        new_scale = self._check_scale(new_scale)
        mode = check_rounding(rounding)  # 无论放大、缩小还是不变，都先校验
        if new_scale >= self._scale:
            if new_scale == self._scale:
                return self
            return self._from_mantissa(self._mul_pow10(self._mantissa, new_scale - self._scale), new_scale,
                                       mask=self._mask)
        divided = self._round_div_pow10(self._mantissa, 0, 1, self._scale - new_scale, mode)
        return self._from_mantissa(divided, new_scale, mask=self._mask)

    def round_to_step(self, step: Any, rounding: str = "half_up") -> "FixedArray":
        """将定点数矩阵离散对齐到指定的步长网格。

        Parameters
        ----------
        step : int, float, Decimal, Sequence 或 FixedArray
            正数步长标量或与 self 广播兼容的步长矩阵。禁止传入字符串。
        rounding : {"half_up", "half_even", "down", "up", "floor", "ceil"} 或 RoundingMode，默认 "half_up"
            取整方式。

        Returns
        -------
        FixedArray
            对齐到步长网格后的定点数矩阵，其 scale 为 max(self.scale, step.scale)。

        Raises
        ------
        TypeError
            step 含有字符串类型或不支持的数据类型。
        ValueError
            step 含有非正数（<= 0）或 rounding 规则不合法。
        OverflowError
            对齐后结果超出 int64 范围。
        """
        step_fa = self._coerce(step)
        a, b = self._align_scales(self, step_fa)
        out_mask = self._combine_masks(a._mask, b._mask)
        valid_step = (b._mantissa <= 0) if out_mask is None else ((b._mantissa <= 0) & (~out_mask))
        if np.any(valid_step):
            raise ValueError("步长 step 必须为正数")
        check_rounding(rounding)
        den = np.where(out_mask, np.int64(1), b._mantissa) if out_mask is not None else b._mantissa
        q = self._divmod_round(a._mantissa, den, mode=rounding)
        res = b.mul_round(q, rounding=None)
        if out_mask is not None:
            return self._from_mantissa(res._mantissa, res._scale, mask=out_mask)
        return res

    def is_on_step(self, step: Any) -> Union[bool, np.ndarray]:
        """检查定点数是否严格位于指定步长的网格点上。

        Parameters
        ----------
        step : int, float, Decimal, Sequence 或 FixedArray
            正数步长标量或与 self 广播兼容的步长矩阵。禁止传入字符串。
        """
        step_fa = self._coerce(step)
        a, b = self._align_scales(self, step_fa)
        out_mask = self._combine_masks(a._mask, b._mask)
        valid_step = (b._mantissa <= 0) if out_mask is None else ((b._mantissa <= 0) & (~out_mask))
        if np.any(valid_step):
            raise ValueError("步长 step 必须为正数")
        den = np.where(out_mask, np.int64(1), b._mantissa) if out_mask is not None else b._mantissa
        rem = a._mantissa % den
        res = (rem == 0)
        if out_mask is not None:
            res = res & (~out_mask)
        return bool(res) if self.ndim == 0 and step_fa.ndim == 0 else res

    # ---------- 比较操作符 ----------

    def _compare_fixed_array(self, other: "FixedArray", op: Callable) -> Any:
        """与另一个 FixedArray 精确比较：标量返回 NumPy bool，数组返回 bool 数组；形状不能广播时报错。"""

        def _cmp_scaled(m_lo: np.ndarray, m_hi: np.ndarray, k: int) -> np.ndarray:
            """比较 m_lo·10^k 与 m_hi（k > 0），返回 -1 / 0 / 1。

            不放大 m_lo，而是把 m_hi 向下整除 10^k 得到 (q, r)，0 ≤ r < 10^k：
            m_lo < q 则小于，m_lo > q 则大于，相等时 r 非 0 则小于、否则相等。任何 scale 差都不会溢出。
            """
            if k > 18:  # |m_hi| < 10^19 ≤ 10^k：商只能是 0 或 -1，余数非 0 当且仅当 m_hi 非 0
                q = np.where(m_hi < 0, -1, 0)
                r_pos = m_hi != 0
            else:
                q, r = np.divmod(m_hi, np.int64(10 ** k))
                r_pos = r != 0
            return np.where(m_lo < q, -1, np.where(m_lo > q, 1, np.where(r_pos, -1, 0)))

        k = other._scale - self._scale
        if k == 0:
            raw_cmp = op(self._mantissa, other._mantissa)
        elif k > 0:
            raw_cmp = op(_cmp_scaled(self._mantissa, other._mantissa, k), 0)
        else:
            raw_cmp = op(-_cmp_scaled(other._mantissa, self._mantissa, -k), 0)

        out_mask = self._combine_masks(self._mask, other._mask)
        if out_mask is None:
            return raw_cmp
        if op is np.not_equal:
            res = np.where(out_mask, True, raw_cmp)
        else:
            res = np.where(out_mask, False, raw_cmp)
        return res[()] if res.ndim == 0 else res

    def _compare(self, other: Any, op: Callable) -> Any:
        """统一比较网关：非数值类型返回 NotImplemented；处理形状广播与不可表示数值。"""
        if isinstance(other, FixedArray):
            return self._compare_fixed_array(other, op)
        try:
            other_fa = self._coerce(other, strict=False)
        except (ValueError, OverflowError):
            if op is np.equal:
                return self._eq_unrepresentable(other)
            if op is np.not_equal:
                eq = self._eq_unrepresentable(other)
                return not eq if isinstance(eq, bool) else ~eq
            raise
        if other_fa is None:
            return NotImplemented
        return self._compare_fixed_array(other_fa, op)

    def _eq_unrepresentable(self, other: Any) -> Any:
        """other 含无法表示的数值（NaN、inf、超出 int64 等）时逐元素判等：这些位置一律不相等。"""
        values = self._expand_nested(other) if isinstance(other, (list, tuple)) else other
        mine, arr = np.broadcast_arrays(self._mantissa, np.asarray(values))  # 形状不齐等真错误照常抛出
        out = np.zeros(arr.shape, dtype=bool)
        flat_m, flat_o, flat_out = mine.ravel(), arr.ravel(), out.reshape(-1)
        if arr.dtype.kind == "f":
            idx = np.flatnonzero(np.isfinite(flat_o) & (np.abs(flat_o) < 2.0 ** 63))
            try:
                other_fa = FixedArray(flat_o[idx])
            except OverflowError:  # 量级悬殊的 float 无法对齐到同一 scale：退回逐个比较
                pass
            else:
                flat_out[idx] = self._from_mantissa(flat_m[idx], self._scale)._compare_fixed_array(other_fa, np.equal)
                idx = ()
        else:
            idx = range(flat_o.size)
        for i in idx:
            try:
                other_fa = FixedArray(flat_o[i])
            except (ValueError, OverflowError):
                continue
            flat_out[i] = self._from_mantissa(flat_m[i], self._scale)._compare_fixed_array(other_fa, np.equal)
        if self._mask is not None:  # 自身的缺失值与任何值都不相等
            out &= ~np.broadcast_to(self._mask, out.shape)
        return out if out.ndim else out[()]

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        return self._compare(other, np.equal)

    def __ne__(self, other: Any) -> Any:  # type: ignore[override]
        return self._compare(other, np.not_equal)

    def __lt__(self, other: Any) -> Any:
        return self._compare(other, np.less)

    def __le__(self, other: Any) -> Any:
        return self._compare(other, np.less_equal)

    def __gt__(self, other: Any) -> Any:
        return self._compare(other, np.greater)

    def __ge__(self, other: Any) -> Any:
        return self._compare(other, np.greater_equal)

    # ---------- 索引、切片与 NumPy 协议 ----------

    def __getitem__(self, key: Any) -> "FixedArray":
        new_m = self._mantissa[key]
        new_mask = self._mask[key] if self._mask is not None else None
        return self._from_mantissa(new_m, self._scale, mask=new_mask)

    def __len__(self) -> int:
        return len(self._mantissa)

    def __bool__(self) -> bool:
        """真值测试：对齐 NumPy 语义，只有恰好包含 1 个元素的数组允许转为 bool，其他情况报错。"""
        if self._mantissa.size != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
        if self._mask is not None and bool(self._mask.item()):
            raise TypeError("缺失值（NA）的真值不明确")
        return bool(self._mantissa.item())

    def __iter__(self):
        """迭代协议：0 维标量禁止迭代（抛出 TypeError），1 维及以上按首维度迭代返回切片视图。"""
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d FixedArray")
        return (self[i] for i in range(len(self)))

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """禁止隐式转成 NumPy 数组（np.asarray、np.concatenate 等），避免悄悄变成 float。"""
        raise TypeError("FixedArray 不能隐式转成数组：要 float 请用 to_float()，要整数尾数请用 .mantissa")

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        return FixedArray._array_function(func, args, kwargs, "FixedArray")

    @staticmethod
    def _array_function(func: Any, args: Any, kwargs: Any, name: str) -> Any:
        """NumPy 函数白名单（FixedArray 与 UnitArray 共用）。

        np.sum、np.mean、np.transpose、np.shape、np.ndim、np.size 转给对象自身的方法；
        np.array_equal、np.array_equiv 精确判等；其余一律报 TypeError——否则像 np.array_equal 这样
        内部吞掉异常的函数会静默返回错误结果。
        """
        if func is np.array_equal:
            return FixedArray._numpy_equal(*args, broadcast=False, **kwargs)
        if func is np.array_equiv:
            return FixedArray._numpy_equal(*args, broadcast=True, **kwargs)
        if func in _NUMPY_DISPATCHED:
            return func._implementation(*args, **kwargs)
        raise TypeError(f"{name} 不支持 numpy.{func.__name__}，请使用 {name} 自身的方法")

    @staticmethod
    def _numpy_equal(a1: Any, a2: Any, broadcast: bool, equal_nan: bool = False) -> bool:
        """np.array_equal（形状相同）与 np.array_equiv（可广播）：逐元素精确相等。

        equal_nan 为 True 时缺失值（Null / NA）之间视为相等。
        """

        def shape(x: Any) -> Tuple[int, ...]:
            if hasattr(x, "shape"):
                return x.shape
            return np.shape(FixedArray._expand_nested(x) if isinstance(x, (list, tuple)) else x)

        if not broadcast and shape(a1) != shape(a2):
            return False
        try:
            if equal_nan:
                fa1 = FixedArray._coerce(a1, strict=False)
                fa2 = FixedArray._coerce(a2, strict=False)
                if fa1 is not None and fa2 is not None:
                    m1 = fa1.isna()
                    m2 = fa2.isna()
                    both_na = m1 & m2
                    one_na = m1 ^ m2
                    if np.any(one_na):
                        return False
                    eq = (fa1 == fa2) | both_na
                    return bool(np.all(eq))
            eq = a1 == a2
        except ValueError:  # 形状不能广播
            return False
        return bool(np.all(eq))

    # ---------- 导出与类型转换（公开 API） ----------

    def to_float(self) -> Union[float, np.ndarray]:
        """转换为标准浮点数（标量返回 float，数组返回 float64 ndarray）。缺失值转为 np.nan。"""
        if self._mantissa.ndim == 0:
            if self._mask is not None and bool(self._mask):
                return float("nan")
            return int(self._mantissa) / (10 ** self._scale)
        m = self._mantissa
        s = self._scale
        if s <= 22 and np.all(np.abs(m) <= 2 ** 53):
            res = m.astype(np.float64) / (10.0 ** s)
        else:
            pow10 = 10 ** s
            fn = np.vectorize(lambda x: int(x) / pow10, otypes=[np.float64])
            res = fn(m)
        if self._mask is not None:
            res = np.where(self._mask, np.nan, res)
        return res

    def to_decimal(self) -> Union[Decimal, None, np.ndarray]:
        """转换为 Python 标准库 Decimal 对象（标量返回 Decimal 或 None，数组返回 object ndarray）。缺失值转为 None。"""
        if self._mantissa.ndim == 0:
            if self._mask is not None and bool(self._mask):
                return None
            return Decimal(int(self._mantissa)).scaleb(-self._scale)
        if self._mask is None:
            fn = np.vectorize(lambda m: Decimal(int(m)).scaleb(-self._scale), otypes=[object])
            return fn(self._mantissa)

        def _cvt(m: int, is_na: bool) -> Any:
            return None if is_na else Decimal(int(m)).scaleb(-self._scale)

        fn = np.vectorize(_cvt, otypes=[object])
        return fn(self._mantissa, self._mask)

    def to_pandas(self, index: Optional[Any] = None, name: Optional[str] = None, as_float: bool = True) -> Any:
        """导出为 pandas.Series 或 pandas.DataFrame。

        Parameters
        ----------
        index : Index-like, optional
            行索引。
        name : str, optional
            Series 名称（仅当导出为 1 维或标量 Series 时生效）。
        as_float : bool, 默认 True
            - True: 导出为 float64 数值（便于统计与绘图）；
            - False: 导出为 Python Decimal 对象（真实数值，零浮点漂移）。

        Returns
        -------
        pandas.Series 或 pandas.DataFrame
            1 维或 0 维返回 Series，2 维及以上返回 DataFrame。
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("需要安装 pandas: pip install pandas") from e

        data = self.to_float() if as_float else self.to_decimal()
        if self.ndim == 0:
            return pd.Series([data], index=index, name=name)
        elif self.ndim == 1:
            return pd.Series(data, index=index, name=name)
        return pd.DataFrame(data, index=index)

    def to_polars(self, name: str = "value", as_float: bool = True) -> Any:
        """导出为 polars.Series。

        Parameters
        ----------
        name : str, 默认 "value"
            Series 名称。
        as_float : bool, 默认 True
            - True: 导出为 polars.Float64 类型；
            - False: 导出为 polars.Decimal(scale=scale) 类型（真实数值，零浮点漂移）。

        Returns
        -------
        polars.Series
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError("需要安装 polars: pip install polars") from e

        if self.ndim > 1:
            raise ValueError(f"to_polars 仅支持 1 维或标量 FixedArray，当前维度为 {self.ndim}")
        if as_float:  # 尾数不含 NaN，NaN 只来自缺失值：转为 null，与 as_float=False 一致
            return pl.Series(name, np.atleast_1d(self.to_float()), nan_to_null=True)
        if self._scale > 38:
            raise ValueError(f"polars.Decimal 最多支持 38 位小数，当前 scale 为 {self._scale}；"
                             "请先 rescale 或使用 as_float=True")
        values = self.to_decimal()
        values = [values] if self.ndim == 0 else values.tolist()
        return pl.Series(name, values, dtype=pl.Decimal(38, self._scale))

    # ---------- 文本与展示 ----------

    def __repr__(self) -> str:
        return f"FixedArray({self}, scale={self._scale})"

    def __str__(self) -> str:
        if self._mantissa.ndim == 0:
            if self._mask is not None and bool(self._mask):
                return "<NA>"
            return self._format_scalar(int(self._mantissa), self._scale)
        if self._mask is None:
            return np.array2string(
                self._mantissa,
                formatter={"all": lambda x: self._format_scalar(int(x), self._scale)},
                separator=", ",
            )
        str_arr = np.empty(self.shape, dtype=object)
        flat_m = self._mantissa.ravel()
        flat_mask = self._mask.ravel()
        flat_str = [
            "<NA>" if flat_mask[i] else self._format_scalar(int(flat_m[i]), self._scale)
            for i in range(flat_m.size)
        ]
        str_arr.ravel()[:] = flat_str
        return np.array2string(
            str_arr,
            formatter={"all": lambda x: str(x)},
            separator=", ",
        )

    @staticmethod
    def _format_scalar(m: int, scale: int) -> str:
        """单个整数 mantissa 格式化为固定小数位字符串。"""
        if scale == 0:
            return str(m)
        sign = "-" if m < 0 else ""
        digits = str(abs(m)).zfill(scale + 1)
        return f"{sign}{digits[:-scale]}.{digits[-scale:]}"


_NUMPY_DISPATCHED = frozenset({np.sum, np.mean, np.transpose, np.shape, np.ndim, np.size})

__all__ = ["FixedArray", "PrecisionLossWarning", "RoundingMode"]
