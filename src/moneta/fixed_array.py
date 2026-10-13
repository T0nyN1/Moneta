"""定点数矩阵计算核心引擎（FixedArray）。

纯数学定点数矩阵系统：数学形式为 v = mantissa * 10**(-scale)。
保持 100% 纯整数底层运算，绝对消除 IEEE 754 浮点累积误差与精度漂移。
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

_MIN = -(2 ** 63)
_MAX = 2 ** 63 - 1
_B = np.uint64(10 ** 9)
_B_SQ = np.uint64(10 ** 18)


class RoundingMode(str, Enum):
    """金融与会计舍入规则枚举（继承自 str，完全兼容裸字符串与强类型枚举）。"""

    HALF_UP = "half_up"  # 四舍五入（逢五远离零，金融零售常用）
    HALF_EVEN = "half_even"  # 银行家舍入（四舍六入五成双，GAAP / IFRS 推荐，消除系统性统计偏差）
    DOWN = "down"  # 向零截断（直接舍去尾数）
    UP = "up"  # 远离零进位
    FLOOR = "floor"  # 向负无穷截断
    CEIL = "ceil"  # 向正无穷进位


_ROUNDINGS = tuple(m.value for m in RoundingMode)


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

    __slots__ = ("_mantissa", "_scale")
    __array_ufunc__ = None  # 阻止 NumPy ufunc 静默运算，确保精度与类型安全
    __hash__ = None  # 可变/数组对象不可哈希

    # ---------- 构造与核心属性 ----------

    def __init__(self, values: Any, scale: Optional[int] = None, ) -> None:
        """构造 FixedArray。支持显式 scale、未知 scale 自动探测、及字符串/对象精确解析。"""
        if isinstance(values, FixedArray):
            source = values if scale is None else values.rescale(scale)
            self._set(source._mantissa, source._scale)  # 尾数只读，直接共享，不必复制
            return
        if scale is not None:
            scale = self._check_scale(scale)

        arr = np.asarray(values)
        kind = arr.dtype.kind

        if kind in "iu":
            # 路径 1：整数
            s = 0 if scale is None else scale
            m = self._as_int64(arr)
            if s == 0 and isinstance(values, np.ndarray) and np.may_share_memory(m, values):
                m = m.copy()  # 不与调用方的数组共享内存，保证构造后不可变
            self._set(self._mul_pow10(m, s), s)
        elif kind == "f":
            if not np.all(np.isfinite(arr)):
                raise ValueError("FixedArray 不能包含 NaN 或无穷大")
            top = float(np.abs(arr).max()) if arr.size else 0.0
            if scale is not None and top * 10.0 ** scale >= 2.0 ** 63:
                raise OverflowError(f"浮点数乘以 10^{scale} 后超出 int64 范围")
            if arr.ndim == 0:
                m, s = self._parse_float_scalar(arr[()])
                m = self._as_int64(m)
            else:
                m, s = self._parse_float_array(arr)
            if scale is not None:
                if s < scale:
                    m = self._mul_pow10(m, scale - s)
                elif s > scale:
                    m = self._divmod_round(m, 10 ** (s - scale), mode="half_up")
                s = scale
            self._set(m, s)
        elif kind in "UO":
            # 路径 5：字符串、Decimal、Fraction 或混合对象（标量与数组同一条路径）
            m, s = self._parse_text_array(arr)
            if scale is not None:
                if s > scale:
                    raise ValueError(f"输入的小数位数 ({s}) 超出指定 scale ({scale})")
                m, s = self._mul_pow10(m, scale - s), scale
            self._set(m, s)
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

    # ---------- 构造收口与类型转换（私有） ----------

    def _set(self, mantissa: Any, scale: int) -> None:
        """收口：所有构造路径最后都经过这里，是唯一给 _mantissa、_scale 赋值的地方。

        尾数统一转成 int64 数组（0 维运算结果会退化成 NumPy 标量，这里包回数组；
        float、object 报 TypeError；超出 int64 报 OverflowError），取视图后设为只读——
        只锁视图，不改动调用方传入的数组。scale 由调用方保证合法。
        """
        view = self._as_int64(mantissa).view()
        view.flags.writeable = False
        self._mantissa = view
        self._scale = int(scale)

    @classmethod
    def _from_mantissa(cls, mantissa: Any, scale: int) -> "FixedArray":
        """直接使用已有的整数尾数矩阵与 scale 构造（内部零拷贝快速路径）。

        int64 数组不复制：调用方之后不应再修改它。
        """
        obj = cls.__new__(cls)
        obj._set(mantissa, scale)
        return obj

    @staticmethod
    def _is_numeric_operand(val: Any) -> bool:
        """运算操作数白名单：数值标量、数值 ndarray，或只含这些元素的 list / tuple / object 数组。

        FixedArray 不认识任何上层类：白名单之外的一切（字符串、布尔、带单位对象等）都不是合法操作数。
        """
        if isinstance(val, (bool, np.bool_)):
            return False
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
        if not cls._is_numeric_operand(other):
            if strict:
                raise TypeError(f"FixedArray 只能与数值类型运算、比较或对齐，收到 {type(other).__name__}")
            return None
        return cls(other)

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
        if exp > 19 or abs(m) * (10 ** exp) > _MAX:  # 先判断位数，避免算出巨大的 10**exp
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

    @classmethod
    def _parse_object_element(cls, val: Any) -> Tuple[int, int]:
        """将单个标量（int、float、Fraction、str、Decimal 等）解析为 (mantissa, scale)。"""
        if isinstance(val, (bool, np.bool_)):
            raise TypeError("FixedArray 不支持布尔类型")
        if isinstance(val, (int, np.integer)):
            return int(val), 0
        if isinstance(val, (float, np.floating)):
            return cls._parse_float_scalar(val)
        if isinstance(val, Fraction):
            return cls._parse_fraction(val)
        return cls._parse_decimal_str(str(val))

    @classmethod
    def _parse_float_array(cls, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        """float 数组 → (尾数, scale)：每个元素取能往返还原自身的最少小数位数，再对齐到最大值。

        逐位试探 s = 0, 1, 2, …：m = rint(x·10^s)，若 m·10^-s 转回 x 的 float 类型后与 x
        完全相等（往返检验），x 的小数位数就是 s。不设容差，大数的零头不会被当成噪声吞掉；
        结果与标量路径的最短写法一致。已通过的元素不再参与后续试探。

        |x·10^s| 超过 2**53 或 s 超过 22 时 float 运算不再精确（10^22 以内才能精确表示），
        剩下的元素改用最短写法逐个解析。
        """
        flat = arr.ravel()
        flat64 = flat.astype(np.float64, copy=False)
        max_abs = float(np.abs(flat64).max()) if flat.size else 0.0
        mantissas = np.zeros(flat.size, dtype=np.int64)
        scales = np.zeros(flat.size, dtype=np.int64)
        pending = None  # None 表示整列仍在试探，此时不做下标筛选，省去复制
        fallback = []
        s = 0
        while s <= 22:
            if max_abs * 10.0 ** s > _MAX:  # 还有元素没通过，scale 至少是 s，最大的数必然溢出
                raise OverflowError(f"浮点数需要超过 {s - 1} 位小数，对齐后超出 int64 范围")
            values = flat if pending is None else flat[pending]
            values64 = flat64 if pending is None else values.astype(np.float64, copy=False)
            m = np.rint(values64 * 10.0 ** s)
            exact = np.abs(m) <= 2.0 ** 53
            ok = exact & ((m / 10.0 ** s).astype(arr.dtype, copy=False) == values)
            if pending is None:
                failed = flat.size - int(np.count_nonzero(ok))
                if failed == 0:  # 整列都在 s 位小数内：直接返回
                    return m.astype(np.int64).reshape(arr.shape), s
                if failed > flat.size // 10 and exact.all():
                    s += 1
                    continue
                pending = np.arange(flat.size)  # 剩余不多，改为只试探没通过的元素
            done = pending[ok]
            mantissas[done] = m[ok]
            scales[done] = s
            fallback.append(pending[~exact])
            pending = pending[exact & ~ok]
            s += 1
            if not pending.size:
                break
        if pending is None:
            pending = np.arange(flat.size)
        fallback.append(pending)
        rest = np.concatenate(fallback)

        # float 运算无法精确判断的元素：按最短写法逐个解析（同值只解析一次）
        rest_parsed = []
        if rest.size:
            distinct, inverse = np.unique(flat[rest], return_inverse=True)
            parsed = [cls._parse_float_scalar(v) for v in distinct]
            rest_parsed = [parsed[i] for i in inverse]
        target = max(int(scales.max()) if flat.size else 0, max((sc for _, sc in rest_parsed), default=0))

        # 对齐到 target：先用整列判断是否溢出，再整列相乘
        shift = target - scales
        nonzero = mantissas != 0
        if np.any(nonzero & (shift > 18)):
            raise OverflowError(f"浮点数对齐到 {target} 位小数后超出 int64 范围")
        factor = np.power(10, np.minimum(shift, 18), dtype=np.int64)
        if np.any(np.abs(mantissas) > _MAX // factor):
            raise OverflowError(f"浮点数对齐到 {target} 位小数后超出 int64 范围")
        result = mantissas * factor
        for i, (m_val, sc) in zip(rest, rest_parsed):
            aligned = m_val * 10 ** (target - sc)
            if not _MIN <= aligned <= _MAX:
                raise OverflowError(f"浮点数对齐到 {target} 位小数后超出 int64 范围")
            result[i] = aligned
        return result.reshape(arr.shape), target

    @classmethod
    def _parse_text_array(cls, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        """文本、Decimal、Fraction 或混合对象数组（含 0 维）→ (对齐后的尾数, scale)。"""
        if arr.ndim == 0:  # 标量直接解析，省去去重的开销
            m, scale = cls._parse_object_element(arr[()])
            return cls._as_int64(m), scale
        flat = arr.ravel()
        if arr.dtype.kind == "U":
            # 字符串可排序：先去重，只解析不同的取值
            distinct, inverse = np.unique(flat, return_inverse=True)
            parsed = [cls._parse_decimal_str(str(v)) for v in distinct]
        else:
            # 对象数组可能混合类型、无法排序：按 (类型, 文本) 去重。不能按值去重，
            # 否则 Decimal("1.1") 与 Decimal("1.100") 相等，谁在前就用谁的小数位数
            index = {}
            parsed = []
            inverse = np.empty(flat.size, dtype=np.intp)
            for i, val in enumerate(flat):
                key = (type(val).__name__, str(val))
                j = index.get(key)
                if j is None:
                    j = index[key] = len(parsed)
                    parsed.append(cls._parse_object_element(val))
                inverse[i] = j
        scale = max((s for _, s in parsed), default=0)
        aligned = []
        for m, s in parsed:
            if m and (scale - s > 18 or abs(m) * 10 ** (scale - s) > _MAX):  # 先判断位数，避免巨大的幂
                raise OverflowError(f"输入对齐到 {scale} 位小数后超出 int64 范围")
            aligned.append(m * 10 ** (scale - s) if m else 0)
        return np.array(aligned, dtype=np.int64)[inverse].reshape(arr.shape), scale

    # ---------- 底层数学与防溢出舍入引擎（私有） ----------

    @staticmethod
    def _check_rounding(mode: Union[RoundingMode, str]) -> str:
        if isinstance(mode, RoundingMode):
            return mode.value
        if mode not in _ROUNDINGS:
            raise ValueError(f"rounding 只能是 {_ROUNDINGS} 之一，收到 {mode!r}")
        return str(mode)

    @staticmethod
    def _check_division(m: np.ndarray, d: np.ndarray) -> None:
        """除数不能含 0；-2**63 除以 -1 的商超出 int64。"""
        if np.any(d == 0):
            raise ZeroDivisionError("除数包含 0")
        if np.any((d == -1) & (m == _MIN)):
            raise OverflowError("-2**63 除以 -1 超出 int64 范围")

    @staticmethod
    def _max_abs(x: np.ndarray) -> int:
        """绝对值的最大值（Python int，-2**63 也不溢出）；空数组为 0。"""
        return max(-int(x.min()), int(x.max())) if x.size else 0

    @staticmethod
    def _mul_pow10(x: np.ndarray, k: int) -> np.ndarray:
        """x·10^k，先用上下界判断，可能越界就报 OverflowError。"""
        if k == 0:
            return x
        top = FixedArray._max_abs(x)
        if top and (k > 18 or top > _MAX // 10 ** k):
            raise OverflowError(f"乘以 10^{k} 后超出 int64 范围")
        return x * 10 ** k

    @staticmethod
    def _decide_bump(mode: str, *, is_negative: Any, odd_q: Any, has_fraction: Any, over_half: Any, exact_half: Any,
                     floor_base: bool = False, ) -> Any:
        """统一收口的舍入进位裁决函数。

        根据截断状态和半数关系，为 6 大舍入模式（half_up, half_even, down, up, floor, ceil）
        统一裁决是否需要对商基准值进位（bump + 1）。

        Parameters
        ----------
        mode : str
            舍入模式名称，必须已通过 _check_rounding 校验。
        is_negative : bool or array_like of bool
            数值是否为负。
        odd_q : bool or array_like of bool
            商（基准整数）是否为奇数。
        has_fraction : bool or array_like of bool
            是否存在非零小数部分。
        over_half : bool or array_like of bool
            小数部分是否严格大于 0.5。
        exact_half : bool or array_like of bool
            小数部分是否恰好等于 0.5。
        floor_base : bool, default False
            基准坐标系：
            - False (Truncation Base): 商 q 是向零截断的绝对值 (|q| >= 0)，bump 作用于 |q|。
              返回 uint64(0) 或 uint64(1)。
            - True (Floor Base): 商 q 是向下取整的带符号数 (q = floor(m/d))，bump 作用于 q。
              返回 bool 或 bool 数组。
        """
        if floor_base:
            if mode == "floor":
                return False
            elif mode == "ceil":
                return has_fraction
            elif mode == "down":
                return has_fraction & is_negative
            elif mode == "up":
                return has_fraction & (~is_negative)
            elif mode == "half_even":
                return np.where(over_half, True, np.where(exact_half, odd_q, False))
            elif mode == "half_up":
                return np.where(is_negative, over_half, over_half | exact_half)
            else:
                raise ValueError(f"rounding 只能是 {_ROUNDINGS} 之一，收到 {mode!r}")
        else:
            if mode == "down":
                return np.uint64(0)
            elif mode == "up":
                return np.where(has_fraction, np.uint64(1), np.uint64(0))
            elif mode == "floor":
                return np.where(is_negative & has_fraction, np.uint64(1), np.uint64(0))
            elif mode == "ceil":
                return np.where((~is_negative) & has_fraction, np.uint64(1), np.uint64(0))
            elif mode == "half_even":
                return np.where(over_half, np.uint64(1), np.where(exact_half & odd_q, np.uint64(1), np.uint64(0)))
            elif mode == "half_up":
                return np.where(over_half | exact_half, np.uint64(1), np.uint64(0))
            else:
                raise ValueError(f"rounding 只能是 {_ROUNDINGS} 之一，收到 {mode!r}")

    @staticmethod
    def _round_quotient(q: np.ndarray, r: np.ndarray, d: np.ndarray, mode: str) -> np.ndarray:
        """由向下取整的商 q 与余数 r（m = q·d + r，r 与 d 同号，|r| < |d|）按 mode 取整。

        真实值为 q + r/d，r/d ∈ [0, 1)。判断 r/d 与 1/2 的大小时比较 r 与 d - r：二者同号
        且 |d - r| ≤ |d|，不会溢出。调用方保证 q + 1 不溢出。
        """
        if mode == "floor":
            return q
        has_fraction = (r != 0)
        is_negative = (q < 0)
        if mode in ("half_even", "half_up"):
            over_half = np.where(d > 0, r > d - r, r < d - r)
            exact_half = (r == d - r)
            odd_q = (q % 2 != 0)
        else:
            over_half = exact_half = odd_q = False
        bump = FixedArray._decide_bump(
            mode,
            is_negative=is_negative,
            odd_q=odd_q,
            has_fraction=has_fraction,
            over_half=over_half,
            exact_half=exact_half,
            floor_base=True,
        )
        return q + bump

    @staticmethod
    def _divmod_round(m: Any, d: Any, mode: str = "half_up") -> Any:
        """整数带余除法舍入：round(m / d)，全程 int64，结果精确。

        先做向下取整的带余除法 ``m = q·d + r``，再由 ``_round_quotient`` 决定 q 是否加 1。

        Parameters
        ----------
        m, d : int 或整数数组
            被除数与除数，必须在 int64 范围内，可广播；d 可正可负，不能为 0。
        mode : {"half_up", "down", "up", "floor", "ceil"}
            ``half_up`` 四舍五入、恰好一半时远离零；``down`` 向零；``up`` 远离零；
            ``floor`` 向负无穷；``ceil`` 向正无穷。

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
        FixedArray._check_rounding(mode)
        scalar = isinstance(m, (int, np.integer)) and isinstance(d, (int, np.integer))
        m, d = FixedArray._as_int64(m), FixedArray._as_int64(d)
        FixedArray._check_division(m, d)
        q, r = np.divmod(m, d)
        q = FixedArray._round_quotient(q, r, d, mode)
        return int(q) if scalar else np.asarray(q)

    @staticmethod
    def _chunked_divmod_round(m: np.ndarray, d: np.ndarray, k: int, mode: str) -> np.ndarray:
        """round(m·10^k / d)，全程 int64 的分段长除法；只有结果本身超出 int64 时才报错。

        直接算 m·10^k 很容易越界（m 超过约 9.2e10 时乘 10^8 就溢出），而结果往往放得下。
        这里先做带余除法 m = q·d + r，再把 10^k 分段乘进去：每段 c 位，用
        (r·10^c) // d 得到商的下 c 位、(r·10^c) % d 得到新余数。|r| < |d|，所以每段
        取 c 使 |d|·10^c ≤ MAX 即可保证不溢出。最后按 mode 由商与余数取整。
        """
        FixedArray._check_division(m, d)
        d_top = FixedArray._max_abs(d)
        if d_top > _MAX // 10:
            raise OverflowError("除数尾数超过 int64 上限的 1/10，无法做长除法")
        chunk = 1
        while chunk < 18 and d_top * 10 ** (chunk + 1) <= _MAX:
            chunk += 1
        q, r = np.divmod(m, d)
        while k:
            c = min(chunk, k)
            p = 10 ** c
            # 下 c 位商在 [0, p) 内：q·p + (p - 1) 不越界，最后进位的 +1 也不越界
            if FixedArray._max_abs(q) > (_MAX - p) // p:
                raise OverflowError("除法结果超出 int64 范围")
            digits, r = np.divmod(r * p, d)
            q = q * p + digits
            k -= c
        return FixedArray._round_quotient(q, r, d, mode)

    @staticmethod
    def _chunked_mul_round(a: Any, b: Any, shift: int, mode: str = "half_up", ) -> np.ndarray:
        """十进制 Base 10^9 分块长乘法与舍入，全程基于纯原生 int64/uint64 整数运算。

        计算 round((a * b) / 10^shift, mode=mode)。
        单块相乘 (10^9 - 1)^2 < 10^18，仅占 int64 上限的 10.8%，在 int64 内部绝对不溢出。
        当且仅当最终舍入结果超出 int64 范围时才抛出 OverflowError。
        """
        a_arr = np.asarray(a, dtype=np.int64)
        b_arr = np.asarray(b, dtype=np.int64)
        a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)

        sign = np.where((a_arr < 0) ^ (b_arr < 0), -1, 1)

        # 安全获取 uint64 绝对值幅度（正确处理 -2**63 无负号溢出）
        with np.errstate(over="ignore"):
            ua = a_arr.view(np.uint64)
            ub = b_arr.view(np.uint64)
            mag_a = np.where(a_arr < 0, (~ua) + np.uint64(1), ua)
            mag_b = np.where(b_arr < 0, (~ub) + np.uint64(1), ub)

        # 拆分为 Base 10^9 的 3 个数字块
        a0 = mag_a % _B
        a1 = (mag_a // _B) % _B
        a2 = mag_a // _B_SQ

        b0 = mag_b % _B
        b1 = (mag_b // _B) % _B
        b2 = mag_b // _B_SQ

        # 卷积项
        c0 = a0 * b0
        c1 = a0 * b1 + a1 * b0
        c2 = a0 * b2 + a1 * b1 + a2 * b0
        c3 = a1 * b2 + a2 * b1
        c4 = a2 * b2

        # 进位传递
        d0 = c0 % _B
        c1 = c1 + c0 // _B
        d1 = c1 % _B
        c2 = c2 + c1 // _B
        d2 = c2 % _B
        c3 = c3 + c2 // _B
        d3 = c3 % _B
        c4 = c4 + c3 // _B
        d4 = c4 % _B
        d5 = c4 // _B

        digits = [d0, d1, d2, d3, d4, d5]

        if shift >= 0:
            k = shift
            if k == 0:
                for x in digits[3:]:
                    if np.any(x > 0):
                        raise OverflowError("乘法结果超出 int64 范围")
                if np.any(digits[2] > 9):
                    raise OverflowError("乘法结果超出 int64 范围")
                val = digits[2] * _B_SQ + digits[1] * _B + digits[0]
                overflow = (sign > 0) & (val > np.uint64(_MAX)) | (sign < 0) & (val > np.uint64(-_MIN))
                if np.any(overflow):
                    raise OverflowError("乘法结果超出 int64 范围")
                return np.where(sign > 0, val.astype(np.int64), -val.astype(np.int64))

            s = k // 9
            r = k % 9

            if s >= len(digits):
                e = [np.zeros_like(mag_a)]
            else:
                e = digits[s:]

            pow10_r = np.uint64(10 ** r)
            rem = np.zeros_like(mag_a)
            q_blocks = [np.zeros_like(mag_a) for _ in range(len(e))]
            for j in range(len(e) - 1, -1, -1):
                val = rem * _B + e[j]
                q_blocks[j] = val // pow10_r
                rem = val % pow10_r

            for j in range(3, len(q_blocks)):
                if np.any(q_blocks[j] > 0):
                    raise OverflowError("融合乘法舍入结果超出 int64 范围")
            q0 = q_blocks[0] if len(q_blocks) > 0 else np.zeros_like(mag_a)
            q1 = q_blocks[1] if len(q_blocks) > 1 else np.zeros_like(mag_a)
            q2 = q_blocks[2] if len(q_blocks) > 2 else np.zeros_like(mag_a)
            if np.any(q2 > 9):
                raise OverflowError("融合乘法舍入结果超出 int64 范围")
            q = q2 * _B_SQ + q1 * _B + q0

            has_fraction = rem > 0
            for d in digits[:s]:
                has_fraction = has_fraction | (d > 0)

            if r > 0:
                first_dropped = (rem // np.uint64(10 ** (r - 1))) % np.uint64(10)
                has_lower = (rem % np.uint64(10 ** (r - 1))) > 0 if r > 1 else np.zeros_like(mag_a, dtype=bool)
                for d in digits[:s]:
                    has_lower = has_lower | (d > 0)
            else:
                if s - 1 < len(digits):
                    first_dropped = (digits[s - 1] // np.uint64(10 ** 8)) % np.uint64(10)
                    has_lower = (digits[s - 1] % np.uint64(10 ** 8)) > 0
                    for d in digits[:s - 1]:
                        has_lower = has_lower | (d > 0)
                else:
                    first_dropped = np.zeros_like(mag_a)
                    has_lower = np.zeros_like(mag_a, dtype=bool)
                    for d in digits:
                        has_lower = has_lower | (d > 0)

            if mode in ("half_even", "half_up"):
                over_half = (first_dropped > 5) | ((first_dropped == 5) & has_lower)
                exact_half = (first_dropped == 5) & (~has_lower)
                odd_q = (q % np.uint64(2) != 0)
            else:
                over_half = exact_half = odd_q = False

            bump = FixedArray._decide_bump(
                mode,
                is_negative=(sign < 0),
                odd_q=odd_q,
                has_fraction=has_fraction,
                over_half=over_half,
                exact_half=exact_half,
                floor_base=False,
            )

            final_abs = q + bump
            overflow = (sign > 0) & (final_abs > np.uint64(_MAX)) | (sign < 0) & (final_abs > np.uint64(-_MIN))
            if np.any(overflow):
                raise OverflowError("融合乘法舍入结果超出 int64 范围")
            return np.where(sign > 0, final_abs.astype(np.int64), -final_abs.astype(np.int64))
        else:
            # shift < 0, 乘以 10**(-shift)
            k = -shift
            for x in digits[3:]:
                if np.any(x > 0):
                    raise OverflowError("乘法结果超出 int64 范围")
            if np.any(digits[2] > 9):
                raise OverflowError("乘法结果超出 int64 范围")
            val = digits[2] * _B_SQ + digits[1] * _B + digits[0]
            if k > 18:
                if np.any(val > 0):
                    raise OverflowError(f"乘法结果乘以 10^{k} 超出 int64 范围")
                return np.zeros_like(a_arr, dtype=np.int64)
            pow10_k = np.uint64(10 ** k)
            limit = np.where(sign > 0, np.uint64(_MAX), np.uint64(-_MIN)) // pow10_k
            if np.any(val > limit):
                raise OverflowError(f"乘法结果乘以 10^{k} 后超出 int64 范围")
            val = val * pow10_k
            return np.where(sign > 0, val.astype(np.int64), -val.astype(np.int64))

    @staticmethod
    def _chunked_sum(arr: np.ndarray, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        """十进制 Base 10^9 分块精确求和，全程纯 int64 运算，消除假阳性误报。

        将输入数组分解为 3 个 Base 10^9 的数字块（r0, r1 < 10^9, r2 in [-10, 9]）。
        各块沿轴求和单项不超过 10^9 * N，在 N <= 9*10^9 内单块绝对不溢出 int64。
        最后通过两次进位折叠判定并重建结果，全程不升级为 Python 任意精度大整数。
        """
        b_base = np.int64(10 ** 9)
        b_sq = np.int64(10 ** 18)
        q1, r0 = np.divmod(arr, b_base)
        r2, r1 = np.divmod(q1, b_base)

        s0 = r0.sum(axis=axis, keepdims=keepdims)
        s1 = r1.sum(axis=axis, keepdims=keepdims)
        s2 = r2.sum(axis=axis, keepdims=keepdims)

        c0, d0 = np.divmod(s0, b_base)
        s1_total = s1 + c0
        c1, d1 = np.divmod(s1_total, b_base)
        s2_total = s2 + c1

        overflow = (s2_total > 9) | (s2_total < -10)
        over_max = (s2_total == 9) & ((d1 > 223372036) | ((d1 == 223372036) & (d0 > 854775807)))
        under_min = (s2_total == -10) & ((d1 < 776627963) | ((d1 == 776627963) & (d0 < 145224192)))
        overflow = overflow | over_max | under_min
        if np.any(overflow):
            raise OverflowError("求和结果超出 int64 范围")

        with np.errstate(over="ignore"):
            low_part = d1 * b_base + d0
            res = np.where(
                s2_total < 0,
                (s2_total + 1) * b_sq + (low_part - b_sq),
                s2_total * b_sq + low_part,
            )
        return np.asarray(res, dtype=np.int64)

    @staticmethod
    def _chunked_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """二进制 Base 2^21 分块矩阵乘法，全程纯 int64 运算，消除假阳性误报。

        将输入矩阵位移拆分为 3 个 21 位的数字块（a0, a1, b0, b1 < 2^21, a2, b2 in signed 22-bit）。
        由于 (2^21)^2 = 2^42，内积在 K <= 2^20 (100万) 内 9 次分块矩阵相乘绝对不溢出 int64。
        最后通过进位折叠判定并重建结果，全程不升级为 Python 任意精度大整数。
        """
        shift = 21
        b_base = np.int64(1 << shift)
        mask = np.int64(b_base - 1)

        a0 = a & mask
        aq1 = a >> shift
        a1 = aq1 & mask
        a2 = aq1 >> shift

        b0 = b & mask
        bq1 = b >> shift
        b1 = bq1 & mask
        b2 = bq1 >> shift

        c00 = np.matmul(a0, b0)
        c01 = np.matmul(a0, b1)
        c02 = np.matmul(a0, b2)

        c10 = np.matmul(a1, b0)
        c11 = np.matmul(a1, b1)
        c12 = np.matmul(a1, b2)

        c20 = np.matmul(a2, b0)
        c21 = np.matmul(a2, b1)
        c22 = np.matmul(a2, b2)

        t0 = c00
        t1 = c01 + c10
        t2 = c02 + c11 + c20
        t3 = c12 + c21
        t4 = c22

        c0, r0 = np.divmod(t0, b_base)
        t1 = t1 + c0
        c1, r1 = np.divmod(t1, b_base)
        t2 = t2 + c1
        c2, r2 = np.divmod(t2, b_base)
        t3 = t3 + c2
        c3, r3 = np.divmod(t3, b_base)
        t4 = t4 + c3

        high = t4 * b_base + r3
        overflow = (high != 0) & (high != -1)
        if np.any(overflow):
            raise OverflowError("矩阵乘法结果超出 int64 范围")

        low63 = (r2 << 42) | (r1 << 21) | r0
        with np.errstate(over="ignore"):
            res = np.where(high == -1, (low63.astype(np.uint64) | np.uint64(1 << 63)).view(np.int64), low63)
        return np.asarray(res, dtype=np.int64)

    # ---------- 代数运算与操作符重载（公开 API） ----------

    def __add__(self, other: Any) -> "FixedArray":
        """加法：自动对齐 scale，纯整数相加，溢出抛出 OverflowError。"""
        b = self._coerce(other, strict=False)
        if b is None:
            return NotImplemented
        a, b_aligned = self._align_scales(self, b)
        res = a._mantissa + b_aligned._mantissa
        if np.any(((a._mantissa ^ res) & (b_aligned._mantissa ^ res)) < 0):  # 同号相加得到异号才是溢出
            raise OverflowError("加法结果超出 int64 范围")
        return self._from_mantissa(res, a._scale)

    def __radd__(self, other: Any) -> "FixedArray":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "FixedArray":
        """减法：自动对齐 scale，纯整数相减，溢出抛出 OverflowError。"""
        b = self._coerce(other, strict=False)
        if b is None:
            return NotImplemented
        a, b_aligned = self._align_scales(self, b)
        res = a._mantissa - b_aligned._mantissa
        if np.any(((a._mantissa ^ b_aligned._mantissa) & (a._mantissa ^ res)) < 0):  # 异号相减结果与被减数异号才是溢出
            raise OverflowError("减法结果超出 int64 范围")
        return self._from_mantissa(res, a._scale)

    def __rsub__(self, other: Any) -> "FixedArray":
        b = self._coerce(other, strict=False)
        if b is None:
            return NotImplemented
        return b.__sub__(self)

    def __neg__(self) -> "FixedArray":
        """取负，溢出抛出 OverflowError。"""
        if np.any(self._mantissa == _MIN):
            raise OverflowError("取负导致数值超出 int64 范围")
        return self._from_mantissa(-self._mantissa, self._scale)

    def __abs__(self) -> "FixedArray":
        """绝对值，溢出抛出 OverflowError。"""
        if np.any(self._mantissa == _MIN):
            raise OverflowError("绝对值导致数值超出 int64 范围")
        return self._from_mantissa(np.abs(self._mantissa), self._scale)

    def mul_round(self, other: Any, scale: Optional[int] = None,
                  rounding: Optional[Union[RoundingMode, str]] = "half_up") -> "FixedArray":
        """融合乘法与可选舍入 (Fused Multiply-Round)，全程纯 int64。

        - 当 ``rounding=None`` 时：不做舍入（精确相乘），默认 ``scale = self.scale + other.scale``；
        - 当 ``rounding`` 非 None 时（默认 ``"half_up"``）：按 ``rounding`` 舍入到目标 ``scale``
          （若 ``scale=None``，默认保留 ``max(self.scale, other.scale)`` 位小数）。

        内部在快速通道不溢出时直接单指令相乘，否则采用十进制 Base 10^9 分块长乘法，
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
            mode = self._check_rounding(rounding)
            target_scale = max(self._scale, other_fa._scale) if scale is None else self._check_scale(scale)

        shift = natural_scale - target_scale

        # 快速通道：若两者全局极值之积在安全范围内且无需额外放大，直接单指令乘法与除法舍入
        if 0 <= shift <= 18 and self._max_abs(self._mantissa) * self._max_abs(other_fa._mantissa) <= _MAX:
            prod = np.multiply(self._mantissa, other_fa._mantissa)
            res_m = prod if shift == 0 else self._divmod_round(prod, 10 ** shift, mode=mode)
            return self._from_mantissa(res_m, target_scale)

        # 纯 int64 分块长乘法通道
        res_m = self._chunked_mul_round(self._mantissa, other_fa._mantissa, shift, mode=mode)
        return self._from_mantissa(res_m, target_scale)

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
            mode = self._check_rounding(rounding)
            target_scale = max(self._scale, other_fa._scale) if scale is None else self._check_scale(scale)

        a, b = self._mantissa, other_fa._mantissa
        terms = a.shape[-1] if a.ndim else 0
        if self._max_abs(a) * self._max_abs(b) * terms <= _MAX:
            result = np.matmul(a, b)  # 形状不合法时由 NumPy 报错
            raw_fa = self._from_mantissa(result, natural_scale)
        else:
            # 慢速保底路径：纯 int64 Base 2^21 分块矩阵乘法判定真实点积，消除正负对冲假阳性误报
            exact_res = self._chunked_matmul(a, b)
            raw_fa = self._from_mantissa(exact_res, natural_scale)

        if target_scale == natural_scale:
            return raw_fa
        return raw_fa.rescale(target_scale, rounding=mode)

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

        全程 int64：商的小数位数比自然精度多时走长除法，只有结果本身超出 int64
        才报错；比自然精度少时把 10 的幂乘到除数上，除数越界就报错。

        Parameters
        ----------
        other : FixedArray 或可构造 FixedArray 的值
            除数，可广播。
        scale : int，可选
            商的小数位数，默认 ``max(self.scale, other.scale)``。
        rounding : {"half_up", "down", "up", "floor", "ceil"}，可选
            取整方式，见 ``_divmod_round``。

        Raises
        ------
        ValueError
            rounding 无效或 scale 不是非负整数。
        ZeroDivisionError
            除数含 0。
        OverflowError
            结果超出 int64 范围。
        """
        self._check_rounding(rounding)
        other = self._coerce(other)
        target = max(self._scale, other._scale) if scale is None else self._check_scale(scale)
        # 商的尾数 = round(a·10^(target + other.scale - self.scale) / b)
        shift = target + other._scale - self._scale
        num, den = self._mantissa, other._mantissa
        if shift > 0:
            quotient = self._chunked_divmod_round(num, den, shift, rounding)
        else:
            self._check_division(num, den)  # 先报除以 0，再判断除数放大后是否越界
            quotient = self._divmod_round(num, self._mul_pow10(den, -shift), rounding)
        return self._from_mantissa(quotient, target)

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

    def __floordiv__(self, other: Any) -> Any:
        """向下取整的商（无量纲整数）：标量返回 int，数组返回 int64 数组。

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
        self._check_division(a._mantissa, b._mantissa)
        q = a._mantissa // b._mantissa
        return int(q) if np.ndim(q) == 0 else q

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
        if np.any(b._mantissa == 0):
            raise ZeroDivisionError("除数包含 0")
        return self._from_mantissa(a._mantissa % b._mantissa, a._scale)

    def __rmod__(self, other: Any) -> "FixedArray":
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return other_fa.__mod__(self)

    def __divmod__(self, other: Any) -> Tuple[Any, "FixedArray"]:
        """(商, 余数)：商同 ``//``，余数同 ``%``。"""
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        a, b = self._align_scales(self, other_fa)
        self._check_division(a._mantissa, b._mantissa)
        q, r = np.divmod(a._mantissa, b._mantissa)
        return (int(q) if np.ndim(q) == 0 else q), self._from_mantissa(r, a._scale)

    def __rdivmod__(self, other: Any) -> Tuple[Any, "FixedArray"]:
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        return other_fa.__divmod__(self)

    # ---------- 张量规约统计（公开 API） ----------

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, out: None = None,
            dtype: None = None) -> "FixedArray":
        """沿轴求和，全程 int64。``np.sum(fa)`` 也会转到这里。

        精确溢出判定：快速路径在最坏情况不超过 int64 时直接计算；超出时使用 Base 10^9
        分块累加判定真实结果，只有最终结果真实超出 int64 时才报错。
        全程维持在 int64 原生整数内，不升级为 Python 任意精度大整数。

        Parameters
        ----------
        axis : int 或 int 元组，可选
            默认对全部元素求和。
        keepdims : bool，可选
            保留被求和的轴。
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
        total = self._mantissa.sum(axis=axis, keepdims=keepdims)
        count = self._mantissa.size // max(np.size(total), 1)  # 每个结果单元格累加的元素个数
        if self._max_abs(self._mantissa) * count <= _MAX:
            return self._from_mantissa(total, self._scale)

        # 慢速保底路径：纯 int64 Base 10^9 分块累加判定真实求和，消除正负对冲假阳性误报
        exact_res = self._chunked_sum(self._mantissa, axis=axis, keepdims=keepdims)
        return self._from_mantissa(exact_res, self._scale)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False,
             scale: Optional[int] = None, rounding: Union[RoundingMode, str] = "half_up", out: None = None,
             dtype: None = None) -> "FixedArray":
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

        mode = self._check_rounding(rounding)
        target_scale = self._scale if scale is None else self._check_scale(scale)
        shift = target_scale - self._scale

        if self._mantissa.size == 0:
            raise ZeroDivisionError("空数组没有均值")

        ndim = self._mantissa.ndim
        AxisError = getattr(np.exceptions, "AxisError", getattr(np, "AxisError", IndexError))
        if axis is None:
            count = self._mantissa.size
        elif isinstance(axis, (int, np.integer)):
            if axis < -ndim or axis >= ndim:
                raise AxisError(f"axis {axis} is out of bounds for array of dimension {ndim}")
            count = self._mantissa.shape[axis]
        elif isinstance(axis, tuple):
            axes = set()
            count = 1
            for a in axis:
                if not isinstance(a, (int, np.integer)):
                    raise TypeError(f"tuple axis must contain integers, got {type(a).__name__}")
                norm_a = int(a if a >= 0 else a + ndim)
                if norm_a < 0 or norm_a >= ndim:
                    raise AxisError(f"axis {a} is out of bounds for array of dimension {ndim}")
                if norm_a in axes:
                    raise ValueError("duplicate value in 'axis'")
                axes.add(norm_a)
                count *= self._mantissa.shape[norm_a]
        else:
            raise TypeError(f"axis must be None, int, or tuple of ints, got {type(axis).__name__}")

        if count == 0:
            raise ZeroDivisionError("空数组没有均值")

        q, r = np.divmod(self._mantissa, count)
        q_sum = q.sum(axis=axis, keepdims=keepdims)
        r_sum = r.sum(axis=axis, keepdims=keepdims)
        q_r, r_tot = np.divmod(r_sum, count)
        q_tot = q_sum + q_r

        if shift == 0:
            res = self._round_quotient(q_tot, r_tot, count, mode)
        elif shift < 0:
            k = -shift
            p = 10 ** k
            q_k, r_k = np.divmod(q_tot, p)
            has_fraction = (r_k != 0) | (r_tot != 0)
            is_negative = (q_k < 0)
            odd_q = (q_k % 2 != 0)
            q_2r, r_2r = np.divmod(2 * r_tot, count)
            comp = 2 * r_k + q_2r
            over_half = (comp > p) | ((comp == p) & (r_2r > 0))
            exact_half = (comp == p) & (r_2r == 0)
            bump = self._decide_bump(
                mode,
                is_negative=is_negative,
                odd_q=odd_q,
                has_fraction=has_fraction,
                over_half=over_half,
                exact_half=exact_half,
                floor_base=True,
            )
            res = q_k + bump
        else:  # shift > 0
            top = self._max_abs(q_tot)
            if shift > 18 or (top and top > _MAX // 10 ** shift):
                raise OverflowError(f"均值乘以 10^{shift} 后超出 int64 范围")
            q_base = q_tot * 10 ** shift
            if count <= _MAX // 10 ** shift:
                q_frac, r_frac = np.divmod(r_tot * 10 ** shift, count)
            else:
                q_frac = np.zeros_like(r_tot)
                cur_r = r_tot.copy()
                k = shift
                chunk = 1
                while chunk < 18 and count * 10 ** (chunk + 1) <= _MAX:
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
            res = self._round_quotient(Q, r_frac, count, mode)
            if np.any((Q == _MAX) & (res < 0)):
                raise OverflowError(f"均值乘以 10^{shift} 后超出 int64 范围")

        return self._from_mantissa(res, target_scale)

    # ---------- 形状变换与转置（公开 API） ----------

    def reshape(self, *shape: Any) -> "FixedArray":
        """矩阵变形。"""
        return self._from_mantissa(self._mantissa.reshape(*shape), self._scale)

    def transpose(self, *axes: int) -> "FixedArray":
        """矩阵转置。"""
        return self._from_mantissa(self._mantissa.transpose(*axes), self._scale)

    @property
    def T(self) -> "FixedArray":
        return self.transpose()

    # ---------- 精度调整与网格对齐（公开 API） ----------

    def rescale(self, new_scale: int, rounding: str = "half_up") -> "FixedArray":
        """调整小数位数到 new_scale；变小时按 rounding 取整，变大时越界报 OverflowError。"""
        new_scale = self._check_scale(new_scale)
        if new_scale >= self._scale:
            if new_scale == self._scale:
                return self
            return self._from_mantissa(self._mul_pow10(self._mantissa, new_scale - self._scale), new_scale)
        divided = self._divmod_round(self._mantissa, 10 ** (self._scale - new_scale), rounding)
        return self._from_mantissa(divided, new_scale)

    def round_to_step(self, step: Any, rounding: str = "half_up") -> "FixedArray":
        """将定点数矩阵离散对齐到指定的步长网格。

        Parameters
        ----------
        step : int, float, Decimal, Sequence 或 FixedArray
            正数步长标量或与 self 广播兼容的步长矩阵。禁止传入字符串。
        rounding : {"half_up", "down", "up", "floor", "ceil"}，默认 "half_up"
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
        if np.any(b._mantissa <= 0):
            raise ValueError("步长 step 必须为正数")
        self._check_rounding(rounding)
        q = self._divmod_round(a._mantissa, b._mantissa, mode=rounding)
        return b.mul_round(q, rounding=None)

    def is_on_step(self, step: Any) -> Union[bool, np.ndarray]:
        """检查定点数是否严格位于指定步长的网格点上。

        Parameters
        ----------
        step : int, float, Decimal, Sequence 或 FixedArray
            正数步长标量或与 self 广播兼容的步长矩阵。禁止传入字符串。
        """
        step_fa = self._coerce(step)
        a, b = self._align_scales(self, step_fa)
        if np.any(b._mantissa <= 0):
            raise ValueError("步长 step 必须为正数")
        rem = a._mantissa % b._mantissa
        res = rem == 0
        return bool(res) if self.ndim == 0 and step_fa.ndim == 0 else res

    # ---------- 比较操作符 ----------

    def _compare(self, other: Any, op: Callable) -> Any:
        """对齐 scale 后比较尾数：标量返回 NumPy bool，数组返回 bool 数组；非数值类型返回 NotImplemented。"""
        other_fa = self._coerce(other, strict=False)
        if other_fa is None:
            return NotImplemented
        a, b = self._align_scales(self, other_fa)
        return op(a._mantissa, b._mantissa)

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        try:
            other_fa = self._coerce(other, strict=False)
        except (ValueError, OverflowError):  # 数值本身无法表示（NaN、超出 int64 等）视为不相等
            return False
        if other_fa is None:
            return NotImplemented
        return self._compare(other_fa, np.equal)  # 形状不匹配是真错误，照常抛出

    def __ne__(self, other: Any) -> Any:  # type: ignore[override]
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq if isinstance(eq, bool) else ~eq

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
        return self._from_mantissa(self._mantissa[key], self._scale)

    def __len__(self) -> int:
        return len(self._mantissa)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """禁止隐式转成 NumPy 数组（np.asarray、np.concatenate 等），避免悄悄变成 float。"""
        raise TypeError("FixedArray 不能隐式转成数组：要 float 请用 to_float()，要整数尾数请用 .mantissa")

    # ---------- 导出与类型转换（公开 API） ----------

    def to_float(self) -> Union[float, np.ndarray]:
        """转换为标准浮点数（标量返回 float，数组返回 float64 ndarray）。"""
        if self._mantissa.ndim == 0:
            return float(self._mantissa) / (10 ** self._scale)
        return self._mantissa.astype(np.float64) / (10 ** self._scale)

    def to_decimal(self) -> Union[Decimal, np.ndarray]:
        """转换为 Python 标准库 Decimal 对象（标量返回 Decimal，数组返回 object ndarray）。"""
        if self._mantissa.ndim == 0:
            return Decimal(int(self._mantissa)).scaleb(-self._scale)
        fn = np.vectorize(lambda m: Decimal(int(m)).scaleb(-self._scale), otypes=[object])
        return fn(self._mantissa)

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

        if self.ndim == 0:
            if as_float:
                return pl.Series(name, [self.to_float()])
            return pl.Series(name, np.array([self._mantissa])).cast(pl.Decimal(scale=self._scale))
        elif self.ndim == 1:
            if as_float:
                return pl.Series(name, self.to_float())
            return pl.Series(name, self._mantissa).cast(pl.Decimal(scale=self._scale))
        else:
            raise ValueError(f"to_polars 仅支持 1 维或标量 FixedArray，当前维度为 {self.ndim}")

    # ---------- 文本与展示 ----------

    def __repr__(self) -> str:
        return f"FixedArray({self}, scale={self._scale})"

    def __str__(self) -> str:
        if self._mantissa.ndim == 0:
            return self._format_scalar(int(self._mantissa), self._scale)
        return np.array2string(
            self._mantissa,
            formatter={"all": lambda x: self._format_scalar(int(x), self._scale)},
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


__all__ = ["FixedArray", "RoundingMode"]
