"""Moneta 资金张量核心体系 (Money Tensor)。

定义广延资金量（Extensive Quantity），绑定单一币种，支持高维张量代数与守恒分配。

约定：
- Money 的精度恒等于币种精度（currency.digits），不存在更高精度的中间金额。
- 构造时 float 按最短写法 half_up 舍入；精确输入（str、Decimal、Fraction、FixedArray 等）超出币种精度时
  由 FixedArray 发出 PrecisionLossWarning 后按 half_up 舍入。
- 运算只接受数值白名单内的操作数，拒绝字符串；运算符遇到不认识的类型返回 NotImplemented。
- 除法结果若不是金额（比率、单价、数量），按 FixedArray.divide 的默认精度：默认保留 max(两侧 scale) + 8 位小数（商放不下 int64 时少保留几位），并去掉末尾的 0；可用 scale 指定。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._rounding import check_rounding, decide_bump
from .currency import Currency, get_currency
from .fixed_array import FixedArray, RoundingMode
from .unit_array import UnitArray


Rounding = Union[RoundingMode, str]


class Money(UnitArray):
    """金融货币金额张量（支持标量与高维 NumPy 矩阵）。

    底层由 FixedArray 承载，强制绑定特定 Currency，其 scale 严格恒等于 currency.digits。
    继承自 UnitArray，自动获得切片、形状变形、容器协议、取负、绝对值、求和及 Pandas 数据导出能力。

    Parameters
    ----------
    values : int, float, str, Decimal, Fraction, FixedArray, Money, Sequence, np.ndarray 或 pandas / polars Series
        以主货币单位（如“元”）表示的数值或数组。float 按最短写法 half_up 舍入到币种精度；
        精确输入超出币种精度时发出 PrecisionLossWarning 后按 half_up 舍入。
    currency : Union[Currency, str]
        所属币种，支持传入 Currency 对象或已注册的代码字符串（如 "CNY"）。
    """

    __slots__ = ("_currency",)

    def __init__(self, values: Any, currency: Union[Currency, str]) -> None:
        curr = get_currency(currency)
        if isinstance(values, Money):
            values._check_compatible_currency(curr)
            self._data = values._data
        else:  # float 静默舍入；精确输入丢精度时由 FixedArray 告警后舍入
            self._data = FixedArray(values, scale=curr.digits, on_excess="warn")
        self._currency = curr

    # ---------- 核心抽象钩子实现 ----------

    def _create_like(self, new_data: FixedArray) -> "Money":
        """从新底层 FixedArray 构建同币种的新 Money 实例；精度必须恰好等于币种精度。"""
        return self._wrap(new_data, self._currency)

    def _check_compatible(self, other: Any) -> None:
        """检查另一个对象是否是同币种的 Money。"""
        if not isinstance(other, Money):
            raise TypeError(f"操作数必须是 Money，得到 {type(other).__name__}")
        if self._currency != other._currency:
            raise TypeError(f"币种不匹配: {self._currency.code} 与 {other._currency.code}")

    def _check_compatible_currency(self, currency: Currency) -> None:
        if self._currency != currency:
            raise TypeError(f"币种不匹配: {self._currency.code} 与 {currency.code}")

    @classmethod
    def _wrap(cls, data: FixedArray, currency: Currency) -> "Money":
        """内部零拷贝构造：data 的精度必须恰好等于币种精度，否则说明内部逻辑漏了舍入。"""
        if data.scale != currency.digits:
            raise ValueError(f"内部错误：金额精度 {data.scale} 与 {currency.code} 的精度 {currency.digits} 不一致")
        obj = cls.__new__(cls)
        obj._data = data
        obj._currency = currency
        return obj

    # ---------- 工厂构造方法 ----------

    @classmethod
    def from_units(cls, units: Any, currency: Union[Currency, str]) -> "Money":
        """由最小原子单位的整数（int 或整数数组）构造；传入的数组会被复制，之后修改原数组不影响 Money。"""
        curr = get_currency(currency)
        m = FixedArray._as_int64(units)
        if isinstance(units, np.ndarray) and np.may_share_memory(m, units):
            m = m.copy()
        return cls._wrap(FixedArray._from_mantissa(m, curr.digits), curr)

    @classmethod
    def zeros(cls, shape: Union[int, Tuple[int, ...]], currency: Union[Currency, str]) -> "Money":
        """创建指定维度的零金额张量。"""
        return cls.from_units(np.zeros(shape, dtype=np.int64), currency)

    @classmethod
    def ones(cls, shape: Union[int, Tuple[int, ...]], currency: Union[Currency, str]) -> "Money":
        """创建指定维度的单位主货币金额张量（各元素均为 1.00... 主单位）。"""
        curr = get_currency(currency)
        return cls.from_units(np.full(shape, 10 ** curr.digits, dtype=np.int64), curr)

    # ---------- 核心属性 ----------

    @property
    def units(self) -> np.ndarray:
        """以原子单位表示的只读纯整数矩阵（int64）。缺失位置为 0，请配合 isna() 使用。"""
        return self.mantissa

    @property
    def currency(self) -> Currency:
        """绑定的币种对象。"""
        return self._currency

    # ---------- 内部工具 ----------

    def _fill_value(self, value: Any) -> FixedArray:
        """数值填充值按 Money 的构造规则换算到币种精度（精确输入超出精度时发出 PrecisionLossWarning）。"""
        self._as_factor(value, "填充值")  # 只做类型校验：float 与精确输入的舍入规则由 Money 构造决定
        return Money(value, self._currency)._data

    def _as_step(self, step: Any) -> FixedArray:
        """校验步长：正数、不是缺失值，且精度不超过币种精度（多余的末尾 0 会被去掉）。"""
        step_fa = self._as_grid(step, "步长")
        if np.any(step_fa.mantissa <= 0):
            raise ValueError("步长 step 必须为正数")
        digits = self._currency.digits
        if step_fa.scale > digits:
            reduced = step_fa.rescale(digits, "down")
            if not np.all(reduced == step_fa):
                raise ValueError(f"步长 {step_fa} 的精度超过 {self._currency.code} 的精度 ({digits})")
            step_fa = reduced
        return step_fa

    def _fused_round_step(self, floor: FixedArray, inexact: Any, step: FixedArray, mode: str) -> FixedArray:
        """由 floor(v)（精度为 step.scale + 1）与「v 是否大于它」一次性把 v 舍入到步长网格。

        所有步长倍数与半步长边界都落在 10^-(step.scale + 1) 的网格上，floor(v) 与 v 之间不会夹着边界，
        且 v 恰在边界上当且仅当没有余数——因此只舍入一次，结果精确。
        """
        unit = FixedArray._mul_pow10(step.mantissa, 1)  # 以 10^-(step.scale + 1) 为单位的步长，越界报 OverflowError
        q, r = np.divmod(floor.mantissa, unit)  # unit > 0：0 ≤ r < unit
        bump = decide_bump(
            mode,
            is_negative=q < 0,
            odd_q=q % 2 != 0,
            has_fraction=(r != 0) | inexact,
            over_half=(r > unit - r) | ((r == unit - r) & inexact),
            exact_half=(r == unit - r) & ~inexact,
            floor_base=True,
        )
        k = FixedArray._from_mantissa(q + bump, 0, mask=floor.mask)  # 缺失值沿用 floor 的掩码
        return k.mul_round(step, rounding=None).rescale(self._currency.digits)

    # ---------- 同币种加减 ----------

    def __add__(self, other: Any) -> "Money":
        """同币种金额相加。币种不匹配时报 TypeError；非 Money 返回 NotImplemented。"""
        if not isinstance(other, Money):
            return NotImplemented
        self._check_compatible(other)
        return self._create_like(self._data + other._data)

    def __sub__(self, other: Any) -> "Money":
        """同币种金额相减。币种不匹配时报 TypeError；非 Money 返回 NotImplemented。"""
        if not isinstance(other, Money):
            return NotImplemented
        self._check_compatible(other)
        return self._create_like(self._data - other._data)

    # ---------- 标量与矩阵乘除 (无量纲系数缩放) ----------

    def multiply(self, other: Any, rounding: Rounding = "half_up", step: Optional[Any] = None) -> "Money":
        """金额乘以无量纲标量或权重矩阵，按 rounding 舍入到币种精度；给出 step 时一次性舍入到步长网格。"""
        factor = self._as_factor(other, "乘数")
        mode = check_rounding(rounding)
        if step is None:
            return self._create_like(self._data.mul_round(factor, scale=self._currency.digits, rounding=mode))
        step_fa = self._as_step(step)
        t = step_fa.scale + 1
        floor = self._data.mul_round(factor, scale=t, rounding="floor")
        inexact = floor != self._data.mul_round(factor, scale=t, rounding="ceil")
        return self._create_like(self._fused_round_step(floor, inexact, step_fa, mode))

    def divide(self, other: Any, rounding: Rounding = "half_up", step: Optional[Any] = None,
               scale: Optional[int] = None) -> Union["Money", FixedArray, Any]:
        """金额除法：
        - Money / 无量纲数值 → Money，按 rounding 舍入到币种精度；给出 step 时一次性舍入到步长网格；
        - Money / 同币种 Money → 无量纲比率 FixedArray；
        - Money / Price → Quantity（数量）；Money / Quantity → Price（单价）。
        结果不是 Money 时默认保留 max(两侧 scale) + 8 位小数（商放不下 int64 时少保留几位），并去掉末尾的 0，可用 scale 指定；此时不接受 step。
        结果是 Money 时精度固定为币种精度，不接受 scale。
        """
        from .price import Price
        from .quantity import Quantity

        mode = check_rounding(rounding)
        if isinstance(other, (Money, Price, Quantity)):
            if step is not None:
                raise ValueError("step 只适用于结果为 Money 的除法")
            target = None if scale is None else FixedArray._check_scale(scale)
            if isinstance(other, Money):
                self._check_compatible(other)
                return self._data.divide(other._data, scale=target, rounding=mode)
            if isinstance(other, Price):
                if other.quote_currency != self._currency:
                    raise TypeError(f"金额币种 {self._currency.code} 与标价计价货币 {other.quote_currency.code} 不一致")
                return Quantity(self._data.divide(other.data, scale=target, rounding=mode), unit=other.base_unit)
            return Price(self._data.divide(other.data, scale=target, rounding=mode),
                         quote_currency=self._currency, base_unit=other.unit)

        if scale is not None:
            raise ValueError("结果为 Money 时精度固定为币种精度，不能指定 scale")
        divisor = self._as_factor(other, "除数")
        if step is None:
            return self._create_like(self._data.divide(divisor, scale=self._currency.digits, rounding=mode))
        step_fa = self._as_step(step)
        t = step_fa.scale + 1
        floor = self._data.divide(divisor, scale=t, rounding="floor")
        inexact = floor != self._data.divide(divisor, scale=t, rounding="ceil")
        return self._create_like(self._fused_round_step(floor, inexact, step_fa, mode))

    def __mul__(self, other: Any) -> "Money":
        """金额乘以无量纲倍数，half_up 舍入到币种精度。

        禁止 Money * Money（货币量纲平方无意义）；其他非数值类型返回 NotImplemented。
        """
        if isinstance(other, Money):
            raise TypeError("禁止两笔金额直接相乘 (Money * Money)：货币量纲平方无物理与金融意义。")
        if not FixedArray._is_numeric_operand(other):
            return NotImplemented
        return self.multiply(other)

    def __rmul__(self, other: Any) -> "Money":
        if not FixedArray._is_numeric_operand(other):
            return NotImplemented
        return self.multiply(other)

    def __truediv__(self, other: Any) -> Union["Money", FixedArray, Any]:
        from .price import Price
        from .quantity import Quantity

        if not isinstance(other, (Money, Price, Quantity)) and not FixedArray._is_numeric_operand(other):
            return NotImplemented
        return self.divide(other)

    def matmul(self, other: Any, rounding: Rounding = "half_up") -> "Money":
        """矩阵内积：Money 向量/矩阵 @ 无量纲权重，结果按 rounding 舍入到币种精度。"""
        weights = self._as_factor(other, "权重")
        return self._create_like(self._data.matmul(weights, scale=self._currency.digits, rounding=rounding))

    def __matmul__(self, other: Any) -> "Money":
        if not FixedArray._is_numeric_operand(other):
            return NotImplemented
        return self.matmul(other)

    def __rmatmul__(self, other: Any) -> "Money":
        """无量纲权重 @ Money 向量/矩阵，half_up 舍入到币种精度。"""
        if not FixedArray._is_numeric_operand(other):
            return NotImplemented
        weights = FixedArray._coerce(other)
        return self._create_like(weights.matmul(self._data, scale=self._currency.digits, rounding="half_up"))

    # ---------- 张量规约 ----------

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False,
             rounding: Rounding = "half_up", skipna: bool = True, out: None = None, dtype: None = None) -> "Money":
        """沿指定轴计算算术平均金额（按 rounding 舍入到币种精度）。``np.mean(m)`` 也会转到这里；skipna 同 FixedArray.mean。"""
        return self._create_like(self._data.mean(axis=axis, keepdims=keepdims, rounding=rounding, skipna=skipna,
                                                 out=out, dtype=dtype))

    # ---------- 离散步长与网格对齐 ----------

    def round_to_step(self, step: Any, rounding: Rounding = "half_up") -> "Money":
        """将金额对齐到步长网格（如 0.05）；步长的精度不能超过币种精度。"""
        return self._create_like(self._data.round_to_step(self._as_step(step), rounding=rounding))

    def is_on_step(self, step: Any) -> Union[bool, np.ndarray]:
        """检查金额是否严格位于指定的步长格点上（缺失值不在任何格点上）。"""
        return self._data.is_on_step(self._as_grid(step, "步长"))

    # ---------- 守恒整除与资产分配 ----------

    def __divmod__(self, other: Any) -> Tuple[Any, "Money"]:
        """守恒带余除法：
        - divmod(Money, 同币种 Money) → (整数份数, Money 零头)，即能分出几份 other；
        - divmod(Money, 正整数 n) → (每份 Money, Money 零头)，即均分为 n 份。
        二者都满足 份数 × 除数 + 零头 == 原金额，向下取整（与 Python divmod 一致，零头 ≥ 0）。
        非整数的数值报 TypeError：按比例分配请使用 allocate。
        """
        if isinstance(other, Money):
            self._check_compatible(other)
            count, rest = divmod(self._data, other._data)
            return count, self._create_like(rest)
        if isinstance(other, UnitArray):
            return NotImplemented
        is_integer = isinstance(other, (int, np.integer)) or (
                    isinstance(other, np.ndarray) and other.dtype.kind in "iu")
        if isinstance(other, (bool, np.bool_)) or not is_integer:
            if FixedArray._is_numeric_operand(other):
                raise TypeError(f"均分份数必须是整数，收到 {type(other).__name__}；按比例分配请使用 allocate")
            return NotImplemented
        parts = FixedArray._as_int64(other)
        if np.any(parts <= 0):
            raise ValueError("均分份数必须是正整数")
        per_part, rest = np.divmod(self.mantissa, parts)
        digits, mask = self._currency.digits, self._data.mask  # 缺失的金额：每份与零头都是缺失值
        return (self._create_like(FixedArray._from_mantissa(per_part, digits, mask=mask)),
                self._create_like(FixedArray._from_mantissa(rest, digits, mask=mask)))

    def allocate(self, ratios: Any) -> List["Money"]:
        """按权重用最大余数法（Hare-Niemeyer）守恒分配：各份之和恰好等于原金额。

        权重可以是 int、float、Decimal、Fraction（有限小数）、FixedArray 或一维数值数组，必须非负且不全为 0。
        余数相同时，排在前面的权重优先多得一个最小单位。
        """
        if self.ndim != 0:
            raise ValueError("allocate 仅支持标量金额 (ndim == 0)")
        if self._data.mask is not None:
            raise ValueError("不能分配缺失的金额")
        weights_fa = self._as_factor(ratios, "分配权重")
        if weights_fa.ndim != 1:
            raise ValueError("分配权重必须是一维序列")
        if weights_fa.mask is not None:
            raise ValueError("分配权重不能是缺失值")
        if weights_fa.size == 0:
            return []
        weights = [int(w) for w in weights_fa.mantissa]
        if any(w < 0 for w in weights):
            raise ValueError("分配权重必须非负")
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("权重总和不能为 0")

        total = int(self.mantissa)
        shares, remainders = [], []
        for w in weights:
            share, rem = divmod(total * w, total_weight)
            shares.append(share)
            remainders.append(rem)
        order = sorted(range(len(weights)), key=lambda i: remainders[i], reverse=True)
        for idx in order[:total - sum(shares)]:
            shares[idx] += 1
        return [self.from_units(s, self._currency) for s in shares]

    def calc_tiered_fee(
            self,
            rate: Union[Any, Sequence[Tuple[Optional[Any], Any]]],
            min_fee: Optional["Money"] = None,
            max_fee: Optional["Money"] = None,
            rounding: Rounding = "half_up",
    ) -> "Money":
        """交易费与印花税计算：单一费率，或分档累进费率表；支持最低收费与封顶，逐元素生效。

        Parameters
        ----------
        rate : 数值 或 [(阈值, 费率), ...]
            单一费率：数值或与金额广播的数值数组。
            分档费率：阈值为该档上限（Money 或数值），必须严格递增且为正，最后一档阈值必须为 None（不封顶）。
            各档费用先精确累加，最后只按 rounding 舍入一次。
        min_fee, max_fee : Money, optional
            同币种的最低收费与封顶，可与金额广播；min_fee 不能大于 max_fee。

        缺失值逐元素传播：金额、费率、min_fee 或 max_fee 缺失的位置，手续费为缺失值；分档阈值不能是缺失值。
        """
        mode = check_rounding(rounding)
        digits = self._currency.digits
        notional = abs(self._data)  # 负金额（如卖出）按绝对值计费，单一费率与分档费率一致
        is_tiers = isinstance(rate, (list, tuple)) and len(rate) > 0 and all(
            isinstance(t, (list, tuple)) and len(t) == 2 for t in rate)
        if is_tiers:
            exact = None
            lower = 0
            for i, (threshold, tier_rate) in enumerate(rate):
                tier_factor = self._as_factor(tier_rate, "费率")
                if threshold is None:
                    if i != len(rate) - 1:
                        raise ValueError("只有最后一档的阈值可以是 None（不封顶）")
                    upper = None
                else:
                    th = threshold if isinstance(threshold, Money) else Money(self._as_factor(threshold, "阈值"),
                                                                              self._currency)
                    self._check_compatible(th)
                    if th.ndim != 0:
                        raise ValueError("分档阈值必须是标量")
                    if th.isna():
                        raise ValueError("分档阈值不能是缺失值")
                    upper = int(th.mantissa)
                    if upper <= lower:
                        raise ValueError("分档阈值必须为正且严格递增")
                clipped = np.maximum(notional.mantissa, lower)
                if upper is not None:
                    clipped = np.minimum(clipped, upper)
                part = FixedArray._from_mantissa(clipped - lower, digits, mask=notional.mask).mul_round(
                    tier_factor, rounding=None)
                exact = part if exact is None else exact + part
                if upper is not None:
                    lower = upper
            if rate[-1][0] is not None:
                raise ValueError("最后一档的阈值必须是 None（不封顶）")
            fee_data = exact.rescale(digits, mode)
        else:
            fee_data = notional.mul_round(self._as_factor(rate, "费率"), scale=digits, rounding=mode)

        fee_m, mask = fee_data.mantissa, fee_data.mask
        for bound in (min_fee, max_fee):
            if bound is not None:
                self._check_compatible(bound)
                mask = FixedArray._combine_masks(mask, bound.data.mask)
        if min_fee is not None and max_fee is not None:
            both = min_fee.notna() & max_fee.notna()
            if np.any((min_fee.mantissa > max_fee.mantissa) & both):
                raise ValueError("min_fee 不能大于 max_fee")
        if min_fee is not None:
            fee_m = np.maximum(fee_m, min_fee.mantissa)
        if max_fee is not None:
            fee_m = np.minimum(fee_m, max_fee.mantissa)
        return self._create_like(FixedArray._from_mantissa(fee_m, digits, mask=mask))

    # ---------- 导出与展示 ----------

    def to_pandas(self, index: Optional[Any] = None, name: Optional[str] = None, as_float: bool = True) -> Any:
        """导出为 pandas.Series 或 pandas.DataFrame（自动注入 currency 元数据）。"""
        res = self.data.to_pandas(index=index, name=name, as_float=as_float)
        res.attrs["currency"] = self.currency.code
        return res

    def format(self, accounting: bool = False, show_symbol: bool = True, show_code: bool = True,
               thousands_sep: str = ",", decimal_sep: str = ".") -> str:
        """格式化标量金额，参数见 MonetaryFormatter.format。"""
        from .formatter import MonetaryFormatter
        return MonetaryFormatter.format(self, accounting=accounting, show_symbol=show_symbol, show_code=show_code,
                                        thousands_sep=thousands_sep, decimal_sep=decimal_sep)

    def __repr__(self) -> str:
        return f"Money({self._data!r}, currency='{self._currency.code}')"

    def __str__(self) -> str:
        return f"{self._data} {self._currency.code}"


__all__ = ["Money"]
