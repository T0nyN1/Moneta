"""UnitArray 基类的行为约定。

已确认的设计：
- 不改变单位的运算（+x、-x、abs、sum、切片、迭代、变形）由基类提供，结果保持原单位。
- ``==`` / ``!=`` 遇到单位不兼容的对象返回 NotImplemented：最终表现为不相等，且不阻断对方的反向比较。
- ``< <= > >=`` 遇到单位不兼容的 UnitArray 由子类报 TypeError；遇到其他类型返回 NotImplemented。
- 真值测试与 NumPy 一致；0 维禁止迭代；禁止 NumPy ufunc 与隐式转数组。
- 判断是否为 0 用 is_zero()，不要写 ``x == 0``（单位对象与裸数字永远不等）。

组织：先用测试专用的最小子类 Length 验证基类约定，再在 Money / Price / Quantity 上验证同样的约定。
"""

from decimal import Decimal

import numpy as np
import pytest

from moneta import Money, Price, Quantity
from moneta.fixed_array import FixedArray as F
from moneta.unit_array import UnitArray

MIN, MAX = -(2 ** 63), 2 ** 63 - 1


class Length(UnitArray):
    """测试专用的最小具体子类：长度，单位为字符串。"""

    __slots__ = ("_unit",)

    def __init__(self, values, unit="m"):
        self._data = values if isinstance(values, F) else F(values)
        self._unit = unit

    def _create_like(self, new_data):
        return Length(new_data, self._unit)

    def _check_compatible(self, other):
        if not isinstance(other, Length):
            raise TypeError(f"操作数必须是 Length，得到 {type(other).__name__}")
        if other._unit != self._unit:
            raise TypeError(f"单位不匹配: {self._unit} 与 {other._unit}")


def L(values, unit="m"):
    return Length(values, unit)


# ===========================================================================
# 1. 抽象钩子
# ===========================================================================


class TestAbstract:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            UnitArray()

    def test_subclass_must_implement_hooks(self):
        class Partial(UnitArray):
            def _create_like(self, new_data):
                return self

        with pytest.raises(TypeError):
            Partial()

    def test_slots(self):
        with pytest.raises(AttributeError):
            L("1").extra = 1


# ===========================================================================
# 2. 属性委托
# ===========================================================================


class TestProperties:
    def test_delegates_to_fixed_array(self):
        a = L([["1.5", "2"], ["3", "4"]])
        assert isinstance(a.data, F) and a.data.scale == 1
        assert a.mantissa.tolist() == [[15, 20], [30, 40]] and not a.mantissa.flags.writeable
        assert (a.scale, a.shape, a.ndim, a.size, a.dtype) == (1, (2, 2), 2, 4, np.int64)


# ===========================================================================
# 3. 容器协议、迭代与变形
# ===========================================================================


class TestContainer:
    def test_len_and_getitem_keep_unit(self):
        a = L(["1", "2", "3"], "km")
        assert len(a) == 3
        assert isinstance(a[1], Length) and a[1]._unit == "km" and a[1].ndim == 0
        assert a[1:].mantissa.tolist() == [2, 3] and a[1:]._unit == "km"
        with pytest.raises(TypeError):
            len(L("1"))

    def test_iteration(self):
        rows = list(L([["1", "2"], ["3", "4"]], "km"))
        assert len(rows) == 2 and all(isinstance(r, Length) and r._unit == "km" for r in rows)
        assert rows[1].mantissa.tolist() == [3, 4]
        with pytest.raises(TypeError, match="iteration over a 0-d Length"):
            iter(L("5"))
        assert list(L(np.array([], dtype=np.int64))) == []

    def test_reshape_and_transpose_keep_unit(self):
        a = L(["1", "2", "3", "4"], "km").reshape(2, 2)
        assert a.shape == (2, 2) and a._unit == "km"
        assert a.T.mantissa.tolist() == [[1, 3], [2, 4]] and a.T._unit == "km"
        assert a.transpose(1, 0).mantissa.tolist() == [[1, 3], [2, 4]]

    def test_truthiness_follows_numpy(self):
        assert bool(L("0")) is False and bool(L("0.5")) is True
        assert bool(L([[3]])) is True
        for ambiguous in (L(["0", "0"]), L(np.array([], dtype=np.int64))):
            with pytest.raises(ValueError, match="ambiguous"):
                bool(ambiguous)


# ===========================================================================
# 4. NumPy 防护
# ===========================================================================


class TestNumpyGuard:
    def test_ufuncs_rejected(self):
        a = L(["1", "2"])
        for op in (lambda: np.add(a, a), lambda: np.sqrt(a), lambda: np.multiply(np.array([1, 2]), a),
                   lambda: np.array([1, 2]) + a, lambda: np.array([1, 2]) * a):
            with pytest.raises(TypeError):
                op()

    def test_no_implicit_array(self):
        a = L(["1", "2"])
        for op in (lambda: np.asarray(a), lambda: np.array(a, dtype=float), lambda: np.concatenate([a, a]),
                   lambda: a.__array__()):
            with pytest.raises(TypeError, match="Length 不能隐式转成数组"):
                op()


# ===========================================================================
# 5. 不改变单位的运算
# ===========================================================================


class TestUnitPreservingOps:
    def test_pos_neg_abs(self):
        a = L(["-1.5", "2"], "km")
        assert +a is a
        assert (-a).mantissa.tolist() == [15, -20] and (-a)._unit == "km"
        assert abs(a).mantissa.tolist() == [15, 20] and abs(a)._unit == "km"
        for op in (lambda x: -x, abs):
            with pytest.raises(OverflowError):
                op(Length(F._from_mantissa(np.array([MIN]), 0)))

    def test_sum(self):
        a = L([["1.5", "2"], ["3", "4.25"]], "km")
        total = a.sum()
        assert isinstance(total, Length) and total._unit == "km" and total.data == F("10.75")
        assert a.sum(axis=0).mantissa.tolist() == [450, 625]
        assert a.sum(axis=1, keepdims=True).shape == (2, 1)
        assert np.sum(a).data == F("10.75") and np.sum(a)._unit == "km"
        assert Length(F._from_mantissa(np.array([MAX, 1, -1]), 0)).sum().mantissa == MAX  # 精确判定溢出
        with pytest.raises(OverflowError):
            Length(F._from_mantissa(np.array([MAX, 1]), 0)).sum()
        with pytest.raises(TypeError):
            np.sum(a, dtype=np.float64)

    def test_is_zero(self):
        assert L("0.00").is_zero() == np.True_ and L("0.01").is_zero() == np.False_
        assert isinstance(L("0").is_zero(), np.bool_)
        assert L(["0", "1.5", "-0"]).is_zero().tolist() == [True, False, True]


class TestAsFactor:
    def test_accepts_whitelist(self):
        assert UnitArray._as_factor(2, "乘数") == F(2)
        assert UnitArray._as_factor([F("1.5"), 2], "乘数").mantissa.tolist() == [15, 20]
        f = F("1.5")
        assert UnitArray._as_factor(f, "乘数") is f

    @pytest.mark.parametrize("value", ["2", True, None, L("1"), [1, "2"]])
    def test_rejects_non_numeric_with_business_message(self, value):
        with pytest.raises(TypeError, match="乘数只接受数值"):
            UnitArray._as_factor(value, "乘数")

    def test_construction_errors_pass_through(self):
        from fractions import Fraction
        with pytest.raises(ValueError, match="不是有限小数"):
            UnitArray._as_factor(Fraction(1, 3), "乘数")
        with pytest.raises(OverflowError):
            UnitArray._as_factor(2 ** 70, "乘数")
        with pytest.raises(ValueError):
            UnitArray._as_factor([1, [2, 3]], "乘数")


# ===========================================================================
# 6. 比较
# ===========================================================================


class TestComparison:
    def test_equality_same_unit(self):
        assert L("1.10") == L("1.1")
        assert (L(["1", "2"]) == L(["1", "3"])).tolist() == [True, False]
        assert (L(["1", "2"]) != L(["1", "3"])).tolist() == [False, True]
        assert isinstance(L("1") == L("1"), np.bool_)

    @pytest.mark.parametrize("other", [L("1", "km"), 1, F("1"), "1", None, Decimal("1")])
    def test_incompatible_is_not_equal(self, other):
        a = L("1")
        assert (a == other) is False and (a != other) is True
        assert a.__eq__(other) is NotImplemented and a.__ne__(other) is NotImplemented

    def test_equality_defers_to_other_operand(self):
        class Probe:
            def __eq__(self, other):
                return "probe-eq"

            def __ne__(self, other):
                return "probe-ne"

        assert (L("1") == Probe()) == "probe-eq"
        assert (L("1") != Probe()) == "probe-ne"

    def test_membership(self):
        assert L("2") in [0, "x", L("2.00")]
        assert L("2") not in [0, L("2", "km")]

    def test_shape_mismatch_is_a_real_error(self):
        with pytest.raises(ValueError):
            L(["1", "2"]) == L(["1", "2", "3"])

    def test_ordering_same_unit(self):
        a, b = L(["1", "2"]), L(["2", "1"])
        assert (a < b).tolist() == [True, False] and (a <= b).tolist() == [True, False]
        assert (a > b).tolist() == [False, True] and (a >= b).tolist() == [False, True]
        assert min(L("3"), L("2")).data == F("2")

    @pytest.mark.parametrize("op", [lambda a, b: a < b, lambda a, b: a <= b, lambda a, b: a > b, lambda a, b: a >= b])
    def test_ordering_incompatible(self, op):
        with pytest.raises(TypeError, match="单位不匹配"):
            op(L("1"), L("1", "km"))  # 不兼容的 UnitArray：子类给出明确原因
        for other in (0, F("1"), "1"):
            with pytest.raises(TypeError, match="not supported"):
                op(L("1"), other)  # 其他类型：返回 NotImplemented，由 Python 报错

    def test_ordering_defers_to_other_operand(self):
        class Probe:
            def __gt__(self, other):
                return "probe-gt"

        assert L("1").__lt__(Probe()) is NotImplemented
        assert (L("1") < Probe()) == "probe-gt"


# ===========================================================================
# 7. 导出
# ===========================================================================


class TestExports:
    def test_float_and_decimal(self):
        a = L(["0.1", "0.25"])
        assert a.to_float().tolist() == [0.1, 0.25]
        assert a.to_decimal().tolist() == [Decimal("0.10"), Decimal("0.25")]
        assert L("1.5").to_float() == 1.5 and L("1.5").to_decimal() == Decimal("1.5")

    def test_pandas(self):
        pytest.importorskip("pandas")
        s = L(["1", "2"]).to_pandas(index=["a", "b"], name="len")
        assert s.name == "len" and s.to_dict() == {"a": 1.0, "b": 2.0}
        assert L(["1.5"]).to_pandas(as_float=False).tolist() == [Decimal("1.5")]

    def test_polars(self):
        pytest.importorskip("polars")
        assert L(["1", "2"]).to_polars(name="len").to_list() == [1.0, 2.0]
        assert L(["1.50"]).to_polars(as_float=False).to_list() == [Decimal("1.50")]


# ===========================================================================
# 8. 在 Money / Price / Quantity 上的同样约定
# ===========================================================================

SUBCLASSES = {
    "Money": (lambda v: Money(v, "USD"), lambda v: Money(v, "CNY"), lambda x: x.currency),
    "Price": (lambda v: Price(v, "USD", tick_size=Decimal("0.01")), lambda v: Price(v, "USD", base_unit="barrel"),
              lambda x: (x.quote_currency, x.base_unit, x.tick_size)),
    "Quantity": (lambda v: Quantity(v, unit="share"), lambda v: Quantity(v, unit="barrel"), lambda x: x.unit),
}


@pytest.mark.parametrize("name", list(SUBCLASSES))
class TestSubclasses:
    def test_numpy_guard(self, name):
        make, _, _ = SUBCLASSES[name]
        a = make(["1", "2"])
        for op in (lambda: np.asarray(a), lambda: np.add(a, a), lambda: np.array([1, 2]) + a):
            with pytest.raises(TypeError):
                op()

    def test_truthiness_and_iteration(self, name):
        make, _, meta = SUBCLASSES[name]
        assert bool(make("0")) is False and bool(make("2")) is True
        with pytest.raises(ValueError):
            bool(make(["0", "0"]))
        with pytest.raises(TypeError, match=f"0-d {name}"):
            list(make("5"))
        items = list(make(["1", "2"]))
        assert len(items) == 2 and all(type(x).__name__ == name and meta(x) == meta(make("1")) for x in items)

    def test_unit_preserving_ops(self, name):
        make, _, meta = SUBCLASSES[name]
        a = make(["-1", "3"])
        for result in (+a, -a, abs(a), a.sum(), a[0], a.reshape(2, 1), a.T):
            assert type(result).__name__ == name and meta(result) == meta(a)
        assert np.all(-a == make(["1", "-3"])) and np.all(abs(a) == make(["1", "3"]))
        assert a.sum() == make("2")
        assert a.is_zero().tolist() == [False, False] and make("0").is_zero()

    def test_numpy_sum_dispatch(self, name):
        make, _, meta = SUBCLASSES[name]
        total = np.sum(make(["-1", "3"]))
        assert total == make("2") and meta(total) == meta(make("1"))

    def test_comparison_rules(self, name):
        make, make_other, _ = SUBCLASSES[name]
        a = make("5")
        assert a == make("5.00") and a != make("6")
        assert (a == 5) is False and (a != 5) is True  # 与裸数字永远不等
        assert (a == make_other("5")) is False and (a != make_other("5")) is True
        assert a < make("6") and a >= make("5")
        with pytest.raises(TypeError):
            a < make_other("6")
        with pytest.raises(TypeError, match="not supported"):
            a < 6
