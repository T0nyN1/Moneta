"""Quantity 的行为约定。标准答案来自 Fraction 与 Decimal。

已确认的设计：
- 数量没有固定精度：运算精确，乘法结果的 scale 为两侧之和；取整到整股、整手须显式调用
  multiply(scale=...)、rescale、round_to_step。
- 除以数值或同单位数量：默认保留 max(两侧 scale) + 8 位小数，可用 scale 指定。
- 构造指定 scale 时与 Money 一致：float 按最短写法 half_up 静默舍入；精确输入超出 scale 时
  发出 PrecisionLossWarning 后 half_up 舍入；不足时补 0。
- 单位为非空字符串（去首尾空白、区分大小写）；"Any" 为通配单位，可与任意单位运算，结果取具体单位。
- 运算只接受数值白名单内的操作数，拒绝字符串；运算符遇到不认识的类型返回 NotImplemented。
  Quantity * Price 交给 Price 计算得到 Money；Quantity * Money、Quantity * Quantity 报 TypeError。

组织：构造、单位、加减、乘法、除法、显式取整、与 Price / Money 的交互、导出与展示。
"""

import random
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from moneta import Money, PrecisionLossWarning, Price, Quantity
from moneta.fixed_array import FixedArray as F, RoundingMode
from moneta.quantity import WILDCARD_UNIT

MODES = [m.value for m in RoundingMode]
MIN, MAX = -(2 ** 63), 2 ** 63 - 1


def fits(v: int) -> bool:
    return MIN <= v <= MAX


def ref_round(x: Fraction, mode: str) -> int:
    """x 按 mode 取整到整数（精确参照）。"""
    floor = x.numerator // x.denominator
    frac = x - floor
    if frac == 0:
        return floor
    if mode == "floor":
        return floor
    if mode == "ceil":
        return floor + 1
    if mode == "down":
        return floor + 1 if x < 0 else floor
    if mode == "up":
        return floor if x < 0 else floor + 1
    half = Fraction(1, 2)
    if frac != half:
        return floor + 1 if frac > half else floor
    if mode == "half_up":
        return floor + 1 if x > 0 else floor
    return floor if floor % 2 == 0 else floor + 1


def value(q) -> Fraction:
    return Fraction(int(q.mantissa), 10 ** q.scale)


def raw(m, scale, unit="share"):
    return Quantity(F._from_mantissa(np.asarray(m, dtype=np.int64), scale), unit=unit)


@pytest.fixture
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ===========================================================================
# 1. 构造
# ===========================================================================


class TestConstruct:
    @pytest.mark.parametrize("values, mantissa, scale", [
        (100, 100, 0), ("1.50", 150, 2), (Decimal("0.001"), 1, 3), (Fraction(1, 8), 125, 3), (1.5, 15, 1),
        (0.1 + 0.2, 30000000000000004, 17), (F("2.5"), 25, 1), (["1", "2.5"], [10, 25], 1),
        (np.array([1, 2]), [1, 2], 0),
    ])
    def test_scale_inferred_from_input(self, values, mantissa, scale, no_warnings):
        q = Quantity(values)
        got = q.mantissa.tolist() if q.ndim else int(q.mantissa)
        assert (got, q.scale, q.unit) == (mantissa, scale, "share")

    @pytest.mark.parametrize("values, scale, mantissa", [(1.5, 0, 2), (2.5, 0, 3), (-1.5, 0, -2), (0.125, 2, 13),
                                                         (np.array([1.5, 0.25]), 1, [15, 3])])
    def test_float_with_scale_rounds_silently(self, values, scale, mantissa, no_warnings):
        q = Quantity(values, scale=scale)
        assert (q.mantissa.tolist() if q.ndim else int(q.mantissa), q.scale) == (mantissa, scale)

    @pytest.mark.parametrize("values, scale, mantissa", [("1.5", 0, 2), (Decimal("-1.25"), 1, -13), (F("0.125"), 2, 13),
                                                         (Fraction(1, 8), 1, 1), (["1.05", "2"], 1, [11, 20])])
    def test_exact_input_beyond_scale_warns_then_rounds(self, values, scale, mantissa):
        with pytest.warns(PrecisionLossWarning, match="超出指定 scale"):
            q = Quantity(values, scale=scale)
        assert (q.mantissa.tolist() if q.ndim else int(q.mantissa), q.scale) == (mantissa, scale)

    def test_trailing_zeros_do_not_warn(self, no_warnings):
        assert Quantity("1.50", scale=1).mantissa == 15
        assert Quantity(["2.000", "-3.10"], scale=1).mantissa.tolist() == [20, -31]
        assert Quantity(F._from_mantissa(0, 40), scale=0).mantissa == 0

    def test_scale_pads(self, no_warnings):
        assert (int(Quantity("1.5", scale=4).mantissa), Quantity(2, scale=3).scale) == (15000, 3)

    def test_invalid_scale(self):
        for bad in (-1, 1.5, True):
            with pytest.raises(ValueError):
                Quantity("1", scale=bad)
            with pytest.raises(ValueError):
                Quantity(1.5, scale=bad)

    def test_from_quantity(self, no_warnings):
        q = Quantity("1.5", unit="barrel")
        assert Quantity(q, unit="barrel") == q
        assert Quantity(Quantity(3, unit=WILDCARD_UNIT), unit="barrel").unit == "barrel"
        with pytest.raises(TypeError, match="单位不匹配"):
            Quantity(q, unit="share")

    @pytest.mark.parametrize("bad", [True, object(), np.array([True])])
    def test_invalid_values(self, bad):
        with pytest.raises((TypeError, ValueError)):
            Quantity(bad)


# ===========================================================================
# 2. 单位
# ===========================================================================


class TestUnit:
    def test_normalization(self):
        assert Quantity(1, unit="  barrel ").unit == "barrel"
        assert Quantity(1, unit="BTC").unit == "BTC"
        assert Quantity(1, unit="BTC") != Quantity(1, unit="btc")  # 区分大小写

    @pytest.mark.parametrize("bad, exc", [(None, TypeError), (1, TypeError), (b"share", TypeError), ("", ValueError),
                                          ("   ", ValueError)])
    def test_invalid(self, bad, exc):
        with pytest.raises(exc):
            Quantity(1, unit=bad)

    def test_wildcard(self):
        a, b = Quantity(3, unit=WILDCARD_UNIT), Quantity(2, unit="barrel")
        assert (a + b).unit == "barrel" and (b + a).unit == "barrel" and (a - b).unit == "barrel"
        assert (a + Quantity(1, unit=WILDCARD_UNIT)).unit == WILDCARD_UNIT
        assert Quantity(5, unit=WILDCARD_UNIT) == Quantity(5, unit="barrel")
        assert Price("2", "USD") * a == Money("6", "USD")

    def test_mismatch(self):
        with pytest.raises(TypeError, match="单位不匹配"):
            Quantity(1) + Quantity(1, unit="barrel")
        with pytest.raises(TypeError, match="单位不匹配"):
            Quantity(1) < Quantity(1, unit="barrel")
        assert (Quantity(1) == Quantity(1, unit="barrel")) is False

    def test_unary_and_slicing_keep_unit(self):
        q = Quantity([-1, 2], unit="barrel")
        for r in (-q, abs(q), +q, q.sum(), q[0], q.reshape(2, 1), q.T):
            assert r.unit == "barrel"


# ===========================================================================
# 3. 加减
# ===========================================================================


class TestAddSub:
    def test_exact(self):
        assert Quantity("1.5") + Quantity(2) == Quantity("3.5")
        assert (Quantity("1.5") + Quantity(2)).scale == 1
        assert Quantity(1) - Quantity("0.001") == Quantity("0.999")
        assert np.all(Quantity([1, 2]) + Quantity(1) == Quantity([2, 3]))

    @pytest.mark.parametrize("other", [1, Decimal("1"), F("1"), "1", None, Money("1", "USD"), Price("1", "USD")])
    def test_non_quantity(self, other):
        assert Quantity(1).__add__(other) is NotImplemented and Quantity(1).__sub__(other) is NotImplemented
        for op in (lambda: Quantity(1) + other, lambda: other + Quantity(1), lambda: Quantity(1) - other):
            with pytest.raises(TypeError):
                op()


# ===========================================================================
# 4. 乘法
# ===========================================================================


class TestMultiply:
    def test_exact_by_default(self):
        assert Quantity(3) * 0.5 == Quantity("1.5") and (Quantity(3) * 0.5).scale == 1
        assert Quantity("1.5") * Decimal("0.33") == Quantity("0.495")
        assert 2 * Quantity("1.25") == Quantity("2.5") and Fraction(1, 4) * Quantity(2) == Quantity("0.5")
        assert np.all(np.array([1, 2]) * Quantity(["1", "2"]) == Quantity([1, 4]))
        assert F("2") * Quantity(3) == Quantity(6)

    def test_with_scale_and_rounding(self):
        assert Quantity("1.37").multiply(3, scale=1) == Quantity("4.1")
        assert Quantity("1.37").multiply(3, scale=0, rounding="down") == Quantity(4)
        assert Quantity(3).multiply(Decimal("0.5"), scale=0, rounding="half_even") == Quantity(2)
        assert Quantity(3).multiply(Decimal("0.5"), scale=3) == Quantity("1.500")
        with pytest.raises(ValueError, match="rounding 只能是"):
            Quantity(1).multiply(2, rounding="nearest")

    def test_random_against_fraction(self):
        rng = random.Random(1)
        for _ in range(2000):
            a, sa = rng.randint(-10 ** 9, 10 ** 9), rng.randint(0, 6)
            b, sb = rng.randint(-10 ** 6, 10 ** 6), rng.randint(0, 6)
            q, f = raw(a, sa), F._from_mantissa(np.int64(b), sb)
            exact = q * f
            assert value(exact) == Fraction(a, 10 ** sa) * Fraction(b, 10 ** sb) and exact.scale == sa + sb
            t, mode = rng.randint(0, 6), rng.choice(MODES)
            want = ref_round(Fraction(a * b, 10 ** (sa + sb)) * 10 ** t, mode)
            if fits(want):
                assert int(q.multiply(f, scale=t, rounding=mode).mantissa) == want
            else:
                with pytest.raises(OverflowError):
                    q.multiply(f, scale=t, rounding=mode)

    @pytest.mark.parametrize("bad", ["2", True, Quantity(2), Money("1", "USD")])
    def test_rejects_non_numeric(self, bad):
        assert Quantity(3).__mul__(bad) is NotImplemented and Quantity(3).__rmul__(bad) is NotImplemented
        with pytest.raises(TypeError):
            Quantity(3) * bad
        if not isinstance(bad, (Quantity, Money)):
            with pytest.raises(TypeError, match="只接受数值"):
                Quantity(3).multiply(bad)


# ===========================================================================
# 5. 除法
# ===========================================================================


class TestDivide:
    def test_by_number(self):
        q = Quantity(10) / 4
        assert q == Quantity("2.5") and q.scale == 8 and q.unit == "share"
        assert (Quantity("1.5") / Decimal("0.25")).scale == 10  # max(1, 2) + 8
        assert Quantity(1).divide(3, scale=2, rounding="up") == Quantity("0.34")
        with pytest.raises(ZeroDivisionError):
            Quantity(1) / 0

    def test_ratio(self):
        r = Quantity(1) / Quantity(3)
        assert isinstance(r, F) and r == F("0.33333333") and r.scale == 8
        assert Quantity(1, unit=WILDCARD_UNIT) / Quantity(4, unit="barrel") == F("0.25")
        assert Quantity("1").divide(Quantity(3), scale=3, rounding="floor") == F("0.333")
        with pytest.raises(TypeError, match="单位不匹配"):
            Quantity(1) / Quantity(1, unit="barrel")

    def test_random_against_fraction(self):
        rng = random.Random(2)
        for _ in range(2000):
            a, sa = rng.randint(-10 ** 9, 10 ** 9), rng.randint(0, 6)
            b, sb = rng.randint(1, 10 ** 6) * rng.choice([1, -1]), rng.randint(0, 6)
            t, mode = rng.randint(0, 10), rng.choice(MODES)
            want = ref_round(Fraction(a, 10 ** sa) / Fraction(b, 10 ** sb) * 10 ** t, mode)
            for divisor in (F._from_mantissa(np.int64(b), sb), raw(b, sb)):  # 除以数值 / 同单位比率
                if fits(want):
                    assert int(raw(a, sa).divide(divisor, scale=t, rounding=mode).mantissa) == want
                else:
                    with pytest.raises(OverflowError):
                        raw(a, sa).divide(divisor, scale=t, rounding=mode)

    @pytest.mark.parametrize("bad", [Money("1", "USD"), Price("1", "USD")])
    def test_by_money_or_price(self, bad):
        assert Quantity(1).__truediv__(bad) is NotImplemented
        with pytest.raises(TypeError):
            Quantity(1) / bad
        with pytest.raises(TypeError, match="数值或同单位"):
            Quantity(1).divide(bad)

    @pytest.mark.parametrize("bad", ["2", True])
    def test_rejects_non_numeric(self, bad):
        assert Quantity(1).__truediv__(bad) is NotImplemented
        with pytest.raises(TypeError):
            Quantity(1) / bad
        with pytest.raises(TypeError, match="只接受数值"):
            Quantity(1).divide(bad)

    def test_number_divided_by_quantity(self):
        with pytest.raises(TypeError):
            1 / Quantity(2)


# ===========================================================================
# 6. 显式取整
# ===========================================================================


class TestExplicitRounding:
    def test_rescale(self):
        assert Quantity("2.5").rescale(0) == Quantity(3) and Quantity("2.5").rescale(0, "half_even") == Quantity(2)
        assert Quantity("2.5").rescale(0, "down").unit == "share"
        assert Quantity(2).rescale(3).scale == 3
        with pytest.raises(ValueError, match="rounding 只能是"):
            Quantity(2).rescale(3, "nearest")

    def test_round_to_lot(self):
        assert Quantity(250).round_to_step(100, "down") == Quantity(200)
        assert Quantity(250).round_to_step(100) == Quantity(300)
        assert Quantity("0.123456").round_to_step(Decimal("0.001"), "floor") == Quantity("0.123")
        assert np.all(Quantity([150, 249]).round_to_step(100) == Quantity([200, 200]))
        assert Quantity(300).is_on_step(100) is True and Quantity(250).is_on_step(100) is False
        with pytest.raises(ValueError, match="必须为正数"):
            Quantity(1).round_to_step(0)
        with pytest.raises(TypeError, match="只接受数值"):
            Quantity(1).round_to_step("100")
        with pytest.raises(TypeError, match="只接受数值"):
            Quantity(1).is_on_step("100")


# ===========================================================================
# 7. 与 Price / Money 的交互
# ===========================================================================


class TestInteraction:
    def test_quantity_times_price_gives_money(self):
        assert Quantity(10) * Price("189.50", "USD") == Money("1895", "USD")
        assert Price("189.50", "USD") * Quantity(10) == Money("1895", "USD")
        with pytest.raises(TypeError, match="单位不匹配"):
            Quantity(1, unit="barrel") * Price("80", "USD")

    def test_money_divisions_produce_quantity(self):
        q = Money("100", "USD") / Price("30", "USD", base_unit="barrel")
        assert isinstance(q, Quantity) and q.unit == "barrel" and q.scale == 10

    def test_quantity_and_money_do_not_mix(self):
        for op in (lambda: Quantity(3) * Money("1", "USD"), lambda: Money("1", "USD") * Quantity(3),
                   lambda: Quantity(3) + Money("1", "USD"), lambda: Quantity(3) / Money("1", "USD")):
            with pytest.raises(TypeError):
                op()


# ===========================================================================
# 8. 导出与展示
# ===========================================================================


class TestExportAndDisplay:
    def test_pandas_carries_unit(self):
        pytest.importorskip("pandas")
        s = Quantity(["1.5", "2"], unit="barrel").to_pandas(name="qty")
        assert s.tolist() == [1.5, 2.0] and s.attrs["unit"] == "barrel" and s.name == "qty"
        assert Quantity([["1"]]).to_pandas(as_float=False).attrs["unit"] == "share"

    def test_repr_str(self):
        assert str(Quantity("1.5", unit="barrel")) == "1.5 barrel"
        assert repr(Quantity(3)) == "Quantity(FixedArray(3, scale=0), unit='share')"

    def test_exports(self):
        import moneta
        assert moneta.Quantity is Quantity and moneta.PrecisionLossWarning is PrecisionLossWarning


# ===========================================================================
# 缺失值（NA）
# ===========================================================================


class TestMissingValues:
    def test_construct_and_propagate(self, no_warnings):
        q = Quantity([None, "1.5", float("nan")], unit="BTC")
        assert q.isna().tolist() == [True, False, True] and q.unit == "BTC" and Quantity(None).isna()
        assert Quantity([None, "1.20"], scale=1).isna().tolist() == [True, False]  # 末尾的 0 不算丢精度
        assert Quantity([F("1.5"), 2], scale=2).mantissa.tolist() == [150, 200]  # 含 FixedArray 元素的序列
        with pytest.warns(PrecisionLossWarning):
            Quantity([F("1.25"), 2], scale=1)  # 精确输入：丢精度要告警
        for got in (q * 2, 2 * q, q + Quantity("1", unit="BTC"), q.multiply(Decimal("0.1"), scale=1), q / 2):
            assert isinstance(got, Quantity) and got.unit == "BTC" and got.isna().tolist() == [True, False, True]
        assert (q * None).isna().tolist() == [True, True, True] and (Quantity(1) / None).isna()
        assert (q / Quantity("0.5", unit="BTC")).isna().tolist() == [True, False, True]  # 比率
        assert (Price("2", "USD", base_unit="BTC") * q).isna().tolist() == [True, False, True]  # → Money

    def test_steps(self):
        q = Quantity([None, "150", "249"])
        assert q.round_to_step(100, "down").isna().tolist() == [True, False, False]
        assert q.round_to_step(100, "down").mantissa.tolist() == [0, 100, 200]
        assert q.is_on_step(50).tolist() == [False, True, False]
        for call in (lambda s: Quantity(5).round_to_step(s), lambda s: Quantity(5).is_on_step(s)):
            for step in (None, float("nan"), [1, None]):
                with pytest.raises(ValueError, match="步长不能是缺失值"):
                    call(step)

    def test_fillna_dropna_and_wildcard(self):
        q = Quantity([None, "2"], unit="BTC")
        assert q.fillna(Decimal("0.5")).mantissa.tolist() == [5, 20] and q.fillna(0).unit == "BTC"
        assert q.fillna(Quantity("0.5", unit=WILDCARD_UNIT)).unit == "BTC"  # 通配单位兼容
        with pytest.raises(TypeError, match="单位不匹配"):
            q.fillna(Quantity(1, unit="ETH"))
        assert q.dropna().mantissa.tolist() == [2] and q.dropna().unit == "BTC"
        assert q.sum(skipna=False).isna() and q.sum() == Quantity(2, unit="BTC")
