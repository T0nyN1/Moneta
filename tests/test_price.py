"""Price 的行为约定。标准答案来自 Fraction 与 Decimal。

已确认的设计：
- 量纲为 [计价货币 / 标的单位]。两个 Price 兼容：计价货币相同，单位相同或任一方为通配单位 "Any"；
  Price 与 Quantity 相乘同样双向通配。运算结果取具体的那个单位。
- 运算精确：乘以数值时 scale 为两侧之和；除法默认保留 max(两侧 scale) + 8 位小数，可用 scale 指定。
  multiply 的返回类型由乘数决定：Price × Quantity → Money（默认 half_up 舍入到币种精度，不接受 scale），
  Price × 数值 → Price。
- 支持 Price ± Price、Price / Price（比率）、Price / 数值；Price * Money、Price * Price 报 TypeError。
- tick_size 是正数值元数据：构造时不要求在网格上，不参与相等比较；两个 Price 运算时 tick 相同则保留，
  否则结果没有 tick；一元运算与乘除数值保留原 tick。
- 构造指定 scale 时与 Money 一致：float 静默 half_up；精确输入超出 scale 时发出 PrecisionLossWarning 后舍入。
- 运算只接受数值白名单内的操作数，拒绝字符串；运算符遇到不认识的类型返回 NotImplemented。

组织：构造、单位与通配、tick、乘法与成交金额、加减、除法、比较、导出与展示。
"""

import random
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from moneta import Money, PrecisionLossWarning, Price, Quantity
from moneta.currency import USD
from moneta.fixed_array import FixedArray as F, RoundingMode

MIN, MAX = -(2 ** 63), 2 ** 63 - 1
MODES = [m.value for m in RoundingMode]
TICK = Decimal("0.01")


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


def fits(v: int) -> bool:
    return MIN <= v <= MAX


def value(x) -> Fraction:
    return Fraction(int(x.mantissa), 10 ** x.scale)


def raw(m, scale, unit="share", ccy="USD"):
    return Price(F._from_mantissa(np.asarray(m, dtype=np.int64), scale), ccy, base_unit=unit)


@pytest.fixture
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ===========================================================================
# 1. 构造
# ===========================================================================


class TestConstruct:
    def test_basic(self, no_warnings):
        p = Price("189.50", "usd", base_unit=" barrel ", tick_size=TICK)
        assert (int(p.mantissa), p.scale, p.quote_currency, p.base_unit) == (18950, 2, USD, "barrel")
        assert p.tick_size == F("0.01") and Price(1, USD).base_unit == "share" and Price(1, USD).tick_size is None
        assert Price(["1", "2.5"], "USD").shape == (2,)

    @pytest.mark.parametrize("values, scale, mantissa", [(1.5, 0, 2), (0.125, 2, 13), (-2.5, 0, -3),
                                                         (np.array([1.005, 2.675]), 2, [101, 268])])
    def test_float_with_scale_rounds_silently(self, values, scale, mantissa, no_warnings):
        p = Price(values, "USD", scale=scale)
        assert (p.mantissa.tolist() if p.ndim else int(p.mantissa), p.scale) == (mantissa, scale)

    @pytest.mark.parametrize("values, scale, mantissa", [("1.234", 2, 123), (Decimal("1.235"), 2, 124),
                                                         (F("1.234"), 2, 123), (Fraction(1, 8), 2, 13)])
    def test_exact_input_beyond_scale_warns_then_rounds(self, values, scale, mantissa):
        with pytest.warns(PrecisionLossWarning, match="超出指定 scale"):
            p = Price(values, "USD", scale=scale)
        assert (int(p.mantissa), p.scale) == (mantissa, scale)

    def test_scale_pads(self, no_warnings):
        assert Price("1.5", "USD", scale=4).scale == 4 and int(Price("1.5", "USD", scale=4).mantissa) == 15000

    def test_from_price(self, no_warnings):
        p = Price("1.5", "USD", tick_size=TICK)
        q = Price(p, "USD")
        assert q == p and q.tick_size == F("0.01")
        assert Price(p, "USD", tick_size=Decimal("0.05")).tick_size == F("0.05")
        assert Price(Price("1", "USD", base_unit="Any"), "USD", base_unit="barrel").base_unit == "barrel"
        with pytest.raises(TypeError, match="价格量纲不匹配"):
            Price(p, "CNY")
        with pytest.raises(TypeError, match="价格量纲不匹配"):
            Price(p, "USD", base_unit="barrel")

    @pytest.mark.parametrize("unit, exc", [(None, TypeError), (1, TypeError), ("", ValueError), ("  ", ValueError)])
    def test_invalid_unit(self, unit, exc):
        with pytest.raises(exc):
            Price("1", "USD", base_unit=unit)

    def test_invalid_currency_and_values(self):
        with pytest.raises(KeyError):
            Price("1", "XYZ")
        with pytest.raises(TypeError):
            Price(True, "USD")


# ===========================================================================
# 2. 单位与通配
# ===========================================================================


class TestUnits:
    def test_price_price_wildcard(self):
        a, b = Price("1", "USD", base_unit="Any"), Price("2", "USD", base_unit="barrel")
        assert (a + b).base_unit == "barrel" and (b + a).base_unit == "barrel" and (b - a).base_unit == "barrel"
        assert (a + Price("1", "USD", base_unit="Any")).base_unit == "Any"
        assert a < b and Price("2", "USD", base_unit="Any") == b
        assert b / a == F("2")

    def test_price_quantity_wildcard_both_ways(self):
        assert Price("2", "USD", base_unit="Any") * Quantity(3, unit="barrel") == Money("6", "USD")
        assert Price("2", "USD", base_unit="barrel") * Quantity(3, unit="Any") == Money("6", "USD")
        assert Quantity(3, unit="barrel") * Price("2", "USD", base_unit="Any") == Money("6", "USD")

    def test_mismatch(self):
        with pytest.raises(TypeError, match="价格量纲不匹配"):
            Price("1", "USD") + Price("1", "USD", base_unit="barrel")
        with pytest.raises(TypeError, match="价格量纲不匹配"):
            Price("1", "USD") - Price("1", "CNY")
        with pytest.raises(TypeError, match="标的单位不匹配"):
            Price("1", "USD") * Quantity(1, unit="barrel")
        assert (Price("1", "USD") == Price("1", "CNY")) is False

    def test_case_sensitive(self):
        with pytest.raises(TypeError):
            Price("1", "USD", base_unit="BTC") * Quantity(1, unit="btc")


# ===========================================================================
# 3. tick
# ===========================================================================


class TestTick:
    @pytest.mark.parametrize("tick", [0, -0.01, Decimal("-1"), np.array([0.01, 0])])
    def test_must_be_positive(self, tick):
        with pytest.raises(ValueError, match="tick_size 必须为正数"):
            Price("1", "USD", tick_size=tick)

    @pytest.mark.parametrize("tick", ["0.01", True, [0.01, "x"]])
    def test_must_be_numeric(self, tick):
        with pytest.raises(TypeError, match="只接受数值"):
            Price("1", "USD", tick_size=tick)

    def test_round_and_check(self):
        p = Price(["1.234", "1.235", "-1.235"], "USD", tick_size=TICK)
        assert p.is_on_step().tolist() == [False, False, False]
        assert np.all(p.round_to_step() == Price(["1.23", "1.24", "-1.24"], "USD"))
        assert np.all(p.round_to_step(rounding="half_even") == Price(["1.23", "1.24", "-1.24"], "USD"))
        assert np.all(p.round_to_step(rounding="down") == Price(["1.23", "1.23", "-1.23"], "USD"))
        assert p.round_to_step().is_on_step().tolist() == [True, True, True]
        assert p.round_to_step().tick_size == F("0.01")
        with pytest.raises(ValueError, match="rounding 只能是"):
            p.round_to_step(rounding="nearest")

    def test_per_element_ticks(self):
        p = Price(["10.23", "10.23"], "USD", tick_size=[Decimal("0.05"), Decimal("0.10")])
        assert np.all(p.round_to_step() == Price(["10.25", "10.20"], "USD"))

    def test_without_tick(self):
        p = Price("1.234", "USD")
        assert p.round_to_step() is p and p.is_on_step() is True
        assert Price(["1", "2"], "USD").is_on_step().tolist() == [True, True]
        with pytest.raises(ValueError, match="rounding 只能是"):
            p.round_to_step(rounding="nearest")

    def test_explicit_step(self):
        p = Price(["10.23", "10.27"], "USD", tick_size=TICK)
        assert np.all(p.round_to_step(Decimal("0.05")) == Price(["10.25", "10.25"], "USD"))
        assert np.all(p.round_to_step(Decimal("0.05"), "down") == Price(["10.20", "10.25"], "USD"))
        assert p.round_to_step(Decimal("0.05")).tick_size == F("0.01")  # 显式 step 不改变自带的 tick
        assert p.is_on_step(Decimal("0.01")).tolist() == [True, True]
        assert p.is_on_step(Decimal("0.05")).tolist() == [False, False]
        assert Price("1.234", "USD").round_to_step(Decimal("0.01")) == Price("1.23", "USD")  # 无 tick 也可显式给 step
        with pytest.raises(ValueError, match="step 必须为正数"):
            p.round_to_step(0)
        with pytest.raises(ValueError, match="step 必须为正数"):
            p.is_on_step(-1)
        with pytest.raises(TypeError, match="只接受数值"):
            p.round_to_step("half_even")  # 第一个位置参数是 step，旧写法 round_to_tick("half_even") 不会被静默误用

    def test_unified_with_money_and_quantity(self):
        assert not hasattr(Price, "round_to_tick") and not hasattr(Price, "is_on_tick")
        for x, step in ((Price("10.23", "USD"), Decimal("0.05")), (Money("10.23", "USD"), Decimal("0.05")),
                        (Quantity(250), 100)):
            assert x.round_to_step(step, "down").is_on_step(step)

    def test_not_enforced_and_not_compared(self, no_warnings):
        p = Price("1.234", "USD", tick_size=TICK)  # 构造时不要求在网格上
        assert p.is_on_step() is False
        assert Price("1", "USD", tick_size=TICK) == Price("1", "USD", tick_size=Decimal("0.05"))
        assert Price("1", "USD", tick_size=TICK) == Price("1", "USD")

    def test_propagation(self):
        a = Price("1", "USD", tick_size=TICK)
        assert (a + Price("2", "USD", tick_size=Decimal("0.010"))).tick_size == F("0.01")  # 数值相同即视为相同
        assert (a + Price("2", "USD", tick_size=Decimal("0.05"))).tick_size is None
        assert (a - Price("2", "USD")).tick_size is None
        for r in (-a, abs(a), +a, a * 2, a / 2, a.multiply(3, scale=1), Price(["1", "2"], "USD", tick_size=TICK)[0],
                  Price(["1", "2"], "USD", tick_size=TICK).sum()):
            assert r.tick_size == F("0.01")


# ===========================================================================
# 4. 乘法与成交金额
# ===========================================================================


class TestMultiply:
    def test_exact_by_default(self):
        p = Price("1.5", "USD") * Decimal("0.33")
        assert p.data == F("0.495") and p.scale == 3 and p.base_unit == "share"
        assert 2 * Price("1.25", "USD") == Price("2.5", "USD")
        assert np.all(np.array([1, 2]) * Price(["1", "2"], "USD") == Price(["1", "4"], "USD"))
        assert F("2") * Price("3", "USD") == Price("6", "USD")

    def test_with_scale_and_rounding(self):
        assert Price("1.37", "USD").multiply(3, scale=1) == Price("4.1", "USD")
        assert Price("1.37", "USD").multiply(3, scale=0, rounding="down") == Price("4", "USD")
        with pytest.raises(ValueError, match="rounding 只能是"):
            Price("1", "USD").multiply(2, rounding="nearest")

    def test_random_against_fraction(self):
        rng = random.Random(1)
        for _ in range(2000):
            a, sa = rng.randint(-10 ** 9, 10 ** 9), rng.randint(0, 6)
            b, sb = rng.randint(-10 ** 6, 10 ** 6), rng.randint(0, 6)
            p, f = raw(a, sa), F._from_mantissa(np.int64(b), sb)
            exact = p * f
            assert value(exact) == Fraction(a, 10 ** sa) * Fraction(b, 10 ** sb) and exact.scale == sa + sb
            t, mode = rng.randint(0, 6), rng.choice(MODES)
            want = ref_round(Fraction(a * b, 10 ** (sa + sb)) * 10 ** t, mode)
            if fits(want):
                assert int(p.multiply(f, scale=t, rounding=mode).mantissa) == want
            else:
                with pytest.raises(OverflowError):
                    p.multiply(f, scale=t, rounding=mode)

    def test_times_quantity_gives_money(self):
        assert Price("189.50", "USD") * Quantity(10) == Money("1895", "USD")
        assert Quantity(10) * Price("189.50", "USD") == Money("1895", "USD")
        assert Price("0.001", "USD") * Quantity("0.4") == Money("0", "USD")  # 默认 half_up
        assert Price("0.001", "USD").multiply(Quantity("0.4"), rounding="up") == Money("0.01", "USD")
        assert np.all(Price(["1.5", "2"], "USD") * Quantity([2, 3]) == Money(["3", "6"], "USD"))
        assert Price("100", "JPY").multiply(Quantity("1.5")) == Money("150", "JPY")
        with pytest.raises(ValueError, match="不能指定 scale"):
            Price("1", "USD").multiply(Quantity(3), scale=4)
        with pytest.raises(TypeError, match="标的单位不匹配"):
            Price("1", "USD").multiply(Quantity(1, unit="barrel"))

    def test_return_type_follows_operand(self):
        assert isinstance(Price("2", "USD").multiply(Quantity(3)), Money)
        assert isinstance(Price("2", "USD").multiply(3), Price)
        assert not hasattr(Price, "value")
        for bad in (Money("1", "USD"), Price("1", "USD")):
            with pytest.raises(TypeError, match="只能乘以 Quantity 或数值"):
                Price("1", "USD").multiply(bad)

    def test_times_quantity_random_against_fraction(self):
        rng = random.Random(2)
        for _ in range(2000):
            a, sa = rng.randint(-10 ** 9, 10 ** 9), rng.randint(0, 8)
            b, sb = rng.randint(-10 ** 6, 10 ** 6), rng.randint(0, 8)
            mode = rng.choice(MODES)
            q = Quantity(F._from_mantissa(np.int64(b), sb))
            got = raw(a, sa).multiply(q, rounding=mode)
            assert got.currency is USD and int(got.units) == ref_round(Fraction(a * b, 10 ** (sa + sb)) * 100, mode)

    @pytest.mark.parametrize("bad", ["2", True, None, Money("1", "USD"), Price("2", "USD")])
    def test_rejects_non_numeric(self, bad):
        assert Price("1", "USD").__mul__(bad) is NotImplemented and Price("1", "USD").__rmul__(bad) is NotImplemented
        with pytest.raises(TypeError):
            Price("1", "USD") * bad
        with pytest.raises(TypeError):
            bad * Price("1", "USD")
        if not isinstance(bad, (Money, Price)):
            with pytest.raises(TypeError, match="只接受数值"):
                Price("1", "USD").multiply(bad)


# ===========================================================================
# 5. 加减
# ===========================================================================


class TestAddSub:
    def test_spread_and_change(self):
        ask, bid = Price("101.25", "USD"), Price("101.20", "USD")
        assert ask - bid == Price("0.05", "USD") and bid + Price("0.05", "USD") == ask
        assert np.all(Price(["1", "2"], "USD") + Price("0.5", "USD") == Price(["1.5", "2.5"], "USD"))
        assert (Price("1.5", "USD") - Price("0.001", "USD")).scale == 3

    @pytest.mark.parametrize("other", [1, Decimal("1"), F("1"), "1", None, Money("1", "USD"), Quantity(1)])
    def test_non_price(self, other):
        assert Price("1", "USD").__add__(other) is NotImplemented and Price("1", "USD").__sub__(other) is NotImplemented
        for op in (lambda: Price("1", "USD") + other, lambda: other + Price("1", "USD"),
                   lambda: Price("1", "USD") - other):
            with pytest.raises(TypeError):
                op()

    def test_overflow(self):
        with pytest.raises(OverflowError):
            raw(MAX, 0) + raw(1, 0)


# ===========================================================================
# 6. 除法
# ===========================================================================


class TestDivide:
    def test_ratio(self):
        r = Price("189.50", "USD") / Price("100", "USD")
        assert isinstance(r, F) and r == F("1.895") and r.scale == 10
        assert Price("1", "USD").divide(Price("3", "USD"), scale=3, rounding="up") == F("0.334")
        with pytest.raises(TypeError, match="价格量纲不匹配"):
            Price("1", "USD") / Price("1", "CNY")

    def test_by_number(self):
        p = Price("189.50", "USD", base_unit="barrel") / 2
        assert isinstance(p, Price) and p.base_unit == "barrel" and p.scale == 10 and p.data == F("94.75")
        assert Price("10", "USD").divide(3, scale=2) == Price("3.33", "USD")
        assert (Price("1.5", "USD") / Decimal("0.25")).scale == 10  # max(1, 2) + 8
        with pytest.raises(ZeroDivisionError):
            Price("1", "USD") / 0

    def test_random_against_fraction(self):
        rng = random.Random(3)
        for _ in range(2000):
            a, sa = rng.randint(-10 ** 9, 10 ** 9), rng.randint(0, 6)
            b, sb = rng.randint(1, 10 ** 6) * rng.choice([1, -1]), rng.randint(0, 6)
            t, mode = rng.randint(0, 10), rng.choice(MODES)
            want = ref_round(Fraction(a, 10 ** sa) / Fraction(b, 10 ** sb) * 10 ** t, mode)
            for divisor in (F._from_mantissa(np.int64(b), sb), raw(b, sb)):
                if fits(want):
                    assert int(raw(a, sa).divide(divisor, scale=t, rounding=mode).mantissa) == want
                else:
                    with pytest.raises(OverflowError):
                        raw(a, sa).divide(divisor, scale=t, rounding=mode)

    @pytest.mark.parametrize("bad", [Money("1", "USD"), Quantity(1)])
    def test_by_money_or_quantity(self, bad):
        assert Price("1", "USD").__truediv__(bad) is NotImplemented
        with pytest.raises(TypeError):
            Price("1", "USD") / bad
        with pytest.raises(TypeError, match="数值或同类 Price"):
            Price("1", "USD").divide(bad)

    @pytest.mark.parametrize("bad", ["2", True, None])
    def test_rejects_non_numeric(self, bad):
        assert Price("1", "USD").__truediv__(bad) is NotImplemented
        with pytest.raises(TypeError, match="只接受数值"):
            Price("1", "USD").divide(bad)

    def test_number_divided_by_price(self):
        with pytest.raises(TypeError):
            1 / Price("2", "USD")


# ===========================================================================
# 7. 比较
# ===========================================================================


class TestCompare:
    def test_rules(self):
        assert Price("1.50", "USD") == Price("1.5", "USD") and Price("1", "USD") < Price("2", "USD")
        assert (Price("1", "USD") == 1) is False and (Price("1", "USD") != Money("1", "USD")) is True
        assert (Price(["1", "3"], "USD") >= Price("2", "USD")).tolist() == [False, True]
        with pytest.raises(TypeError, match="价格量纲不匹配"):
            Price("1", "USD") < Price("2", "CNY")
        with pytest.raises(TypeError):
            Price("1", "USD") < 2


# ===========================================================================
# 8. 导出与展示
# ===========================================================================


class TestExportAndDisplay:
    def test_pandas_carries_metadata(self):
        pytest.importorskip("pandas")
        s = Price(["1.5", "2"], "USD", base_unit="barrel").to_pandas(name="px")
        assert s.tolist() == [1.5, 2.0] and s.name == "px"
        assert s.attrs == {"quote_currency": "USD", "base_unit": "barrel"}

    def test_repr_str(self):
        assert str(Price("1.5", "USD", base_unit="barrel")) == "1.5 USD/barrel"
        assert repr(Price("1.5", "USD")) == "Price(FixedArray(1.5, scale=1), quote_currency='USD', base_unit='share')"
