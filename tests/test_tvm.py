"""tvm 模块测试：pv / fv / npv / discount_factors，与 50 位 Decimal 参考值比较。"""

import random
from datetime import date, timedelta
from decimal import Decimal as D, ROUND_HALF_UP, localcontext
from fractions import Fraction

import numpy as np
import pytest

import moneta
from moneta import CashFlow, FixedArray, Money
from moneta.conventions import DayCountConvention, year_fraction
from moneta.currency import Currency
from moneta.tvm import discount_factors, fv, npv, pv

COMPOUNDINGS = ("annual", "continuous", "simple")


def usd(v):
    return Money(v, "USD")


# ---------- 参考实现 ----------

def ref_growth(r, t, compounding) -> D:
    with localcontext() as ctx:
        ctx.prec = 50
        r, t = D(repr(float(r))), D(repr(float(t)))
        if compounding == "annual":
            return (1 + r) ** t
        if compounding == "continuous":
            return (r * t).exp()
        return 1 + r * t


def ref_mul(amount, factor: D) -> D:
    with localcontext() as ctx:
        ctx.prec = 50
        return D(amount) * factor


def assert_rounded(result: Money, exact: D, rel: str = "1e-13"):
    """result 应是 exact 按 half_up 舍入到币种精度的值；仅当 exact 与平局点的距离小于浮点相对误差时允许取另一侧。"""
    assert result.ndim == 0
    q = D(1).scaleb(-result.currency.digits)
    got = result.data.to_decimal()
    with localcontext() as ctx:
        ctx.prec = 50
        assert abs(got - exact) <= q / 2 + abs(exact) * D(rel), (got, exact)
        if abs(abs(got - exact) - q / 2) > abs(exact) * D(rel):
            assert got == exact.quantize(q, rounding=ROUND_HALF_UP), (got, exact)


# ---------- pv / fv ----------

class TestPvFv:
    @pytest.mark.parametrize("func, amount, rate, periods, compounding, expected", [
        (pv, "110.00", 0.10, 1, "annual", "100.00"),
        (fv, "100.00", 0.10, 1, "annual", "110.00"),
        (pv, "121.00", 0.10, 2, "annual", "100.00"),
        (fv, "100.00", 0.05, 2, "continuous", "110.52"),   # 100·e^0.1 = 110.517…
        (pv, "110.52", 0.05, 2, "continuous", "100.00"),
        (pv, "110.00", 0.10, 1, "simple", "100.00"),
        (fv, "100.00", 0.10, 2.5, "simple", "125.00"),
        (pv, "100.00", 0.10, 0, "annual", "100.00"),
        (pv, "100.00", 0.00, 30, "annual", "100.00"),
        (pv, "100.00", 0.10, -1, "annual", "110.00"),      # 负期数即向后复利
    ])
    def test_examples(self, func, amount, rate, periods, compounding, expected):
        assert func(usd(amount), rate, periods, compounding=compounding) == usd(expected)

    def test_default_is_annual(self):
        assert pv(usd("110"), 0.1, 1) == pv(usd("110"), 0.1, 1, compounding="annual")

    def test_large_amount_precision(self):
        """旧实现把因子截到 8 位小数，此例误差 2.85 美元。"""
        result = pv(usd("1000000000.00"), 0.07, 30)
        assert result == usd("131367117.15")
        assert_rounded(result, ref_mul("1000000000.00", 1 / ref_growth(0.07, 30, "annual")))

    def test_tiny_and_huge_factors_keep_precision(self):
        assert_rounded(pv(usd("1000000000000.00"), 1.0, 40), ref_mul("1000000000000", 1 / ref_growth(1.0, 40, "annual")))
        assert_rounded(fv(usd("0.01"), 1.0, 55), ref_mul("0.01", ref_growth(1.0, 55, "annual")))
        assert pv(usd("1000000000"), 1.0, 60) == usd("0")

    def test_jpy_and_three_digits(self):
        assert pv(Money("10000", "JPY"), 0.1, 1) == Money("9091", "JPY")
        kwd = Currency("KWD", 3)
        assert pv(Money("1", kwd), 0.1, 1) == Money("0.909", kwd)

    @pytest.mark.parametrize("compounding", COMPOUNDINGS)
    def test_random_against_reference(self, compounding):
        rng = random.Random(17 + COMPOUNDINGS.index(compounding))
        for _ in range(1500):
            amount = D(rng.randint(-10 ** 12, 10 ** 12)).scaleb(-2)
            rate = round(rng.uniform(-0.5, 0.3), rng.randint(2, 8))
            periods = round(rng.uniform(-5, 40), rng.choice([0, 0, 3, 6]))
            if compounding == "simple" and 1 + rate * periods <= 0.05:
                continue
            g = ref_growth(rate, periods, compounding)
            if abs(amount) * max(g, 1 / g) > 10 ** 15:  # 结果超出 int64 尾数
                continue
            m = usd(str(amount))
            assert_rounded(pv(m, rate, periods, compounding=compounding), ref_mul(amount, 1 / g))
            assert_rounded(fv(m, rate, periods, compounding=compounding), ref_mul(amount, g))

    def test_rate_forms_agree(self):
        expected = pv(usd("1000"), 0.05, 3)
        for rate in ("0.05", " 0.05 ", FixedArray("0.05"), Fraction(1, 20), D("0.05"), np.float64(0.05)):
            assert pv(usd("1000"), rate, 3) == expected
        assert pv(usd("1000"), 1, 1) == usd("500") and pv(usd("1000"), np.int64(1), 1) == usd("500")
        assert pv(usd("1000"), 0.05, np.int32(3)) == expected == pv(usd("1000"), 0.05, D("3"))

    def test_broadcasting(self):
        out = pv(usd("100"), 0.1, [0, 1, 2])
        assert (out == usd(["100.00", "90.91", "82.64"])).all()
        out = fv(usd(["100", "200"]), [0.1, 0.2], 1)
        assert (out == usd(["110.00", "240.00"])).all()
        out = pv(usd([["100"], ["200"]]), 0.1, np.array([1, 2]))
        assert out.shape == (2, 2) and (out == usd([["90.91", "82.64"], ["181.82", "165.29"]])).all()

    def test_array_factor_precision_follows_largest(self):
        """同一数组内因子相差很大时，每个元素仍应正确舍入。"""
        periods = np.array([0.0, 10.0, 30.0, 60.0])
        out = pv(usd("1000000000.00"), 0.07, periods)
        for i, t in enumerate(periods):
            assert_rounded(out[i], ref_mul("1000000000.00", 1 / ref_growth(0.07, t, "annual")))

    def test_na_propagates(self):
        out = pv(usd(["100", None]), 0.1, 1)
        assert out[0] == usd("90.91") and bool(out.isna()[1])

    def test_inverse(self):
        m = usd("12345.67")
        assert pv(fv(m, 0.07, 10), 0.07, 10) == m


class TestPvFvValidation:
    @pytest.mark.parametrize("func", [pv, fv])
    def test_amount_must_be_money(self, func):
        name = "future_value" if func is pv else "present_value"
        with pytest.raises(TypeError, match=f"{name} 必须是 Money，收到 float"):
            func(100.0, 0.1, 1)

    @pytest.mark.parametrize("rate", [True, np.bool_(False), None, [True], ["0.1"], [Fraction(1, 10)], object()])
    def test_rate_type(self, rate):
        with pytest.raises(TypeError, match="rate 必须是实数"):
            pv(usd("1"), rate, 1)

    @pytest.mark.parametrize("periods", [True, None, [None], ["1"]])
    def test_periods_type(self, periods):
        with pytest.raises(TypeError, match="periods 必须是实数"):
            pv(usd("1"), 0.1, periods)

    @pytest.mark.parametrize("rate", [float("nan"), float("inf"), "nan", "1e400", [0.1, float("nan")]])
    def test_rate_must_be_finite(self, rate):
        with pytest.raises(ValueError, match="rate 必须是有限数"):
            pv(usd("1"), rate, 1)

    def test_periods_must_be_finite(self):
        with pytest.raises(ValueError, match="periods 必须是有限数"):
            fv(usd("1"), 0.1, float("-inf"))

    def test_invalid_string_and_na(self):
        with pytest.raises(ValueError, match="rate 不是合法的数值: 'abc'"):
            pv(usd("1"), "abc", 1)
        with pytest.raises(ValueError, match="rate 不能包含缺失值"):
            pv(usd("1"), FixedArray(["0.1", None]), 1)

    @pytest.mark.parametrize("rate", [-1, -1.5])
    def test_annual_requires_rate_above_minus_one(self, rate):
        for func in (pv, fv):
            with pytest.raises(ValueError, match="annual 复利要求 rate > -1"):
                func(usd("100"), rate, 0.5)

    def test_simple_requires_positive_growth(self):
        for func in (pv, fv):
            with pytest.raises(ValueError, match="simple 计息要求"):
                func(usd("100"), -0.5, [1, 2], compounding="simple")

    def test_continuous_accepts_any_finite_rate(self):
        assert pv(usd("100"), -2.0, 1, compounding="continuous") == usd("738.91")

    def test_growth_out_of_range(self):
        with pytest.raises(ValueError, match="增长因子超出浮点数范围"):
            fv(usd("1"), 0.5, 10000)
        with pytest.raises(ValueError, match="增长因子超出浮点数范围"):
            pv(usd("1"), 0.5, -10000)
        with pytest.raises(OverflowError, match="增长因子 .* 超出定点数范围"):
            fv(usd("1"), 1.0, 70)
        with pytest.raises(OverflowError):
            fv(usd("1000000"), 0.5, 100)

    @pytest.mark.parametrize("compounding", ["periodic", "Annual", "", None, ["annual"]])
    def test_compounding(self, compounding):
        for func in (pv, fv):
            with pytest.raises(ValueError, match="不支持的复利方式"):
                func(usd("1"), 0.1, 1, compounding=compounding)

    def test_removed_parameters(self):
        for kwargs in (dict(nper=1), dict(convention="ACT/365")):
            with pytest.raises(TypeError):
                pv(usd("1"), 0.1, 1, **kwargs)
        with pytest.raises(TypeError):
            fv(usd("1"), 0.1, nper=1)
        with pytest.raises(TypeError):
            pv(usd("1"), 0.1, 1, "annual")  # compounding 仅限关键字


# ---------- npv ----------

def _ref_npv(rate, amounts, times, compounding="annual") -> D:
    with localcontext() as ctx:
        ctx.prec = 50
        return sum((D(str(a)) / ref_growth(rate, t, compounding) for a, t in zip(amounts, times)), D(0))


class TestNpv:
    def test_money_flows_use_index_periods(self):
        flows = [usd("-1000"), usd("550"), usd("605")]
        assert npv(0.1, flows) == usd("0.00")
        assert npv(0.1, tuple(flows)) == npv(0.1, iter(flows)) == npv(0.1, (f for f in flows)) == usd("0")

    def test_dated_flows(self):
        flows = [CashFlow(usd("-1000"), "2026-01-01"), CashFlow(usd("1100"), "2026-07-01")]
        result = npv(0.1, flows, "2026-01-01")
        assert result == usd("49.22")
        assert_rounded(result, _ref_npv(0.1, ["-1000", "1100"], [0, 181 / 365]))

    def test_convention_and_compounding(self):
        flows = [CashFlow(usd("1000000"), "2027-01-01")]
        act365 = npv(0.05, flows, "2026-01-01")
        act360 = npv(0.05, flows, "2026-01-01", convention="act/360")
        assert act365 == pv(usd("1000000"), 0.05, 1) and act360 == pv(usd("1000000"), 0.05, 365 / 360)
        assert act360 < act365
        cont = npv(0.05, flows, date(2026, 1, 1), convention=DayCountConvention.ACT_ACT, compounding="continuous")
        assert cont == pv(usd("1000000"), 0.05, 1, compounding="continuous")

    def test_flows_before_as_of_date_are_compounded(self):
        flows = [CashFlow(usd("100"), "2025-01-01")]
        assert npv(0.1, flows, "2026-01-01") == usd("110.00")

    def test_single_rounding(self):
        """逐笔舍入再求和会得到 0.02；精确值约 0.0300，只舍入一次得到 0.03。"""
        flows = [usd("0.01")] * 1000
        assert npv(0.5, flows) == usd("0.03")
        assert sum((pv(f, 0.5, i) for i, f in enumerate(flows)), usd("0")) == usd("0.02")

    @pytest.mark.parametrize("compounding", COMPOUNDINGS)
    def test_random_money_flows(self, compounding):
        rng = random.Random(len(compounding) * 101)
        for _ in range(300):
            n = rng.randint(1, 40)
            amounts = [D(rng.randint(-10 ** 9, 10 ** 9)).scaleb(-2) for _ in range(n)]
            rate = round(rng.uniform(0, 0.2), rng.randint(2, 6))
            result = npv(rate, [usd(str(a)) for a in amounts], compounding=compounding)
            assert_rounded(result, _ref_npv(rate, amounts, range(n), compounding), rel="1e-12")

    def test_random_dated_flows(self):
        rng = random.Random(99)
        for _ in range(300):
            as_of = date(2020, 1, 1) + timedelta(days=rng.randrange(3000))
            conv = rng.choice(list(DayCountConvention))
            n = rng.randint(1, 30)
            days = [as_of + timedelta(days=rng.randrange(-400, 10000)) for _ in range(n)]
            amounts = [D(rng.randint(-10 ** 9, 10 ** 9)).scaleb(-2) for _ in range(n)]
            rate = round(rng.uniform(0, 0.15), 4)
            flows = [CashFlow(usd(str(a)), d) for a, d in zip(amounts, days)]
            times = [year_fraction(as_of, d, conv) for d in days]
            assert_rounded(npv(rate, flows, as_of, convention=conv), _ref_npv(rate, amounts, times), rel="1e-12")

    def test_array_amounts_and_rates(self):
        flows = [CashFlow(usd(["-100", "-200"]), "2026-01-01"), CashFlow(usd(["110", "220"]), "2027-01-01")]
        assert (npv(0.1, flows, "2026-01-01") == usd(["0", "0"])).all()
        assert (npv([0.1, 0.0], [usd("-100"), usd("110")]) == usd(["0.00", "10.00"])).all()

    def test_keyword_call(self):
        assert npv(rate=0.1, cash_flows=[usd("-100"), usd("110")]) == usd("0")


class TestNpvValidation:
    def test_cash_flows_container(self):
        for bad in (usd("1"), CashFlow(usd("1"), "2026-01-01"), "abc", 5, None):
            with pytest.raises(TypeError, match="cash_flows 必须是 Money 或 CashFlow 的序列"):
                npv(0.1, bad)
        with pytest.raises(ValueError, match="cash_flows 不能为空"):
            npv(0.1, [])

    def test_dated_flows_require_as_of_date(self):
        with pytest.raises(ValueError, match="必须给出 as_of_date"):
            npv(0.1, [usd("1"), CashFlow(usd("1"), "2026-01-01")])

    def test_as_of_date_requires_cash_flow(self):
        with pytest.raises(ValueError, match="必须是带日期的 CashFlow"):
            npv(0.1, [CashFlow(usd("1"), "2026-01-01"), usd("1")], "2026-01-01")

    def test_element_types(self):
        with pytest.raises(TypeError, match="cash_flows 元素必须是 Money，收到 float"):
            npv(0.1, [usd("1"), 1.0])
        with pytest.raises(TypeError, match="cash_flows 元素必须是 CashFlow，收到 int"):
            npv(0.1, [CashFlow(usd("1"), "2026-01-01"), 1], "2026-01-01")

    def test_currency_mismatch(self):
        with pytest.raises(TypeError, match="币种不匹配: USD 与 EUR"):
            npv(0.1, [usd("1"), Money("1", "EUR")])

    def test_as_of_date_label(self):
        with pytest.raises(ValueError, match="^as_of_date 不是合法的 ISO 日期"):
            npv(0.1, [CashFlow(usd("1"), "2026-01-01")], "x")

    def test_convention_validated_even_without_date(self):
        with pytest.raises(ValueError, match="不支持的计日惯例"):
            npv(0.1, [usd("1")], convention="ACT/365F")

    def test_rate_and_compounding(self):
        with pytest.raises(TypeError, match="rate 必须是实数"):
            npv(None, [usd("1")])
        with pytest.raises(ValueError, match="不支持的复利方式"):
            npv(0.1, [usd("1")], compounding="periodic")
        with pytest.raises(ValueError, match="annual 复利要求"):
            npv(-1, [usd("1")])

    def test_old_call_styles_rejected(self):
        flows = [CashFlow(usd("1"), "2026-01-01")]
        with pytest.raises(TypeError, match="cash_flows 必须是 Money 或 CashFlow 的序列，收到 float"):
            npv(flows, 0.1)  # 旧的现金流在前写法
        with pytest.raises(TypeError):
            npv(0.1, flows, base_date="2026-01-01")
        with pytest.raises(TypeError):
            npv(0.1, flows, "2026-01-01", "ACT/360")  # convention 仅限关键字


# ---------- discount_factors ----------

class TestDiscountFactors:
    @pytest.mark.parametrize("compounding", COMPOUNDINGS)
    def test_values(self, compounding):
        rates, times = [0.01, 0.03, 0.05, 0.08], [0.25, 1, 10, 1000]
        if compounding == "simple":
            times = [0.25, 1, 10, 30]
        out = discount_factors(rates, times, compounding=compounding)
        assert isinstance(out, np.ndarray) and out.dtype == np.float64 and out.shape == (4,)
        for r, t, x in zip(rates, times, out):
            # 1 + r 本身的浮点误差会被 t 次幂放大约 t 倍
            assert abs(D(float(x)) * ref_growth(r, t, compounding) - 1) < D("1e-15") * (10 + D(t))

    def test_default_annual_and_scalar(self):
        out = discount_factors(0.05, 1)
        assert isinstance(out, np.float64) and out == 1 / 1.05
        assert discount_factors(0.05, 1000) > 0  # 旧实现截到 8 位后为 0

    def test_broadcasting(self):
        out = discount_factors(np.array([[0.01], [0.02]]), [1, 2, 3])
        assert out.shape == (2, 3)
        assert np.allclose(out, 1 / (1 + np.array([[0.01], [0.02]])) ** np.array([1, 2, 3]), rtol=1e-15)
        with pytest.raises(ValueError):
            discount_factors([0.05, 0.04, 0.03], [1, 2])

    def test_input_forms(self):
        assert discount_factors("0.05", 1) == discount_factors(FixedArray("0.05"), 1) == discount_factors(0.05, 1)

    def test_validation(self):
        with pytest.raises(ValueError, match="rates 必须是有限数"):
            discount_factors(float("nan"), 1)
        with pytest.raises(TypeError, match="times 必须是实数"):
            discount_factors(0.05, True)
        with pytest.raises(ValueError, match="annual 复利要求"):
            discount_factors(-1.0, 1.0)
        with pytest.raises(ValueError, match="不支持的复利方式"):
            discount_factors(0.05, 1, compounding="periodic")

    def test_removed_parameters(self):
        for kwargs in (dict(zero_rates=0.05, maturities=1), dict(rate=0.05, times=1)):
            with pytest.raises(TypeError):
                discount_factors(**kwargs)


def test_exports():
    import moneta.tvm as mod

    assert mod.__all__ == ["pv", "fv", "npv", "discount_factors"]
    for name in mod.__all__:
        assert getattr(moneta, name) is getattr(mod, name)
