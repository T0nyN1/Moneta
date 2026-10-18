"""Money 的行为约定。标准答案来自 Fraction 与 Decimal。

已确认的设计：
- Money 的精度恒等于币种精度，没有 scale 参数，也不存在更高精度的中间金额。
- 构造：float 按最短写法 half_up 舍入；精确输入（str、Decimal、Fraction、FixedArray 等）超出币种精度时
  发出 PrecisionLossWarning 后按 half_up 舍入。from_units 会复制传入的数组。
- 运算只接受数值白名单内的操作数，拒绝字符串；运算符遇到不认识的类型返回 NotImplemented。
- 除法结果若不是金额（比率、单价、数量），默认保留 币种精度 + 8 位小数，可用 scale 指定；
  结果是 Money 时精度固定，不接受 scale。
- multiply / divide 的 step：一次性舍入到步长网格（不先舍入到币种精度）；步长精度不能超过币种精度。
- divmod(Money, Money) → (整数份数, 零头)；divmod(Money, 正整数 n) → (每份, 零头)，向下取整；
  非整数报错并提示使用 allocate。
- allocate：最大余数法，权重为非负数值，结果之和恰好等于原金额。
- calc_tiered_fee：分档阈值严格递增、最后一档为 None；各档精确累加后只舍入一次；min/max 逐元素生效。

组织：构造、内部不变式、加减、乘法、除法、矩阵乘法、规约与符号、步长、divmod、allocate、费用、导出与展示。
"""

import random
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from moneta import Money, Price, Quantity
from moneta.currency import JPY, USD, Currency
from moneta.fixed_array import FixedArray as F, RoundingMode
from moneta.money import PrecisionLossWarning

MIN, MAX = -(2 ** 63), 2 ** 63 - 1
MODES = [m.value for m in RoundingMode]


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


def usd(v):
    return Money(v, "USD")


def units(m):
    return np.ravel(m.units).tolist() if m.ndim else int(m.units)


@pytest.fixture
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ===========================================================================
# 1. 构造
# ===========================================================================


class TestConstruct:
    @pytest.mark.parametrize("value, want", [
        ("12.34", 1234), ("12.3", 1230), (12, 1200), (Decimal("0.10"), 10), (Fraction(1, 4), 25),
        (F("1.5"), 150), (["1", "2.5"], [100, 250]), (np.array([1, 2]), [100, 200]),
        (np.array([Decimal("1.1"), 2], dtype=object), [110, 200]),
    ])
    def test_exact_inputs_within_precision(self, value, want, no_warnings):
        m = usd(value)
        assert units(m) == want and m.scale == 2 and m.currency is USD

    @pytest.mark.parametrize("value, want", [(1.005, 101), (0.125, 13), (-0.125, -13), (12.345, 1235), (0.1 + 0.2, 30),
                                             (np.array([1.005, 2.675]), [101, 268])])
    def test_float_rounds_half_up_silently(self, value, want, no_warnings):
        assert units(usd(value)) == want

    def test_float_jpy(self, no_warnings):
        assert units(Money(2.5, "JPY")) == 3 and units(Money(np.array([0.4, -2.5]), JPY)) == [0, -3]

    @pytest.mark.parametrize("value, want", [("1.005", 101), (Decimal("-1.005"), -101), (F("0.125"), 13),
                                             (Fraction(1, 8), 13), (["1.001", "2"], [100, 200]),
                                             (np.array([1.005, "2"], dtype=object), [101, 200])])
    def test_exact_inputs_beyond_precision_warn_then_round(self, value, want):
        with pytest.warns(PrecisionLossWarning, match="超出 USD 的精度"):
            m = usd(value)
        assert units(m) == want and m.scale == 2

    def test_precision_warning_is_a_user_warning_and_can_be_an_error(self):
        assert issubclass(PrecisionLossWarning, UserWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PrecisionLossWarning)
            with pytest.raises(PrecisionLossWarning):
                usd("0.001")

    def test_from_money(self, no_warnings):
        m = usd("1.5")
        assert Money(m, "USD") == m and Money(m, USD).currency is USD
        with pytest.raises(TypeError, match="币种不匹配"):
            Money(m, "CNY")

    def test_scale_parameter_removed(self):
        with pytest.raises(TypeError):
            Money("1", "USD", scale=3)

    def test_currency_resolution(self):
        assert usd("1").currency is USD and Money("1", "usd").currency is USD
        with pytest.raises(KeyError):
            Money("1", "XYZ")
        with pytest.raises(ValueError, match="不一致"):
            Money("1", Currency("USD", 3))
        custom = Currency("XYZ", 4)
        assert Money("1.2345", custom).scale == 4

    @pytest.mark.parametrize("bad", [True, np.array([True]), None, object()])
    def test_invalid_values(self, bad):
        with pytest.raises((TypeError, ValueError)):
            usd(bad)

    def test_from_units_copies_and_validates(self):
        arr = np.array([100, 200], dtype=np.int64)
        m = Money.from_units(arr, "USD")
        arr[0] = 999
        assert units(m) == [100, 200] and not m.units.flags.writeable
        assert units(Money.from_units(5, "JPY")) == 5 and Money.from_units(5, "JPY").scale == 0
        assert units(Money.from_units([1, 2], "USD")) == [1, 2]
        for bad in (1.5, np.array([1.5]), True):
            with pytest.raises(TypeError):
                Money.from_units(bad, "USD")
        with pytest.raises(OverflowError):
            Money.from_units(2 ** 63, "USD")

    def test_zeros_ones(self):
        assert units(Money.zeros(3, "USD")) == [0, 0, 0]
        assert Money.zeros((2, 2), "USD").shape == (2, 2)
        assert units(Money.ones(2, "USD")) == [100, 100] and units(Money.ones((), "JPY")) == 1

    def test_from_pandas_series(self, no_warnings):
        pd = pytest.importorskip("pandas")
        assert units(Money(pd.Series([1.5, 2.25]), "USD")) == [150, 225]
        assert units(Money(pd.Series([Decimal("1.50"), Decimal("2")], dtype=object), "USD")) == [150, 200]

    def test_from_polars_series(self, no_warnings):
        pl = pytest.importorskip("polars")
        assert units(Money(pl.Series([1.5, 2.25]), "USD")) == [150, 225]

    def test_removed_redundant_apis(self):
        import moneta
        assert not hasattr(Money, "from_series") and not hasattr(Money, "to_series")
        assert not hasattr(moneta, "calc_tiered_fee")

    def test_properties(self):
        m = usd(["1", "2"])
        assert m.units.tolist() == [100, 200] and m.currency is USD and m.scale == 2


# ===========================================================================
# 2. 内部不变式：精度恒等于币种精度
# ===========================================================================


class TestInvariant:
    def test_wrap_rejects_other_scales(self):
        with pytest.raises(ValueError, match="内部错误"):
            Money._wrap(F("1.001"), USD)
        with pytest.raises(ValueError, match="内部错误"):
            usd("1")._create_like(F("1"))

    def test_every_operation_keeps_currency_digits(self):
        m = usd(["1.23", "-4.56"])
        results = [m + m, m - m, -m, abs(m), +m, m * 3, m * Decimal("0.333"), m / 7, m.sum(), m.mean(),
                   m.round_to_step(Decimal("0.05")), m[0], m.reshape(2, 1), m @ F(["0.5", "0.5"]),
                   divmod(m, 3)[0], divmod(m, 3)[1], divmod(m, usd("1"))[1], usd("1").allocate([1, 2])[0],
                   m.calc_tiered_fee(Decimal("0.001")), m.multiply(F("1.5"), step=Decimal("0.05"))]
        assert all(isinstance(r, Money) and r.scale == 2 and r.currency is USD for r in results)


# ===========================================================================
# 3. 加减
# ===========================================================================


class TestAddSub:
    def test_same_currency(self):
        assert usd("1.5") + usd("0.25") == usd("1.75")
        assert usd("1.5") - usd("0.25") == usd("1.25")
        assert np.all(usd([["1"], ["2"]]) + usd(["0.1", "0.2"]) == usd([["1.1", "1.2"], ["2.1", "2.2"]]))

    def test_currency_mismatch(self):
        with pytest.raises(TypeError, match="币种不匹配"):
            usd("1") + Money("1", "CNY")
        with pytest.raises(TypeError, match="币种不匹配"):
            usd("1") - Money("1", "CNY")

    @pytest.mark.parametrize("other", [1, 1.5, Decimal("1"), F("1"), "1", None, Price("1", "USD"), Quantity(1)])
    def test_non_money_operands(self, other):
        assert usd("1").__add__(other) is NotImplemented and usd("1").__sub__(other) is NotImplemented
        for op in (lambda: usd("1") + other, lambda: other + usd("1"), lambda: usd("1") - other,
                   lambda: other - usd("1")):
            with pytest.raises(TypeError):
                op()

    def test_builtin_sum_needs_start(self):
        parts = [usd("1"), usd("2")]
        with pytest.raises(TypeError):
            sum(parts)
        assert sum(parts, Money.zeros((), "USD")) == usd("3")

    def test_overflow(self):
        with pytest.raises(OverflowError):
            Money.from_units(MAX, "USD") + Money.from_units(1, "USD")

    def test_ordering_with_other_unit_array(self):
        with pytest.raises(TypeError, match="操作数必须是 Money"):
            usd("1") < Price("1", "USD")


# ===========================================================================
# 4. 乘法
# ===========================================================================


class TestMultiply:
    def test_operator_half_up(self):
        assert usd("10") * Decimal("0.333") == usd("3.33")
        assert usd("1.25") * Decimal("0.5") == usd("0.63")  # 0.625 → 0.63
        assert 3 * usd("1.11") == usd("3.33") and usd("1.11") * 3 == usd("3.33")
        assert usd("2") * 1.1 == usd("2.2")  # float 乘数按最短写法
        assert np.all(np.array([1, 2]) * usd(["1", "2"]) == usd(["1", "4"]))  # ndarray 左操作数 → __rmul__
        assert F("2") * usd("3") == usd("6")

    @pytest.mark.parametrize("mode", list(RoundingMode))
    def test_rounding_modes(self, mode):
        for text in ("0.25", "0.35", "-0.25", "-0.35"):
            want = ref_round(Fraction(Decimal(text)) * 100 / 10, mode.value)
            assert units(usd(text).multiply(Decimal("0.1"), rounding=mode)) == want

    def test_random_against_fraction(self):
        rng = random.Random(1)
        for _ in range(3000):
            a, fm, fs = rng.randint(-10 ** 12, 10 ** 12), rng.randint(-10 ** 6, 10 ** 6), rng.randint(0, 8)
            mode = rng.choice(MODES)
            got = Money.from_units(a, "USD").multiply(F._from_mantissa(np.int64(fm), fs), rounding=mode)
            assert units(got) == ref_round(Fraction(a * fm, 10 ** fs), mode), (a, fm, fs, mode)

    def test_money_times_money_forbidden(self):
        with pytest.raises(TypeError, match="禁止两笔金额直接相乘"):
            usd("1") * usd("2")

    @pytest.mark.parametrize("bad", ["0.05", b"1", True, None, ["1", 2]])
    def test_strings_and_non_numeric_rejected(self, bad):
        assert usd("1").__mul__(bad) is NotImplemented and usd("1").__rmul__(bad) is NotImplemented
        with pytest.raises(TypeError):
            usd("1") * bad
        with pytest.raises(TypeError, match="只接受数值"):
            usd("1").multiply(bad)

    def test_unknown_types_get_reflected(self):
        class Custom:
            def __rmul__(self, other):
                return "reflected"

        assert usd("1") * Custom() == "reflected"
        assert usd("1").__mul__(Price("1", "USD")) is NotImplemented
        assert usd("1").__mul__(Quantity(1)) is NotImplemented

    def test_step_single_rounding(self):
        # 先舍入到分会把 0.0499 变成 0.05，再按 0.10 的 half_up 得到 0.10；一次性舍入应得 0.00
        assert usd("1").multiply(Decimal("0.0499"), step=Decimal("0.10")) == usd("0")
        assert usd("1").multiply(Decimal("0.05"), step=Decimal("0.10")) == usd("0.10")  # 恰好一半：half_up
        assert usd("1").multiply(Decimal("0.05"), step=Decimal("0.10"), rounding="half_even") == usd("0")
        assert usd("-1").multiply(Decimal("0.0751"), step=Decimal("0.05")) == usd("-0.10")
        assert np.all(usd(["1", "2"]).multiply(Decimal("0.123"), step=Decimal("0.05")) == usd(["0.10", "0.25"]))

    def test_step_random_against_fraction(self):
        rng = random.Random(2)
        for _ in range(3000):
            a, fm, fs = rng.randint(-10 ** 10, 10 ** 10), rng.randint(-10 ** 6, 10 ** 6), rng.randint(2, 8)
            step_m, step_s = rng.choice([1, 2, 3, 5, 7, 10, 25]), rng.randint(0, 2)
            mode = rng.choice(MODES)
            step = Fraction(step_m, 10 ** step_s)
            want = ref_round(Fraction(a * fm, 10 ** (fs + 2)) / step, mode) * step
            got = Money.from_units(a, "USD").multiply(F._from_mantissa(np.int64(fm), fs), rounding=mode,
                                                      step=F._from_mantissa(np.int64(step_m), step_s))
            assert Fraction(units(got), 100) == want, (a, fm, fs, step_m, step_s, mode)

    def test_step_validation(self):
        assert usd("1").multiply(1, step=F("0.050")) == usd("1.00")  # 末尾 0 不算精度
        with pytest.raises(ValueError, match="精度超过"):
            usd("1").multiply(1, step=Decimal("0.003"))
        for bad in (0, -1, Decimal("-0.05")):
            with pytest.raises(ValueError, match="必须为正数"):
                usd("1").multiply(1, step=bad)
        with pytest.raises(TypeError):
            usd("1").multiply(1, step="0.05")

    def test_overflow(self):
        with pytest.raises(OverflowError):
            Money.from_units(MAX, "USD") * 2


# ===========================================================================
# 5. 除法
# ===========================================================================


class TestDivide:
    def test_by_number_gives_money(self):
        assert usd("10") / 3 == usd("3.33") and usd("10") / Decimal("4") == usd("2.50")
        assert usd("-10").divide(3, rounding="floor") == usd("-3.34")
        with pytest.raises(ValueError, match="不能指定 scale"):
            usd("10").divide(3, scale=4)
        with pytest.raises(ZeroDivisionError):
            usd("1") / 0

    def test_ratio_precision(self):
        r = usd("1") / usd("3")
        assert isinstance(r, F) and r.scale == 10 and r == F("0.3333333333")
        assert Money("1", "JPY") / Money("3", "JPY") == F("0.33333333")  # JPY：0 + 8 位
        assert usd("1").divide(usd("3"), scale=4, rounding="up") == F("0.3334")
        with pytest.raises(TypeError, match="币种不匹配"):
            usd("1") / Money("1", "CNY")

    def test_unit_price_and_quantity(self):
        p = usd("100") / Quantity(3)
        assert isinstance(p, Price) and p.scale == 10 and p.base_unit == "share" and p.quote_currency is USD
        assert p.data == F("33.3333333333")
        q = usd("100") / Price("30", "USD", base_unit="barrel")
        assert isinstance(q, Quantity) and q.unit == "barrel" and q.data == F("3.3333333333")
        assert usd("1895") / Price("189.50", "USD") == Quantity(10)
        assert usd("100").divide(Quantity(3), scale=2).data == F("33.33")
        with pytest.raises(TypeError, match="计价货币"):
            usd("100") / Price("30", "CNY")
        with pytest.raises(ValueError, match="step 只适用于"):
            usd("100").divide(usd("3"), step=Decimal("0.05"))

    def test_step_single_rounding(self):
        assert usd("1").divide(Decimal("20.04"), step=Decimal("0.10")) == usd("0")  # 0.0499... → 0.00
        assert usd("1").divide(20, step=Decimal("0.10")) == usd("0.10")  # 恰好 0.05：half_up

    def test_step_random_against_fraction(self):
        rng = random.Random(3)
        for _ in range(2000):
            a, dm, ds = rng.randint(-10 ** 10, 10 ** 10), rng.randint(1, 10 ** 6) * rng.choice([1, -1]), rng.randint(0,
                                                                                                                     4)
            step_m, step_s = rng.choice([1, 2, 5, 10, 25, 3]), rng.randint(0, 2)
            mode = rng.choice(MODES)
            step = Fraction(step_m, 10 ** step_s)
            want = ref_round(Fraction(a, 100) / Fraction(dm, 10 ** ds) / step, mode) * step
            got = Money.from_units(a, "USD").divide(F._from_mantissa(np.int64(dm), ds), rounding=mode,
                                                    step=F._from_mantissa(np.int64(step_m), step_s))
            assert Fraction(units(got), 100) == want, (a, dm, ds, step_m, step_s, mode)

    @pytest.mark.parametrize("bad", ["2", None, True])
    def test_non_numeric(self, bad):
        assert usd("1").__truediv__(bad) is NotImplemented
        with pytest.raises(TypeError):
            usd("1") / bad
        with pytest.raises(TypeError, match="只接受数值"):
            usd("1").divide(bad)

    def test_number_divided_by_money(self):
        with pytest.raises(TypeError):
            1 / usd("1")


# ===========================================================================
# 6. 矩阵乘法
# ===========================================================================


class TestMatmul:
    def test_money_at_weights(self):
        m = usd(["10.00", "20.00", "30.00"])
        assert m @ F(["0.2", "0.3", "0.5"]) == usd("23.00")
        assert m.matmul([Decimal("0.333")] * 3, rounding="down") == usd("19.98")
        assert m @ [1, 1, 1] == usd("60")

    def test_weights_at_money(self):
        m = usd(["1.00", "2.00"])
        w = [[1, 0], [Decimal("0.5"), Decimal("0.5")]]
        assert np.all(F(w) @ m == usd(["1.00", "1.50"]))
        assert np.all(np.array([[1, 1]]) @ m == usd(["3.00"]))
        assert np.all(w @ m == usd(["1.00", "1.50"]))

    @pytest.mark.parametrize("bad", ["1", usd(["1"]), None])
    def test_rejects_non_numeric(self, bad):
        assert usd(["1"]).__matmul__(bad) is NotImplemented and usd(["1"]).__rmatmul__(bad) is NotImplemented
        with pytest.raises(TypeError):
            usd(["1"]) @ bad


# ===========================================================================
# 7. 规约与符号
# ===========================================================================


class TestReductionsAndSign:
    def test_sum(self):
        m = usd([["1.5", "2"], ["3", "4.25"]])
        assert m.sum() == usd("10.75") and np.sum(m) == usd("10.75")
        assert np.all(m.sum(axis=0) == usd(["4.5", "6.25"]))
        assert np.all(np.sum(m, axis=1, keepdims=True) == usd([["3.5"], ["7.25"]]))

    def test_mean(self):
        assert usd(["1", "2", "4"]).mean() == usd("2.33")
        assert usd(["0.01", "0.02"]).mean() == usd("0.02") and usd(["0.01", "0.02"]).mean(rounding="half_even") == usd(
            "0.02")
        assert usd(["0.01", "0.02"]).mean(rounding="floor") == usd("0.01")
        assert np.mean(usd(["1", "2"])) == usd("1.5")
        assert np.all(np.mean(usd([["1", "2"], ["3", "5"]]), axis=0) == usd(["2", "3.5"]))
        with pytest.raises(TypeError):
            np.mean(usd(["1"]), dtype=float)

    def test_sign(self):
        assert -usd("1.5") == usd("-1.5") and abs(usd("-1.5")) == usd("1.5") and +usd("1") == usd("1")
        with pytest.raises(OverflowError):
            -Money.from_units(MIN, "USD")


# ===========================================================================
# 8. 步长
# ===========================================================================


class TestSteps:
    def test_round_to_step(self):
        assert usd("10.23").round_to_step(Decimal("0.05")) == usd("10.25")
        assert usd("10.23").round_to_step(Decimal("0.05"), rounding="down") == usd("10.20")
        assert usd("345").round_to_step(200) == usd("400")
        assert usd("1.00").round_to_step(F("0.10")) == usd("1.00")
        assert np.all(usd(["10.21", "10.24"]).round_to_step(Decimal("0.05")) == usd(["10.20", "10.25"]))

    def test_step_finer_than_currency_rejected(self):
        with pytest.raises(ValueError, match="精度超过"):
            usd("1.00").round_to_step(Decimal("0.003"))
        with pytest.raises(ValueError, match="必须为正数"):
            usd("1").round_to_step(0)
        with pytest.raises(TypeError):
            usd("1").round_to_step("0.05")

    def test_is_on_step(self):
        assert usd("10.25").is_on_step(Decimal("0.05")) is True and usd("10.23").is_on_step(Decimal("0.05")) is False
        assert usd("1.00").is_on_step(Decimal("0.003")) is False
        assert usd(["1.00", "1.05"]).is_on_step(Decimal("0.10")).tolist() == [True, False]
        with pytest.raises(TypeError):
            usd("1").is_on_step("0.05")


# ===========================================================================
# 9. divmod
# ===========================================================================


class TestDivmod:
    def test_by_money_counts_portions(self):
        count, rest = divmod(usd("10.00"), usd("3.00"))
        assert count == 3 and type(count) is int and rest == usd("1.00")
        count, rest = divmod(usd(["10", "-10"]), usd("3"))
        assert count.tolist() == [3, -4] and np.all(rest == usd(["1", "2"]))
        with pytest.raises(TypeError, match="币种不匹配"):
            divmod(usd("1"), Money("1", "CNY"))
        with pytest.raises(ZeroDivisionError):
            divmod(usd("1"), usd("0"))

    def test_by_integer_splits_evenly(self):
        assert divmod(usd("10"), 3) == (usd("3.33"), usd("0.01"))
        assert divmod(usd("-10"), 3) == (usd("-3.34"), usd("0.02"))  # 向下取整，零头 ≥ 0
        assert divmod(Money("10", "JPY"), np.int64(4)) == (Money("2", "JPY"), Money("2", "JPY"))
        per, rest = divmod(usd(["10", "1"]), np.array([3, 7]))
        assert np.all(per == usd(["3.33", "0.14"])) and np.all(rest == usd(["0.01", "0.02"]))

    def test_split_random(self):
        rng = random.Random(4)
        for _ in range(2000):
            a, n = rng.randint(-10 ** 15, 10 ** 15), rng.randint(1, 10 ** 6)
            per, rest = divmod(Money.from_units(a, "USD"), n)
            assert units(per) == a // n and units(rest) == a % n and units(per) * n + units(rest) == a

    @pytest.mark.parametrize("n", [0, -3, np.array([2, 0])])
    def test_parts_must_be_positive(self, n):
        with pytest.raises(ValueError, match="正整数"):
            divmod(usd("10"), n)

    @pytest.mark.parametrize("n", [3.0, 1.5, Decimal("3"), Fraction(3), F("3"), np.array([1.5])])
    def test_non_integer_suggests_allocate(self, n):
        with pytest.raises(TypeError, match="allocate"):
            divmod(usd("10"), n)

    @pytest.mark.parametrize("bad", ["3", True, None, Price("1", "USD")])
    def test_non_numeric(self, bad):
        assert usd("10").__divmod__(bad) is NotImplemented
        with pytest.raises(TypeError):
            divmod(usd("10"), bad)


# ===========================================================================
# 10. allocate
# ===========================================================================


class TestAllocate:
    def test_examples(self):
        assert usd("100").allocate([1, 1, 1]) == [usd("33.34"), usd("33.33"), usd("33.33")]
        assert usd("100").allocate([0.5, 0.5]) == [usd("50"), usd("50")]
        assert usd("100").allocate([1.5, 1]) == [usd("60"), usd("40")]
        assert usd("1").allocate([Decimal("0.1"), Decimal("0.2")]) == [usd("0.33"), usd("0.67")]
        assert usd("1").allocate(np.array([1, 2])) == [usd("0.33"), usd("0.67")]
        assert usd("1").allocate(F(["1", "0", "1"])) == [usd("0.50"), usd("0"), usd("0.50")]
        assert usd("-1").allocate([1, 1, 1]) == [usd("-0.33"), usd("-0.33"), usd("-0.34")]
        assert usd("1").allocate([]) == []

    def test_random_conservation_and_bound(self):
        rng = random.Random(5)
        for _ in range(1000):
            a = rng.randint(-10 ** 12, 10 ** 12)
            ws = [rng.choice([0, 1, 2, 3, 0.5, 1.25, Decimal("0.1"), Fraction(1, 8)]) for _ in range(rng.randint(1, 8))]
            total = sum(Fraction(str(w)) if not isinstance(w, Fraction) else w for w in ws)
            if total == 0:
                continue
            parts = Money.from_units(a, "USD").allocate(ws)
            got = [units(p) for p in parts]
            exact = [Fraction(a) * (Fraction(str(w)) if not isinstance(w, Fraction) else w) / total for w in ws]
            assert sum(got) == a
            assert all(abs(g - e) < 1 for g, e in zip(got, exact))

    def test_errors(self):
        with pytest.raises(ValueError, match="标量"):
            usd(["1", "2"]).allocate([1, 1])
        with pytest.raises(ValueError, match="一维"):
            usd("1").allocate([[1, 2]])
        with pytest.raises(ValueError, match="非负"):
            usd("1").allocate([1, -1])
        with pytest.raises(ValueError, match="不能为 0"):
            usd("1").allocate([0, 0])
        with pytest.raises(TypeError, match="只接受数值"):
            usd("1").allocate(["1", "2"])
        with pytest.raises(ValueError):
            usd("1").allocate([Fraction(1, 3)])  # 不是有限小数


# ===========================================================================
# 11. 费用
# ===========================================================================

TIERS = [(Decimal("1000"), Decimal("0.01")), (Decimal("5000"), Decimal("0.005")), (None, Decimal("0.001"))]


class TestTieredFee:
    def test_single_rate(self):
        assert usd("1000").calc_tiered_fee(Decimal("0.0003")) == usd("0.30")
        assert usd("1001").calc_tiered_fee(Decimal("0.0005"), rounding="up") == usd("0.51")
        assert np.all(usd(["100", "200"]).calc_tiered_fee([Decimal("0.01"), Decimal("0.02")]) == usd(["1", "4"]))

    def test_tiers(self):
        assert usd("6000").calc_tiered_fee(TIERS) == usd("31.00")  # 10 + 20 + 1
        assert usd("500").calc_tiered_fee(TIERS) == usd("5.00")
        assert usd("-500").calc_tiered_fee(TIERS) == usd("0")
        assert np.all(usd(["500", "6000", "0"]).calc_tiered_fee(TIERS) == usd(["5", "31", "0"]))
        money_tiers = [(usd("1000"), Decimal("0.01")), (None, Decimal("0.001"))]
        assert usd("1500").calc_tiered_fee(money_tiers) == usd("10.50")

    def test_tiers_round_once(self):
        # 各档 0.004 + 0.004 = 0.008：分别舍入为 0 + 0，一次舍入为 0.01
        tiers = [(Decimal("0.4"), Decimal("0.01")), (None, Decimal("0.01"))]
        assert usd("0.8").calc_tiered_fee(tiers) == usd("0.01")

    def test_tiers_random_against_fraction(self):
        rng = random.Random(6)
        for _ in range(1000):
            th = sorted(rng.sample(range(1, 10 ** 6), 3))
            rates = [Decimal(rng.randint(0, 999)).scaleb(-rng.randint(2, 5)) for _ in range(4)]
            tiers = [(Decimal(t).scaleb(-2), r) for t, r in zip(th, rates)] + [(None, rates[3])]
            amt, mode = rng.randint(-10 ** 6, 3 * 10 ** 6), rng.choice(MODES)
            exact, lower = Fraction(0), 0
            for t, r in tiers:
                upper = None if t is None else int(t * 100)
                x = max(amt, lower) if upper is None else min(max(amt, lower), upper)
                exact += (x - lower) * Fraction(r)
                lower = lower if upper is None else upper
            assert units(Money.from_units(amt, "USD").calc_tiered_fee(tiers, rounding=mode)) == ref_round(exact, mode)

    @pytest.mark.parametrize("tiers, msg", [
        ([(Decimal("100"), Decimal("0.1"))], "最后一档"),
        ([(None, Decimal("0.1")), (Decimal("100"), Decimal("0.1"))], "只有最后一档"),
        ([(Decimal("100"), Decimal("0.1")), (Decimal("50"), Decimal("0.2")), (None, Decimal("0.3"))], "严格递增"),
        ([(Decimal("0"), Decimal("0.1")), (None, Decimal("0.2"))], "严格递增"),
        ([(usd(["1", "2"]), Decimal("0.1")), (None, Decimal("0.2"))], "标量"),
    ])
    def test_tier_validation(self, tiers, msg):
        with pytest.raises(ValueError, match=msg):
            usd("150").calc_tiered_fee(tiers)

    def test_tier_types(self):
        with pytest.raises(TypeError, match="只接受数值"):
            usd("150").calc_tiered_fee([("100", Decimal("0.1")), (None, Decimal("0.2"))])
        with pytest.raises(TypeError, match="只接受数值"):
            usd("150").calc_tiered_fee([(Decimal("100"), "0.1"), (None, Decimal("0.2"))])
        with pytest.raises(TypeError, match="币种不匹配"):
            usd("150").calc_tiered_fee([(Money("100", "CNY"), Decimal("0.1")), (None, Decimal("0.2"))])
        with pytest.raises(TypeError, match="只接受数值"):
            usd("150").calc_tiered_fee("0.001")

    def test_min_and_max_fee_elementwise(self):
        m = usd(["10", "1000", "100000"])
        fee = m.calc_tiered_fee(Decimal("0.001"), min_fee=usd("5"), max_fee=usd("50"))
        assert np.all(fee == usd(["5", "5", "50"]))
        per_row_min = usd(["0", "2", "0"])
        assert np.all(m.calc_tiered_fee(Decimal("0.001"), min_fee=per_row_min) == usd(["0.01", "2", "100"]))
        with pytest.raises(ValueError, match="min_fee 不能大于 max_fee"):
            m.calc_tiered_fee(Decimal("0.001"), min_fee=usd("5"), max_fee=usd("2"))
        with pytest.raises(TypeError, match="币种不匹配"):
            m.calc_tiered_fee(Decimal("0.001"), min_fee=Money("5", "CNY"))


# ===========================================================================
# 12. 导出与展示
# ===========================================================================


class TestExportAndDisplay:
    def test_pandas_carries_currency(self):
        pytest.importorskip("pandas")
        s = usd(["1.5", "2"]).to_pandas(name="amt")
        assert s.tolist() == [1.5, 2.0] and s.attrs["currency"] == "USD" and s.name == "amt"
        assert usd([["1"]]).to_pandas(as_float=False).attrs["currency"] == "USD"

    def test_polars(self):
        pytest.importorskip("polars")
        assert usd(["1.50"]).to_polars(as_float=False).to_list() == [Decimal("1.50")]

    def test_float_and_decimal(self):
        assert usd("1.5").to_float() == 1.5 and usd("1.5").to_decimal() == Decimal("1.50")

    def test_repr_str_format(self):
        assert str(usd("1.5")) == "1.50 USD"
        assert repr(usd("1.5")) == "Money(FixedArray(1.50, scale=2), currency='USD')"
        assert "1.50" in usd("1.5").format()
