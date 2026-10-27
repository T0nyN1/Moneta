"""cashflow 测试：CashFlow 与 generate_cash_flow_schedule。

约定（cashflow.py）：
- CashFlow(amount, pay_date, label="") 不可变；amount 必须是 Money，label 必须是 str。
- 付息日 = 起始日 + k 个周期（不逐期累加）；起始日是月末时所有付息日取月末；到期日必须正好落在付息日序列上。
- 票息 = principal × coupon_rate / 每年付息次数，half_up 精确舍入一次。
"""

import random
from datetime import date, datetime
from decimal import Decimal
from fractions import Fraction

import pytest

import moneta
import moneta.cashflow as cashflow_module
from moneta import Money
from moneta.cashflow import CashFlow, generate_cash_flow_schedule as schedule


def usd(v):
    return Money(v, "USD")


def dates(flows, label="Coupon"):
    return [str(cf.pay_date) for cf in flows if cf.label == label]


def half_up(x: Fraction) -> int:
    """对精确值做 half_up（远离零）舍入。"""
    q, r = divmod(abs(x.numerator), x.denominator)
    q += 2 * r >= x.denominator
    return q if x >= 0 else -q


# ===========================================================================
# 1. CashFlow
# ===========================================================================


class TestCashFlow:
    def test_construct(self):
        cf = CashFlow(usd("100.00"), datetime(2026, 6, 1, 9), "Coupon")
        assert (cf.amount, cf.currency.code, cf.pay_date, cf.label) == (usd(100), "USD", date(2026, 6, 1), "Coupon")
        assert CashFlow(usd(1), "2026-06-01").label == ""
        assert CashFlow(usd([1, None]), date(2026, 6, 1)).amount.isna().tolist() == [False, True]  # 金额可以是数组

    @pytest.mark.parametrize("args, err, msg", [
        ((100, "2026-06-01"), TypeError, "amount 必须是 Money"),
        ((Money(1, "USD"), "2026-02-30"), ValueError, "^pay_date 不是合法的 ISO 日期"),
        ((Money(1, "USD"), None), TypeError, "^pay_date 必须是"),
        ((Money(1, "USD"), "2026-06-01", None), TypeError, "label 必须是 str"),
        ((Money(1, "USD"), "2026-06-01", 1), TypeError, "label 必须是 str"),
    ])
    def test_validation(self, args, err, msg):
        with pytest.raises(err, match=msg):
            CashFlow(*args)

    def test_removed_aliases(self):
        with pytest.raises(TypeError):
            CashFlow(usd(1), "2026-06-01", description="Coupon")
        cf = CashFlow(usd(1), "2026-06-01")
        assert not hasattr(cf, "date") and not hasattr(cf, "description")

    def test_immutable_and_display(self):
        cf = CashFlow(usd("100.00"), "2026-06-01", "Coupon")
        with pytest.raises(AttributeError, match="禁止设置属性"):
            cf._amount = usd(999)
        with pytest.raises(AttributeError, match="禁止删除属性"):
            del cf._label
        assert cf.amount == usd(100)
        assert repr(cf) == ("CashFlow(amount=Money(FixedArray(100.00, scale=2), currency='USD'), "
                            "pay_date='2026-06-01', label='Coupon')")
        assert str(cf) == "CashFlow(100.00 USD on 2026-06-01 [Coupon])"


# ===========================================================================
# 2. 付息日
# ===========================================================================


class TestScheduleDates:
    @pytest.mark.parametrize("freq", ["monthly", "Monthly", " MONTHLY ", 12])
    def test_freq_forms(self, freq):
        flows = schedule("2024-01-15", "2024-04-15", freq, coupon=usd(1))
        assert dates(flows) == ["2024-02-15", "2024-03-15", "2024-04-15"]

    @pytest.mark.parametrize("freq, n", [("quarterly", 4), ("semi-annual", 2), ("semiannual", 2), ("annual", 1),
                                         ("annually", 1), (4, 4), (2, 2), (1, 1)])
    def test_one_year(self, freq, n):
        assert len(schedule("2024-03-10", "2025-03-10", freq, coupon=usd(1))) == n

    @pytest.mark.parametrize("freq", [3, 0, 6, True, 4.0, "weekly", "", None])
    def test_unsupported_freq(self, freq):
        with pytest.raises(ValueError, match="不支持的付息频率"):
            schedule("2024-01-01", "2025-01-01", freq, coupon=usd(1))

    def test_no_month_end_drift(self):
        # 起始日是月末：所有付息日取月末，不会漂到 29 号
        assert dates(schedule("2024-01-31", "2024-06-30", "monthly", coupon=usd(1))) == \
            ["2024-02-29", "2024-03-31", "2024-04-30", "2024-05-31", "2024-06-30"]
        assert dates(schedule("2024-02-29", "2024-08-31", "quarterly", coupon=usd(1))) == ["2024-05-31", "2024-08-31"]
        assert dates(schedule("2023-11-30", "2024-02-29", "quarterly", coupon=usd(1))) == ["2024-02-29"]
        # 起始日不是月末：日超出当月天数时取月末，下一期回到原来的日
        assert dates(schedule("2024-01-30", "2024-04-30", "monthly", coupon=usd(1))) == \
            ["2024-02-29", "2024-03-30", "2024-04-30"]
        assert dates(schedule("2024-01-31", "2024-04-30", "monthly", coupon=usd(1))) == \
            ["2024-02-29", "2024-03-31", "2024-04-30"]  # 曾经生成 4 期

    def test_long_schedule_is_computed_from_start(self):
        eom = schedule("2024-08-31", "2054-08-31", "monthly", coupon=usd(1))
        assert len(eom) == 360 and all(date.fromordinal(cf.pay_date.toordinal() + 1).day == 1 for cf in eom)
        mid = schedule("2024-01-30", "2054-01-30", "monthly", coupon=usd(1))
        assert len(mid) == 360 and all(cf.pay_date.day == (30 if cf.pay_date.month != 2 else
                                                           (29 if cf.pay_date.year % 4 == 0 else 28)) for cf in mid)

    @pytest.mark.parametrize("start, end, freq", [
        ("2024-01-01", "2024-05-01", "quarterly"),  # 曾经在 05-01 再付一整期
        ("2024-01-31", "2024-06-29", "monthly"),
        ("2024-01-15", "2024-03-14", "monthly"),
        ("2024-01-01", "2024-06-01", "annual"),  # 不足一期
    ])
    def test_end_must_be_on_schedule(self, start, end, freq):
        with pytest.raises(ValueError, match="不在付息日序列上"):
            schedule(start, end, freq, coupon=usd(1))

    def test_off_schedule_message(self):
        with pytest.raises(ValueError, match=r"^end_date \(2024-05-01\) 不在付息日序列上（位于付息日 2024-04-01 与付息日 "
                                             r"2024-07-01 之间），不支持不足一期的付息$"):
            schedule("2024-01-01", "2024-05-01", "quarterly", coupon=usd(1))
        with pytest.raises(ValueError, match=r"（位于起息日 2024-01-01 与付息日 2025-01-01 之间）"):
            schedule("2024-01-01", "2024-06-01", "annual", coupon=usd(1))  # 早于第一个付息日

    def test_dates_validation(self):
        with pytest.raises(ValueError, match="必须严格晚于 start_date"):
            schedule("2024-01-01", "2024-01-01", "monthly", coupon=usd(1))
        with pytest.raises(ValueError, match="必须严格晚于"):
            schedule("2025-01-01", "2024-01-01", "monthly", coupon=usd(1))
        with pytest.raises(ValueError, match="^end_date 不是合法的 ISO 日期"):
            schedule("2024-01-01", "2024-13-01", "monthly", coupon=usd(1))
        with pytest.raises(TypeError, match="^start_date 必须是"):
            schedule(None, "2024-12-01", "monthly", coupon=usd(1))


# ===========================================================================
# 3. 金额
# ===========================================================================


class TestScheduleAmounts:
    def test_coupon_and_principal(self):
        flows = schedule(date(2026, 1, 1), datetime(2027, 1, 1), 2, coupon=usd(25), principal=usd(1000))
        assert [(str(cf.pay_date), cf.amount, cf.label) for cf in flows] == [
            ("2026-07-01", usd(25), "Coupon"), ("2027-01-01", usd(25), "Coupon"),
            ("2027-01-01", usd(1000), "Principal")]
        only_principal = schedule("2026-01-01", "2027-01-01", "quarterly", principal=usd(1000))
        assert [(str(cf.pay_date), cf.label) for cf in only_principal] == [("2027-01-01", "Principal")]

    @pytest.mark.parametrize("rate", ["0.05", 0.05, Decimal("0.05"), Fraction(1, 20), moneta.FixedArray("0.050")])
    def test_coupon_rate_forms(self, rate):
        flows = schedule("2026-01-01", "2027-01-01", "semi-annual", principal=usd(1000), coupon_rate=rate)
        assert [cf.amount for cf in flows] == [usd("25.00"), usd("25.00"), usd(1000)]
        assert flows[0].amount.data.scale == 2

    def test_coupon_rate_examples(self):
        def first(principal, rate, freq):
            return schedule("2024-01-01", "2025-01-01", freq, principal=principal, coupon_rate=rate)[0].amount

        assert first(usd(1200), "0.06", 4) == usd(18)  # 曾经 frequency=3 时静默按季付 24
        assert first(usd(1000), "0.07", 12) == usd("5.83")  # 70 / 12 = 5.8333…
        assert first(usd("1000.20"), "0.03", 12) == usd("2.50")  # 30.006 / 12 = 2.5005 → 2.50
        assert first(usd(-1000), "0.07", 12) == usd("-5.83")
        assert first(Money(100000, "JPY"), "0.0125", 12) == Money(104, "JPY")  # 104.1666… JPY 无小数
        assert first(usd(1000), 0, 12) == usd(0)

    @pytest.mark.parametrize("rate, want", [
        ("0.06", "0.01"), ("0.0600000001", "0.01"), ("0.0599999999", "0.00"),  # 1 × 0.06 / 12 = 0.005 恰为平局
        ("0.18", "0.02"), ("0.1799999999", "0.01"),  # 0.015 与略低于 0.015
    ])
    def test_coupon_rate_ties(self, rate, want):
        def first(principal):
            return schedule("2024-01-01", "2025-01-01", 12, principal=principal, coupon_rate=rate)[0].amount

        assert first(usd(1)) == usd(want) and first(usd(-1)) == -usd(want)  # 平局远离零，截断不会越过平局点

    def test_coupon_rate_exact_random(self):
        rng = random.Random(11)
        for _ in range(3000):
            cents = rng.randint(-10 ** 13, 10 ** 13)
            rate = Fraction(rng.randint(0, 10 ** 6), 10 ** rng.randint(4, 9))  # 0 ~ 100
            n = rng.choice([1, 2, 4, 12])
            got = schedule("2024-01-01", "2025-01-01", n, principal=Money.from_units(cents, "USD"),
                           coupon_rate=Decimal(rate.numerator) / Decimal(rate.denominator))[0].amount
            assert int(got.units) == half_up(Fraction(cents) * rate / n)

    def test_float_rate_with_many_digits_does_not_overflow(self):
        # 0.07 / 12 = 0.005833333333333334（18 位小数）：精确乘积会超出 int64，按 digits + 1 位截断后再舍入
        got = schedule("2024-01-01", "2025-01-01", 1, principal=usd("1000000000.00"), coupon_rate=0.07 / 12)
        assert got[0].amount == usd("5833333.33")  # 10^9 × 0.005833333333333334 = 5833333.333…

    def test_vectorized_principal(self):
        flows = schedule("2024-01-01", "2025-01-01", "annual", principal=usd(["100", None, "300"]), coupon_rate="0.1")
        assert flows[0].amount.isna().tolist() == [False, True, False]
        assert (flows[0].amount.fillna(usd(0)) == usd(["10", "0", "30"])).all()

    @pytest.mark.parametrize("kwargs, err, msg", [
        (dict(), ValueError, "至少需要给出一种现金流"),
        (dict(coupon_rate="0.05"), ValueError, "必须同时给出 principal"),
        (dict(coupon=Money(1, "USD"), principal=Money(100, "USD"), coupon_rate="0.05"), ValueError, "只能给出一个"),
        (dict(coupon=Money(1, "USD"), principal=Money(100, "CNY")), ValueError, "币种 USD 与本金币种 CNY 不一致"),
        (dict(principal=Money(100, "USD"), coupon_rate=float("nan")), ValueError, "coupon_rate 不能是缺失值"),
        (dict(principal=Money(100, "USD"), coupon_rate="abc"), ValueError, "无法解析"),
        (dict(principal=Money(100, "USD"), coupon_rate=True), TypeError, "bool"),
        (dict(coupon=1), TypeError, "coupon 必须是 Money"),
        (dict(principal="100"), TypeError, "principal 必须是 Money"),
    ])
    def test_validation(self, kwargs, err, msg):
        with pytest.raises(err, match=msg):
            schedule("2024-01-01", "2025-01-01", "annual", **kwargs)

    def test_removed_aliases(self):
        for kwargs in (dict(maturity_date="2025-01-01"), dict(frequency=2), dict(periodic_amount=usd(1)),
                       dict(final_principal=usd(1)), dict(convention="ACT/365")):
            with pytest.raises(TypeError):
                schedule("2024-01-01", "2025-01-01", "annual", coupon=usd(1), **kwargs)
        with pytest.raises(TypeError):
            schedule("2024-01-01", "2025-01-01")  # freq 必填
        with pytest.raises(TypeError):
            schedule("2024-01-01", "2025-01-01", "annual", usd(1))  # coupon 等只能用关键字


def test_exports():
    assert cashflow_module.__all__ == ["CashFlow", "generate_cash_flow_schedule"]
    assert moneta.CashFlow is CashFlow and moneta.generate_cash_flow_schedule is schedule
