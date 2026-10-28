"""conventions 模块测试：计日惯例解析与 year_fraction 计算。"""

import random
from datetime import date, datetime, timedelta
from fractions import Fraction

import pytest

import moneta
from moneta.conventions import DayCountConvention, year_fraction
from moneta.currency import USD
from moneta.lot import PositionLot
from moneta.money import Money

C = DayCountConvention


# ---------- 参考实现（按惯例定义逐条写出，用 Fraction 精确计算） ----------

def _ref_30_360(d1: date, d2: date) -> Fraction:
    """30/360 Bond Basis（ISDA 2006 4.16(f)）。"""
    day1 = 30 if d1.day == 31 else d1.day
    day2 = 30 if d2.day == 31 and day1 == 30 else d2.day
    return Fraction(360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (day2 - day1), 360)


def _ref_act_act_isda(d1: date, d2: date) -> Fraction:
    """ACT/ACT ISDA：逐日累加 1 / 当年天数。"""
    total, d = Fraction(0), d1
    while d < d2:
        total += Fraction(1, 366 if (d.year % 4 == 0 and d.year % 100 != 0) or d.year % 400 == 0 else 365)
        d += timedelta(days=1)
    return total


def _ref(d1: date, d2: date, conv: DayCountConvention) -> Fraction:
    if d1 > d2:
        return -_ref(d2, d1, conv)
    if conv is C.ACT_360:
        return Fraction((d2 - d1).days, 360)
    if conv is C.ACT_365:
        return Fraction((d2 - d1).days, 365)
    if conv is C.THIRTY_360:
        return _ref_30_360(d1, d2)
    return _ref_act_act_isda(d1, d2)


def _random_date(rng: random.Random) -> date:
    return date(1999, 1, 1) + timedelta(days=rng.randrange(0, 365 * 8))


# ---------- 枚举 ----------

class TestEnum:
    def test_members_and_values(self):
        assert [(m.name, m.value) for m in C] == [
            ("ACT_360", "ACT/360"), ("ACT_365", "ACT/365"), ("THIRTY_360", "30/360"), ("ACT_ACT", "ACT/ACT")]

    def test_exports(self):
        import moneta.conventions as mod

        assert mod.__all__ == ["DayCountConvention", "year_fraction"]
        assert moneta.DayCountConvention is C
        assert moneta.year_fraction is year_fraction


# ---------- 惯例参数解析 ----------

class TestConventionParsing:
    @pytest.mark.parametrize("text, conv", [
        ("ACT/360", C.ACT_360), ("act/360", C.ACT_360), ("ACT_360", C.ACT_360), ("act_360", C.ACT_360),
        ("  ACT/360\t", C.ACT_360), ("ACT/365", C.ACT_365), ("Act_365", C.ACT_365),
        ("30/360", C.THIRTY_360), ("thirty_360", C.THIRTY_360), (" 30/360 ", C.THIRTY_360),
        ("ACT/ACT", C.ACT_ACT), ("act/act", C.ACT_ACT), ("ACT_ACT", C.ACT_ACT),
    ])
    def test_string_forms(self, text, conv):
        d1, d2 = date(2023, 11, 30), date(2025, 2, 28)
        assert year_fraction(d1, d2, text) == year_fraction(d1, d2, conv)

    def test_default_is_act_365(self):
        assert year_fraction("2024-01-01", "2025-01-01") == 366 / 365

    @pytest.mark.parametrize("text", ["", "ACT", "ACT/365F", "ACT 360", "ACT-360", "30E/360", "ACT/ACT ICMA"])
    def test_unknown_string(self, text):
        with pytest.raises(ValueError) as exc:
            year_fraction("2024-01-01", "2024-07-01", text)
        msg = str(exc.value)
        assert msg.startswith(f"不支持的计日惯例 {text!r}")
        assert "'ACT/360'" in msg and "'THIRTY_360'" in msg

    @pytest.mark.parametrize("value", [None, 5, 1.0, True, b"ACT/360", ["ACT/360"]])
    def test_wrong_type(self, value):
        with pytest.raises(TypeError, match=f"convention 必须是 DayCountConvention 或 str，收到 {type(value).__name__}"):
            year_fraction("2024-01-01", "2024-07-01", value)


# ---------- 日期参数 ----------

class TestDates:
    def test_accepts_str_date_datetime(self):
        expected = 182 / 360
        assert year_fraction("2024-01-01", "2024-07-01", C.ACT_360) == expected
        assert year_fraction(date(2024, 1, 1), date(2024, 7, 1), C.ACT_360) == expected
        assert year_fraction(datetime(2024, 1, 1, 23, 59), datetime(2024, 7, 1, 0, 1), C.ACT_360) == expected

    @pytest.mark.parametrize("args, name", [
        (("x", "2024-01-01"), "start_date"),
        (("2024-01-01", "2024-13-01"), "end_date"),
    ])
    def test_invalid_date_names_field(self, args, name):
        with pytest.raises(ValueError, match=f"^{name} 不是合法的 ISO 日期"):
            year_fraction(*args)

    @pytest.mark.parametrize("args, name", [
        ((None, "2024-01-01"), "start_date"),
        (("2024-01-01", 20240101), "end_date"),
    ])
    def test_wrong_date_type_names_field(self, args, name):
        with pytest.raises(TypeError, match=f"^{name} 必须是"):
            year_fraction(*args)


# ---------- 计算 ----------

class TestYearFraction:
    @pytest.mark.parametrize("conv", list(C))
    def test_same_date_is_zero(self, conv):
        result = year_fraction("2024-02-29", "2024-02-29", conv)
        assert result == 0.0 and isinstance(result, float)

    @pytest.mark.parametrize("d1, d2, conv, expected", [
        # ACT/360、ACT/365 Fixed：实际天数除以固定分母，闰年也除以 365
        ("2026-01-01", "2026-07-01", C.ACT_360, Fraction(181, 360)),
        ("2026-01-01", "2026-07-01", C.ACT_365, Fraction(181, 365)),
        ("2024-01-01", "2025-01-01", C.ACT_365, Fraction(366, 365)),
        # 30/360 Bond Basis
        ("2026-01-01", "2026-07-01", C.THIRTY_360, Fraction(1, 2)),
        ("2026-01-31", "2026-02-28", C.THIRTY_360, Fraction(28, 360)),   # D1=31→30
        ("2026-01-30", "2026-03-31", C.THIRTY_360, Fraction(60, 360)),   # D1=30 时 D2=31→30
        ("2026-01-31", "2026-03-31", C.THIRTY_360, Fraction(60, 360)),
        ("2026-01-15", "2026-03-31", C.THIRTY_360, Fraction(76, 360)),   # D1<30 时 D2=31 保留
        ("2024-02-29", "2024-08-31", C.THIRTY_360, Fraction(182, 360)),  # 无 2 月月末规则
        ("2023-12-31", "2025-12-31", C.THIRTY_360, Fraction(2)),
        # ACT/ACT ISDA
        ("2024-01-01", "2025-01-01", C.ACT_ACT, Fraction(1)),
        ("2023-01-01", "2024-01-01", C.ACT_ACT, Fraction(1)),
        ("2024-03-01", "2024-09-01", C.ACT_ACT, Fraction(184, 366)),
        ("2023-07-01", "2024-07-01", C.ACT_ACT, Fraction(184, 365) + Fraction(182, 366)),
        ("2023-12-31", "2026-01-02", C.ACT_ACT, Fraction(1, 365) + 2 + Fraction(1, 365)),
    ])
    def test_known_values(self, d1, d2, conv, expected):
        assert year_fraction(d1, d2, conv) == float(expected)
        assert year_fraction(d2, d1, conv) == -float(expected)

    @pytest.mark.parametrize("conv", list(C))
    def test_random_against_reference(self, conv):
        """随机日期对与参考实现比较：结果应等于精确值正确舍入后的 float。"""
        rng = random.Random(20261027 + list(C).index(conv))
        for _ in range(2000):
            d1, d2 = _random_date(rng), _random_date(rng)
            assert year_fraction(d1, d2, conv) == float(_ref(d1, d2, conv)), (d1, d2, conv)

    @pytest.mark.parametrize("conv", list(C))
    def test_month_ends_against_reference(self, conv):
        """所有月末对月末（含 2 月 28/29 日）组合，覆盖 30/360 的调整与 ACT/ACT 的跨年分段。"""
        ends = [date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
                for y in (2023, 2024, 2025) for m in range(1, 13)]
        for d1 in ends:
            for d2 in ends:
                assert year_fraction(d1, d2, conv) == float(_ref(d1, d2, conv)), (d1, d2, conv)

    @pytest.mark.parametrize("conv", list(C))
    def test_antisymmetric(self, conv):
        rng = random.Random(7)
        for _ in range(500):
            d1, d2 = _random_date(rng), _random_date(rng)
            assert year_fraction(d2, d1, conv) == -year_fraction(d1, d2, conv)

    def test_act_act_additive_across_years(self):
        d1, mid, d2 = date(2023, 5, 17), date(2024, 1, 1), date(2024, 11, 3)
        exact = _ref_act_act_isda(d1, mid) + _ref_act_act_isda(mid, d2)
        assert year_fraction(d1, d2, C.ACT_ACT) == float(exact)


# ---------- 下游调用 ----------

def test_position_lot_accepts_string_convention():
    lot = PositionLot(USD, Money("1", "USD"), "2024-01-01")
    assert lot.year_fraction("2024-07-01", "act/360") == 182 / 360
    assert lot.year_fraction("2024-07-01", "ACT_ACT") == 182 / 366
    with pytest.raises(ValueError, match="不支持的计日惯例"):
        lot.year_fraction("2024-07-01", "ACT/365F")
