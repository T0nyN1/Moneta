"""formatter 模块测试：MonetaryFormatter.format / format_array 与 Money.format。"""

import random

import numpy as np
import pytest

import moneta
from moneta import FixedArray, Money
from moneta.currency import Currency
from moneta.formatter import MonetaryFormatter as MF


def usd(v):
    return Money(v, "USD")


# ---------- 数字部分 ----------

class TestNumber:
    @pytest.mark.parametrize("value, text", [
        ("0", "$0.00 USD"), ("0.05", "$0.05 USD"), ("1", "$1.00 USD"), ("999.99", "$999.99 USD"),
        ("1000", "$1,000.00 USD"), ("1234567.89", "$1,234,567.89 USD"),
        ("92233720368547758.07", "$92,233,720,368,547,758.07 USD"),
    ])
    def test_grouping(self, value, text):
        assert MF.format(usd(value)) == text

    def test_zero_digit_and_three_digit_currencies(self):
        assert MF.format(Money("-1234567", "JPY")) == "-¥1,234,567 JPY"
        kwd = Currency("KWD", 3, symbol="KD")
        assert MF.format(Money("-1234.005", kwd)) == "-KD 1,234.005 KWD"

    def test_int64_extremes(self):
        lo = Money._wrap(FixedArray._from_mantissa(np.int64(np.iinfo(np.int64).min), 2), moneta.USD)
        assert MF.format(lo, show_symbol=False, show_code=False) == "-92,233,720,368,547,758.08"

    def test_separators(self):
        m = usd("-1234567.89")
        assert MF.format(m, thousands_sep=".", decimal_sep=",") == "-$1.234.567,89 USD"
        assert MF.format(m, thousands_sep=" ", decimal_sep=",") == "-$1 234 567,89 USD"
        assert MF.format(m, thousands_sep="") == "-$1234567.89 USD"
        assert MF.format(m, thousands_sep="'", decimal_sep="·") == "-$1'234'567·89 USD"

    def test_random_against_reference(self):
        """数字部分去掉分隔符后应与 str(Decimal) 一致。"""
        rng = random.Random(20261027)
        for _ in range(2000):
            mantissa = rng.randint(-10 ** 17, 10 ** 17)
            m = Money._wrap(FixedArray._from_mantissa(np.int64(mantissa), 2), moneta.USD)
            text = MF.format(m, show_symbol=False, show_code=False)
            assert text.replace(",", "") == str(m.data.to_decimal()), mantissa
            int_part = text.lstrip("-").split(".")[0]
            assert all(len(g) == 3 for g in int_part.split(",")[1:]) and 1 <= len(int_part.split(",")[0]) <= 3


# ---------- 符号、代码与负数 ----------

class TestSymbolAndSign:
    @pytest.mark.parametrize("kwargs, pos, neg", [
        ({}, "$1,234.50 USD", "-$1,234.50 USD"),
        ({"accounting": True}, "$1,234.50 USD", "($1,234.50 USD)"),
        ({"show_code": False}, "$1,234.50", "-$1,234.50"),
        ({"show_symbol": False}, "1,234.50 USD", "-1,234.50 USD"),
        ({"show_symbol": False, "show_code": False}, "1,234.50", "-1,234.50"),
        ({"show_symbol": False, "show_code": False, "accounting": True}, "1,234.50", "(1,234.50)"),
    ])
    def test_combinations(self, kwargs, pos, neg):
        assert MF.format(usd("1234.5"), **kwargs) == pos
        assert MF.format(usd("-1234.5"), **kwargs) == neg

    def test_negative_zero_has_no_sign(self):
        assert MF.format(usd("-0.00"), accounting=True) == "$0.00 USD"

    def test_symbol_equal_to_code_shown_once(self):
        chf = Money("-1234.5", "CHF")
        assert MF.format(chf) == "-1,234.50 CHF"
        assert MF.format(chf, accounting=True) == "(1,234.50 CHF)"
        assert MF.format(chf, show_code=False) == "-CHF 1,234.50"
        assert MF.format(chf, show_symbol=False) == "-1,234.50 CHF"

    def test_alphabetic_symbol_gets_space(self):
        fr = Currency("XFR", 2, symbol="Fr")
        assert MF.format(Money("1", fr)) == "Fr 1.00 XFR"
        assert MF.format(Money("1", "HKD"), show_code=False) == "HK$1.00"

    def test_currency_without_symbol(self):
        bare = Currency("XBR", 2)
        assert MF.format(Money("-1", bare)) == "-1.00 XBR"
        assert MF.format(Money("-1", bare), show_code=False, accounting=True) == "(1.00)"

    def test_yen_symbol_is_disambiguated_by_code(self):
        assert MF.format(Money("1", "CNY")) == "¥1.00 CNY"
        assert MF.format(Money("1", "JPY")) == "¥1 JPY"


# ---------- 缺失值、校验 ----------

class TestValidation:
    def test_na_scalar(self):
        assert MF.format(usd(["1.00", None])[1]) == "<NA>"
        assert MF.format(usd(["1.00", None])[1], accounting=True, show_code=False) == "<NA>"

    def test_format_requires_scalar(self):
        with pytest.raises(ValueError, match="format_array"):
            MF.format(usd(["1", "2"]))

    @pytest.mark.parametrize("value", [1.5, "1.00 USD", FixedArray("1.00"), None])
    def test_requires_money(self, value):
        for func in (MF.format, MF.format_array):
            with pytest.raises(TypeError, match=f"money 必须是 Money，收到 {type(value).__name__}"):
                func(value)

    def test_separator_validation(self):
        with pytest.raises(TypeError, match="thousands_sep 必须是 str，收到 NoneType"):
            MF.format(usd("1"), thousands_sep=None)
        with pytest.raises(TypeError, match="decimal_sep 必须是 str，收到 int"):
            MF.format_array(usd(["1"]), decimal_sep=0)
        with pytest.raises(ValueError, match="decimal_sep 不能为空"):
            MF.format(usd("1"), decimal_sep="")
        with pytest.raises(ValueError, match="不能相同"):
            MF.format(usd("1"), thousands_sep=".")

    def test_removed_parameters(self):
        for kwargs in (dict(locale="de_DE"), dict(use_parentheses=True), dict(use_symbol=True)):
            with pytest.raises(TypeError):
                MF.format(usd("1"), **kwargs)
            with pytest.raises(TypeError):
                MF.format_array(usd(["1"]), **kwargs)
            with pytest.raises(TypeError):
                usd("1").format(**kwargs)


# ---------- format_array ----------

class TestFormatArray:
    def test_shapes_and_values(self):
        m = usd([["1.00", None], ["-2.5", "1000"]])
        out = MF.format_array(m, accounting=True)
        assert out.shape == (2, 2) and out.dtype == object
        assert out.tolist() == [["$1.00 USD", "<NA>"], ["($2.50 USD)", "$1,000.00 USD"]]

    def test_matches_scalar_format(self):
        m = usd(["-1234.5", "0", "7", None, "-0.01"])
        for kwargs in ({}, {"accounting": True, "show_code": False}, {"thousands_sep": ".", "decimal_sep": ","}):
            out = MF.format_array(m, **kwargs)
            assert list(out) == [MF.format(m[i], **kwargs) for i in range(len(out))]

    def test_scalar_and_empty(self):
        out = MF.format_array(usd("1"))
        assert out.shape == () and out[()] == "$1.00 USD"
        assert MF.format_array(usd([])).shape == (0,)


# ---------- Money.format 与导出 ----------

def test_money_format_forwards_all_parameters():
    m = usd("-1234.5")
    assert m.format() == MF.format(m)
    assert m.format(accounting=True, show_symbol=False, show_code=True, thousands_sep=" ", decimal_sep=",") \
        == "(1 234,50 USD)"


def test_exports():
    import moneta.formatter as mod

    assert mod.__all__ == ["MonetaryFormatter"]
    assert moneta.MonetaryFormatter is MF
