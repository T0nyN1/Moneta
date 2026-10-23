"""_units 测试：通配单位常量、单位校验 check_unit 与兼容性判定 units_compatible。

约定（_units.py）：
- 单位必须是非空 str，去掉首尾空白后返回；大小写敏感。
- 两个单位兼容当且仅当相同，或至少一方为通配单位 "Any"。
- Quantity、Price、Instrument 的单位参数都按同一规则校验（Instrument 另外禁止通配单位，见 test_instrument）。
"""

import pytest

from moneta import Instrument, Price, Quantity
from moneta._units import WILDCARD_UNIT, check_unit, units_compatible

UNIT_CONSUMERS = {
    "Quantity": lambda unit: Quantity(1, unit=unit).unit,
    "Price": lambda unit: Price("1", "USD", base_unit=unit).base_unit,
    "Instrument": lambda unit: Instrument("X", "USD", base_unit=unit).base_unit,
}


class TestWildcard:
    def test_constant(self):
        assert WILDCARD_UNIT == "Any"

    def test_public_reexports(self):
        from moneta import quantity
        assert quantity.WILDCARD_UNIT is WILDCARD_UNIT


class TestCheckUnit:
    @pytest.mark.parametrize("unit, want", [("share", "share"), ("  barrel \t\n", "barrel"), ("BTC", "BTC"),
                                            ("btc", "btc"), ("Any", "Any"), ("合约", "合约")])
    def test_normalization(self, unit, want):
        assert check_unit(unit) == want

    @pytest.mark.parametrize("bad", [None, 1, 1.5, b"share", ["share"]])
    def test_rejects_non_str(self, bad):
        with pytest.raises(TypeError, match="单位必须是 str"):
            check_unit(bad)

    @pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
    def test_rejects_blank(self, bad):
        with pytest.raises(ValueError, match="单位不能为空"):
            check_unit(bad)

    @pytest.mark.parametrize("consumer", list(UNIT_CONSUMERS))
    def test_all_consumers_share_the_rule(self, consumer):
        make = UNIT_CONSUMERS[consumer]
        assert make("  barrel ") == "barrel" and make("BTC") == "BTC"
        for bad in (None, 1, b"share"):
            with pytest.raises(TypeError, match="单位必须是 str"):
                make(bad)
        for bad in ("", "   "):
            with pytest.raises(ValueError, match="单位不能为空"):
                make(bad)


class TestUnitsCompatible:
    @pytest.mark.parametrize("a, b, want", [
        ("share", "share", True), ("barrel", "barrel", True), ("share", "barrel", False),
        ("share", "Share", False),  # 大小写敏感
        ("Any", "share", True), ("barrel", "Any", True), ("Any", "Any", True),
        ("any", "share", False),  # 只有精确的 "Any" 是通配单位
    ])
    def test_rule(self, a, b, want):
        assert units_compatible(a, b) is want and units_compatible(b, a) is want  # 对称

    def test_consumers_follow_the_rule(self):
        assert (Quantity(1, unit="Any") + Quantity(1, unit="barrel")).unit == "barrel"
        assert Price("2", "USD", base_unit="Any") * Quantity(3, unit="barrel") == Price("2", "USD") * Quantity(3)
        with pytest.raises(TypeError):
            Quantity(1, unit="any") + Quantity(1, unit="barrel")
