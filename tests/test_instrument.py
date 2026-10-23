"""Instrument 测试：构造与校验、不可变与值语义、按标的构造数量 / 价格 / 名义金额。

约定（instrument.py）：
- 相等与哈希只看 (symbol, quote_currency)；名称、乘数等参考数据不参与比较。
- 代码去空白转大写且不能为空；基准单位不能是通配单位 "Any"；乘数与 tick 必须是正的数值标量。
- notional = 价格 × 数量 × 合约乘数，只舍入一次到计价货币精度；裸数字按本标的的计价货币与单位解释。
"""

import copy
import pickle
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from moneta import Instrument, Money, PrecisionLossWarning, Price, Quantity
from moneta.currency import CNY, USD
from moneta.fixed_array import FixedArray as F


def aapl(**kw):
    return Instrument("AAPL", "USD", **kw)


# ===========================================================================
# 1. 构造与校验
# ===========================================================================


class TestConstruct:
    def test_normalization_and_defaults(self):
        inst = Instrument("  aapl ", "usd")
        assert (inst.symbol, inst.quote_currency, inst.base_unit) == ("AAPL", USD, "share")
        assert inst.multiplier == F(1) and inst.tick_size is None
        assert (inst.name, inst.asset_class) == ("AAPL", "Equity")
        full = Instrument("cl2412", CNY, base_unit=" barrel ", multiplier=1000, tick_size=Decimal("0.5"),
                          name=" Crude Oil ", asset_class="Commodity")
        assert (full.base_unit, full.name, full.asset_class) == ("barrel", "Crude Oil", "Commodity")
        assert full.quote_currency is CNY and full.multiplier == F(1000) and full.tick_size == F("0.5")
        assert Instrument("X", "USD", name="  ", asset_class=" ").name == "X"
        assert Instrument("X", "USD", asset_class=" ").asset_class == "Equity"

    @pytest.mark.parametrize("value, want", [(1000, F(1000)), (Decimal("0.1"), F("0.1")), (Fraction(1, 4), F("0.25")),
                                             (2.5, F("2.5")), (F("10.00"), F("10.00")), (np.int64(5), F(5))])
    def test_multiplier_and_tick_keep_exact_value(self, value, want):
        inst = aapl(multiplier=value, tick_size=value)
        assert inst.multiplier == want and inst.multiplier.scale == want.scale  # 不再静默舍入到整数
        assert inst.tick_size == want

    def test_symbol_validation(self):
        for bad in (None, 123, b"AAPL"):
            with pytest.raises(TypeError, match="标的代码必须是 str"):
                Instrument(bad, "USD")
        for bad in ("", "   "):
            with pytest.raises(ValueError, match="标的代码不能为空"):
                Instrument(bad, "USD")

    def test_quote_currency_is_required_and_resolved(self):
        with pytest.raises(TypeError):
            Instrument("AAPL")  # 没有默认的 USD
        with pytest.raises(KeyError, match="未注册"):
            Instrument("AAPL", "XYZ")

    def test_base_unit_cannot_be_wildcard(self):  # 一般的单位校验见 test_units
        with pytest.raises(ValueError, match="通配单位"):
            aapl(base_unit="Any")
        assert aapl(base_unit="any").base_unit == "any"  # 只有精确的 "Any" 是通配单位

    @pytest.mark.parametrize("field", ["multiplier", "tick_size"])
    def test_positive_scalar_fields(self, field):
        for bad, err, msg in (("1", TypeError, "只接受数值"), (True, TypeError, "只接受数值"),
                              (0, ValueError, "必须为正数"), (-1, ValueError, "必须为正数"),
                              (float("nan"), ValueError, "不能是缺失值"), ([1, 2], ValueError, "必须是标量"),
                              (Money("1", "USD"), TypeError, "只接受数值")):
            with pytest.raises(err, match=msg):
                aapl(**{field: bad})
        if field == "multiplier":
            with pytest.raises(ValueError, match="不能是缺失值"):
                aapl(multiplier=None)  # None 是缺失值；tick_size=None 才表示不限制

    def test_text_fields_must_be_str(self):
        with pytest.raises(TypeError, match="name 必须是 str"):
            aapl(name=1)
        with pytest.raises(TypeError, match="asset_class 必须是 str"):
            aapl(asset_class=None)


# ===========================================================================
# 2. 不可变与值语义
# ===========================================================================


class TestValueSemantics:
    def test_immutable(self):
        inst = aapl()
        for attr in ("_symbol", "symbol", "new_attr"):
            with pytest.raises(AttributeError, match="禁止设置属性"):
                setattr(inst, attr, "MSFT")
        with pytest.raises(AttributeError, match="禁止删除属性"):
            del inst._symbol
        assert inst.symbol == "AAPL"

    def test_equality_uses_symbol_and_currency_only(self):
        a = Instrument("aapl", "USD", name="Apple Inc.", multiplier=1)
        b = Instrument("AAPL", USD, name="Apple", multiplier=100, tick_size=Decimal("0.01"))
        assert a == b and hash(a) == hash(b) and len({a, b}) == 1
        assert {a: 1}[b] == 1  # 可作为持仓字典的键
        assert a != Instrument("AAPL", "CNY") and a != Instrument("MSFT", "USD")
        assert a != "AAPL" and (a == "AAPL") is False

    def test_copy_and_pickle(self):
        inst = Instrument("CL2412", "USD", base_unit="barrel", multiplier=1000, tick_size=Decimal("0.01"),
                          name="Crude", asset_class="Commodity")
        assert copy.copy(inst) is inst and copy.deepcopy(inst) is inst
        restored = pickle.loads(pickle.dumps(inst))
        assert restored == inst and repr(restored) == repr(inst)
        assert restored.multiplier.scale == 0 and restored.tick_size == F("0.01")

    def test_repr_and_str(self):
        inst = Instrument("CL2412", "USD", base_unit="barrel", multiplier=1000, tick_size=Decimal("0.01"), name="Crude")
        assert repr(inst) == ("Instrument(symbol='CL2412', quote_currency='USD', base_unit='barrel', "
                              "multiplier='1000', tick_size='0.01', name='Crude', asset_class='Equity')")
        assert "tick_size=None" in repr(aapl())
        assert str(inst) == "CL2412 (Crude, USD/barrel)"


# ===========================================================================
# 3. 按标的构造数量、价格与名义金额
# ===========================================================================


class TestQuantityAndPrice:
    def test_quantity(self):
        q = Instrument("CL", "USD", base_unit="barrel").as_quantity(["1.5", 2])
        assert isinstance(q, Quantity) and q.unit == "barrel" and q.mantissa.tolist() == [15, 20]
        assert aapl().as_quantity(Decimal("3.20"), scale=1).mantissa == 32
        with pytest.warns(PrecisionLossWarning):  # 精确输入丢精度：与 Quantity 构造一致
            assert aapl().as_quantity(Decimal("3.25"), scale=1).mantissa == 33

    def test_price_from_numbers(self):
        p = Instrument("CL", "CNY", base_unit="barrel", tick_size=Decimal("0.5")).as_price("80.25")
        assert isinstance(p, Price) and p.quote_currency is CNY and p.base_unit == "barrel"
        assert p.tick_size == F("0.5") and p.round_to_step() == Price("80.5", "CNY", base_unit="barrel")
        assert aapl().as_price(189.505, scale=2).mantissa == 18951  # float 按最短写法 half_up

    def test_price_passthrough_and_checks(self):
        inst = aapl()
        given = Price("189.5", "USD")
        assert inst.as_price(given) is given
        assert inst.as_price(Price("1.5", "USD", base_unit="Any")).base_unit == "Any"  # 通配单位兼容
        assert inst.as_price(given, scale=3).mantissa == 189500
        with pytest.raises(TypeError, match="计价货币"):
            inst.as_price(Price("189.5", "CNY"))
        with pytest.raises(TypeError, match="价格单位"):
            inst.as_price(Price("189.5", "USD", base_unit="barrel"))


class TestNotional:
    def test_with_multiplier(self):
        cl = Instrument("CL2412", "USD", base_unit="contract", multiplier=1000)
        got = cl.calc_notional(Price("80.25", "USD", base_unit="contract"), Quantity(3, unit="contract"))
        assert isinstance(got, Money) and got == Money("240750", "USD")
        assert cl.calc_notional("80.25", 3) == Money("240750", "USD")  # 裸数字按计价货币与基准单位解释

    def test_rounds_once(self):
        # 0.333 × 3 × 0.5 = 0.4995：一次舍入得 0.50；若先把 0.999 × 0.5 舍入两次会差 1 分
        inst = aapl(multiplier=Decimal("0.5"))
        assert inst.calc_notional(Decimal("0.333"), 3) == Money("0.50", "USD")
        assert inst.calc_notional(Decimal("0.333"), 3, rounding="down") == Money("0.49", "USD")
        assert Instrument("X", "JPY", multiplier=Decimal("0.1")).calc_notional("12.34", 5) == Money("6", "JPY")

    def test_arrays_and_missing_values(self):
        got = aapl(multiplier=100).calc_notional(Price([None, "1.5"], "USD"), Quantity([2, 3]))
        assert got.isna().tolist() == [True, False] and got.units.tolist() == [0, 45000]

    def test_unit_and_currency_checks(self):
        inst = Instrument("CL", "USD", base_unit="barrel")
        assert inst.calc_notional(Price("2", "USD", base_unit="barrel"), Quantity(3, unit="Any")) == Money("6", "USD")
        with pytest.raises(TypeError, match="数量单位"):
            inst.calc_notional("2", Quantity(3, unit="share"))
        with pytest.raises(TypeError, match="计价货币"):
            inst.calc_notional(Price("2", "CNY", base_unit="barrel"), 3)
        assert inst.calc_notional("2", "3") == Money("6", "USD")  # 与构造函数一致：文本数值可以构造
        with pytest.raises(TypeError):
            inst.calc_notional(Money("2", "USD"), 3)  # 带单位的对象不能当作价格数值
