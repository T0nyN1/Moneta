"""Currency 与 CurrencyRegistry 的行为约定。

已确认的设计：
- Currency 是不可变值对象：构造时校验代码（3 到 5 个字母，去空白、转大写）与 digits（0 到 18 的整数，
  不接受 bool、float）；之后禁止修改任何属性。
- 相等比较代码与 digits；与非 Currency 比较返回 NotImplemented（因此 USD == "USD" 为 False）。
- 注册表是唯一事实来源：重复注册完全相同的定义无副作用；同一代码的任何字段不同即报 ValueError，
  不提供覆盖开关，内置币种永远不可修改。get_currency 遇到与注册表冲突的实例同样报错。
- copy / deepcopy 返回自身；pickle 还原时回到注册表中的同一单例，未注册的还原为独立实例。

组织：构造校验、不可变、相等与哈希、单例语义、注册表、内置币种、与 Money 的集成。
"""

import copy
import pickle

import numpy as np
import pytest

from moneta import Money
from moneta.currency import (
    AUD,
    CAD,
    CHF,
    CNY,
    EUR,
    GBP,
    HKD,
    JPY,
    SGD,
    USD,
    Currency,
    CurrencyRegistry,
    get_currency,
)


@pytest.fixture(autouse=True)
def isolated_registry():
    """每个测试结束后恢复注册表，避免自定义币种在测试之间泄漏。"""
    yield
    CurrencyRegistry.clear()


# ===========================================================================
# 1. 构造校验
# ===========================================================================


class TestConstruct:
    def test_normalizes_code_and_defaults(self):
        c = Currency(" usdt ", 6)
        assert (c.code, c.digits, c.symbol, c.name) == ("USDT", 6, "", "USDT")
        assert Currency("BHD", 3, "BD", "Bahraini Dinar").name == "Bahraini Dinar"

    @pytest.mark.parametrize("digits", [0, 18, np.int64(3), np.uint8(2)])
    def test_accepts_integer_digits(self, digits):
        c = Currency("ABC", digits)
        assert c.digits == int(digits) and type(c.digits) is int

    @pytest.mark.parametrize("code", ["", "US", "ABCDEF", "US1", "U-S", "美元", "  ", "ＵＳＤ", "ußd", "usdı", "ǆab"])
    def test_rejects_invalid_code(self, code):
        with pytest.raises(ValueError, match="3 到 5 个英文字母"):
            Currency(code, 2)

    @pytest.mark.parametrize("code", [None, 840, b"USD", USD])
    def test_rejects_non_str_code(self, code):
        with pytest.raises(TypeError, match="货币代码必须是 str"):
            Currency(code, 2)

    @pytest.mark.parametrize("digits", [-1, 19, 100])
    def test_rejects_out_of_range_digits(self, digits):
        with pytest.raises(ValueError, match="digits 必须在 0 到 18 之间"):
            Currency("ABC", digits)

    @pytest.mark.parametrize("digits", [True, np.True_, 2.0, 2.9, "2", None])
    def test_rejects_non_integer_digits(self, digits):
        with pytest.raises(TypeError):
            Currency("ABC", digits)

    @pytest.mark.parametrize("field", ["symbol", "name"])
    @pytest.mark.parametrize("value", [None, 1, b"$"])
    def test_rejects_non_str_symbol_and_name(self, field, value):
        with pytest.raises(TypeError, match=f"{field} 必须是 str"):
            Currency("ABC", 2, **{field: value})


# ===========================================================================
# 2. 不可变
# ===========================================================================


class TestImmutable:
    @pytest.mark.parametrize("attr", ["_code", "_digits", "_symbol", "_name", "code", "digits", "extra"])
    def test_cannot_set(self, attr):
        with pytest.raises(AttributeError, match="禁止设置属性"):
            setattr(USD, attr, 5)
        assert USD.digits == 2 and USD.code == "USD"

    def test_cannot_delete(self):
        with pytest.raises(AttributeError, match="禁止删除属性 '_digits'"):
            del USD._digits
        assert USD.digits == 2


# ===========================================================================
# 3. 相等与哈希
# ===========================================================================


class TestEquality:
    def test_code_and_digits_define_equality(self):
        assert Currency("ABC", 2) == Currency("abc", 2)
        assert Currency("ABC", 2, "A$", "Alpha") == Currency("ABC", 2)  # 展示字段不影响相等
        assert Currency("ABC", 2) != Currency("ABC", 3)
        assert Currency("ABC", 2) != Currency("ABD", 2)

    def test_non_currency_is_not_equal(self):
        assert (USD == "USD") is False and (USD != "USD") is True
        assert USD.__eq__("USD") is NotImplemented
        assert USD != 2 and USD is not None

    def test_hash_consistent_with_equality(self):
        assert hash(Currency("ABC", 2)) == hash(Currency("ABC", 2, "A$"))
        assert len({USD, get_currency("usd"), Currency("USD", 2, "$", "US Dollar"), CNY}) == 2
        assert {USD: 1}[get_currency("USD")] == 1

    def test_display(self):
        assert str(USD) == "USD"
        assert repr(USD) == "Currency(code='USD', digits=2, symbol='$', name='US Dollar')"


# ===========================================================================
# 4. 单例语义：copy / deepcopy / pickle
# ===========================================================================


class TestSingleton:
    def test_copy_and_deepcopy_return_self(self):
        assert copy.copy(USD) is USD
        assert copy.deepcopy(USD) is USD
        assert copy.deepcopy({"ccy": USD, "list": [CNY]})["list"][0] is CNY

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_registered_returns_singleton(self, protocol):
        assert pickle.loads(pickle.dumps(USD, protocol=protocol)) is USD
        assert pickle.loads(pickle.dumps([JPY, JPY], protocol=protocol))[1] is JPY

    def test_pickle_unregistered_returns_independent_instance(self):
        custom = Currency("XYZ", 4, "X", "Xyz")
        restored = pickle.loads(pickle.dumps(custom))
        assert restored == custom and restored is not custom
        assert (restored.symbol, restored.name) == ("X", "Xyz")
        assert not CurrencyRegistry.has("XYZ")  # 还原不会自动注册

    def test_pickle_restores_registered_custom_singleton(self):
        custom = Currency("XYZ", 4)
        CurrencyRegistry.register(custom)
        assert pickle.loads(pickle.dumps(Currency("XYZ", 4))) is custom

    def test_pickle_conflicting_with_registry_raises(self):
        data = pickle.dumps(Currency("XYZ", 4))
        CurrencyRegistry.register(Currency("XYZ", 2))
        with pytest.raises(ValueError, match="不一致"):
            pickle.loads(data)


# ===========================================================================
# 5. 注册表
# ===========================================================================


class TestRegistry:
    def test_get_by_code_normalizes(self):
        assert get_currency("usd") is USD
        assert get_currency(" eur ") is EUR
        assert CurrencyRegistry.get("JPY") is JPY

    def test_get_unknown_code(self):
        with pytest.raises(KeyError, match="未注册的货币代码"):
            get_currency("XYZ")

    @pytest.mark.parametrize("bad", [None, 840, b"USD", ["USD"]])
    def test_get_rejects_non_str(self, bad):
        with pytest.raises(TypeError, match="必须是 str 或 Currency"):
            get_currency(bad)

    def test_get_currency_instance(self):
        assert get_currency(USD) is USD
        assert get_currency(Currency("USD", 2, "$", "US Dollar")) is USD  # 等价实例 → 注册表单例
        custom = Currency("XYZ", 3)
        assert get_currency(custom) is custom  # 未注册：原样返回

    @pytest.mark.parametrize("conflict", [Currency("USD", 3, "$", "US Dollar"), Currency("USD", 2, "US$", "US Dollar"),
                                          Currency("USD", 2, "$", "Dollar")])
    def test_get_conflicting_instance_raises(self, conflict):
        with pytest.raises(ValueError, match="不一致"):
            get_currency(conflict)

    def test_register_new_and_idempotent(self):
        custom = Currency("XYZ", 3, "X", "Xyz")
        CurrencyRegistry.register(custom)
        CurrencyRegistry.register(custom)
        CurrencyRegistry.register(Currency("xyz", 3, "X", "Xyz"))  # 完全相同的定义：无副作用
        assert get_currency("XYZ") is custom

    @pytest.mark.parametrize("conflict", [Currency("XYZ", 4, "X", "Xyz"), Currency("XYZ", 3, "Y", "Xyz"),
                                          Currency("XYZ", 3, "X", "Other")])
    def test_register_conflict_raises(self, conflict):
        CurrencyRegistry.register(Currency("XYZ", 3, "X", "Xyz"))
        with pytest.raises(ValueError, match="已注册为"):
            CurrencyRegistry.register(conflict)
        assert get_currency("XYZ").digits == 3

    def test_builtin_cannot_be_overridden(self):
        with pytest.raises(ValueError, match="已注册为"):
            CurrencyRegistry.register(Currency("USD", 4, "$", "US Dollar"))
        assert get_currency("USD") is USD and USD.digits == 2

    @pytest.mark.parametrize("bad", ["USD", None, 1])
    def test_register_rejects_non_currency(self, bad):
        with pytest.raises(TypeError, match="只能注册 Currency 实例"):
            CurrencyRegistry.register(bad)

    def test_has(self):
        assert CurrencyRegistry.has("usd") and CurrencyRegistry.has(USD)
        assert not CurrencyRegistry.has("XYZ")
        assert not CurrencyRegistry.has(Currency("USD", 3))  # 与注册定义不一致
        assert not CurrencyRegistry.has(Currency("XYZ", 2))
        assert not CurrencyRegistry.has(None) and not CurrencyRegistry.has(840)

    def test_clear_restores_defaults_only(self):
        CurrencyRegistry.register(Currency("XYZ", 3))
        CurrencyRegistry.clear()
        assert not CurrencyRegistry.has("XYZ")
        assert get_currency("USD") is USD


# ===========================================================================
# 6. 内置币种
# ===========================================================================


class TestBuiltins:
    @pytest.mark.parametrize("ccy, code, digits", [
        (USD, "USD", 2), (CNY, "CNY", 2), (EUR, "EUR", 2), (JPY, "JPY", 0), (GBP, "GBP", 2),
        (HKD, "HKD", 2), (AUD, "AUD", 2), (CAD, "CAD", 2), (CHF, "CHF", 2), (SGD, "SGD", 2),
    ])
    def test_iso_4217_digits_and_registry_identity(self, ccy, code, digits):
        assert (ccy.code, ccy.digits) == (code, digits)
        assert get_currency(code) is ccy
        assert ccy.symbol and ccy.name != code

    def test_exports(self):
        import moneta
        for name in ("Currency", "CurrencyRegistry", "get_currency", "USD", "CNY", "JPY"):
            assert getattr(moneta, name) is globals()[name]


# ===========================================================================
# 7. 与 Money 的集成
# ===========================================================================


class TestMoneyIntegration:
    def test_money_resolves_to_registry_singleton(self):
        assert Money("1", "usd").currency is USD
        assert Money("1", Currency("USD", 2, "$", "US Dollar")).currency is USD

    def test_money_rejects_conflicting_currency(self):
        # 不会再出现 3 位小数的 USD
        with pytest.raises(ValueError, match="不一致"):
            Money(1, Currency("USD", 3))

    def test_unregistered_custom_currency(self):
        xyz = Currency("XYZ", 4)
        m = Money("1.2345", xyz)
        assert m.currency is xyz and m.scale == 4
        assert (m + Money("1", Currency("XYZ", 4))).currency == xyz

    def test_same_code_different_digits_never_mix(self):
        with pytest.raises(TypeError, match="币种不匹配"):
            Money(1, Currency("XYZ", 2)) + Money(1, Currency("XYZ", 3))
