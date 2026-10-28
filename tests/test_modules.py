"""测试模块拆分后的导入完整性与向后兼容性（conventions、formatter、tvm 的功能测试见各自的测试文件）。"""

import pytest


def test_moneta_exports():
    import moneta

    # 核心
    assert hasattr(moneta, "FixedArray")
    assert hasattr(moneta, "UnitArray")
    # 币种
    assert hasattr(moneta, "Currency")
    assert hasattr(moneta, "CurrencyRegistry")
    assert hasattr(moneta, "get_currency")
    assert hasattr(moneta, "USD")
    assert hasattr(moneta, "CNY")
    # 资金与标价
    assert hasattr(moneta, "Money")
    assert hasattr(moneta, "Price")
    assert hasattr(moneta, "Quantity")
    # 批次与资产组合
    assert hasattr(moneta, "PositionLot")
    assert hasattr(moneta, "Portfolio")
    assert hasattr(moneta, "Instrument")
    assert hasattr(moneta, "LotMatchingRule")
    assert hasattr(moneta, "match_lots")
    # 现金流与计日
    assert hasattr(moneta, "CashFlow")
    assert hasattr(moneta, "DayCountConvention")
    assert hasattr(moneta, "generate_cash_flow_schedule")
    # 清算与分配
    assert hasattr(moneta.Money, "allocate")
    assert hasattr(moneta.Money, "round_to_step")
    assert hasattr(moneta.Money, "is_on_step")
    # 时间价值
    assert hasattr(moneta, "pv")
    assert hasattr(moneta, "fv")
    assert hasattr(moneta, "npv")
    assert hasattr(moneta, "discount_factors")
    # 格式化
    assert hasattr(moneta, "MonetaryFormatter")
    # 计日与金融惯例
    assert hasattr(moneta, "DayCountConvention")
    assert hasattr(moneta, "year_fraction")
    # 外汇
    assert hasattr(moneta, "ExchangeRate")
    assert hasattr(moneta, "ExchangeRateNotFoundError")
    assert hasattr(moneta, "FXProvider")
    assert hasattr(moneta, "FXEngine")


def test_submodule_direct_imports():
    from moneta.conventions import DayCountConvention, year_fraction
    from moneta.cashflow import CashFlow, generate_cash_flow_schedule
    from moneta.formatter import MonetaryFormatter
    from moneta.lot import LotMatchingRule, PositionLot
    from moneta.money import Money
    from moneta.portfolio import Portfolio
    from moneta.price import Price
    from moneta.quantity import Quantity
    from moneta.instrument import Instrument
    from moneta.tvm import discount_factors, fv, npv, pv
    from moneta.unit_array import UnitArray
    from moneta.fx import ExchangeRate, ExchangeRateNotFoundError, FXEngine

    assert DayCountConvention.ACT_365.value == "ACT/365"
    assert LotMatchingRule.FIFO.value == "FIFO"
    assert hasattr(Money, "calc_tiered_fee")
    assert hasattr(Quantity, "data")
    assert callable(year_fraction)
    assert issubclass(ExchangeRateNotFoundError, KeyError)
    assert issubclass(Money, UnitArray)
    assert issubclass(Price, UnitArray)
    assert issubclass(Quantity, UnitArray)


def test_conversion_apis():
    from moneta import FixedArray, Money, Price, Quantity, UnitArray

    for cls in (FixedArray, Money, Price, Quantity, UnitArray):
        assert hasattr(cls, "to_float")
        assert hasattr(cls, "to_decimal")
        assert hasattr(cls, "to_pandas")
        assert hasattr(cls, "to_polars")


def test_money_semantics_and_float_removal():
    from moneta import Money

    # 1. Money 必须彻底删除 __float__，防止隐式 float 污染
    assert not hasattr(Money, "__float__")

    # 2. 禁止两笔金额相乘 (Money * Money)
    m = Money("100.00", "USD")
    with pytest.raises(TypeError, match="禁止两笔金额直接相乘"):
        _ = m * m


def test_price_quantity_unit_check():
    from moneta import Price, Quantity

    p = Price("189.50", "USD", base_unit="share")
    q_mismatch = Quantity(10, unit="barrel")

    with pytest.raises(TypeError, match="标的单位不匹配"):
        _ = p * q_mismatch

    # 正常单位相乘应得到 Money
    q_match = Quantity(10, unit="share")
    m = p * q_match
    assert str(m) == "1895.00 USD"


def test_instrument_and_portfolio_contract():
    from moneta import Instrument, Portfolio, Quantity

    # 1. 验证 Instrument 纯不可变参考数据构造与属性
    inst = Instrument("aapl", quote_currency="USD", base_unit="share", name="Apple Inc.", asset_class="Equity")
    assert inst.symbol == "AAPL"
    assert inst.name == "Apple Inc."
    assert inst.asset_class == "Equity"
    assert inst.base_unit == "share"
    assert inst.quote_currency.code == "USD"
    assert "AAPL" in str(inst)

    # 2. 验证 Portfolio 不支持直接减法 (强制通过 withdraw 满足 FIFO 会计批次)
    p1 = Portfolio("USD")
    p2 = Portfolio("USD")
    with pytest.raises(TypeError, match="Portfolio 不支持相减"):
        _ = p1 - p2


def test_unit_array_inheritance_and_no_stubs():
    from decimal import Decimal
    from moneta import Money, Price, Quantity

    # Money 继承自 UnitArray 的切片、比较与转换
    m = Money("100.50", "USD")
    assert m == Money("100.50", "USD")
    assert not (m == Money("100.50", "CNY"))
    assert m.to_float() == 100.50
    assert m.to_decimal() == Decimal("100.50")
    assert m != "other"

    # Quantity 继承自 UnitArray
    q1 = Quantity(100)
    q2 = Quantity(50)
    assert q1 > q2
    assert q1 >= q2
    assert q2 < q1
    assert q1 != q2

    # Price 继承自 UnitArray
    p = Price("189.50", "USD")
    assert p.to_float() == 189.50


def test_default_units_and_wildcard_alignment():
    from moneta import Price, Quantity, Money

    # 默认单位完全一致且自动匹配 (Price.base_unit='share', Quantity.unit='share')
    p = Price("189.50", "USD")
    q = Quantity(100)
    m = p * q
    assert isinstance(m, Money)
    assert str(m) == "18950.00 USD"

    # 通配单位 Any 与具体单位相加继承具体单位
    q_any = Quantity(50, unit="Any")
    q_share = Quantity(100, unit="share")
    q_res = q_any + q_share
    assert q_res.unit == "share"
    assert q_res == Quantity(150, unit="share")


def test_money_advanced_operations():
    from decimal import Decimal
    from moneta import Money, Price, Quantity, FixedArray

    # 1. Money / Price -> Quantity
    m = Money("1895.00", "USD")
    p = Price("189.50", "USD", base_unit="share")
    q = m / p
    assert isinstance(q, Quantity)
    assert q.unit == "share"
    assert q == Quantity(10, unit="share")

    # 2. Money / Quantity -> Price
    p_derived = m / Quantity(10, unit="share")
    assert isinstance(p_derived, Price)
    assert p_derived.quote_currency.code == "USD"
    assert p_derived.base_unit == "share"
    assert p_derived == Price("189.50", "USD")

    # 3. allocate (Hamilton / Largest Remainder Method)
    total = Money("100.00", "USD")
    parts = total.allocate([1, 1, 1])
    assert len(parts) == 3
    assert sum([part.data for part in parts], FixedArray(0, scale=2)) == total.data
    assert parts[0] == Money("33.34", "USD")
    assert parts[1] == Money("33.33", "USD")
    assert parts[2] == Money("33.33", "USD")

    # 4. calc_tiered_fee 阶梯费率
    tiers = [
        (Decimal("1000.00"), Decimal("0.01")),   # 前 1000 费率 1%
        (Decimal("5000.00"), Decimal("0.005")),  # 1000~5000 费率 0.5%
        (None, Decimal("0.001")),                # 5000 以上费率 0.1%
    ]
    fee = Money("6000.00", "USD").calc_tiered_fee(tiers)
    # 1000 * 0.01 = 10, 4000 * 0.005 = 20, 1000 * 0.001 = 1 -> 总计 31.00 USD
    assert fee == Money("31.00", "USD")

    # 5. sum, mean, __matmul__, __divmod__
    m_arr = Money(["10.00", "20.00", "30.00"], "USD")
    assert m_arr.sum() == Money("60.00", "USD")
    assert m_arr.mean() == Money("20.00", "USD")

    weights = FixedArray(["0.2", "0.3", "0.5"])
    assert m_arr @ weights == Money("23.00", "USD")

    q_div, r_div = divmod(Money("10.00", "USD"), Money("3.00", "USD"))
    assert q_div == 3
    assert r_div == Money("1.00", "USD")
