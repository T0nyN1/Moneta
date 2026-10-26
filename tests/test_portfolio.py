"""Portfolio 测试：以资产为键的持仓、入账与批次结转、估值、再平衡。

约定（portfolio.py）：
- 现金以 Currency、证券以 Instrument 为键；持仓数量由批次汇总，批次是唯一事实来源。
- 基准货币必填；任何换汇都必须显式给出 fx（ExchangeRate 或 FXEngine），没有静默快照或 1:1 换算。
- 证券估值必须给出市价，裸数字按标的计价货币解释并乘以合约乘数。
- withdraw 返回 (平掉的批次, 已实现盈亏)，盈亏以基准货币计。
"""

from datetime import date
from decimal import Decimal

import pytest

from moneta import Instrument, Money, Portfolio, Price, Quantity
from moneta.currency import CNY, EUR, USD
from moneta.fixed_array import FixedArray as F
from moneta.fx import ExchangeRate, FXEngine, UserDictFXProvider
from moneta.lot import LotMatchingRule, PositionLot

AAPL = Instrument("AAPL", "USD")
CL = Instrument("CL", "USD", base_unit="barrel", multiplier=1000)
TENCENT = Instrument("0700", "HKD")
USDCNY_7 = ExchangeRate("USD", "CNY", "7.0")
# 不带日期的汇率用于不传日期的估值；入账 / 出账按交易日严格匹配，需要当天的汇率
ENGINE = FXEngine(UserDictFXProvider({
    "USD/CNY": "7.2", "EUR/USD": "1.1",
    **{day: {"USD/CNY": "7.2", "EUR/USD": "1.1"} for day in ("2024-01-02", "2024-02-01", "2024-03-01")},
}))


def cny(v):
    return Money(v, "CNY")


def usd(v):
    return Money(v, "USD")


def sample():
    """CNY 组合：1000 CNY 现金、100 股 AAPL（100 USD，汇率 7.0）、50 USD 现金（汇率 7.1）。"""
    p = Portfolio("CNY")
    p.deposit("CNY", 1000, "2024-01-01")
    p.deposit(AAPL, 100, "2024-01-02", price="100", fx=USDCNY_7)
    p.deposit(USD, usd(50), "2024-01-03", fx=ExchangeRate("USD", "CNY", "7.1"))
    return p


# ===========================================================================
# 1. 构造与查询
# ===========================================================================


class TestQueries:
    def test_base_currency_required(self):
        assert Portfolio("cny").base_currency is CNY and Portfolio(USD).base_currency is USD
        with pytest.raises(TypeError):
            Portfolio()
        with pytest.raises(KeyError):
            Portfolio("XYZ")

    def test_assets_and_positions(self):
        p = sample()
        assert p.assets == [CNY, AAPL, USD] and p.currencies == [CNY, USD] and p.instruments == [AAPL] and len(p) == 3
        assert p.position(CNY) == cny(1000) and p.position(AAPL) == Quantity(100) and p.position(USD) == usd(50)
        assert p["cny"] == cny(1000) and p["aapl"] == Quantity(100) and p[Instrument("aapl", "USD")] == Quantity(100)
        assert list(p) == [(CNY, cny(1000)), (AAPL, Quantity(100)), (USD, usd(50))]

    def test_missing_assets(self):
        p = sample()
        assert p.position(EUR) == Money(0, "EUR") and p["EUR"] == Money(0, "EUR")  # 已注册币种：持仓为 0
        assert p.position(CL) == Quantity(0, unit="barrel")  # 用标的自己的单位，而不是 share
        with pytest.raises(KeyError, match="没有代码为 'MSFT'"):
            p["MSFT"]
        assert p.get("MSFT", "none") == "none" and p.get(EUR) is None and p.get(AAPL) == Quantity(100)
        assert "AAPL" in p and AAPL in p and "msft" not in p and EUR not in p and 1 not in p
        with pytest.raises(TypeError, match="资产必须是"):
            p.position(1)

    def test_symbol_and_currency_codes_do_not_collide(self):
        p = Portfolio("USD")
        cad_stock = Instrument("CAD", "USD")
        p.deposit(cad_stock, 3, "2024-01-02", price=10)
        assert p["CAD"] == Quantity(3) and p.position(Instrument("CAD", "USD")) == Quantity(3)
        with pytest.raises(ValueError, match="price 或 cost"):
            p.deposit("CAD", 5, "2024-01-02")  # 字符串先匹配已持有的资产：这里是股票 CAD，不会悄悄当成现金
        from moneta.currency import CAD
        p.deposit(CAD, 5, "2024-01-02", fx=ExchangeRate("CAD", "USD", "0.75"))
        with pytest.raises(KeyError, match="多个持仓"):
            p["CAD"]  # 同一代码既是现金又是标的：必须用对象区分
        assert p.position(CAD) == Money(5, "CAD") and p.position(cad_stock) == Quantity(3)

    def test_same_symbol_in_two_currencies(self):
        p = Portfolio("USD")
        a_usd, a_cny = Instrument("A", "USD"), Instrument("A", "CNY")
        p.deposit(a_usd, 1, "2024-01-02", price=1)
        p.deposit(a_cny, 2, "2024-01-02", price=1, fx=ExchangeRate("CNY", "USD", "0.14"))
        assert len(p) == 2 and p.position(a_usd) == Quantity(1) and p.position(a_cny) == Quantity(2)

    def test_lots_are_copies(self):
        p = sample()
        p.lots(AAPL).clear()
        p.lots().clear()
        assert len(p.lots(AAPL)) == 1 and len(p.lots()) == 3 and p.lots(EUR) == []


# ===========================================================================
# 2. 入账
# ===========================================================================


class TestDeposit:
    def test_cash(self):
        p = Portfolio("CNY")
        lot = p.deposit(CNY, cny("100.50"), date(2024, 1, 1), label="init")
        assert lot.cost_base == cny("100.50") and lot.label == "init" and p.lots(CNY) == [lot]
        eur = p.deposit("EUR", 100, "2024-01-02", fx=ExchangeRate("EUR", "CNY", "7.8"))
        assert eur.cost_base == cny(780) and p["EUR"] == Money(100, "EUR")
        assert p.deposit("EUR", 10, "2024-01-03", cost_base=cny("79")).cost_base == cny(79)  # 直接给基准货币成本
        assert p["EUR"] == Money(110, "EUR") and len(p.lots(EUR)) == 2

    def test_security(self):
        p = Portfolio("CNY")
        lot = p.deposit(AAPL, Quantity(10), "2024-01-02", price=Price("189.5", "USD"), fx=USDCNY_7)
        assert (lot.cost, lot.cost_base) == (usd(1895), cny(13265))
        fut = p.deposit(CL, 2, "2024-01-02", price="80.25", fx=ENGINE)  # FXEngine 按入账日取汇率
        assert (fut.cost, fut.cost_base) == (usd(160500), cny(1155600))  # 含合约乘数
        fee = p.deposit(AAPL, 1, "2024-01-03", cost=usd("190.99"), cost_base=cny("1375"))  # 成本含手续费
        assert fee.cost == usd("190.99") and p[AAPL] == Quantity(11)
        local = Portfolio("USD")
        assert local.deposit(AAPL, 1, "2024-01-02", price=100).cost_base == usd(100)  # 同币种：不需要 fx

    def test_requires_explicit_fx(self):
        p = Portfolio("CNY")
        with pytest.raises(ValueError, match="需要 USD/CNY 的汇率"):
            p.deposit(AAPL, 1, "2024-01-02", price=100)
        with pytest.raises(ValueError, match="需要 EUR/CNY 的汇率"):
            p.deposit(EUR, 1, "2024-01-02")
        with pytest.raises(TypeError, match="不匹配"):
            p.deposit(AAPL, 1, "2024-01-02", price=100, fx=ExchangeRate("EUR", "USD", "1.1"))
        assert len(p) == 0  # 失败时不会留下半截状态

    def test_validation(self):
        p = Portfolio("USD")
        with pytest.raises(ValueError, match="不接受 price / cost"):
            p.deposit(USD, 1, "2024-01-01", price=1)
        with pytest.raises(ValueError, match="price 或 cost 之一"):
            p.deposit(AAPL, 1, "2024-01-01")
        with pytest.raises(ValueError, match="price 或 cost 之一"):
            p.deposit(AAPL, 1, "2024-01-01", price=1, cost=usd(1))
        with pytest.raises(TypeError, match="单位"):
            p.deposit(CL, Quantity(1), "2024-01-01", price=1)  # share ≠ barrel
        with pytest.raises(ValueError, match="必须为正数"):
            p.deposit(USD, 0, "2024-01-01")
        with pytest.raises(ValueError, match="acquired_date"):
            p.deposit(USD, 1, "2024-02-30")
        with pytest.raises(KeyError):
            p.deposit("MSFT", 1, "2024-01-01", price=1)  # 未持有的标的必须传 Instrument
        assert len(p) == 0

    def test_add_lot(self):
        p = Portfolio("CNY")
        p.add_lot(PositionLot(AAPL, Quantity(1), "2024-01-02", cost=usd(100), cost_base=cny(700)))
        assert p[AAPL] == Quantity(1)
        with pytest.raises(ValueError, match="基准货币"):
            p.add_lot(PositionLot(AAPL, Quantity(1), "2024-01-02", cost=usd(100)))  # cost_base 是 USD
        with pytest.raises(TypeError, match="PositionLot"):
            p.add_lot("lot")


# ===========================================================================
# 3. 出账与已实现盈亏
# ===========================================================================


class TestWithdraw:
    def portfolio(self):
        p = Portfolio("CNY")
        p.deposit(AAPL, 10, "2024-01-01", price=100, fx=USDCNY_7)
        p.deposit(AAPL, 10, "2024-02-01", price=120, fx=USDCNY_7)
        return p

    @pytest.mark.parametrize("rule, cost_base, left", [
        (LotMatchingRule.FIFO, 7000 + 4200, [(5, 4200)]),
        (LotMatchingRule.LIFO, 8400 + 3500, [(5, 3500)]),
        (LotMatchingRule.WAC, 5600 + 5880, [(2, 1400), (3, 2520)]),  # 整股按 10:10 分 15 股：8 + 7
    ])
    def test_rules_and_realized_pnl(self, rule, cost_base, left):
        p = self.portfolio()
        closed, realized = p.withdraw(AAPL, 15, "2024-03-01", price=130, fx=ExchangeRate("USD", "CNY", "7.2"), rule=rule)
        assert sum((lot.cost_base for lot in closed), cny(0)) == cny(cost_base)
        assert realized == cny(15 * 130 * Decimal("7.2")) - cny(cost_base)
        assert [(lot.amount, lot.cost_base) for lot in p.lots(AAPL)] == [(Quantity(q), cny(c)) for q, c in left]
        assert p[AAPL] == Quantity(5)

    def test_without_price_is_a_transfer(self):
        p = self.portfolio()
        closed, realized = p.withdraw("AAPL", Quantity(5), "2024-03-01")
        assert realized is None and closed[0].amount == Quantity(5)
        with pytest.raises(ValueError, match="需要 USD/CNY 的汇率"):
            p.withdraw(AAPL, 1, "2024-03-01", price=130)  # 给了成交价却缺汇率：盈亏算不完整，报错

    def test_full_withdraw_removes_asset(self):
        p = sample()
        closed, realized = p.withdraw(USD, usd(50), "2024-02-01", fx=ENGINE)
        assert realized == cny(5) and USD not in p and p.position(USD) == usd(0) and p.currencies == [CNY]
        _, realized = p.withdraw(CNY, 1000, "2024-02-01")
        assert realized == cny(0) and p.assets == [AAPL]
        p.withdraw(AAPL, 100, "2024-02-01")
        assert len(p) == 0 and p.lots() == []

    def test_foreign_cash_without_fx_has_no_pnl(self):
        p = sample()
        closed, realized = p.withdraw("USD", 20, "2024-02-01")
        assert realized is None and p["USD"] == usd(30) and closed[0].cost_base == cny(142)

    def test_errors(self):
        p = self.portfolio()
        with pytest.raises(ValueError, match="超过批次总量"):
            p.withdraw(AAPL, 21, "2024-03-01")
        with pytest.raises(ValueError, match="没有持有"):
            p.withdraw(CL, 1, "2024-03-01")
        with pytest.raises(ValueError, match="trade_date"):
            p.withdraw(AAPL, 1, "2024-13-01")
        with pytest.raises(TypeError, match="LotMatchingRule"):
            p.withdraw(AAPL, 1, "2024-03-01", rule="FIFO")
        assert p[AAPL] == Quantity(20)  # 失败时持仓不变


# ===========================================================================
# 4. 估值
# ===========================================================================


class TestValuation:
    def test_total_value_cost_and_pnl(self):
        p = sample()
        prices = {"AAPL": 120}
        assert p.total_value(prices, ENGINE) == cny(1000 + 100 * 120 * Decimal("7.2") + 50 * Decimal("7.2"))
        assert p.total_cost() == cny(1000 + 70000 + 355)
        assert p.unrealized_pnl({AAPL: Price("120", "USD")}, ENGINE) == cny(16405)
        rate = p.return_rate({"aapl": 120}, ENGINE)
        assert isinstance(rate, F) and rate == F("0.229906804") and rate.scale == 9  # 2 + 8 位后去掉末尾的 0

    def test_multiplier_and_dated_fx(self):
        p = Portfolio("USD")
        p.deposit(CL, 2, "2024-01-02", price=80)
        assert p.total_value({CL: Price("81.5", "USD", base_unit="barrel")}) == usd(163000)
        dated = FXEngine(UserDictFXProvider({"2024-01-02": {"EUR/USD": "1.10"}, "2024-02-01": {"EUR/USD": "1.20"}}))
        p.deposit(EUR, 100, "2024-01-02", fx=dated)
        assert p.lots(EUR)[0].cost_base == usd(110)
        assert p.total_value({"CL": 80}, dated, as_of_date="2024-02-01") == usd(160000 + 120)

    def test_requires_prices_and_fx(self):
        p = sample()
        with pytest.raises(ValueError, match="缺少标的 AAPL 的市价"):
            p.total_value(fx=ENGINE)  # 不会静默用成本价
        with pytest.raises(ValueError, match="需要 USD/CNY 的汇率"):
            p.total_value({"AAPL": 120})  # 不会静默用快照汇率
        with pytest.raises(TypeError, match="计价货币"):
            p.total_value({"AAPL": Price("120", "CNY")}, ENGINE)  # 价格币种必须是标的计价货币

    def test_cash_only_and_empty(self):
        assert Portfolio("USD").total_value() == usd(0) and Portfolio("USD").total_cost() == usd(0)
        p = Portfolio("USD")
        p.deposit(USD, 100, "2024-01-01")
        assert p.total_value() == usd(100) and p.unrealized_pnl() == usd(0) and p.return_rate() == 0
        with pytest.raises(ZeroDivisionError):
            Portfolio("USD").return_rate()


# ===========================================================================
# 5. 再平衡
# ===========================================================================


class TestRebalance:
    def test_adjustments(self):
        p = sample()
        got = p.rebalance({"CNY": Decimal("0.5"), AAPL: Decimal("0.3"), CL: Decimal("0.2")}, {"AAPL": 120, "CL": 80},
                          ENGINE)
        total = 1000 + 86400 + 360  # 87760 CNY
        assert list(got) == [CNY, AAPL, USD, CL]
        assert got[CNY] == cny(total / 2) - cny(1000)
        assert got[USD] == usd(-50)  # 未出现在权重中：目标为 0
        assert got[AAPL] == Quantity("30.47222222222") - Quantity(100)  # 26328 / (7.2 × 120)，只舍入一次（3 + 8 位）
        assert got[CL].unit == "barrel" and got[CL] == Quantity("0.03047222222", unit="barrel")  # 17552 / (7.2 × 80 × 1000)，3 + 8 位

    def test_weights_validation(self):
        p = sample()
        with pytest.raises(ValueError, match="合计必须为 1"):
            p.rebalance({"CNY": Decimal("0.3")}, {"AAPL": 120}, ENGINE)
        with pytest.raises(ValueError, match="非负"):
            p.rebalance({"CNY": Decimal("1.5"), AAPL: Decimal("-0.5")}, {"AAPL": 120}, ENGINE)
        with pytest.raises(ValueError, match="缺失值"):
            p.rebalance({"CNY": None, AAPL: 1}, {"AAPL": 120}, ENGINE)
        with pytest.raises(TypeError, match="只接受数值"):
            p.rebalance({"CNY": "1"}, {"AAPL": 120}, ENGINE)
        with pytest.raises(ValueError, match="重复"):
            p.rebalance({"CNY": Decimal("0.5"), CNY: Decimal("0.5")}, {"AAPL": 120}, ENGINE)
        with pytest.raises(KeyError):
            p.rebalance({"MSFT": 1}, {"AAPL": 120}, ENGINE)

    def test_rate_in_either_direction(self):
        p = Portfolio("CNY")
        p.deposit(CNY, 8000, "2024-01-01")
        for rate in (ExchangeRate("USD", "CNY", "8"), ExchangeRate("CNY", "USD", "0.125")):
            got = p.rebalance({CNY: Decimal("0.5"), USD: Decimal("0.25"), AAPL: Decimal("0.25")}, {AAPL: 50}, rate)
            assert got == {CNY: cny(-4000), USD: usd(250), AAPL: Quantity(5)}  # 2000 CNY = 250 USD = 5 股
        with pytest.raises(TypeError, match="不匹配"):
            p.rebalance({CNY: Decimal("0.5"), AAPL: Decimal("0.5")}, {AAPL: 50}, ExchangeRate("EUR", "USD", "1.1"))

    def test_single_currency(self):
        p = Portfolio("USD")
        p.deposit(USD, 1000, "2024-01-01")
        got = p.rebalance({USD: Decimal("0.25"), AAPL: Decimal("0.75")}, {AAPL: Decimal("150")})
        assert got == {USD: usd(-750), AAPL: Quantity(5)}


# ===========================================================================
# 6. 运算符与展示
# ===========================================================================


class TestOperatorsAndDisplay:
    def test_arithmetic_rejected(self):
        with pytest.raises(TypeError, match="不支持相加"):
            Portfolio("USD") + Portfolio("USD")
        with pytest.raises(TypeError, match="不支持相减"):
            Portfolio("USD") - Portfolio("USD")

    def test_str_and_repr(self):
        assert str(sample()) == "Portfolio(CNY: 1000.00 CNY, AAPL: 100 share, USD: 50.00 USD, base=CNY)"
        assert str(Portfolio("USD")) == "Portfolio(Empty, base=USD)"
        assert repr(Portfolio("USD")) == "Portfolio(base_currency='USD', lots=[])"
        assert repr(sample()).startswith("Portfolio(base_currency='CNY', lots=[PositionLot(asset='CNY'")
