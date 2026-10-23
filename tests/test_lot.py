"""PositionLot 与 match_lots 测试。

约定（lot.py）：
- 批次标的为 Currency（现金）或 Instrument（证券），数量为正的非缺失标量；证券数量统一存为基准单位。
- cost 为计价货币总成本，cost_base 为基准货币总成本（省略时等于 cost）；不做隐含的 1:1 换汇。
- split 按数量比例拆分，两部分成本之和恰好等于原值；match_lots 支持 FIFO / LIFO / WAC。
- 市值与盈亏必须显式给出价格（证券）与汇率（计价货币不是基准货币时）；收益率为比率 FixedArray。
"""

import random
from datetime import date, datetime
from decimal import Decimal
from fractions import Fraction

import pytest

from moneta import Instrument, Money, Price, Quantity
from moneta.conventions import DayCountConvention
from moneta.currency import CNY, EUR, USD
from moneta.fixed_array import FixedArray as F
from moneta.fx import ExchangeRate
from moneta.lot import LotMatchingRule, PositionLot, match_lots

AAPL = Instrument("AAPL", "USD")
CL = Instrument("CL", "USD", base_unit="barrel", multiplier=1000)
FIFO, LIFO, WAC = LotMatchingRule.FIFO, LotMatchingRule.LIFO, LotMatchingRule.WAC


def usd(v):
    return Money(v, "USD")


def cny(v):
    return Money(v, "CNY")


def stock(qty, cost, day="2024-01-02", cost_base=None, label=""):
    return PositionLot(AAPL, Quantity(qty), day, cost=usd(cost), cost_base=cost_base, label=label)


# ===========================================================================
# 1. 构造与校验
# ===========================================================================


class TestConstruct:
    def test_cash_lot(self):
        lot = PositionLot(USD, usd("100"), "2024-01-02")
        assert lot.is_cash and lot.asset is USD and lot.amount == usd("100")
        assert lot.cost == usd("100") and lot.cost_base == usd("100") and lot.base_currency is USD
        assert lot.acquired_date == date(2024, 1, 2) and lot.label == ""
        eur = PositionLot(EUR, Money("100", "EUR"), date(2024, 1, 2), cost_base=usd("108.50"))
        assert eur.cost == Money("100", "EUR") and eur.cost_base == usd("108.50") and eur.base_currency is USD
        assert PositionLot(USD, usd("1"), "2024-01-02", cost=usd("1.00")).cost == usd("1")

    def test_security_lot(self):
        lot = PositionLot(AAPL, Quantity(10), datetime(2024, 1, 2, 15, 30), cost=usd("1895"),
                          cost_base=cny("13644"), label="T-1")
        assert not lot.is_cash and lot.asset == AAPL and lot.amount == Quantity(10)
        assert lot.cost == usd("1895") and lot.cost_base == cny("13644") and lot.base_currency is CNY
        assert lot.acquired_date == date(2024, 1, 2) and lot.label == "T-1"
        wildcard = PositionLot(CL, Quantity("2.5", unit="Any"), "2024-01-02", cost=usd("200000"))
        assert wildcard.amount.unit == "barrel"  # 通配单位统一为基准单位
        assert PositionLot(AAPL, Quantity(1), "2024-01-02", cost=usd(0)).cost == usd(0)  # 成本可以为 0（如赠股）

    @pytest.mark.parametrize("kwargs, err, msg", [
        (dict(asset="USD", amount=usd(1)), TypeError, "Currency 或 Instrument"),
        (dict(asset=USD, amount=Quantity(1)), TypeError, "必须是 Money"),
        (dict(asset=USD, amount=cny(1)), TypeError, "不一致"),
        (dict(asset=USD, amount=usd(1), cost=usd(2)), ValueError, "成本就是金额本身"),
        (dict(asset=USD, amount=usd(1), cost=cny(1)), ValueError, "成本就是金额本身"),
        (dict(asset=AAPL, amount=usd(1), cost=usd(1)), TypeError, "必须是 Quantity"),
        (dict(asset=AAPL, amount=Quantity(1, unit="barrel"), cost=usd(1)), TypeError, "基准单位"),
        (dict(asset=AAPL, amount=Quantity(1)), TypeError, "必须给出计价货币总成本"),
        (dict(asset=AAPL, amount=Quantity(1), cost=Price("1", "USD")), TypeError, "必须给出计价货币总成本"),
        (dict(asset=AAPL, amount=Quantity(1), cost=cny(1)), TypeError, "计价货币"),
        (dict(asset=AAPL, amount=Quantity(1), cost=usd(1), cost_base=7), TypeError, "cost_base 必须是 Money"),
        (dict(asset=USD, amount=usd(0)), ValueError, "持仓数量 必须为正数"),
        (dict(asset=USD, amount=usd(-1)), ValueError, "持仓数量 必须为正数"),
        (dict(asset=USD, amount=usd(None)), ValueError, "持仓数量 不能是缺失值"),
        (dict(asset=USD, amount=usd([1, 2])), ValueError, "持仓数量 必须是标量"),
        (dict(asset=AAPL, amount=Quantity(1), cost=usd(-1)), ValueError, "cost 必须非负"),
        (dict(asset=AAPL, amount=Quantity(1), cost=usd(None)), ValueError, "cost 不能是缺失值"),
        (dict(asset=AAPL, amount=Quantity(1), cost=usd(1), cost_base=cny(-1)), ValueError, "cost_base 必须非负"),
        (dict(asset=USD, amount=usd(1), label=1), TypeError, "label 必须是 str"),
    ])
    def test_validation(self, kwargs, err, msg):
        kwargs.setdefault("acquired_date", "2024-01-02")
        with pytest.raises(err, match=msg):
            PositionLot(**kwargs)

    @pytest.mark.parametrize("day, err", [("2024-13-01", ValueError), (None, TypeError)])  # 解析规则见 test_utils
    def test_bad_dates(self, day, err):
        with pytest.raises(err, match="acquired_date"):
            PositionLot(USD, usd(1), day)

    def test_immutable_and_repr(self):
        lot = stock(10, "1895", label="T-1")
        with pytest.raises(AttributeError, match="禁止设置属性"):
            lot._amount = Quantity(1)
        with pytest.raises(AttributeError, match="禁止删除属性"):
            del lot._cost
        assert repr(lot) == ("PositionLot(asset='AAPL', amount=10 share, date='2024-01-02', cost=1895.00 USD, "
                             "cost_base=1895.00 USD, label='T-1')")
        assert repr(PositionLot(USD, usd(1), "2024-01-02")).startswith("PositionLot(asset='USD'")


# ===========================================================================
# 2. 持有期
# ===========================================================================


class TestHoldingPeriod:
    def test_days_and_year_fraction(self):
        lot = PositionLot(USD, usd(1), "2024-01-01")
        assert lot.holding_days("2024-03-01") == 60 and lot.holding_days(datetime(2024, 1, 1, 23)) == 0
        assert lot.year_fraction("2025-01-01") == pytest.approx(366 / 365)
        assert lot.year_fraction(date(2024, 7, 1), DayCountConvention.ACT_360) == pytest.approx(182 / 360)
        with pytest.raises(TypeError):
            lot.holding_days()  # as_of_date 必填
        with pytest.raises(TypeError, match="as_of_date"):
            lot.year_fraction(None)


# ===========================================================================
# 3. 拆分
# ===========================================================================


class TestSplit:
    def test_proportional_cost(self):
        lot = stock(3, "100.00", cost_base=cny("719.99"))
        part, rest = lot.split(Quantity(1))
        assert part.amount == Quantity(1) and rest.amount == Quantity(2)
        assert part.cost + rest.cost == lot.cost and part.cost_base + rest.cost_base == lot.cost_base
        assert part.cost == usd("33.33") and rest.cost == usd("66.67")  # 最大余数法：余数大的一方多得 1 分
        assert part.cost_base == cny("240.00") and rest.cost_base == cny("479.99")
        assert part.acquired_date == rest.acquired_date == lot.acquired_date and part.asset == AAPL

    def test_full_and_wildcard_and_cash(self):
        lot = stock("2.5", "250")
        assert lot.split(Quantity("2.50")) == (lot, None)
        part, rest = lot.split(Quantity("0.5", unit="Any"))
        assert part.amount.unit == "share" and (part.cost, rest.cost) == (usd(50), usd(200))
        cash = PositionLot(EUR, Money("100", "EUR"), "2024-01-02", cost_base=usd("108.51"))
        part, rest = cash.split(Money("40", "EUR"))
        assert part.is_cash and part.cost == Money("40", "EUR") and part.cost_base + rest.cost_base == usd("108.51")

    @pytest.mark.parametrize("amount, err, msg", [
        (Quantity(4), ValueError, "超过"), (Quantity(0), ValueError, "必须为正数"),
        (Quantity(-1), ValueError, "必须为正数"),
        (Quantity(None), ValueError, "缺失值"), (Quantity(1, unit="barrel"), TypeError, "单位"),
        (usd(1), TypeError, "Quantity"), (1, TypeError, "Quantity")])
    def test_invalid(self, amount, err, msg):
        with pytest.raises(err, match=msg):
            stock(3, "100").split(amount)

    def test_cash_split_requires_same_currency(self):
        for amount in (Money(1, "EUR"), Quantity(1)):
            with pytest.raises(TypeError, match="USD 的 Money"):
                PositionLot(USD, usd(3), "2024-01-02").split(amount)

    def test_random_conservation(self):
        rng = random.Random(7)
        for _ in range(500):
            qty = rng.randint(2, 10 ** 6)
            cost = Money.from_units(rng.randint(0, 10 ** 12), "USD")
            base = Money.from_units(rng.randint(0, 10 ** 12), "CNY")
            lot = PositionLot(AAPL, Quantity(qty), "2024-01-02", cost=cost, cost_base=base)
            k = rng.randint(1, qty - 1)
            part, rest = lot.split(Quantity(k))
            assert part.cost + rest.cost == cost and part.cost_base + rest.cost_base == base
            exact = Fraction(int(cost.units) * k, qty)
            assert abs(int(part.cost.units) - exact) < 1  # 与精确比例相差不到 1 个最小单位


# ===========================================================================
# 4. 市值与盈亏
# ===========================================================================


class TestValuation:
    def test_security_in_quote_currency(self):
        lot = stock(10, "1000")
        assert lot.market_value(Price("110", "USD")) == usd(1100) and lot.market_value(110) == usd(1100)
        assert lot.profit_loss("110") == usd(100) and lot.pnl("110") == usd(100)
        assert PositionLot.pnl is PositionLot.profit_loss
        rate = lot.return_rate(Price("110", "USD"))
        assert isinstance(rate, F) and rate == F("0.1") and rate.scale == 10

    def test_multiplier(self):
        lot = PositionLot(CL, Quantity(2, unit="barrel"), "2024-01-02", cost=usd(160000))
        assert lot.market_value(Price("81.5", "USD", base_unit="barrel")) == usd(163000)
        assert lot.pnl("79") == usd(-2000)

    def test_foreign_quote_needs_current_fx(self):
        lot = stock(10, "1000", cost_base=cny("7000"))  # 买入时 USD/CNY = 7.0
        now = ExchangeRate("7.2", "USD", "CNY")
        assert lot.market_value(100, now) == cny("7200")  # 用当前汇率，而不是入场汇率
        assert lot.pnl(100, now) == cny("200") and lot.return_rate(100, now) == F("0.0285714286")  # 2 + 8 位小数
        # 任一方向的汇率都可以：反方向时直接除以汇率，只舍入一次
        assert lot.market_value(100, ExchangeRate("0.125", "CNY", "USD")) == cny("8000")
        assert lot.market_value(100, ExchangeRate("8", "USD", "CNY")) == cny("8000")
        assert lot.market_value(Decimal("100.01"), ExchangeRate("0.3", "CNY", "USD")) == cny("3333.67")
        with pytest.raises(ValueError, match="需要 USD/CNY 的汇率"):
            lot.market_value(100)
        with pytest.raises(TypeError, match="不匹配"):
            lot.market_value(100, ExchangeRate("1.1", "EUR", "USD"))

    def test_cash(self):
        eur = PositionLot(EUR, Money("100", "EUR"), "2024-01-02", cost_base=usd("108.00"))
        now = ExchangeRate("1.10", "EUR", "USD")
        assert eur.market_value(fx=now) == usd(110) and eur.pnl(fx=now) == usd(2)
        local = PositionLot(USD, usd(100), "2024-01-02")
        assert local.market_value() == usd(100) and local.pnl() == usd(0) and local.return_rate() == 0
        with pytest.raises(ValueError, match="不接受 price"):
            local.market_value(1)
        with pytest.raises(ValueError, match="不需要 fx"):
            local.market_value(fx=now)

    def test_missing_price_and_value(self):
        lot = stock(10, "1000", cost_base=cny("7000"))
        with pytest.raises(ValueError, match="缺少标的 AAPL 的市价"):
            lot.market_value()
        na_value = lot.market_value(Price(None, "USD"), ExchangeRate("7.2", "USD", "CNY"))
        assert na_value.isna() and na_value.currency is CNY  # 缺失的市价得到缺失的市值，而不是 0
        assert stock(10, "1000").pnl(Price(None, "USD")).isna()

    def test_zero_cost_return_rate(self):
        gift = PositionLot(AAPL, Quantity(1), "2024-01-02", cost=usd(0))
        assert gift.pnl(10) == usd(10)
        with pytest.raises(ZeroDivisionError):
            gift.return_rate(10)

    def test_price_checks(self):
        with pytest.raises(TypeError, match="计价货币"):
            stock(1, "1").market_value(Price("1", "CNY"))


# ===========================================================================
# 5. match_lots
# ===========================================================================


def lots3():
    return [stock(10, "1000", "2024-01-01"), stock(5, "600", "2024-02-01"), stock(5, "700", "2024-03-01")]


class TestMatchLots:
    def test_fifo(self):
        closed, remaining = match_lots(lots3(), Quantity(12), FIFO)
        assert [l.amount for l in closed] == [Quantity(10), Quantity(2)]
        assert [l.cost for l in closed] == [usd(1000), usd(240)]
        assert [(l.amount, l.cost, str(l.acquired_date)) for l in remaining] == \
               [(Quantity(3), usd(360), "2024-02-01"), (Quantity(5), usd(700), "2024-03-01")]

    def test_lifo(self):
        closed, remaining = match_lots(lots3(), Quantity(7), LIFO)
        assert [(l.amount, l.cost) for l in closed] == [(Quantity(5), usd(700)), (Quantity(2), usd(240))]
        assert [l.amount for l in remaining] == [Quantity(10), Quantity(3)]

    def test_wac(self):
        closed, remaining = match_lots(lots3(), Quantity(4), WAC)  # 按 10:5:5 扣减 2:1:1
        assert [l.amount for l in closed] == [Quantity(2), Quantity(1), Quantity(1)]
        assert sum((l.cost for l in closed), usd(0)) == usd(460)  # 平均成本 115 × 4
        assert [l.amount for l in remaining] == [Quantity(8), Quantity(4), Quantity(4)]

    def test_wac_rounding_to_smallest_unit(self):
        lots = [stock(1, "100"), stock(1, "200"), stock(1, "300")]
        closed, remaining = match_lots(lots, Quantity(1), WAC)  # 1/3 份分不开：余数相同时靠前的批次优先
        assert [l.amount for l in closed] == [Quantity(1)] and closed[0].cost == usd(100)
        closed, _ = match_lots(lots, Quantity("1.0"), WAC)  # 按更细的 scale 分配
        assert [l.amount for l in closed] == [Quantity("0.4"), Quantity("0.3"), Quantity("0.3")]
        cash = [PositionLot(USD, usd("0.03"), "2024-01-01"), PositionLot(USD, usd("0.03"), "2024-01-02")]
        closed, remaining = match_lots(cash, usd("0.03"), WAC)
        assert [l.amount for l in closed] == [usd("0.02"), usd("0.01")] and [l.amount for l in remaining] == \
               [usd("0.01"), usd("0.02")]

    @pytest.mark.parametrize("rule", [FIFO, LIFO, WAC])
    def test_full_and_conservation(self, rule):
        lots = lots3()
        closed, remaining = match_lots(lots, Quantity(20), rule)
        assert remaining == [] and sum((l.cost for l in closed), usd(0)) == usd(2300)
        rng = random.Random(11)
        for _ in range(100):
            k = Quantity(F._from_mantissa(rng.randint(1, 2000), 2))
            closed, remaining = match_lots(lots, k, rule)
            assert sum((l.amount for l in closed), Quantity(0)) == k
            assert sum((l.cost for l in closed + remaining), usd(0)) == usd(2300)
            assert sum((l.amount for l in closed + remaining), Quantity(0)) == Quantity(20)

    def test_errors(self):
        with pytest.raises(ValueError, match="超过批次总量"):
            match_lots(lots3(), Quantity(21))
        with pytest.raises(ValueError, match="没有可扣减的批次"):
            match_lots([], Quantity(1))
        with pytest.raises(ValueError, match="必须为正数"):
            match_lots(lots3(), Quantity(0))
        with pytest.raises(ValueError, match="同一标的"):
            match_lots([stock(1, "1"), PositionLot(USD, usd(1), "2024-01-01")], Quantity(1))
        with pytest.raises(ValueError, match="同一标的"):
            match_lots([stock(1, "1"), stock(1, "1", cost_base=cny(7))], Quantity(1))
        with pytest.raises(TypeError, match="LotMatchingRule"):
            match_lots(lots3(), Quantity(1), "FIFO")
        with pytest.raises(TypeError, match="PositionLot"):
            match_lots([stock(1, "1"), "lot"], Quantity(1))

    def test_rule_enum(self):
        assert [r.value for r in LotMatchingRule] == ["FIFO", "LIFO", "WAC"]
