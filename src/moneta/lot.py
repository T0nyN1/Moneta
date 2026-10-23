"""Moneta 持仓批次与平仓会计规则 (Position Lot & Lot Matching)。

约定：
- PositionLot 是不可变的单笔持仓批次：标的为 Currency（现金）或 Instrument（证券），数量为正的标量。
- 批次显式记录两份总成本：cost（计价货币，证券为 成交价 × 数量 × 合约乘数，现金为金额本身）与
  cost_base（组合基准货币，由调用方按买入日汇率换算后传入）。不做任何隐含的 1:1 换汇，也没有默认币种。
- split 按数量比例拆分批次，两部分的成本用最大余数法分配，之和恰好等于原成本。
- 市值、盈亏必须显式传入价格（证券）与汇率（计价货币不是基准货币时，任一方向的 ExchangeRate，按中间价
  乘或除一次舍入）；收益率返回比率 FixedArray。
- match_lots 按 FIFO / LIFO / WAC 从批次中扣减数量，WAC 按比例从所有批次扣减，等价于按加权平均成本平仓。
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union

from ._units import units_compatible
from ._utils import to_date as _to_date
from .conventions import DayCountConvention, year_fraction as _year_fraction
from .currency import Currency
from .fixed_array import FixedArray
from .instrument import Instrument
from .money import Money
from .quantity import Quantity

DateLike = Union[str, date, datetime]
Asset = Union[Currency, Instrument]
Amount = Union[Money, Quantity]


class LotMatchingRule(Enum):
    """平仓批次结转规则（Lot Matching）。"""

    FIFO = "FIFO"  # 先进先出：优先平掉最早买入的批次
    LIFO = "LIFO"  # 后进先出：优先平掉最近买入的批次
    WAC = "WAC"  # 加权平均成本：按比例从所有批次扣减，平掉部分的成本即平均成本


class PositionLot:
    """单笔持仓批次（不可变）。

    Parameters
    ----------
    asset : Currency 或 Instrument
        持有的现金币种或证券标的；同一标的的批次在组合中归在一起。
    amount : Money 或 Quantity
        持有数量：现金为该币种的 Money，证券为 Quantity（单位须为标的基准单位或通配单位，统一存为基准单位）。
        必须是正的、非缺失的标量。
    acquired_date : date、datetime 或 ISO 字符串
        买入 / 入账日期。
    cost : Money, optional
        计价货币表示的总成本（证券必填，币种须为标的计价货币，非负）；现金批次的成本就是金额本身，
        可省略，给出时必须等于 amount。
    cost_base : Money, optional
        组合基准货币表示的总成本（非负），由调用方按买入日汇率换算；省略时等于 cost（即基准货币就是计价货币）。
    label : str, optional
        交易编号等备注，不参与标识。
    """

    __slots__ = ("_asset", "_amount", "_acquired_date", "_cost", "_cost_base", "_label")

    @staticmethod
    def _check_scalar(value: Any, what: str, positive: bool) -> None:
        """标量、非缺失；positive 为 True 时必须 > 0，否则必须 ≥ 0。"""
        if value.ndim != 0:
            raise ValueError(f"{what} 必须是标量")
        if value.isna():
            raise ValueError(f"{what} 不能是缺失值")
        m = int(value.mantissa)
        if m < 0 or (positive and m == 0):
            raise ValueError(f"{what} 必须{'为正数' if positive else '非负'}，收到 {value}")

    def __init__(self, asset: Asset, amount: Amount, acquired_date: DateLike, cost: Optional[Money] = None,
                 cost_base: Optional[Money] = None, label: str = "") -> None:
        if isinstance(asset, Currency):
            if not isinstance(amount, Money):
                raise TypeError(f"现金批次的数量必须是 Money，收到 {type(amount).__name__}")
            if amount.currency != asset:
                raise TypeError(f"金额币种 {amount.currency.code} 与批次币种 {asset.code} 不一致")
            if cost is not None and not (isinstance(cost, Money) and cost.currency == asset and bool(cost == amount)):
                raise ValueError("现金批次的成本就是金额本身，cost 省略或必须等于 amount")
            cost = amount
        elif isinstance(asset, Instrument):
            if not isinstance(amount, Quantity):
                raise TypeError(f"证券批次的数量必须是 Quantity，收到 {type(amount).__name__}")
            if not units_compatible(amount.unit, asset.base_unit):
                raise TypeError(f"数量单位 '{amount.unit}' 与标的 {asset.symbol} 的基准单位 '{asset.base_unit}' 不一致")
            amount = Quantity(amount, unit=asset.base_unit)
            if not isinstance(cost, Money):
                raise TypeError(f"证券批次必须给出计价货币总成本 cost (Money)，收到 {type(cost).__name__}")
            if cost.currency != asset.quote_currency:
                raise TypeError(f"成本币种 {cost.currency.code} 与标的 {asset.symbol} 的计价货币 "
                                f"{asset.quote_currency.code} 不一致")
        else:
            raise TypeError(f"批次标的必须是 Currency 或 Instrument，收到 {type(asset).__name__}")
        if cost_base is None:
            cost_base = cost
        elif not isinstance(cost_base, Money):
            raise TypeError(f"cost_base 必须是 Money，收到 {type(cost_base).__name__}")
        self._check_scalar(amount, "持仓数量", positive=True)
        self._check_scalar(cost, "cost", positive=False)
        self._check_scalar(cost_base, "cost_base", positive=False)
        if not isinstance(label, str):
            raise TypeError(f"label 必须是 str，收到 {type(label).__name__}")

        for name, value in (("_asset", asset), ("_amount", amount),
                            ("_acquired_date", _to_date(acquired_date, "acquired_date")),
                            ("_cost", cost), ("_cost_base", cost_base), ("_label", label)):
            object.__setattr__(self, name, value)

    # ---------- 属性 ----------

    @property
    def asset(self) -> Asset:
        """持有的现金币种（Currency）或证券标的（Instrument）。"""
        return self._asset

    @property
    def is_cash(self) -> bool:
        """是否为现金批次。"""
        return isinstance(self._asset, Currency)

    @property
    def amount(self) -> Amount:
        """持有数量（现金为 Money，证券为基准单位的 Quantity）。"""
        return self._amount

    @property
    def acquired_date(self) -> date:
        """买入 / 入账日期。"""
        return self._acquired_date

    @property
    def cost(self) -> Money:
        """计价货币表示的总成本（现金批次即金额本身）。"""
        return self._cost

    @property
    def cost_base(self) -> Money:
        """组合基准货币表示的总成本。"""
        return self._cost_base

    @property
    def base_currency(self) -> Currency:
        """组合基准货币（即 cost_base 的币种）。"""
        return self._cost_base.currency

    @property
    def label(self) -> str:
        """交易编号等备注。"""
        return self._label

    # ---------- 持有期 ----------

    def holding_days(self, as_of_date: DateLike) -> int:
        """自买入日至 as_of_date 的日历天数。"""
        return (_to_date(as_of_date, "as_of_date") - self._acquired_date).days

    def year_fraction(self, as_of_date: DateLike, convention: DayCountConvention = DayCountConvention.ACT_365) -> float:
        """按计日惯例计算自买入日至 as_of_date 的年化分度。"""
        return _year_fraction(self._acquired_date, _to_date(as_of_date, "as_of_date"), convention)

    # ---------- 拆分 ----------

    def split(self, amount: Amount) -> Tuple["PositionLot", Optional["PositionLot"]]:
        """拆出 amount（0 < amount ≤ 持仓数量）：返回 (拆出的批次, 剩余批次)，全部拆出时剩余为 None。

        两部分的 cost 与 cost_base 按数量比例用最大余数法分配，之和恰好等于原值；买入日与备注保持不变。
        """
        part = self._coerce_amount(amount)
        self._check_scalar(part, "拆分数量", positive=True)
        if part > self._amount:
            raise ValueError(f"拆分数量 {part} 超过批次持仓 {self._amount}")
        if part == self._amount:
            return self, None
        rest = self._amount - part
        weights = FixedArray([part.data, rest.data])
        cost_part, cost_rest = self._cost.allocate(weights)
        base_part, base_rest = self._cost_base.allocate(weights)
        return self._derive(part, cost_part, base_part), self._derive(rest, cost_rest, base_rest)

    def _coerce_amount(self, amount: Any) -> Amount:
        """与本批次同类、同币种或同单位（可为通配单位）的数量，统一为本批次的单位。"""
        if self.is_cash:
            if not isinstance(amount, Money) or amount.currency != self._asset:
                raise TypeError(f"数量必须是 {self._asset.code} 的 Money，收到 {amount!r}")
            return amount
        if not isinstance(amount, Quantity) or not units_compatible(amount.unit, self._amount.unit):
            raise TypeError(f"数量必须是单位为 '{self._amount.unit}' 的 Quantity，收到 {amount!r}")
        return Quantity(amount, unit=self._amount.unit)

    def _derive(self, amount: Amount, cost: Money, cost_base: Money) -> "PositionLot":
        return PositionLot(self._asset, amount, self._acquired_date, None if self.is_cash else cost, cost_base,
                           self._label)

    # ---------- 市值与盈亏 ----------

    def market_value(self, price: Any = None, fx: Any = None) -> Money:
        """按基准货币计的市值。

        price：证券必填（Price 或按标的计价货币解释的裸数字），现金不接受；
        fx：计价货币不是基准货币时必填，为两种货币之间的 ExchangeRate（任一方向）；按中间价换算，
        反方向时直接除以汇率，只舍入一次（不经过 inverse 的二次舍入），缺失值保持为缺失值。
        """
        if self.is_cash:
            if price is not None:
                raise ValueError("现金批次的市值就是金额本身，不接受 price")
            local = self._amount
        else:
            if price is None:
                raise ValueError(f"缺少标的 {self._asset.symbol} 的市价 price")
            local = self._asset.calc_notional(price, self._amount)
        base = self.base_currency
        if local.currency == base:
            if fx is not None:
                raise ValueError("计价货币就是基准货币，不需要 fx")
            return local
        return _convert_at_rate(local, fx, base)

    def profit_loss(self, price: Any = None, fx: Any = None) -> Money:
        """按给定价格与汇率计的盈亏（基准货币）= 市值 − cost_base；平仓批次传入成交价即为已实现盈亏。"""
        return self.market_value(price, fx) - self._cost_base

    pnl = profit_loss

    def return_rate(self, price: Any = None, fx: Any = None) -> FixedArray:
        """收益率 = 盈亏 / cost_base（比率 FixedArray，保留基准货币精度 + 8 位小数）；成本为 0 时报 ZeroDivisionError。"""
        return self.profit_loss(price, fx).divide(self._cost_base)

    # ---------- 不可变与展示 ----------

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(f"PositionLot 是不可变对象，禁止设置属性 '{key}'")

    def __delattr__(self, key: str) -> None:
        raise AttributeError(f"PositionLot 是不可变对象，禁止删除属性 '{key}'")

    def __repr__(self) -> str:
        asset = self._asset.code if self.is_cash else self._asset.symbol
        label = f", label='{self._label}'" if self._label else ""
        return (f"PositionLot(asset='{asset}', amount={self._amount}, date='{self._acquired_date}', "
                f"cost={self._cost}, cost_base={self._cost_base}{label})")


def _convert_at_rate(money: Money, rate: Any, target: Currency) -> Money:
    """按 money 与 target 两种货币之间的 ExchangeRate（任一方向）把金额换算为 target，只舍入一次。

    正方向乘以中间价、反方向直接除以中间价（不经过 inverse 的二次舍入），缺失值保持为缺失值。
    """
    if rate is None:
        raise ValueError(f"需要 {money.currency.code}/{target.code} 的汇率 fx 才能换算为 {target.code}")
    pair = (rate.base_currency, rate.quote_currency)
    if pair == (money.currency, target):
        converted = money.data.mul_round(rate.rate, scale=target.digits)
    elif pair == (target, money.currency):
        converted = money.data.divide(rate.rate, scale=target.digits)
    else:
        raise TypeError(f"汇率 {rate.base_currency.code}/{rate.quote_currency.code} 与 "
                        f"{money.currency.code}/{target.code} 不匹配")
    return Money._wrap(converted, target)


def match_lots(lots: Sequence[PositionLot], amount: Amount,
               rule: LotMatchingRule = LotMatchingRule.FIFO) -> Tuple[List[PositionLot], List[PositionLot]]:
    """按规则从同一标的的批次中扣减 amount，返回 (平掉的批次, 剩余批次)。

    lots 按买入顺序排列：FIFO 从前往后、LIFO 从后往前依次扣减，最后一个被扣到的批次按比例拆分；
    WAC 按各批次数量比例从所有批次扣减（用最大余数法分配到最小单位，余数相同时靠前的批次优先）。
    剩余批次保持原来的顺序。amount 必须为正且不超过批次总量，所有批次的标的与基准货币必须相同。
    """
    if not isinstance(rule, LotMatchingRule):
        raise TypeError(f"rule 必须是 LotMatchingRule，收到 {rule!r}")
    if not lots:
        raise ValueError("没有可扣减的批次")
    first = lots[0]
    for lot in lots:
        if not isinstance(lot, PositionLot):
            raise TypeError(f"lots 只能包含 PositionLot，收到 {type(lot).__name__}")
        if lot.asset != first.asset or lot.base_currency != first.base_currency:
            raise ValueError("match_lots 的批次必须属于同一标的且基准货币相同")
    target = first._coerce_amount(amount)
    PositionLot._check_scalar(target, "扣减数量", positive=True)
    total = first.amount
    for lot in lots[1:]:
        total = total + lot.amount
    if target > total:
        raise ValueError(f"扣减数量 {target} 超过批次总量 {total}")

    if rule is LotMatchingRule.WAC:
        shares = _proportional_shares(lots, target)
    else:
        order = range(len(lots)) if rule is LotMatchingRule.FIFO else range(len(lots) - 1, -1, -1)
        shares, left = [None] * len(lots), target
        for i in order:
            if left.is_zero():
                break
            shares[i] = lots[i].amount if lots[i].amount <= left else left
            left = left - shares[i]

    closed, remaining = [], []
    for lot, share in zip(lots, shares):
        if share is None or share.is_zero():
            remaining.append(lot)
            continue
        part, rest = lot.split(share)
        closed.append(part)
        if rest is not None:
            remaining.append(rest)
    if rule is LotMatchingRule.LIFO:
        closed.reverse()  # 按扣减顺序返回
    return closed, remaining


def _proportional_shares(lots: Sequence[PositionLot], target: Amount) -> List[Amount]:
    """把 target 按各批次数量比例分到最小单位（共同的 scale 上），最大余数法，余数相同时靠前的批次优先。"""
    scale = max([lot.amount.scale for lot in lots] + [target.scale])
    units = [int(lot.amount.data.rescale(scale).mantissa) for lot in lots]
    goal, whole = int(target.data.rescale(scale).mantissa), sum(units)
    shares, remainders = [], []
    for u in units:
        q, r = divmod(goal * u, whole)
        shares.append(q)
        remainders.append(r)
    for i in sorted(range(len(units)), key=lambda i: remainders[i], reverse=True)[:goal - sum(shares)]:
        shares[i] += 1  # 每份不超过 ceil(goal·u/whole) ≤ u
    unit = lots[0].amount
    if isinstance(unit, Money):
        return [Money.from_units(s, unit.currency) for s in shares]
    return [Quantity(FixedArray._from_mantissa(s, scale), unit=unit.unit) for s in shares]


__all__ = ["LotMatchingRule", "PositionLot", "match_lots"]
