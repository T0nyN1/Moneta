"""Moneta 多币种资产组合 (Portfolio)。

约定：
- 持仓以资产为键：现金为 Currency，证券为 Instrument（相等只看代码与计价货币）。批次 PositionLot 是唯一事实来源，
  某资产的持仓数量即其全部批次数量之和，不另存一份余额。
- 基准货币必填。任何换汇都必须显式给出汇率：fx 可以是一个 ExchangeRate，也可以是 FXEngine（按交易日或
  估值日取汇率）；不提供时报错，不会静默使用快照汇率或按 1:1 换算。
- 证券估值必须给出市价；裸数字价格按标的的计价货币解释，并乘以合约乘数。
- withdraw 按 FIFO / LIFO / WAC 扣减批次，给出成交价（现金为汇率）时返回按基准货币计的已实现盈亏。
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from ._utils import to_date
from .currency import Currency, CurrencyRegistry, get_currency
from .fixed_array import FixedArray
from .instrument import Instrument
from .lot import LotMatchingRule, PositionLot, _convert_at_rate, match_lots
from .money import Money
from .quantity import Quantity

Asset = Union[Currency, Instrument]
Position = Union[Money, Quantity]


class Portfolio:
    """多币种资产组合：现金与证券持仓、批次结转、估值与再平衡。

    Parameters
    ----------
    base_currency : Currency 或 str
        组合基准（记账）货币，如 "USD" 或 "CNY"。
    """

    __slots__ = ("_base_currency", "_lots")

    def __init__(self, base_currency: Union[Currency, str]) -> None:
        self._base_currency = get_currency(base_currency)
        self._lots: Dict[Asset, List[PositionLot]] = {}

    # ---------- 资产与持仓查询 ----------

    @property
    def base_currency(self) -> Currency:
        """组合基准货币。"""
        return self._base_currency

    @property
    def assets(self) -> List[Asset]:
        """当前持有的全部资产（按首次入账顺序）。"""
        return list(self._lots)

    @property
    def currencies(self) -> List[Currency]:
        """当前持有现金的币种。"""
        return [a for a in self._lots if isinstance(a, Currency)]

    @property
    def instruments(self) -> List[Instrument]:
        """当前持有的证券标的。"""
        return [a for a in self._lots if isinstance(a, Instrument)]

    def lots(self, asset: Optional[Union[Asset, str]] = None) -> List[PositionLot]:
        """某资产的批次（按入账顺序）；不传 asset 时返回全部批次。返回副本，批次本身不可变。"""
        if asset is None:
            return [lot for lots in self._lots.values() for lot in lots]
        return list(self._lots.get(self._resolve(asset), []))

    def position(self, asset: Union[Asset, str]) -> Position:
        """某资产的持仓数量：现金为 Money，证券为基准单位的 Quantity；未持有时为 0。"""
        key = self._resolve(asset)
        lots = self._lots.get(key)
        if not lots:
            return Money(0, key) if isinstance(key, Currency) else key.as_quantity(0)
        total = lots[0].amount
        for lot in lots[1:]:
            total = total + lot.amount
        return total

    def _resolve(self, asset: Union[Asset, str]) -> Asset:
        """Currency / Instrument 原样返回；字符串先在已持有资产中按币种代码或标的代码唯一匹配，
        找不到时若是已注册的币种代码则视为该币种，否则报 KeyError。"""
        if isinstance(asset, (Currency, Instrument)):
            return asset
        if not isinstance(asset, str):
            raise TypeError(f"资产必须是 Currency、Instrument 或代码字符串，收到 {type(asset).__name__}")
        code = asset.strip().upper()
        matches = [a for a in self._lots if (a.code if isinstance(a, Currency) else a.symbol) == code]
        if len(matches) > 1:
            raise KeyError(f"代码 {code!r} 对应多个持仓 {matches}，请传入 Currency 或 Instrument")
        if matches:
            return matches[0]
        if CurrencyRegistry.has(code):
            return get_currency(code)
        raise KeyError(f"组合中没有代码为 {code!r} 的持仓")

    def __getitem__(self, asset: Union[Asset, str]) -> Position:
        """portfolio[USD]、portfolio["USD"]、portfolio[aapl]、portfolio["AAPL"] → 持仓数量。"""
        return self.position(asset)

    def __contains__(self, asset: Any) -> bool:
        try:
            return self._resolve(asset) in self._lots
        except (KeyError, TypeError):
            return False

    def get(self, asset: Union[Asset, str], default: Any = None) -> Any:
        """持有该资产时返回持仓数量，否则返回 default。"""
        return self.position(asset) if asset in self else default

    def __iter__(self) -> Iterator[Tuple[Asset, Position]]:
        """依次产出 (资产, 持仓数量)。"""
        return ((asset, self.position(asset)) for asset in list(self._lots))

    def __len__(self) -> int:
        return len(self._lots)

    # ---------- 入账与出账 ----------

    def add_lot(self, lot: PositionLot) -> None:
        """加入一个已构造好的批次；批次的基准货币必须与组合一致。"""
        if not isinstance(lot, PositionLot):
            raise TypeError(f"只能加入 PositionLot，收到 {type(lot).__name__}")
        if lot.base_currency != self._base_currency:
            raise ValueError(f"批次的基准货币 {lot.base_currency.code} 与组合基准货币 {self._base_currency.code} 不一致")
        self._lots.setdefault(lot.asset, []).append(lot)

    def deposit(self, asset: Union[Asset, str], amount: Any, acquired_date: Any, *, price: Any = None,
                cost: Optional[Money] = None, cost_base: Optional[Money] = None, fx: Any = None,
                label: str = "") -> PositionLot:
        """入账一笔现金或证券，返回新建的批次。

        - 现金：asset 为 Currency（或币种代码），amount 为该币种的 Money 或数值，不接受 price / cost。
        - 证券：asset 为 Instrument，amount 为 Quantity 或数值（按基准单位）；price（成交价）与 cost
          （计价货币总成本，如含手续费）二选一，给 price 时成本 = 名义金额（含合约乘数）。
        - cost_base：基准货币总成本。省略时，计价货币就是基准货币则等于成本，否则按 fx 在 acquired_date
          的汇率换算；两者都没有时报错。
        """
        key = self._resolve(asset)
        if isinstance(key, Currency):
            if price is not None or cost is not None:
                raise ValueError("现金入账的成本就是金额本身，不接受 price / cost")
            amount = amount if isinstance(amount, Money) else Money(amount, key)
            local_cost = amount
        else:
            amount = amount if isinstance(amount, Quantity) else key.as_quantity(amount)
            if (price is None) == (cost is None):
                raise ValueError("证券入账必须给出 price 或 cost 之一")
            local_cost = cost if cost is not None else key.calc_notional(price, amount)
        if cost_base is None:
            cost_base = self._to_base(local_cost, fx, acquired_date)
        lot = PositionLot(key, amount, acquired_date, cost=None if isinstance(key, Currency) else local_cost,
                          cost_base=cost_base, label=label)
        self.add_lot(lot)
        return lot

    def withdraw(self, asset: Union[Asset, str], amount: Any, trade_date: Any, *, price: Any = None,
                 fx: Any = None, rule: LotMatchingRule = LotMatchingRule.FIFO
                 ) -> Tuple[List[PositionLot], Optional[Money]]:
        """按规则扣减批次，返回 (平掉的批次, 已实现盈亏)。

        已实现盈亏以基准货币计：证券给出 price（成交价）时计算，现金总是计算；计价货币不是基准货币时
        需要 fx（在 trade_date 的汇率），证券给了 price 却缺少 fx 报错，外币现金缺少 fx 时盈亏为 None；
        证券不给 price（如转出）时盈亏为 None。持仓全部扣完后，该资产从组合中移除。
        """
        to_date(trade_date, "trade_date")  # 交易日必须是合法日期，保证结转可重现
        key = self._resolve(asset)
        lots = self._lots.get(key)
        if not lots:
            raise ValueError(f"组合中没有持有 {key}")
        if isinstance(key, Currency):
            amount = amount if isinstance(amount, Money) else Money(amount, key)
        else:
            amount = amount if isinstance(amount, Quantity) else key.as_quantity(amount)
        closed, remaining = match_lots(lots, amount, rule)

        realized = None
        local_currency = key if isinstance(key, Currency) else key.quote_currency
        foreign = local_currency != self._base_currency
        if isinstance(key, Instrument) and price is not None:
            rate = self._rate(fx, local_currency, trade_date) if foreign else None
            realized = self._sum(lot.profit_loss(price, rate) for lot in closed)
        elif isinstance(key, Currency) and (not foreign or fx is not None):
            rate = self._rate(fx, local_currency, trade_date) if foreign else None
            realized = self._sum(lot.profit_loss(fx=rate) for lot in closed)

        if remaining:
            self._lots[key] = remaining
        else:
            del self._lots[key]
        return closed, realized

    # ---------- 估值 ----------

    def total_value(self, prices: Optional[Mapping[Any, Any]] = None, fx: Any = None,
                    as_of_date: Any = None) -> Money:
        """按基准货币计的组合总市值。证券必须在 prices 中有市价（键为 Instrument 或标的代码）；
        非基准货币的资产需要 fx（FXEngine 时按 as_of_date 取汇率）。"""
        return self._sum(self._position_value(asset, prices, fx, as_of_date) for asset in self._lots)

    def total_cost(self) -> Money:
        """全部批次按基准货币计的总成本。"""
        return self._sum(lot.cost_base for lot in self.lots())

    def unrealized_pnl(self, prices: Optional[Mapping[Any, Any]] = None, fx: Any = None,
                       as_of_date: Any = None) -> Money:
        """未实现盈亏（基准货币）= 总市值 − 总成本。"""
        return self.total_value(prices, fx, as_of_date) - self.total_cost()

    def return_rate(self, prices: Optional[Mapping[Any, Any]] = None, fx: Any = None,
                    as_of_date: Any = None) -> FixedArray:
        """收益率 = 未实现盈亏 / 总成本（比率 FixedArray，精度同 FixedArray.divide 的默认规则）；总成本为 0 时报错。"""
        return self.unrealized_pnl(prices, fx, as_of_date).divide(self.total_cost())

    def _position_value(self, asset: Asset, prices: Optional[Mapping[Any, Any]], fx: Any, as_of_date: Any) -> Money:
        position = self.position(asset)
        if isinstance(asset, Currency):
            local = position
        else:
            local = asset.calc_notional(self._price_of(asset, prices), position)
        return self._to_base(local, fx, as_of_date)

    @staticmethod
    def _price_of(instrument: Instrument, prices: Optional[Mapping[Any, Any]]) -> Any:
        """在 prices 中按 Instrument 或标的代码（不区分大小写）查找市价；缺失时报错。"""
        prices = prices or {}
        if instrument in prices:
            return prices[instrument]
        for key, value in prices.items():
            if isinstance(key, str) and key.strip().upper() == instrument.symbol:
                return value
        raise ValueError(f"缺少标的 {instrument.symbol} 的市价")

    # ---------- 再平衡 ----------

    def rebalance(self, target_weights: Mapping[Any, Any], prices: Optional[Mapping[Any, Any]] = None,
                  fx: Any = None, as_of_date: Any = None) -> Dict[Asset, Position]:
        """按目标权重计算各资产需要调整的数量：正数为买入 / 存入，负数为卖出 / 取出。

        权重键为 Currency、Instrument 或已持有资产的代码，值为非负数值，合计必须恰好为 1；
        未出现在权重中的已持有资产目标为 0。现金差额为该币种的 Money，证券差额为基准单位的 Quantity
        （精度同 FixedArray.divide 的默认规则，需要整手时再 round_to_step）。
        """
        weights: Dict[Asset, FixedArray] = {}
        for key, weight in target_weights.items():
            asset = self._resolve(key)
            w = Money._as_grid(weight, "权重")
            if w.ndim != 0 or w.mantissa < 0:
                raise ValueError(f"{asset} 的权重必须是非负标量，收到 {weight!r}")
            if asset in weights:
                raise ValueError(f"{asset} 的权重重复给出")
            weights[asset] = w
        total_weight = sum(weights.values(), FixedArray(0))
        if total_weight != 1:
            raise ValueError(f"目标权重合计必须为 1，收到 {total_weight}")

        total = self.total_value(prices, fx, as_of_date)
        adjustments: Dict[Asset, Position] = {}
        for asset in list(self._lots) + [a for a in weights if a not in self._lots]:
            # 目标价值 = 总市值 × 权重（基准货币，精确）；换到本地货币与除以每单位价值都合并成一次除法
            num = total.data.mul_round(weights.get(asset, FixedArray(0)), rounding=None)
            local = asset if isinstance(asset, Currency) else asset.quote_currency
            num, den = self._base_to_local(num, local, fx, as_of_date)
            if isinstance(asset, Currency):
                target = Money._wrap(num.divide(den, scale=local.digits), local)
                adjustments[asset] = target - self.position(asset)
            else:  # 每单位价值 = 价格 × 合约乘数（精确）
                unit_value = asset.as_price(self._price_of(asset, prices)).data.mul_round(asset.multiplier, rounding=None)
                units = num.divide(den.mul_round(unit_value, rounding=None))
                adjustments[asset] = asset.as_quantity(units) - self.position(asset)
        return adjustments

    # ---------- 换汇辅助 ----------

    def _rate(self, fx: Any, currency: Currency, as_of_date: Any) -> Any:
        """currency 与基准货币之间的汇率：fx 为 FXEngine 时按日期取，为 ExchangeRate 时原样使用。"""
        if fx is None:
            raise ValueError(f"需要 {currency.code}/{self._base_currency.code} 的汇率：请传入 fx")
        if hasattr(fx, "get_rate"):
            when = to_date(as_of_date, "as_of_date", optional=True)
            return fx.get_rate(currency, self._base_currency, as_of_date=when)
        return fx

    def _base_to_local(self, value: FixedArray, currency: Currency, fx: Any, as_of_date: Any
                       ) -> Tuple[FixedArray, FixedArray]:
        """基准货币价值 → 本地货币价值，表示为精确的分子 / 分母（不舍入）。"""
        one = FixedArray(1)
        if currency == self._base_currency:
            return value, one
        rate = self._rate(fx, currency, as_of_date)
        pair = (rate.base_currency, rate.quote_currency)
        if pair == (self._base_currency, currency):
            return value.mul_round(rate.rate, rounding=None), one
        if pair == (currency, self._base_currency):
            return value, rate.rate
        raise TypeError(f"汇率 {rate.base_currency.code}/{rate.quote_currency.code} 与 "
                        f"{currency.code}/{self._base_currency.code} 不匹配")

    def _to_base(self, money: Money, fx: Any, as_of_date: Any) -> Money:
        if money.currency == self._base_currency:
            return money
        return _convert_at_rate(money, self._rate(fx, money.currency, as_of_date), self._base_currency)

    def _sum(self, values: Any) -> Money:
        total = Money(0, self._base_currency)
        for value in values:
            total = total + value
        return total

    # ---------- 运算符与展示 ----------

    def __add__(self, other: Any) -> Any:
        raise TypeError("Portfolio 不支持相加：批次的买入日期与成本不能随意合并，请用 deposit / add_lot 逐笔入账")

    def __sub__(self, other: Any) -> Any:
        raise TypeError("Portfolio 不支持相减：减少持仓必须通过 withdraw 按规则核销批次")

    def __repr__(self) -> str:
        return f"Portfolio(base_currency='{self._base_currency.code}', lots={self.lots()!r})"

    def __str__(self) -> str:
        items = ", ".join(f"{a.code if isinstance(a, Currency) else a.symbol}: {p}" for a, p in self) or "Empty"
        return f"Portfolio({items}, base={self._base_currency.code})"


__all__ = ["Portfolio"]
