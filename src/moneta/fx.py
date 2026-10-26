"""Moneta 外汇汇率与换汇 (FX)。

包含：
1. ExchangeRate：货币对汇率（中间价与可选的买卖价），换汇时只舍入一次；
2. 汇率数据源：FXProvider 抽象基类、用户字典 UserDictFXProvider、内置参考快照 SnapshotFXProvider、
   在线数据源 FrankfurterFXProvider 与按顺序查询的 CompositeFXProvider；
3. FXEngine：按数据源查汇率，没有直盘汇率时经锚定货币三角套算；
4. 汇兑损益分解 decompose_fx_pnl 与抛补利率平价远期汇率 covered_interest_parity_forward。

约定：
- 汇率必须为正，买卖价满足 bid ≤ rate ≤ ask；同一币种的汇率恒为 1。
- 按日期严格匹配：传入 as_of_date 时只使用当天的汇率，不传时只使用不带日期的汇率，查不到就报
  ExchangeRateNotFoundError。不会向前沿用旧汇率，不会使用未来汇率，也不会静默使用快照或网络数据——
  数据源需要自行提供每一天的汇率。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ._rounding import RoundingMode, check_rounding
from ._utils import to_date
from .currency import Currency, get_currency
from .fixed_array import FixedArray
from .money import Money

DateLike = Union[str, date, datetime]
CurrencyLike = Union[Currency, str]
Rounding = Union[RoundingMode, str]
_SIDES = ("mid", "bid", "ask")


class ExchangeRateNotFoundError(KeyError):
    """数据源中没有所需货币对（在所需日期）的汇率。"""


class ExchangeRate:
    """货币对汇率（不可变）：1 单位基准货币 base 兑换 rate 单位计价货币 quote。

    Parameters
    ----------
    base_currency, quote_currency : Currency 或 str
        基准货币与计价货币，如 USD/CNY 中的 USD 与 CNY。
    rate : 数值
        中间价，必须为正；同一币种时必须为 1。
    bid, ask : 数值, optional
        买价与卖价（报价方的视角）：bid 是报价方买入 1 单位 base 支付的 quote（你卖出 base 得到的价格），
        ask 是报价方卖出 1 单位 base 收取的 quote（你买入 base 支付的价格）。须满足 0 < bid ≤ rate ≤ ask。
    as_of_date : date、datetime 或 ISO 字符串, optional
        汇率日期。
    """

    __slots__ = ("_base_currency", "_quote_currency", "_rate", "_bid", "_ask", "_as_of_date")

    def __init__(self, base_currency: CurrencyLike, quote_currency: CurrencyLike, rate: Any, *, bid: Any = None,
                 ask: Any = None, as_of_date: Optional[DateLike] = None) -> None:
        base, quote = get_currency(base_currency), get_currency(quote_currency)
        mid = self._check_rate(rate, "rate")
        bid = None if bid is None else self._check_rate(bid, "bid")
        ask = None if ask is None else self._check_rate(ask, "ask")
        if bid is not None and np.any(bid > mid):
            raise ValueError(f"bid ({bid}) 不能高于中间价 ({mid})")
        if ask is not None and np.any(ask < mid):
            raise ValueError(f"ask ({ask}) 不能低于中间价 ({mid})")
        if base == quote and any(np.any(x != 1) for x in (mid, bid, ask) if x is not None):
            raise ValueError(f"同一币种 {base.code}/{quote.code} 的汇率必须为 1")
        for name, value in (("_base_currency", base), ("_quote_currency", quote), ("_rate", mid), ("_bid", bid),
                            ("_ask", ask), ("_as_of_date", to_date(as_of_date, "as_of_date", optional=True))):
            object.__setattr__(self, name, value)

    @staticmethod
    def parse_currency_pair(pair: Any) -> Tuple[Currency, Currency]:
        """'USD/CNY' → (USD, CNY)；必须恰好一个 '/'，两侧为已注册的币种代码。"""
        if not isinstance(pair, str):
            raise TypeError(f"货币对必须是 'BASE/QUOTE' 形式的 str，收到 {type(pair).__name__}")
        parts = pair.split("/")
        if len(parts) != 2:
            raise ValueError(f"货币对必须是 'BASE/QUOTE' 形式，收到 {pair!r}")
        return get_currency(parts[0]), get_currency(parts[1])

    @staticmethod
    def _check_rate(value: Any, what: str) -> FixedArray:
        """正的、非缺失的数值（标量或数组）；接受构造 FixedArray 时可用的所有输入。"""
        fa = value if isinstance(value, FixedArray) else FixedArray(value)
        if fa.mask is not None:
            raise ValueError(f"{what} 不能是缺失值")
        if np.any(fa.mantissa <= 0):
            raise ValueError(f"{what} 必须为正数，收到 {fa}")
        return fa

    @classmethod
    def from_pair(cls, pair: str, rate: Any, **kwargs: Any) -> "ExchangeRate":
        """由 'BASE/QUOTE' 形式的货币对构造，如 ExchangeRate.from_pair("USD/CNY", "7.2")。"""
        base, quote = cls.parse_currency_pair(pair)
        return cls(base, quote, rate, **kwargs)

    # ---------- 属性 ----------

    @property
    def base_currency(self) -> Currency:
        """基准货币。"""
        return self._base_currency

    @property
    def quote_currency(self) -> Currency:
        """计价货币。"""
        return self._quote_currency

    @property
    def pair(self) -> str:
        """货币对，如 'USD/CNY'。"""
        return f"{self._base_currency.code}/{self._quote_currency.code}"

    @property
    def rate(self) -> FixedArray:
        """中间价。"""
        return self._rate

    @property
    def bid(self) -> Optional[FixedArray]:
        """买价（未提供为 None）。"""
        return self._bid

    @property
    def ask(self) -> Optional[FixedArray]:
        """卖价（未提供为 None）。"""
        return self._ask

    @property
    def spread(self) -> Optional[FixedArray]:
        """买卖价差 ask − bid（任一未提供为 None）。"""
        return None if self._bid is None or self._ask is None else self._ask - self._bid

    @property
    def as_of_date(self) -> Optional[date]:
        """汇率日期（未提供为 None）。"""
        return self._as_of_date

    # ---------- 换算 ----------

    def _side_rate(self, side: str) -> FixedArray:
        if side not in _SIDES:
            raise ValueError(f"side 只能是 {_SIDES} 之一，收到 {side!r}")
        rate = {"mid": self._rate, "bid": self._bid, "ask": self._ask}[side]
        if rate is None:
            raise ValueError(f"汇率 {self.pair} 没有提供 {side} 价")
        return rate

    def inverse(self) -> "ExchangeRate":
        """反向汇率 QUOTE/BASE：中间价取倒数，买卖价互换后取倒数（精度按 FixedArray 默认除法规则）。"""
        one = FixedArray(1)
        return ExchangeRate(self._quote_currency, self._base_currency, one / self._rate,
                            bid=None if self._ask is None else one / self._ask,
                            ask=None if self._bid is None else one / self._bid, as_of_date=self._as_of_date)

    def convert(self, money: Money, side: str = "mid", rounding: Rounding = "half_up") -> Money:
        """把 base 金额换成 quote（乘以汇率），或把 quote 金额换成 base（直接除以汇率），只舍入一次。

        side 选择所用的价格："mid"（默认）、"bid"、"ask"；例如卖出 base 换 quote 用 bid，用 quote 买入 base 用 ask。
        缺失值保持为缺失值。
        """
        rate, mode = self._side_rate(side), check_rounding(rounding)
        if money.currency == self._base_currency:
            target = self._quote_currency
            data = money.data.mul_round(rate, scale=target.digits, rounding=mode)
        elif money.currency == self._quote_currency:
            target = self._base_currency
            data = money.data.divide(rate, scale=target.digits, rounding=mode)
        else:
            raise TypeError(f"金额币种 {money.currency.code} 与汇率 {self.pair} 不匹配")
        return Money._wrap(data, target)

    # ---------- 不可变与展示 ----------

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(f"ExchangeRate 是不可变对象，禁止设置属性 '{key}'")

    def __delattr__(self, key: str) -> None:
        raise AttributeError(f"ExchangeRate 是不可变对象，禁止删除属性 '{key}'")

    def __repr__(self) -> str:
        extra = "".join(f", {name}='{value}'" for name, value in
                        (("bid", self._bid), ("ask", self._ask), ("as_of_date", self._as_of_date)) if value is not None)
        return f"ExchangeRate('{self._base_currency.code}', '{self._quote_currency.code}', '{self._rate}'{extra})"

    def __str__(self) -> str:
        return f"{self.pair} = {self._rate}"


# ============================================================================
# 汇率数据源
# ============================================================================


class FXProvider(ABC):
    """汇率数据源抽象基类。"""

    @abstractmethod
    def get_rate(self, base: CurrencyLike, quote: CurrencyLike,
                 as_of_date: Optional[DateLike] = None) -> Optional[ExchangeRate]:
        """返回与 as_of_date 严格匹配的汇率：传日期时只用当天的汇率，不传时只用不带日期的汇率；没有时返回 None。"""

    def get_rates_matrix(self, currencies: Sequence[CurrencyLike],
                         as_of_date: Optional[DateLike] = None) -> Dict[Tuple[str, str], ExchangeRate]:
        """currencies 两两之间（含自身）能查到的全部汇率，键为 (base 代码, quote 代码)。"""
        resolved = [get_currency(c) for c in currencies]
        matrix = {}
        for b in resolved:
            for q in resolved:
                rate = self.get_rate(b, q, as_of_date)
                if rate is not None:
                    matrix[(b.code, q.code)] = rate
        return matrix

    @staticmethod
    def _identity(base: Currency, quote: Currency, day: Optional[date]) -> Optional[ExchangeRate]:
        """同一币种的汇率恒为 1；不同币种返回 None。"""
        return ExchangeRate(base, quote, 1, as_of_date=day) if base == quote else None


class UserDictFXProvider(FXProvider):
    """用户提供的汇率表（离线、回测）。

    rates 的键可以是：
    - 'BASE/QUOTE' 或 (base, quote)：不带日期的汇率；
    - (base, quote, 日期)：带日期的汇率；
    - 日期（date、datetime 或 ISO 字符串）：值为 {'BASE/QUOTE': 汇率, ...}，即当天的一组汇率。
    汇率可以是数值或 ExchangeRate（货币对须一致；未带日期时使用键中的日期）。
    同一天同一货币对（无论方向）只能给出一次；反方向的查询自动取倒数。
    """

    def __init__(self, rates: Mapping[Any, Any]) -> None:
        self._rates: Dict[Tuple[str, str, Optional[date]], ExchangeRate] = {}
        for key, value in rates.items():
            if isinstance(key, tuple):
                if len(key) not in (2, 3):
                    raise ValueError(f"元组键必须是 (base, quote) 或 (base, quote, 日期)，收到 {key!r}")
                self._add(get_currency(key[0]), get_currency(key[1]), value,
                          to_date(key[2], "日期") if len(key) == 3 else None)
            elif isinstance(key, str) and "/" in key:
                self._add(*ExchangeRate.parse_currency_pair(key), value, None)
            elif isinstance(key, (str, date)):
                day = to_date(key, "日期键")
                if not isinstance(value, Mapping):
                    raise TypeError(f"日期键 {key!r} 的值必须是 {{'BASE/QUOTE': 汇率}} 字典")
                for pair, rate in value.items():
                    self._add(*ExchangeRate.parse_currency_pair(pair), rate, day)
            else:
                raise TypeError(f"无法识别的汇率键 {key!r}")

    def _add(self, base: Currency, quote: Currency, value: Any, day: Optional[date]) -> None:
        if isinstance(value, ExchangeRate):
            if (value.base_currency, value.quote_currency) != (base, quote):
                raise ValueError(f"汇率 {value.pair} 与键 {base.code}/{quote.code} 不一致")
            if value.as_of_date not in (None, day):
                raise ValueError(f"汇率日期 {value.as_of_date} 与键的日期 {day} 不一致")
            rate = ExchangeRate(base, quote, value.rate, bid=value.bid, ask=value.ask, as_of_date=day)
        else:
            rate = ExchangeRate(base, quote, value, as_of_date=day)
        if (base.code, quote.code, day) in self._rates or (quote.code, base.code, day) in self._rates:
            raise ValueError(f"{base.code}/{quote.code}" + (f" 在 {day}" if day else "") + " 的汇率重复给出")
        self._rates[(base.code, quote.code, day)] = rate

    def get_rate(self, base: CurrencyLike, quote: CurrencyLike,
                 as_of_date: Optional[DateLike] = None) -> Optional[ExchangeRate]:
        b, q, day = get_currency(base), get_currency(quote), to_date(as_of_date, "as_of_date", optional=True)
        if b == q:
            return self._identity(b, q, day)
        direct = self._rates.get((b.code, q.code, day))
        if direct is not None:
            return direct
        reverse = self._rates.get((q.code, b.code, day))
        return None if reverse is None else reverse.inverse()


_SNAPSHOT_RATES: Dict[str, str] = {
    "USD/CNY": "7.2350",
    "EUR/USD": "1.0850",
    "GBP/USD": "1.2750",
    "USD/JPY": "155.50",
    "USD/HKD": "7.8200",
    "USD/SGD": "1.3450",
    "USD/CHF": "0.9050",
    "AUD/USD": "0.6650",
    "USD/CAD": "1.3650",
}


class SnapshotFXProvider(UserDictFXProvider):
    """内置的一组不带日期的参考汇率，仅供示例与离线测试显式使用；数值是过时的固定值，不能用于真实估值。"""

    def __init__(self) -> None:
        super().__init__(_SNAPSHOT_RATES)


class FrankfurterFXProvider(FXProvider):
    """在线数据源：欧洲央行参考汇率（Frankfurter 服务，https://api.frankfurter.dev/v1/）。

    会发起真实的网络请求，只有显式创建并传给 FXEngine 时才会使用。不传日期时取最新汇率；传日期时
    只接受当天发布的汇率（非交易日服务返回的是之前的汇率，视为没有）。查询结果缓存在内存中。
    不支持的币种或日期返回 None；网络错误照常抛出，不会被静默吞掉。
    """

    _API = "https://api.frankfurter.dev/v1/"

    def __init__(self, timeout_seconds: float = 3.0, cache_enabled: bool = True) -> None:
        self._timeout = timeout_seconds
        self._cache: Optional[Dict[Tuple[str, str, Optional[date]], Optional[ExchangeRate]]] = \
            {} if cache_enabled else None

    def get_rate(self, base: CurrencyLike, quote: CurrencyLike,
                 as_of_date: Optional[DateLike] = None) -> Optional[ExchangeRate]:
        b, q, day = get_currency(base), get_currency(quote), to_date(as_of_date, "as_of_date", optional=True)
        if b == q:
            return self._identity(b, q, day)
        key = (b.code, q.code, day)
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        rate = self._fetch(b, q, day)
        if self._cache is not None:
            self._cache[key] = rate
        return rate

    def _fetch(self, base: Currency, quote: Currency, day: Optional[date]) -> Optional[ExchangeRate]:
        query = urllib.parse.urlencode({"base": base.code, "symbols": quote.code})
        url = f"{self._API}{day.isoformat() if day else 'latest'}?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "Moneta"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in (404, 422):  # 不支持的币种或日期
                return None
            raise
        published = to_date(data["date"], "date")
        value = data.get("rates", {}).get(quote.code)
        if value is None or (day is not None and published != day):
            return None
        return ExchangeRate(base, quote, str(value), as_of_date=published)


class CompositeFXProvider(FXProvider):
    """按顺序查询多个数据源，返回第一个查到的汇率。"""

    def __init__(self, providers: Sequence[FXProvider]) -> None:
        for provider in providers:
            if not isinstance(provider, FXProvider):
                raise TypeError(f"providers 只能包含 FXProvider，收到 {type(provider).__name__}")
        self._providers = list(providers)

    def get_rate(self, base: CurrencyLike, quote: CurrencyLike,
                 as_of_date: Optional[DateLike] = None) -> Optional[ExchangeRate]:
        for provider in self._providers:
            rate = provider.get_rate(base, quote, as_of_date)
            if rate is not None:
                return rate
        return None


# ============================================================================
# 换汇引擎
# ============================================================================


class FXEngine:
    """按数据源查询汇率并换汇；没有直盘汇率时，经锚定货币（默认 USD）三角套算。查不到时报错。"""

    def __init__(self, provider: FXProvider, base_anchor_currency: CurrencyLike = "USD") -> None:
        if not isinstance(provider, FXProvider):
            raise TypeError(f"provider 必须是 FXProvider，收到 {type(provider).__name__}")
        self._provider = provider
        self._base_anchor = get_currency(base_anchor_currency)

    @property
    def provider(self) -> FXProvider:
        return self._provider

    @property
    def base_anchor_currency(self) -> Currency:
        return self._base_anchor

    def get_rate(self, base: CurrencyLike, quote: CurrencyLike, as_of_date: Optional[DateLike] = None) -> ExchangeRate:
        """base/quote 的汇率：先查直盘，再经锚定货币套算（交叉汇率为两段汇率的精确乘积）。"""
        b, q, day = get_currency(base), get_currency(quote), to_date(as_of_date, "as_of_date", optional=True)
        direct = self._provider.get_rate(b, q, day)
        if direct is not None:
            return direct
        anchor = self._base_anchor
        if anchor not in (b, q):
            first, second = self._provider.get_rate(b, anchor, day), self._provider.get_rate(anchor, q, day)
            if first is not None and second is not None:
                def product(x: Optional[FixedArray], y: Optional[FixedArray]) -> Optional[FixedArray]:
                    return None if x is None or y is None else x * y

                return ExchangeRate(b, q, first.rate * second.rate, bid=product(first.bid, second.bid),
                                    ask=product(first.ask, second.ask), as_of_date=day)
        raise ExchangeRateNotFoundError(f"未找到 {b.code}/{q.code}" + (f" 在 {day}" if day else "") + " 的汇率")

    def convert(self, money: Money, to_currency: CurrencyLike, as_of_date: Optional[DateLike] = None,
                side: str = "mid", rounding: Rounding = "half_up") -> Money:
        """把金额换成 to_currency（同币种原样返回），规则同 ExchangeRate.convert。"""
        target = get_currency(to_currency)
        if money.currency == target:
            return money
        return self.get_rate(money.currency, target, as_of_date).convert(money, side=side, rounding=rounding)


# ============================================================================
# 外汇金融计算
# ============================================================================


def decompose_fx_pnl(v0: Money, v1: Money, fx0: ExchangeRate, fx1: ExchangeRate,
                     rounding: Rounding = "half_up") -> Tuple[Money, Money, Money]:
    """把外币资产从期初 v0 到期末 v1 的损益（以本币计）分解为三项，三项之和恰好等于总损益：

    - 本地损益 = (v1 − v0) × fx0；
    - 汇兑损益 = v0 × (fx1 − fx0)；
    - 交叉项 = 总损益 − 本地损益 − 汇兑损益，其中总损益 = v1 × fx1 − v0 × fx0。

    fx0、fx1 是外币与本币之间的 ExchangeRate（任一方向，两者的本币须相同）。每次换算都用 ExchangeRate.convert，
    舍入产生的残差由交叉项吸收（汇兑损益与精确值至多相差本币最小单位 1 个）。缺失值保持为缺失值。
    """
    if not isinstance(v0, Money) or not isinstance(v1, Money):
        raise TypeError("v0、v1 必须是 Money")
    if v0.currency != v1.currency:
        raise ValueError(f"期初与期末币种不一致: {v0.currency.code} 与 {v1.currency.code}")
    for fx in (fx0, fx1):
        if not isinstance(fx, ExchangeRate):
            raise TypeError(f"fx0、fx1 必须是 ExchangeRate，收到 {type(fx).__name__}")
    start, end = fx0.convert(v0, rounding=rounding), fx1.convert(v0, rounding=rounding)
    if start.currency != end.currency:
        raise ValueError(f"两个汇率的本币不一致: {start.currency.code} 与 {end.currency.code}")
    total = fx1.convert(v1, rounding=rounding) - start
    local = fx0.convert(v1 - v0, rounding=rounding)
    fx = end - start
    return local, fx, total - local - fx


def covered_interest_parity_forward(spot: ExchangeRate, r_base: Any, r_quote: Any, tenor_years: Any) -> ExchangeRate:
    """抛补利率平价远期汇率（单利）：F = S × (1 + r_quote × T) / (1 + r_base × T)。

    利率与期限为数值（float 按最短写法精确解析）；全程定点计算，商的精度按 FixedArray 默认除法规则，
    买卖价按同一比例换算。
    """
    rb, rq, tenor = (FixedArray._coerce(x) for x in (r_base, r_quote, tenor_years))
    if np.any(tenor.mantissa < 0):
        raise ValueError("tenor_years 不能为负数")
    numerator, denominator = 1 + rq * tenor, 1 + rb * tenor
    if np.any(denominator.mantissa <= 0) or np.any(numerator.mantissa <= 0):
        raise ValueError("1 + 利率 × 期限 必须为正")

    def forward(rate: Optional[FixedArray]) -> Optional[FixedArray]:
        return None if rate is None else (rate * numerator).divide(denominator)

    return ExchangeRate(spot.base_currency, spot.quote_currency, forward(spot.rate), bid=forward(spot.bid),
                        ask=forward(spot.ask), as_of_date=spot.as_of_date)


__all__ = [
    "CompositeFXProvider",
    "ExchangeRate",
    "ExchangeRateNotFoundError",
    "FXEngine",
    "FXProvider",
    "FrankfurterFXProvider",
    "SnapshotFXProvider",
    "UserDictFXProvider",
    "covered_interest_parity_forward",
    "decompose_fx_pnl",
]
