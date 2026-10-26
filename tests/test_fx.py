"""fx 测试：ExchangeRate、数据源（严格按日期匹配）、FXEngine、汇兑损益分解与远期汇率。

约定（fx.py）：
- ExchangeRate(base, quote, rate, *, bid, ask, as_of_date)：汇率为正、bid ≤ rate ≤ ask、同一币种为 1；不可变。
- convert 正方向乘以汇率、反方向直接除以汇率，只舍入一次，保留缺失值；side 严格校验。
- 数据源按日期严格匹配：传日期只用当天的汇率，不传日期只用不带日期的汇率，查不到返回 None，FXEngine 报错。
"""

import io
import json
import random
import urllib.error
from datetime import date, datetime
from decimal import Decimal
from fractions import Fraction

import pytest

import moneta
import moneta.fx as fx_module
from moneta import Money
from moneta.currency import CNY, EUR, JPY, USD
from moneta.fixed_array import FixedArray as F
from moneta.fx import (CompositeFXProvider, ExchangeRate, ExchangeRateNotFoundError, FrankfurterFXProvider, FXEngine,
                       FXProvider, SnapshotFXProvider, UserDictFXProvider, covered_interest_parity_forward,
                       decompose_fx_pnl)
from moneta.fixed_array import RoundingMode

MODES = [m.value for m in RoundingMode]
HALF = Fraction(1, 2)


def ref_round(x: Fraction, mode: str) -> int:
    floor = x.numerator // x.denominator
    frac = x - floor
    if frac == 0 or mode == "floor":
        return floor
    if mode == "ceil":
        return floor + 1
    if mode == "down":
        return floor + 1 if x < 0 else floor
    if mode == "up":
        return floor if x < 0 else floor + 1
    if frac != HALF:
        return floor + 1 if frac > HALF else floor
    if mode == "half_up":
        return floor + 1 if x > 0 else floor
    return floor if floor % 2 == 0 else floor + 1


def usd(v):
    return Money(v, "USD")


def cny(v):
    return Money(v, "CNY")


# ===========================================================================
# 1. ExchangeRate
# ===========================================================================


class TestExchangeRate:
    def test_construct(self):
        r = ExchangeRate("usd", CNY, "7.2", bid=Decimal("7.19"), ask=7.21, as_of_date=datetime(2024, 1, 2, 9))
        assert (r.base_currency, r.quote_currency, r.pair) == (USD, CNY, "USD/CNY")
        assert (r.rate, r.bid, r.ask, r.spread) == (F("7.2"), F("7.19"), F("7.21"), F("0.02"))
        assert r.as_of_date == date(2024, 1, 2)
        plain = ExchangeRate("USD", "CNY", 7)
        assert plain.bid is None and plain.ask is None and plain.spread is None and plain.as_of_date is None
        same =ExchangeRate.from_pair("EUR/USD", "1.1", as_of_date="2024-01-02")
        assert (same.pair, same.rate, same.as_of_date) == ("EUR/USD", F("1.1"), date(2024, 1, 2))
        assert ExchangeRate("USD", "USD", 1).rate == 1 and ExchangeRate("USD", "CNY", F("7.20")).rate.scale == 2

    @pytest.mark.parametrize("kwargs, err, msg", [
        (dict(rate=0), ValueError, "rate 必须为正数"), (dict(rate=-7), ValueError, "rate 必须为正数"),
        (dict(rate=None), ValueError, "rate 不能是缺失值"), (dict(rate=float("nan")), ValueError, "缺失值"),
        (dict(rate=True), TypeError, "bool"), (dict(rate="abc"), ValueError, "无法解析"),
        (dict(rate="7.2", bid="7.3"), ValueError, "bid .* 不能高于中间价"),
        (dict(rate="7.2", ask="7.1"), ValueError, "ask .* 不能低于中间价"),
        (dict(rate="7.2", bid=0), ValueError, "bid 必须为正数"),
        (dict(rate="7.2", as_of_date=20240102), TypeError, "as_of_date"),
        (dict(rate="7.2", as_of_date="2024-02-30"), ValueError, "as_of_date"),
    ])
    def test_validation(self, kwargs, err, msg):
        with pytest.raises(err, match=msg):
            ExchangeRate("USD", "CNY", **kwargs)

    def test_currency_validation(self):
        with pytest.raises(ValueError, match="同一币种"):
            ExchangeRate("USD", "USD", "7.2")
        with pytest.raises(ValueError, match="同一币种"):
            ExchangeRate("USD", "USD", 1, ask="1.01")
        with pytest.raises(KeyError):
            ExchangeRate("USD", "XYZ", 1)
        with pytest.raises(TypeError):
            ExchangeRate(Decimal("7.2"), "USD", "CNY")  # 不再猜参数顺序
        with pytest.raises(TypeError):
            ExchangeRate("USD", "CNY", "7.2", Decimal("7.1"))  # bid / ask / as_of_date 只能用关键字
        for bad in ("USDCNY", "USD/CNY/EUR", "USD-CNY", 1):
            with pytest.raises((ValueError, TypeError)):
                ExchangeRate.from_pair(bad, 1)

    def test_parse_currency_pair(self):
        assert ExchangeRate.parse_currency_pair("USD/CNY") == (USD, CNY)
        assert ExchangeRate.parse_currency_pair("eur/usd") == (EUR, USD)
        with pytest.raises(TypeError, match="货币对必须是 'BASE/QUOTE' 形式的 str"):
            ExchangeRate.parse_currency_pair(123)
        with pytest.raises(ValueError, match="货币对必须是 'BASE/QUOTE' 形式"):
            ExchangeRate.parse_currency_pair("USDCNY")
        with pytest.raises(ValueError, match="货币对必须是 'BASE/QUOTE' 形式"):
            ExchangeRate.parse_currency_pair("USD/CNY/EUR")

    def test_check_rate(self):
        assert ExchangeRate._check_rate("7.2", "rate") == F("7.2")
        assert ExchangeRate._check_rate(F("7.2"), "rate") == F("7.2")
        with pytest.raises(ValueError, match="rate 不能是缺失值"):
            ExchangeRate._check_rate(None, "rate")
        with pytest.raises(ValueError, match="rate 必须为正数"):
            ExchangeRate._check_rate(0, "rate")
        with pytest.raises(ValueError, match="rate 必须为正数"):
            ExchangeRate._check_rate("-1.5", "rate")
        assert (ExchangeRate._check_rate(["7.2", "7.3"], "rate") == F(["7.2", "7.3"])).all()
        with pytest.raises(ValueError, match="bid 必须为正数"):
            ExchangeRate._check_rate(["7.2", "0"], "bid")  # 数组中任一元素不为正
        with pytest.raises(ValueError, match="ask 不能是缺失值"):
            ExchangeRate._check_rate(F(["7.2", None]), "ask")  # FixedArray 原样使用，同样检查缺失值

    def test_immutable_and_display(self):
        r = ExchangeRate("USD", "CNY", "7.2", bid="7.19", as_of_date="2024-01-02")
        with pytest.raises(AttributeError, match="禁止设置属性"):
            r._rate = F(1)
        with pytest.raises(AttributeError, match="禁止删除属性"):
            del r._rate
        assert repr(r) == "ExchangeRate('USD', 'CNY', '7.2', bid='7.19', as_of_date='2024-01-02')"
        assert repr(ExchangeRate("USD", "CNY", 7)) == "ExchangeRate('USD', 'CNY', '7')"
        assert str(r) == "USD/CNY = 7.2"

    def test_inverse(self):
        r = ExchangeRate("USD", "CNY", "8", bid="7.9", ask="8.1", as_of_date="2024-01-02").inverse()
        assert (r.pair, r.rate, r.as_of_date) == ("CNY/USD", F("0.125"), date(2024, 1, 2))
        assert r.bid == F(1) / F("8.1") and r.ask == F(1) / F("7.9") and r.bid <= r.rate <= r.ask  # 买卖价互换
        assert ExchangeRate("USD", "CNY", 3).inverse().rate == F("0.33333333")

    def test_convert(self):
        r = ExchangeRate("USD", "CNY", "7.2", bid="7.19", ask="7.21")
        assert r.convert(usd(100)) == cny(720) and r.convert(usd(100), side="bid") == cny(719)
        assert r.convert(cny(720)) == usd(100) and r.convert(cny(721), side="ask") == usd(100)
        assert (r.convert(usd(["1.01", "2"]), rounding="down") == cny(["7.27", "14.40"])).all()
        got = r.convert(usd([None, "1"]))
        assert got.isna().tolist() == [True, False] and got.currency is CNY  # 保留缺失值
        jpy = ExchangeRate("USD", "JPY", "155.123457")
        assert jpy.convert(Money("1000000000", "JPY")) == usd("6446478.30")  # 反方向直接除，只舍入一次
        with pytest.raises(TypeError, match="不匹配"):
            r.convert(Money(1, "EUR"))
        with pytest.raises(ValueError, match="side 只能是"):
            r.convert(usd(1), side="BID")
        with pytest.raises(ValueError, match="没有提供 bid"):
            ExchangeRate("USD", "CNY", "7.2").convert(usd(1), side="bid")
        with pytest.raises(ValueError, match="rounding"):
            r.convert(usd(1), rounding="nearest")

    def test_convert_random_against_fraction(self):
        rng = random.Random(5)
        for _ in range(2000):
            rate = Fraction(rng.randint(1, 10 ** 7), 10 ** rng.randint(0, 6))
            amount = rng.randint(-10 ** 12, 10 ** 12)
            mode = rng.choice(MODES)
            r = ExchangeRate("USD", "JPY", F(Decimal(rate.numerator) / Decimal(rate.denominator)))
            assert int(r.convert(Money.from_units(amount, "USD"), rounding=mode).units) == \
                ref_round(Fraction(amount, 100) * rate, mode)
            assert int(r.convert(Money.from_units(amount, "JPY"), rounding=mode).units) == \
                ref_round(Fraction(amount) / rate * 100, mode)


# ===========================================================================
# 2. 数据源
# ===========================================================================


class TestUserDictProvider:
    def provider(self):
        return UserDictFXProvider({
            "USD/CNY": "7.25",
            ("EUR", "USD"): "1.08",
            "2024-01-02": {"USD/CNY": "7.10", "EUR/USD": "1.10"},
            date(2024, 1, 3): {"USD/CNY": ExchangeRate("USD", "CNY", "7.12", bid="7.11")},
            ("USD", "CNY", "2024-01-05"): "7.15",
        })

    def test_undated_and_dated_lookups(self):
        p = self.provider()
        assert p.get_rate("USD", "CNY").rate == F("7.25") and p.get_rate(EUR, USD).rate == F("1.08")
        assert p.get_rate("USD", "CNY", "2024-01-02").rate == F("7.10")
        assert p.get_rate("USD", "CNY", datetime(2024, 1, 3, 8)).bid == F("7.11")
        assert p.get_rate("USD", "CNY", date(2024, 1, 5)).as_of_date == date(2024, 1, 5)
        assert p.get_rate("CNY", "USD", "2024-01-02").rate == F(1) / F("7.10")  # 反方向取倒数
        assert p.get_rate("USD", "USD", "2024-01-02").rate == 1 and p.get_rate(CNY, CNY).rate == 1

    def test_strict_date_matching(self):
        p = self.provider()
        assert p.get_rate("USD", "CNY", "2024-01-04") is None  # 不向前沿用 1 月 3 日
        assert p.get_rate("USD", "CNY", "2023-12-31") is None  # 不使用未来汇率
        assert p.get_rate("USD", "CNY", "2030-01-01") is None
        assert p.get_rate("EUR", "USD", "2024-01-03") is None  # 不回落到不带日期的汇率
        only_dated = UserDictFXProvider({"2024-01-02": {"USD/CNY": "7.1"}})
        assert only_dated.get_rate("USD", "CNY") is None  # 不传日期不会取某一天的汇率
        assert p.get_rate("USD", "JPY") is None

    @pytest.mark.parametrize("rates, err, msg", [
        ({"USDCNY": "7.2"}, ValueError, "日期键 不是合法的 ISO 日期"),  # 写错的货币对不会被静默丢弃
        ({"USD-CNY": "7.2"}, ValueError, "日期键"),
        ({"USD/CNY/EUR": "7.2"}, ValueError, "BASE/QUOTE"),
        ({"USD/XYZ": "7.2"}, KeyError, "未注册"),
        ({("USD",): "7.2"}, ValueError, "元组键"),
        ({"2024-01-02": "7.2"}, TypeError, "字典"),
        ({1: "7.2"}, TypeError, "无法识别"),
        ({"USD/CNY": "7.2", ("USD", "CNY"): "7.3"}, ValueError, "重复"),
        ({"USD/CNY": "7.2", "CNY/USD": "0.14"}, ValueError, "重复"),  # 两个方向也算重复
        ({"USD/CNY": -1}, ValueError, "正数"),
        ({"USD/CNY": ExchangeRate("CNY", "USD", 1)}, ValueError, "不一致"),
        ({"2024-01-02": {"USD/CNY": ExchangeRate("USD", "CNY", 7, as_of_date="2024-01-03")}}, ValueError, "日期"),
    ])
    def test_invalid_tables(self, rates, err, msg):
        with pytest.raises(err, match=msg):
            UserDictFXProvider(rates)

    def test_rates_matrix(self):
        m = self.provider().get_rates_matrix(["USD", CNY, "EUR"])
        assert set(m) == {("USD", "USD"), ("CNY", "CNY"), ("EUR", "EUR"), ("USD", "CNY"), ("CNY", "USD"),
                          ("EUR", "USD"), ("USD", "EUR")}
        dated = self.provider().get_rates_matrix(["USD", "CNY"], "2024-01-05")
        assert set(dated) == {("USD", "USD"), ("CNY", "CNY"), ("USD", "CNY"), ("CNY", "USD")}

    def test_snapshot_is_explicit_and_undated(self):
        snap = SnapshotFXProvider()
        assert snap.get_rate("USD", "CNY").rate == F("7.2350") and snap.get_rate("USD", "CNY", "2024-01-02") is None
        assert isinstance(snap, UserDictFXProvider)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestFrankfurterProvider:
    def install(self, monkeypatch, payloads):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request.full_url, timeout))
            body = payloads.pop(0)
            if isinstance(body, Exception):
                raise body
            return FakeResponse(json.dumps(body).encode())

        monkeypatch.setattr(fx_module.urllib.request, "urlopen", fake_urlopen)
        return calls

    def test_latest_and_dated(self, monkeypatch):
        calls = self.install(monkeypatch, [{"date": "2024-05-03", "rates": {"CNY": 7.2369}},
                                           {"date": "2024-01-02", "rates": {"JPY": 141.5}}])
        p = FrankfurterFXProvider(timeout_seconds=2.0)
        latest = p.get_rate("usd", "CNY")
        assert latest.rate == F("7.2369") and latest.as_of_date == date(2024, 5, 3)
        assert p.get_rate("USD", "CNY") is latest  # 内存缓存，不再请求
        dated = p.get_rate(EUR, JPY, "2024-01-02")
        assert dated.rate == F("141.5") and dated.as_of_date == date(2024, 1, 2)
        assert calls == [("https://api.frankfurter.dev/v1/latest?base=USD&symbols=CNY", 2.0),
                         ("https://api.frankfurter.dev/v1/2024-01-02?base=EUR&symbols=JPY", 2.0)]
        assert p.get_rate("USD", "USD") .rate == 1 and len(calls) == 2  # 同币种不发请求

    def test_strict_date_and_missing(self, monkeypatch):
        self.install(monkeypatch, [{"date": "2024-01-05", "rates": {"CNY": 7.1}},  # 1 月 6 日是周六：返回的是周五
                                   {"date": "2024-01-02", "rates": {}},
                                   urllib.error.HTTPError("u", 404, "not found", None, None)])
        p = FrankfurterFXProvider(cache_enabled=False)
        assert p.get_rate("USD", "CNY", "2024-01-06") is None
        assert p.get_rate("USD", "CNY", "2024-01-02") is None
        assert p.get_rate("USD", "CNY", "1900-01-01") is None

    def test_errors_are_not_swallowed(self, monkeypatch):
        self.install(monkeypatch, [urllib.error.URLError("offline"), urllib.error.HTTPError("u", 500, "boom", None, None)])
        p = FrankfurterFXProvider()
        with pytest.raises(urllib.error.URLError):
            p.get_rate("USD", "CNY")
        with pytest.raises(urllib.error.HTTPError):
            p.get_rate("USD", "EUR")


class TestCompositeProvider:
    def test_first_hit_wins(self):
        first = UserDictFXProvider({"USD/CNY": "7.1"})
        second = UserDictFXProvider({"USD/CNY": "7.2", "EUR/USD": "1.1"})
        c = CompositeFXProvider([first, second])
        assert c.get_rate("USD", "CNY").rate == F("7.1") and c.get_rate("EUR", "USD").rate == F("1.1")
        assert c.get_rate("USD", "JPY") is None and ("EUR", "USD") in c.get_rates_matrix(["EUR", "USD"])
        with pytest.raises(TypeError, match="FXProvider"):
            CompositeFXProvider([first, {"USD/CNY": "7"}])
        assert issubclass(CompositeFXProvider, FXProvider)


# ===========================================================================
# 3. FXEngine
# ===========================================================================


class TestEngine:
    def engine(self):
        return FXEngine(UserDictFXProvider({
            "USD/CNY": "7.2000", "EUR/USD": "1.0850", "USD/JPY": "155.5",
            "2024-01-02": {"USD/CNY": "7.10", "EUR/USD": "1.10"},
        }))

    def test_direct_and_cross(self):
        e = self.engine()
        assert e.get_rate("USD", "CNY").rate == F("7.2") and e.get_rate("CNY", "USD").rate == F(1) / F("7.2000")
        cross = e.get_rate("EUR", "JPY")
        assert cross.pair == "EUR/JPY" and cross.rate == F("168.7175")  # 两段汇率的精确乘积
        assert e.get_rate("EUR", "CNY", "2024-01-02").rate == F("7.81")
        assert e.get_rate(USD, USD).rate == 1 and e.base_anchor_currency is USD and isinstance(e.provider, UserDictFXProvider)

    def test_cross_bid_ask(self):
        e = FXEngine(UserDictFXProvider({"EUR/USD": ExchangeRate("EUR", "USD", "1.1", bid="1.09", ask="1.11"),
                                         "USD/JPY": ExchangeRate("USD", "JPY", "150", bid="149", ask="151")}))
        r = e.get_rate("EUR", "JPY")
        assert (r.rate, r.bid, r.ask) == (F("165"), F("162.41"), F("167.61"))
        assert FXEngine(UserDictFXProvider({"EUR/USD": "1.1", "USD/JPY": ExchangeRate("USD", "JPY", "150", bid="149")})
                        ).get_rate("EUR", "JPY").bid is None

    def test_not_found_is_an_error(self):
        e = self.engine()
        with pytest.raises(ExchangeRateNotFoundError, match="USD/CNY 在 2024-01-03"):
            e.get_rate("USD", "CNY", "2024-01-03")  # 严格按日期：没有就报错
        with pytest.raises(ExchangeRateNotFoundError, match="EUR/JPY 在 2024-01-02"):
            e.get_rate("EUR", "JPY", "2024-01-02")  # 套算的一段缺失
        with pytest.raises(KeyError):
            e.get_rate("USD", "GBP")
        with pytest.raises(TypeError, match="FXProvider"):
            FXEngine("USD/CNY")
        with pytest.raises(TypeError):
            FXEngine()  # provider 必填

    def test_convert(self):
        e = self.engine()
        assert e.convert(usd(100), "CNY") == cny(720) and e.convert(cny(720), USD) == usd(100)
        assert e.convert(usd(100), "CNY", as_of_date="2024-01-02") == cny(710)
        assert e.convert(usd(7), USD) == usd(7)
        assert e.convert(Money(100, "EUR"), "JPY", rounding="down") == Money(16871, "JPY")
        with pytest.raises(ExchangeRateNotFoundError):
            e.convert(usd(1), "CNY", "2024-01-03")


# ===========================================================================
# 4. 汇兑损益分解与远期汇率
# ===========================================================================


class TestDecomposeFxPnl:
    def test_example(self):
        local, fx, cross = decompose_fx_pnl(usd(1000), usd(1200), ExchangeRate("USD", "CNY", "7.0"),
                                            ExchangeRate("USD", "CNY", "7.2"))
        assert (local, fx, cross) == (cny(1400), cny(200), cny(40))
        local2, fx2, cross2 = decompose_fx_pnl(usd(1000), usd(1200), ExchangeRate("CNY", "USD", "0.125"),
                                               ExchangeRate("CNY", "USD", "0.1"))  # 反方向：8 → 10
        assert (local2, fx2, cross2) == (cny(1600), cny(2000), cny(400))

    def test_conservation_random(self):
        rng = random.Random(9)
        for _ in range(500):
            v0, v1 = (Money.from_units(rng.randint(0, 10 ** 9), "USD") for _ in range(2))
            r0, r1 = (Decimal(rng.randint(1, 10 ** 6)).scaleb(-rng.randint(0, 5)) for _ in range(2))
            fx0 = ExchangeRate("USD", "JPY", r0) if rng.random() < 0.5 else ExchangeRate("JPY", "USD", r0)
            fx1 = ExchangeRate("USD", "JPY", r1) if rng.random() < 0.5 else ExchangeRate("JPY", "USD", r1)
            mode = rng.choice(MODES)
            local, fx, cross = decompose_fx_pnl(v0, v1, fx0, fx1, rounding=mode)
            total = fx1.convert(v1, rounding=mode) - fx0.convert(v0, rounding=mode)
            assert local + fx + cross == total and local.currency is JPY
            f0, f1 = (Fraction(r) if f.base_currency is USD else 1 / Fraction(r) for r, f in ((r0, fx0), (r1, fx1)))
            assert local.units == ref_round(Fraction(int((v1 - v0).units), 100) * f0, mode)  # 本地损益只舍入一次
            assert abs(int(fx.units) - Fraction(int(v0.units), 100) * (f1 - f0)) <= 1  # 汇兑损益至多差 1 个最小单位

    def test_high_precision_inverse_rates(self):
        # 两个 inverse() 得到的 12 位小数汇率：旧实现精确计算 d0·d1 会超出 int64
        fx0, fx1 = ExchangeRate("USD", "CNY", "7.2").inverse(), ExchangeRate("USD", "CNY", "7.3").inverse()
        v0, v1 = usd("1000000000"), usd("1100000000")
        local, fx, cross = decompose_fx_pnl(v0, v1, fx0, fx1)
        assert local + fx + cross == fx1.convert(v1) - fx0.convert(v0)
        assert local == fx0.convert(v1 - v0) and fx == fx1.convert(v0) - fx0.convert(v0)

    def test_missing_values_and_errors(self):
        local, fx, cross = decompose_fx_pnl(usd([None, 1000]), usd([None, 1200]), ExchangeRate("USD", "CNY", 7),
                                            ExchangeRate("USD", "CNY", "7.2"))
        assert local.isna().tolist() == [True, False] and cross.isna().tolist() == [True, False]
        with pytest.raises(ValueError, match="币种不一致"):
            decompose_fx_pnl(usd(1), cny(1), ExchangeRate("USD", "CNY", 7), ExchangeRate("USD", "CNY", 7))
        with pytest.raises(ValueError, match="本币不一致"):
            decompose_fx_pnl(usd(1), usd(2), ExchangeRate("USD", "CNY", 7), ExchangeRate("USD", "JPY", 150))
        with pytest.raises(TypeError, match="不匹配"):
            decompose_fx_pnl(usd(1), usd(2), ExchangeRate("EUR", "CNY", 7), ExchangeRate("USD", "CNY", 7))
        with pytest.raises(TypeError, match="ExchangeRate"):
            decompose_fx_pnl(usd(1), usd(2), "7.0", ExchangeRate("USD", "CNY", 7))
        with pytest.raises(TypeError, match="Money"):
            decompose_fx_pnl(1, usd(2), ExchangeRate("USD", "CNY", 7), ExchangeRate("USD", "CNY", 7))


class TestCoveredInterestParity:
    def test_forward(self):
        spot = ExchangeRate("USD", "CNY", "7.2", bid="7.19", ask="7.21", as_of_date="2024-01-02")
        fwd = covered_interest_parity_forward(spot, 0.05, 0.02, 1)
        assert fwd.pair == "USD/CNY" and fwd.as_of_date == date(2024, 1, 2)
        assert fwd.rate == F("6.99428571429") and fwd.rate.scale == 11  # 7.344 / 1.05，按默认除法精度（3 + 8 位）
        assert fwd.bid == (F("7.19") * F("1.02")).divide(F("1.05")) and fwd.ask > fwd.rate > fwd.bid
        half_year = covered_interest_parity_forward(ExchangeRate("EUR", "USD", "1.1"), Decimal("0.03"),
                                                    Decimal("0.05"), Decimal("0.5"))
        assert half_year.rate == (F("1.1") * F("1.025")).divide(F("1.015"))
        assert covered_interest_parity_forward(spot, 0, 0, 0).rate == spot.rate

    def test_validation(self):
        spot = ExchangeRate("USD", "CNY", "7.2")
        with pytest.raises(ValueError, match="不能为负数"):
            covered_interest_parity_forward(spot, 0.05, 0.02, -1)
        with pytest.raises(ValueError, match="必须为正"):
            covered_interest_parity_forward(spot, -2, 0.02, 1)
        with pytest.raises(TypeError):
            covered_interest_parity_forward(spot, "0.05", 0.02, 1)


# ===========================================================================
# 5. 导出
# ===========================================================================


def test_exports():
    assert set(fx_module.__all__) == {"CompositeFXProvider", "ExchangeRate", "ExchangeRateNotFoundError", "FXEngine",
                                      "FXProvider", "FrankfurterFXProvider", "SnapshotFXProvider",
                                      "UserDictFXProvider", "covered_interest_parity_forward", "decompose_fx_pnl"}
    for name in fx_module.__all__:
        assert getattr(moneta, name) is getattr(fx_module, name)
    for removed in ("LiveFXWatcher", "convert_currency"):
        assert not hasattr(moneta, removed) and not hasattr(fx_module, removed)
    assert issubclass(ExchangeRateNotFoundError, KeyError)
