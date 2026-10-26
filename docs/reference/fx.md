# fx — 外汇汇率、多源汇率引擎、汇兑损益分解与远期平价

`moneta.fx` 模块提供精确的定点数外汇汇率表示（[`ExchangeRate`](#exchangeratebase_currency-quote_currency-rate--bidnone-asknone-as_of_datenone)）、可插拔的多源汇率提供者体系（[`FXProvider`](#fxprovider)、[`UserDictFXProvider`](#userdictfxprovider)、[`SnapshotFXProvider`](#snapshotfxprovider)、[`FrankfurterFXProvider`](#frankfurterfxprovider)、[`CompositeFXProvider`](#compositefxprovider)）、支持锚定货币精确三角套算的汇率引擎（[`FXEngine`](#fxengineprovider-base_anchor_currencyusd)），以及汇兑损益三因子分解（[`decompose_fx_pnl`](#decompose_fx_pnl)）与抛补利率平价远期汇率计算（[`covered_interest_parity_forward`](#covered_interest_parity_forward)）。

```python
>>> from moneta import Money, CNY, EUR, JPY, USD
>>> from moneta.fx import (
...     ExchangeRate, FXEngine, UserDictFXProvider,
...     covered_interest_parity_forward, decompose_fx_pnl,
... )
>>> rate = ExchangeRate("USD", "CNY", "7.2", bid="7.19", as_of_date="2024-01-02")
>>> rate.pair, repr(rate)
('USD/CNY', "ExchangeRate('USD', 'CNY', '7.2', bid='7.19', as_of_date='2024-01-02')")
>>> rate.convert(Money("100.00", "USD")), rate.convert(Money("720.00", "CNY"))
(Money(FixedArray(720.00, scale=2), currency='CNY'), Money(FixedArray(100.00, scale=2), currency='USD'))
>>> engine = FXEngine(UserDictFXProvider({"EUR/USD": "1.085", "USD/JPY": "155.5"}))
>>> engine.get_rate("EUR", "JPY")  # 经锚定货币 USD 精确套算：1.085 × 155.5 = 168.7175
ExchangeRate('EUR', 'JPY', '168.7175')
>>> decompose_fx_pnl(Money("1000", "USD"), Money("1200", "USD"), ExchangeRate("USD", "CNY", "7.0"), ExchangeRate("USD", "CNY", "7.2"))
(Money(FixedArray(1400.00, scale=2), currency='CNY'), Money(FixedArray(200.00, scale=2), currency='CNY'), Money(FixedArray(40.00, scale=2), currency='CNY'))
>>> covered_interest_parity_forward(ExchangeRate("USD", "CNY", "7.2"), 0.05, 0.02, 1)
ExchangeRate('USD', 'CNY', '6.99428571429')
```

[← API 参考](index.md)

## 核心机制与设计约束

### 1. 货币对语义、买卖价差不变量与正反向单次舍入换算（`ExchangeRate`）

- **标准金融报价语义（`BASE/QUOTE`）**：`ExchangeRate(base_currency, quote_currency, rate, *, bid=None, ask=None, as_of_date=None)` 表示 **$1$ 单位基准货币（`base_currency`）兑换 `rate` 单位计价货币（`quote_currency`）**。货币对与中间价在前，可选的买价 `bid`、卖价 `ask` 与日期 `as_of_date` 只能通过**关键字参数**传入；同时提供 [`from_pair("USD/CNY", rate, ...)`](#exchangeratefrom_pair) 类方法与公开静态方法 [`parse_currency_pair`](#exchangerateparse_currency_pair)。
- **严格校验不变量**：
  1. `rate`、`bid`、`ask` 均通过 [`_check_rate`](#exchangerate_check_rate-与-exchangerate_side_rate) 转换为非缺失、严格大于 `0` 的 [`FixedArray`](fixed_array.md) 定点数（不能有缺失值，且所有元素必须 $> 0$）；
  2. 买卖价与中间价必须满足 **$0 < \text{bid} \le \text{rate} \le \text{ask}$**（`bid` 是报价方买入 1 单位 `base` 支付的 `quote`，即卖出 `base` 得到的价格；`ask` 是报价方卖出 1 单位 `base` 收取的 `quote`，即买入 `base` 支付的价格）；
  3. 同一币种（`base_currency == quote_currency`）的汇率必须恒为 `1`（若提供了 `bid` 或 `ask` 也必须为 `1`）。
- **正反向单次舍入换算（[`convert`](#exchangerateconvert)）**：
  - `side` 只能是 `"mid"`（默认）、`"bid"` 或 `"ask"`，缺少对应价格时直接抛出 `ValueError`；
  - **正方向（`money.currency == base_currency`）**：乘以所选价格 `rate`，按 `quote_currency.digits` 舍入一次（例如卖出 `base` 换 `quote` 传 `side="bid"`）；
  - **反方向（`money.currency == quote_currency`）**：**直接除以所选价格 `rate`**，按 `base_currency.digits` 舍入一次，不经过 `inverse()` 的二次舍入（例如用 `quote` 买入 `base` 传 `side="ask"`）；缺失值 `<NA>` 保持为缺失值。
- **倒数汇率（[`inverse`](#exchangerateinverse)）**：返回反向汇率 `QUOTE/BASE`，中间价取倒数 `1 / rate`，**买卖价互换后取倒数**（新 `bid = 1 / ask`，新 `ask = 1 / bid`），商的精度均遵循 [`FixedArray`](fixed_array.md#fixedarraydivide) 的默认除法规则。

### 2. 数据源责任链与严格日期匹配契约（`FXProvider`）

所有汇率数据源均继承自抽象基类 [`FXProvider`](#fxprovider)，并遵循统一的**严格按日期匹配规则**：

- **严格按日期匹配**：传入 `as_of_date` 时只使用**当天**的汇率，不传时只使用**不带日期**的汇率；查不到时数据源返回 `None`（由 [`FXEngine`](#fxengineprovider-base_anchor_currencyusd) 抛出 [`ExchangeRateNotFoundError`](#exchangeratenotfounderror)）。绝不向前沿用前一天的旧汇率，不会使用未来汇率，也不会静默回退到快照或网络数据。
- **内置数据源对比**：

| 数据源类 | 适用场景 | 键形式与方向处理 | 冲突、异常与缓存行为 |
|---|---|---|---|
| [`UserDictFXProvider`](#userdictfxprovider) | 离线估值、回测与自定义汇率表注入 | 支持三种键形式：① `'BASE/QUOTE'` 或 `(base, quote)`（不带日期）；② `(base, quote, 日期)`（带日期）；③ `日期: {'BASE/QUOTE': 汇率, ...}`（当天一组汇率）。反方向查询自动取 `.inverse()` | **同一天同一货币对重复给出（无论方向）立即抛 `ValueError`** |
| [`SnapshotFXProvider`](#snapshotfxprovider) | 示例与离线测试（只能显式使用） | 继承自 `UserDictFXProvider`，内置 9 组不带日期的参考汇率（`_SNAPSHOT_RATES`）；传入具体日期时返回 `None` | 只能显式构造传给 `FXEngine`，不能用于真实估值 |
| [`FrankfurterFXProvider`](#frankfurterfxprovider) | 在线查询欧洲央行（ECB）参考汇率 | 不传日期查 `/latest`，传日期查 `/{YYYY-MM-DD}`；**返回的发布日期 `published` 与请求日期不同时（如非交易日返回前一交易日）视为没有并返回 `None`** | HTTP `404` / `422` 返回 `None`；其他网络错误照常抛出；查询结果缓存在内存中（`_cache`） |
| [`CompositeFXProvider`](#compositefxprovider) | 多数据源按优先级组合 | 按顺序查询各数据源，返回**第一个查到的汇率** | 校验所有元素均为 `FXProvider` |

### 3. 必填数据源与精确三角套算（`FXEngine`）

[`FXEngine(provider, base_anchor_currency="USD")`](#fxengineprovider-base_anchor_currencyusd) 的 `provider` 参数必填，按以下顺序解析 `base/quote` 在 `as_of_date` 的汇率：

1. **直盘查询**：先调用 `provider.get_rate(base, quote, day)`（同币种恒返回 `1`，反向汇率由数据源自动取倒数）；
2. **经锚定货币精确乘积套算**：若没有直盘汇率且 `anchor`（默认 `USD`）不等于 `base` 或 `quote`，则查询两段汇率 `first = provider.get_rate(base, anchor, day)` 与 `second = provider.get_rate(anchor, quote, day)`，并用**两段汇率的精确乘积**合成交叉汇率：
   $$\text{cross\_rate} = \text{first.rate} \times \text{second.rate}$$
   例如 `EUR/USD = 1.085` 与 `USD/JPY = 155.5` 套算得到 `1.085 × 155.5 = 168.7175`（若两段均带有 `bid` 或 `ask`，买卖价也同样取精确乘积 `first.bid * second.bid`、`first.ask * second.ask`）；
3. **查不到时报错**：若直盘与套算均未查到，抛出 [`ExchangeRateNotFoundError`](#exchangeratenotfounderror)（`KeyError` 的子类）。

### 4. 守恒汇兑损益三因子分解与抛补利率平价

- **汇兑损益分解（[`decompose_fx_pnl(v0, v1, fx0, fx1, rounding="half_up")`](#decompose_fx_pnl)）**：
  - 不需要传 `target_currency`，也不接受纯数字汇率：`fx0` 与 `fx1` 是外币与本币之间的 `ExchangeRate`（任一方向均可，只要两者换算出的本币一致）；
  - 每次换算均调用 `ExchangeRate.convert`：
    - 本地损益 `local = fx0.convert(v1 - v0, rounding=rounding)`；
    - 汇兑损益 `fx = fx1.convert(v0, rounding=rounding) - fx0.convert(v0, rounding=rounding)`（与理论精确值 `v0 × (fx1 - fx0)` 最多相差 $1$ 个最小货币单位）；
    - 交叉项 `cross = total - local - fx`，其中总损益 `total = fx1.convert(v1, rounding=rounding) - fx0.convert(v0, rounding=rounding)`；
  - **三项之和精确等于总损益**（`local + fx + cross == total`），舍入残差由交叉项吸收。
- **抛补利率平价远期汇率（[`covered_interest_parity_forward(spot, r_base, r_quote, tenor_years)`](#covered_interest_parity_forward)）**：
  - 按单利公式 $F = S \times (1 + r_{\text{quote}} \cdot T) / (1 + r_{\text{base}} \cdot T)$ 全程定点计算；
  - 商的精度遵循 [`FixedArray`](fixed_array.md#fixedarraydivide) 的默认除法规则（例如 `7.2 × 1.02 / 1.05 = 6.99428571429`），买卖价按同一比例换算。

## API 速查表

| 分类 | 接口 / 属性 / 异常 | 功能说明 |
|---|---|---|
| **汇率对象** | [`ExchangeRate(base_currency, quote_currency, rate, *, bid=None, ask=None, as_of_date=None)`](#exchangeratebase_currency-quote_currency-rate--bidnone-asknone-as_of_datenone) | 构造不可变的货币对汇率对象（货币对在前，`bid`/`ask`/`as_of_date` 仅限关键字参数） |
| | [`ExchangeRate.from_pair`](#exchangeratefrom_pair) | 由 `"USD/CNY"` 形式的货币对字符串构造 `ExchangeRate` |
| | [`ExchangeRate.parse_currency_pair`](#exchangerateparse_currency_pair) | 解析 `"BASE/QUOTE"` 字符串并返回 `(base_currency, quote_currency)` 二元组 |
| | [`base_currency`、`quote_currency`、`pair`、`rate`、`bid`、`ask`、`spread`、`as_of_date`](#exchangerate-核心属性与文本表示) | 货币对、中间价、买卖价、买卖价差（`ask - bid`）与汇率日期只读属性及不可变保护 |
| | [`ExchangeRate.inverse`](#exchangerateinverse) | 计算反向汇率 `QUOTE/BASE`（买卖价互换后取倒数，精度按默认除法规则） |
| | [`ExchangeRate.convert`](#exchangerateconvert) | 按 `mid` / `bid` / `ask` 将 `Money` 正向乘或反向直接除单次舍入换算为另一币种 |
| **异常类型** | [`ExchangeRateNotFoundError`](#exchangeratenotfounderror) | 数据源中没有所需货币对（在所需日期）的汇率时抛出的异常（`KeyError` 的子类） |
| **汇率数据源** | [`FXProvider`](#fxprovider) | 汇率数据源抽象基类（定义 `get_rate` 与 `get_rates_matrix`） |
| | [`UserDictFXProvider`](#userdictfxprovider) | 用户字典汇率数据源（支持三种键形式，拦截同一天同一货币对正反向重复给出） |
| | [`SnapshotFXProvider`](#snapshotfxprovider) | 内置不带日期的参考快照数据源（仅供示例与离线测试显式使用） |
| | [`FrankfurterFXProvider`](#frankfurterfxprovider) | 欧洲央行（Frankfurter 服务）在线汇率数据源（严格校验发布日期、内存缓存） |
| | [`CompositeFXProvider`](#compositefxprovider) | 多数据源组合器（按顺序查询，返回第一个查到的汇率） |
| **换汇引擎** | [`FXEngine(provider, base_anchor_currency="USD")`](#fxengineprovider-base_anchor_currencyusd) | 按数据源查汇率并换汇；无直盘时经锚定货币用两段汇率精确乘积三角套算 |
| | [`FXEngine.get_rate`](#fxengineget_rate) / [`FXEngine.convert`](#fxengineconvert) | 查询直盘或交叉套算汇率 / 将 `Money` 换算为目标币种 |
| **外汇金融计算** | [`decompose_fx_pnl`](#decompose_fx_pnl) | 将外币资产跨期本币总损益守恒分解为 `(本地损益, 汇兑损益, 交叉项)` |
| | [`covered_interest_parity_forward`](#covered_interest_parity_forward) | 基于抛补利率平价（CIP）计算理论远期汇率 |
| **内部辅助（私有 API）** | [`DateLike`、`CurrencyLike`、`Rounding`、`_SIDES`、`_SNAPSHOT_RATES`](#模块级类型别名与内部常量) | 模块内部类型注解别名、合法报价档位元组与内置参考汇率表 |
| | [`ExchangeRate._check_rate` 与 `ExchangeRate._side_rate`](#exchangerate_check_rate-与-exchangerate_side_rate) | 正数非缺失汇率校验 / 按 `side` 提取报价 |
| | [`FXProvider._identity`、`UserDictFXProvider._add`、`FrankfurterFXProvider._fetch`](#数据源内部辅助方法) | 同币种恒等汇率构造、字典条目解析去重与 HTTP 请求抓取 |

## 汇率对象（`ExchangeRate`）

### `ExchangeRate(base_currency, quote_currency, rate, *, bid=None, ask=None, as_of_date=None)`

```python
class moneta.fx.ExchangeRate(
    base_currency,
    quote_currency,
    rate,
    *,
    bid=None,
    ask=None,
    as_of_date=None,
)
```

构造不可变的货币对汇率对象，表示 `1` 单位基准货币 `base_currency` 兑换 `rate` 单位计价货币 `quote_currency`（同时由顶层 `moneta` 与 `moneta.fx` 导出）。

**参数**

- **base_currency** : *Currency 或 str* — 基准货币（如 `"USD"` 或 [`USD`](currency.md)），通过 [`get_currency`](currency.md#get_currency) 解析。
- **quote_currency** : *Currency 或 str* — 计价货币（如 `"CNY"` 或 [`CNY`](currency.md)），通过 [`get_currency`](currency.md#get_currency) 解析。
- **rate** : *数值或字符串* — 中间价，必须为正且不能有缺失值；同一币种时必须为 `1`。
- **bid** : *数值或字符串，仅关键字参数，可选* — 买价（报价方买入 1 单位 `base` 支付的 `quote`，即卖出 `base` 得到的价格），须满足 `0 < bid <= rate`。
- **ask** : *数值或字符串，仅关键字参数，可选* — 卖价（报价方卖出 1 单位 `base` 收取的 `quote`，即买入 `base` 支付的价格），须满足 `rate <= ask`。
- **as_of_date** : *date, datetime, str 或 None，仅关键字参数，可选* — 汇率日期，通过 [`to_date(as_of_date, "as_of_date", optional=True)`](_utils.md#to_date) 解析为 `datetime.date` 或 `None`。

**返回值**

- **out** : *ExchangeRate* — 不可变的汇率实例。

**异常**

- **TypeError** — `bid`、`ask`、`as_of_date` 未使用关键字传入，或币种/数值/日期类型非法。
- **KeyError** — `base_currency` 或 `quote_currency` 是未注册的货币代码。
- **ValueError** — `rate`、`bid` 或 `ask` 含有缺失值或 $\le 0$；违反 `bid <= rate <= ask`；或当 `base_currency == quote_currency` 时 `rate`、`bid`、`ask` 不等于 `1`。

**示例**

```python
>>> from datetime import date
>>> from moneta import ExchangeRate, CNY, USD
>>> r = ExchangeRate("USD", "CNY", "7.2", bid="7.19", as_of_date="2024-01-02")
>>> (r.base_currency is USD, r.quote_currency is CNY, r.rate, r.as_of_date)
(True, True, FixedArray(7.2, scale=1), datetime.date(2024, 1, 2))
>>> ExchangeRate("USD", "USD", 1)
ExchangeRate('USD', 'USD', '1')
>>> ExchangeRate("USD", "CNY", "7.20", bid="7.25")  # 违反 bid <= rate
Traceback (most recent call last):
    ...
ValueError: bid (7.25) 不能高于中间价 (7.20)
>>> ExchangeRate("USD", "USD", "1.01")  # 同一币种汇率必须为 1
Traceback (most recent call last):
    ...
ValueError: 同一币种 USD/USD 的汇率必须为 1
```

**另请参阅**

- [`ExchangeRate.from_pair`](#exchangeratefrom_pair)
- [`ExchangeRate.parse_currency_pair`](#exchangerateparse_currency_pair)
- [`ExchangeRate.convert`](#exchangerateconvert)
- [`ExchangeRate.inverse`](#exchangerateinverse)

### `ExchangeRate.from_pair`

```python
@classmethod
ExchangeRate.from_pair(pair, rate, **kwargs)
```

由 `"BASE/QUOTE"` 形式的货币对字符串构造 [`ExchangeRate`](#exchangeratebase_currency-quote_currency-rate--bidnone-asknone-as_of_datenone)，内部先调用 [`parse_currency_pair(pair)`](#exchangerateparse_currency_pair) 解析出 `(base, quote)`，再调用 `cls(base, quote, rate, **kwargs)`。

**参数**

- **pair** : *str* — 形如 `"USD/CNY"` 的货币对字符串。
- **rate, \*\*kwargs** — 中间价 `rate` 以及可选的关键字参数 `bid`、`ask`、`as_of_date`。

**返回值**

- **out** : *ExchangeRate*

**示例**

```python
>>> from moneta.fx import ExchangeRate
>>> ExchangeRate.from_pair("USD/CNY", "7.2", bid="7.19", ask="7.21")
ExchangeRate('USD', 'CNY', '7.2', bid='7.19', ask='7.21')
```

### `ExchangeRate.parse_currency_pair`

```python
@staticmethod
ExchangeRate.parse_currency_pair(pair)
```

解析 `"BASE/QUOTE"` 形式的货币对字符串（如 `"USD/CNY"`），返回 `(base_currency, quote_currency)` 二元组。字符串中必须恰好包含一个 `'/'`，且两侧均为已注册的币种代码。

**参数**

- **pair** : *str* — `'BASE/QUOTE'` 形式的货币对字符串。

**返回值**

- **base, quote** : *Tuple[Currency, Currency]*

**异常**

- **TypeError** — `pair` 不是 `str`（报 `f"货币对必须是 'BASE/QUOTE' 形式的 str，收到 {type(pair).__name__}"`）。
- **ValueError** — `pair.split("/")` 的段数不等于 `2`（报 `f"货币对必须是 'BASE/QUOTE' 形式，收到 {pair!r}"`）。
- **KeyError** — 解析出的币种代码（包括空字符串 `""`）未在 [`CurrencyRegistry`](currency.md#currencyregistry) 中注册。

**示例**

```python
>>> from moneta import CNY, USD
>>> from moneta.fx import ExchangeRate
>>> ExchangeRate.parse_currency_pair(" usd / cny ") == (USD, CNY)
True
>>> ExchangeRate.parse_currency_pair("USDCNY")
Traceback (most recent call last):
    ...
ValueError: 货币对必须是 'BASE/QUOTE' 形式，收到 'USDCNY'
>>> ExchangeRate.parse_currency_pair("USD/")
Traceback (most recent call last):
    ...
KeyError: "未注册的货币代码: ''"
>>> ExchangeRate.parse_currency_pair(123)
Traceback (most recent call last):
    ...
TypeError: 货币对必须是 'BASE/QUOTE' 形式的 str，收到 int
```

### `ExchangeRate` 核心属性与文本表示

```python
rate.base_currency   # Currency
rate.quote_currency  # Currency
rate.pair            # str ("BASE/QUOTE")
rate.rate            # FixedArray
rate.bid             # Optional[FixedArray]
rate.ask             # Optional[FixedArray]
rate.spread          # Optional[FixedArray] (ask - bid)
rate.as_of_date      # Optional[datetime.date]
str(rate)
repr(rate)
```

`ExchangeRate` 通过六个只读槽位 `__slots__ = ("_base_currency", "_quote_currency", "_rate", "_bid", "_ask", "_as_of_date")` 存储状态，构造后完全**不可变**（重写了 `__setattr__` 与 `__delattr__`，任何修改或删除均抛出 `AttributeError`）：

- **`base_currency`** / **`quote_currency`**：基准货币与计价货币 [`Currency`](currency.md) 单例。
- **`pair`**：货币对字符串 `f"{self._base_currency.code}/{self._quote_currency.code}"`。
- **`rate`** / **`bid`** / **`ask`**：中间价、买价与卖价（[`FixedArray`](fixed_array.md)；未提供 `bid` 或 `ask` 时为 `None`）。
- **`spread`**：买卖价差 `self._ask - self._bid`（`bid` 或 `ask` 任一未提供时为 `None`）。
- **`as_of_date`**：汇率日期（`datetime.date`，未提供时为 `None`）。
- **`str(rate)`**：返回 `f"{self.pair} = {self._rate}"`。
- **`repr(rate)`**：返回形如 `"ExchangeRate('USD', 'CNY', '7.2', bid='7.19', as_of_date='2024-01-02')"` 的构造表示（仅附加非 `None` 的 `bid`、`ask`、`as_of_date` 关键字项）。

**示例**

```python
>>> from moneta.fx import ExchangeRate
>>> r = ExchangeRate("USD", "CNY", "7.2", bid="7.19", as_of_date="2024-01-02")
>>> str(r), repr(r)
('USD/CNY = 7.2', "ExchangeRate('USD', 'CNY', '7.2', bid='7.19', as_of_date='2024-01-02')")
>>> r.spread is None
True
>>> r2 = ExchangeRate("USD", "CNY", "7.20", bid="7.19", ask="7.22")
>>> r2.spread
FixedArray(0.03, scale=2)
>>> r.rate = 8
Traceback (most recent call last):
    ...
AttributeError: ExchangeRate 是不可变对象，禁止设置属性 'rate'
```

### `ExchangeRate.inverse`

```python
ExchangeRate.inverse()
```

返回反向汇率 `QUOTE/BASE`：中间价取倒数 `FixedArray(1) / self._rate`，**买卖价互换后取倒数**（`bid = 1 / self._ask`，`ask = 1 / self._bid`），精度按 [`FixedArray`](fixed_array.md#fixedarraydivide) 默认除法规则（默认保留 `scale + 8` 位小数，放不下 `int64` 时少保留，最后去掉末尾 `0`）。

**返回值**

- **inv** : *ExchangeRate* — 保留原 `as_of_date` 的反向汇率实例。

**示例**

```python
>>> from moneta.fx import ExchangeRate
>>> r = ExchangeRate("USD", "CNY", "8", bid="7.8", ask="8.0", as_of_date="2024-01-02")
>>> inv = r.inverse()
>>> inv
ExchangeRate('CNY', 'USD', '0.125', bid='0.125', ask='0.128205128', as_of_date='2024-01-02')
>>> bool(inv.bid <= inv.rate <= inv.ask)
True
```

### `ExchangeRate.convert`

```python
ExchangeRate.convert(money, side="mid", rounding="half_up")
```

把 `base_currency` 金额换成 `quote_currency`（乘以汇率），或把 `quote_currency` 金额换成 `base_currency`（直接除以汇率），全程**只舍入一次**；缺失值 `<NA>` 保持为缺失值。

**参数**

- **money** : *Money* — 待换算的金额张量，其币种必须等于 `self.base_currency` 或 `self.quote_currency`。
- **side** : *`{"mid", "bid", "ask"}`，默认 `"mid"`* — 选择所用的价格档位（例如卖出 `base` 换 `quote` 用 `"bid"`，用 `quote` 买入 `base` 用 `"ask"`）。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 换算到目标币种精度（`target.digits`）时的舍入规则。

**返回值**

- **out** : *Money* — 换算后的目标币种金额。

**异常**

- **ValueError** — `side` 不是 `("mid", "bid", "ask")` 之一，或当前汇率没有提供所选的 `bid` / `ask` 价。
- **TypeError** — `money` 的币种与当前汇率的货币对不匹配。

**示例**

正反向单次舍入换算、买卖价选择与缺失值保持：

```python
>>> from moneta import Money
>>> from moneta.fx import ExchangeRate
>>> r = ExchangeRate("USD", "CNY", "7.20", bid="7.18", ask="7.22")
>>> r.convert(Money("100.00", "USD"))  # 正向 mid: 100 × 7.20 = 720.00 CNY
Money(FixedArray(720.00, scale=2), currency='CNY')
>>> r.convert(Money("100.00", "USD"), side="bid")  # 卖出 USD 换 CNY 用 bid (7.18)
Money(FixedArray(718.00, scale=2), currency='CNY')
>>> r.convert(Money("722.00", "CNY"), side="ask")  # 用 CNY 买入 USD 用 ask (7.22): 722 / 7.22 = 100.00 USD
Money(FixedArray(100.00, scale=2), currency='USD')
>>> ExchangeRate("CNY", "USD", "0.3").convert(Money("1000.10", "USD"))  # 反向直接除一次：1000.10 / 0.3 = 3333.67 CNY
Money(FixedArray(3333.67, scale=2), currency='CNY')
>>> r.convert(Money([None, "10.00"], "USD"))
Money(FixedArray([<NA>, 72.00], scale=2), currency='CNY')
```

缺少对应报价或 `side` 非法时的异常拦截：

```python
>>> ExchangeRate("USD", "CNY", "7.20").convert(Money("100", "USD"), side="bid")
Traceback (most recent call last):
    ...
ValueError: 汇率 USD/CNY 没有提供 bid 价
>>> r.convert(Money("100", "USD"), side="last")
Traceback (most recent call last):
    ...
ValueError: side 只能是 ('mid', 'bid', 'ask') 之一，收到 'last'
```

**另请参阅**

- [`FXEngine.convert`](#fxengineconvert)
- [`_convert_at_rate`](lot.md#_convert_at_rate)

## 汇率异常与数据源体系（`FXProvider`）

### `ExchangeRateNotFoundError`

```python
class moneta.fx.ExchangeRateNotFoundError(KeyError)
```

数据源中没有所需货币对（在所需日期）的汇率时抛出的异常（`KeyError` 的子类，同时由顶层 `moneta` 与 `moneta.fx` 导出）。

**示例**

```python
>>> from moneta import ExchangeRateNotFoundError
>>> issubclass(ExchangeRateNotFoundError, KeyError)
True
```

### `FXProvider`

```python
class moneta.fx.FXProvider(ABC)
```

汇率数据源抽象基类。子类实现抽象方法 `get_rate(base, quote, as_of_date=None)`；基类同时提供 `get_rates_matrix` 方法与 `_identity` 静态方法。

**核心方法**

- **`get_rate(base, quote, as_of_date=None)`**（抽象方法）：返回与 `as_of_date` 严格匹配的汇率（传入日期时只用当天的汇率，不传时只用不带日期的汇率；没有时返回 `None`）。
- **`get_rates_matrix(currencies, as_of_date=None)`**：返回 `currencies` 两两之间（含自身）能查到的全部汇率字典 `Dict[Tuple[str, str], ExchangeRate]`，键为 `(base.code, quote.code)`。

**示例**

```python
>>> from moneta.fx import UserDictFXProvider
>>> p = UserDictFXProvider({"USD/CNY": "7.20"})
>>> mat = p.get_rates_matrix(["USD", "CNY", "EUR"])
>>> sorted(mat.keys())
[('CNY', 'CNY'), ('CNY', 'USD'), ('EUR', 'EUR'), ('USD', 'CNY'), ('USD', 'USD')]
```

### `UserDictFXProvider`

```python
class moneta.fx.UserDictFXProvider(rates)
```

用户提供的内存汇率表数据源（适用于离线核算与回测）。

**参数**

- **rates** : *Mapping[Any, Any]* — 汇率字典，键支持以下三种形式：
  1. **`'BASE/QUOTE'` 或 `(base, quote)`**：不带日期的汇率（如 `"USD/CNY": "7.2"` 或 `(USD, CNY): "7.2"`）；
  2. **`(base, quote, 日期)`**：带日期的三元组键（如 `("USD", "CNY", "2024-01-02"): "7.10"`）；
  3. **`日期`（`date`、`datetime` 或 ISO 字符串）**：值为 `{'BASE/QUOTE': 汇率, ...}` 字典，表示当天的一组汇率。
  - 字典的值可以是数值/字符串，也可以是 `ExchangeRate` 实例：其货币对必须与键一致；当 `ExchangeRate.as_of_date` 为 `None` 时自动使用键中的日期（不带日期的键则为 `None`），若自带日期（`as_of_date` 非 `None`），则该日期必须与键的日期完全一致（对不带日期的键即不能自带日期），否则抛出 `ValueError`。

**严格日期匹配与重复检测**

- **同一天同一货币对（无论方向）只能给出一次**：若在同一日期（或同为无日期）下重复给出同一货币对（例如同时给出 `"USD/CNY"` 与 `("USD", "CNY")`，或同时给出正向 `"USD/CNY"` 与反向 `"CNY/USD"`），构造器立即抛出 `ValueError`。
- **反方向自动取倒数**：查询 `get_rate(base, quote, as_of_date)` 时，若正向未存储但反向已存储，自动返回反向汇率的 `.inverse()`。

**示例**

三种键形式、正向与反向查询及严格日期匹配：

```python
>>> from moneta import CNY, EUR, USD
>>> from moneta.fx import ExchangeRate, UserDictFXProvider
>>> p = UserDictFXProvider({
...     "USD/CNY": "7.25",
...     (EUR, USD): ExchangeRate("EUR", "USD", "1.085", bid="1.084", ask="1.086"),
...     ("USD", "JPY", "2024-01-02"): "155.0",
...     "2024-01-02": {"USD/CNY": "7.10"},
... })
>>> p.get_rate("USD", "CNY"), p.get_rate("USD", "CNY", "2024-01-02"), p.get_rate("USD", "CNY", "2024-01-03")
(ExchangeRate('USD', 'CNY', '7.25'), ExchangeRate('USD', 'CNY', '7.10', as_of_date='2024-01-02'), None)
>>> p.get_rate("CNY", "USD")  # 反方向查询自动取倒数
ExchangeRate('CNY', 'USD', '0.1379310345')
```

同一天同一货币对重复给出（无论方向）或 `ExchangeRate` 日期与键不一致时报错：

```python
>>> UserDictFXProvider({"USD/CNY": "7.2", "CNY/USD": "0.14"})
Traceback (most recent call last):
    ...
ValueError: CNY/USD 的汇率重复给出
>>> UserDictFXProvider({"2024-01-02": {"USD/CNY": "7.1"}, ("USD", "CNY", "2024-01-02"): "7.2"})
Traceback (most recent call last):
    ...
ValueError: USD/CNY 在 2024-01-02 的汇率重复给出
>>> UserDictFXProvider({"USD/CNY": ExchangeRate("USD", "CNY", 7, as_of_date="2024-01-02")})
Traceback (most recent call last):
    ...
ValueError: 汇率日期 2024-01-02 与键的日期 None 不一致
```

### `SnapshotFXProvider`

```python
class moneta.fx.SnapshotFXProvider()
```

内置的一组不带日期的参考汇率数据源（继承自 [`UserDictFXProvider`](#userdictfxprovider)），**仅供示例与离线测试显式使用**；数值是过时的固定值，不能用于真实估值。

**内置汇率表与日期行为**

- 内置 `_SNAPSHOT_RATES` 包含 9 组无日期汇率：`"USD/CNY": "7.2350"`、`"EUR/USD": "1.0850"`、`"GBP/USD": "1.2750"`、`"USD/JPY": "155.50"`、`"USD/HKD": "7.8200"`、`"USD/SGD": "1.3450"`、`"USD/CHF": "0.9050"`、`"AUD/USD": "0.6650"`、`"USD/CAD": "1.3650"`。
- 由于全部为不带日期的汇率，传入非空 `as_of_date` 时（不同币种）始终返回 `None`。

**示例**

```python
>>> from moneta.fx import SnapshotFXProvider
>>> snap = SnapshotFXProvider()
>>> snap.get_rate("USD", "CNY"), snap.get_rate("EUR", "USD")
(ExchangeRate('USD', 'CNY', '7.2350'), ExchangeRate('EUR', 'USD', '1.0850'))
>>> snap.get_rate("USD", "CNY", as_of_date="2024-01-02") is None
True
```

### `FrankfurterFXProvider`

```python
class moneta.fx.FrankfurterFXProvider(
    timeout_seconds=3.0,
    cache_enabled=True,
)
```

在线数据源：查询欧洲央行参考汇率（基于 Frankfurter 服务 `https://api.frankfurter.dev/v1/`）。只有显式创建并传给 [`FXEngine`](#fxengineprovider-base_anchor_currencyusd) 时才会发起网络请求。

**参数**

- **timeout_seconds** : *float，默认 `3.0`* — HTTP 请求超时时间（秒）。
- **cache_enabled** : *bool，默认 `True`* — 是否将查询结果缓存在内存字典 `_cache` 中。

**查询、缓存与错误处理规则**

- **严格校验发布日期**：不传 `as_of_date` 时请求 `/latest`；传入 `as_of_date` 时请求 `/{day.isoformat()}`，并校验响应中的 `"date"` 字段 `published == day`——若非交易日服务返回的是之前交易日的汇率（`published != day`），视为当天没有汇率并返回 `None`。
- **HTTP 错误分流**：遇到 HTTP `404` 或 `422`（不支持的币种或日期）时返回 `None`；其他网络错误或 HTTP 异常照常抛出，不会被静默吞掉。
- **内存缓存**：启用缓存时，`(base.code, quote.code, day)` 的查询结果（包括 `None`）会缓存在 `self._cache` 中避免重复网络请求。

**示例**

```python
>>> from moneta.fx import FrankfurterFXProvider
>>> provider = FrankfurterFXProvider(timeout_seconds=3.0, cache_enabled=True)
>>> provider.get_rate("USD", "USD")
ExchangeRate('USD', 'USD', '1')
```

### `CompositeFXProvider`

```python
class moneta.fx.CompositeFXProvider(providers)
```

按顺序查询多个 [`FXProvider`](#fxprovider) 数据源，返回**第一个查到的汇率**；若所有数据源均返回 `None` 则返回 `None`。

**参数**

- **providers** : *Sequence[FXProvider]* — 数据源序列（若包含非 `FXProvider` 元素则抛出 `TypeError`）。

**示例**

```python
>>> from moneta.fx import CompositeFXProvider, SnapshotFXProvider, UserDictFXProvider
>>> primary = UserDictFXProvider({"USD/CNY": "7.18"})
>>> fallback = SnapshotFXProvider()
>>> comp = CompositeFXProvider([primary, fallback])
>>> comp.get_rate("USD", "CNY")  # 命中 primary
ExchangeRate('USD', 'CNY', '7.18')
>>> comp.get_rate("USD", "JPY")  # primary 未命中，回退命中 fallback
ExchangeRate('USD', 'JPY', '155.50')
```

## 换汇引擎（`FXEngine`）

### `FXEngine(provider, base_anchor_currency="USD")`

```python
class moneta.fx.FXEngine(provider, base_anchor_currency="USD")
```

按数据源查询汇率并换汇；没有直盘汇率时，经锚定货币（默认 `USD`）三角套算；查不到时抛出 [`ExchangeRateNotFoundError`](#exchangeratenotfounderror)。

**参数**

- **provider** : *FXProvider* — **必填**的汇率数据源实例（不接受 `None`）。
- **base_anchor_currency** : *Currency 或 str，默认 `"USD"`* — 用于交叉套算的中心锚定货币。

**属性**

- **`engine.provider`** : *FXProvider* — 绑定的汇率数据源。
- **`engine.base_anchor_currency`** : *Currency* — 锚定货币 [`Currency`](currency.md) 单例。

**异常**

- **TypeError** — `provider` 不是 `FXProvider` 实例，或 `base_anchor_currency` 类型非法。
- **KeyError** — `base_anchor_currency` 是未注册的货币代码。

**示例**

```python
>>> from moneta import USD
>>> from moneta.fx import FXEngine, SnapshotFXProvider
>>> engine = FXEngine(SnapshotFXProvider())
>>> engine.base_anchor_currency is USD
True
>>> FXEngine(None)
Traceback (most recent call last):
    ...
TypeError: provider 必须是 FXProvider，收到 NoneType
```

### `FXEngine.get_rate`

```python
FXEngine.get_rate(base, quote, as_of_date=None)
```

查询 `base/quote` 在 `as_of_date` 的汇率：先查数据源直盘（含反向），若未命中且 `base`、`quote` 均不等于 `self.base_anchor_currency`，则经锚定货币查询 `first = (base -> anchor)` 与 `second = (anchor -> quote)`，并以**两段汇率的精确乘积** `first.rate * second.rate` 合成交叉汇率（若两段均带有 `bid` 或 `ask`，买卖价也取精确乘积）。

**参数**

- **base** : *Currency 或 str* — 基准货币。
- **quote** : *Currency 或 str* — 计价货币。
- **as_of_date** : *date, datetime, str 或 None，默认 `None`* — 汇率日期（严格按日期匹配）。

**返回值**

- **rate** : *ExchangeRate*

**异常**

- **ExchangeRateNotFoundError** — 直盘与经锚定货币套算均未找到对应日期的汇率。

**示例**

```python
>>> from moneta.fx import ExchangeRate, FXEngine, UserDictFXProvider
>>> p = UserDictFXProvider({
...     "EUR/USD": ExchangeRate("EUR", "USD", "1.085", bid="1.084", ask="1.086"),
...     "USD/JPY": ExchangeRate("USD", "JPY", "155.5", bid="155.4", ask="155.6"),
... })
>>> engine = FXEngine(p)
>>> engine.get_rate("EUR", "JPY")  # 1.085 × 155.5 = 168.7175
ExchangeRate('EUR', 'JPY', '168.7175', bid='168.4536', ask='168.9816')
>>> engine.get_rate("EUR", "JPY", as_of_date="2024-01-02")
Traceback (most recent call last):
    ...
moneta.fx.ExchangeRateNotFoundError: '未找到 EUR/JPY 在 2024-01-02 的汇率'
```

### `FXEngine.convert`

```python
FXEngine.convert(
    money,
    to_currency,
    as_of_date=None,
    side="mid",
    rounding="half_up",
)
```

把金额 `money` 换成 `to_currency`（同币种时原样返回 `money`），换算规则同 [`ExchangeRate.convert`](#exchangerateconvert)。

**参数**

- **money** : *Money* — 待换算的金额（标量或数组，支持缺失值 `<NA>`）。
- **to_currency** : *Currency 或 str* — 目标币种。
- **as_of_date** : *date, datetime, str 或 None，默认 `None`* — 汇率日期。
- **side** : *`{"mid", "bid", "ask"}`，默认 `"mid"`* — 报价档位。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入规则。

**返回值**

- **out** : *Money*

**示例**

```python
>>> from moneta import Money
>>> from moneta.fx import FXEngine, UserDictFXProvider
>>> engine = FXEngine(UserDictFXProvider({"EUR/USD": "1.085", "USD/JPY": "155.5"}))
>>> engine.convert(Money(["100.00", "200.00"], "EUR"), "JPY")  # 100 × 168.7175 -> 16872 JPY
Money(FixedArray([16872, 33744], scale=0), currency='JPY')
```

## 外汇金融计算

### `decompose_fx_pnl`

```python
moneta.fx.decompose_fx_pnl(v0, v1, fx0, fx1, rounding="half_up")
```

把外币资产从期初 `v0` 到期末 `v1` 的损益（以本币计）分解为**本地损益**、**汇兑损益**与**交叉项**三项，三项之和恰好等于总损益：

- **本地损益** $= (v_1 - v_0) \times \text{fx}_0$；
- **汇兑损益** $= v_0 \times (\text{fx}_1 - \text{fx}_0)$（由 `fx1.convert(v0) - fx0.convert(v0)` 计算，与理论精确值至多相差本币最小单位 1 个）；
- **交叉项** $=$ 总损益 $-$ 本地损益 $-$ 汇兑损益，其中总损益 $= v_1 \times \text{fx}_1 - v_0 \times \text{fx}_0$。

**参数**

- **v0** : *Money* — 期初外币金额（标量或张量，支持缺失值 `<NA>`）。
- **v1** : *Money* — 期末外币金额（须与 `v0` 同币种）。
- **fx0** : *ExchangeRate* — 期初外币与本币之间的 `ExchangeRate`（任一方向均可，不接受纯数字汇率）。
- **fx1** : *ExchangeRate* — 期末外币与本币之间的 `ExchangeRate`（任一方向均可，换算出的本币须与 `fx0` 相同）。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 每次调用 `ExchangeRate.convert` 所用的舍入规则。

**返回值**

- **local_pnl, fx_pnl, cross_pnl** : *Tuple[Money, Money, Money]* — 以本币计价的三项损益，满足 `local_pnl + fx_pnl + cross_pnl == total_pnl`。

**异常**

- **TypeError** — `v0`、`v1` 不是 `Money`；`fx0`、`fx1` 不是 `ExchangeRate`；或汇率货币对不包含外币 `v0.currency`。
- **ValueError** — `v0` 与 `v1` 币种不一致，或 `fx0` 与 `fx1` 换算出的目标本币不一致。

**示例**

```python
>>> from moneta import Money
>>> from moneta.fx import ExchangeRate, decompose_fx_pnl
>>> v0, v1 = Money("1000", "USD"), Money("1200", "USD")
>>> fx0, fx1 = ExchangeRate("USD", "CNY", "7.0"), ExchangeRate("USD", "CNY", "7.2")
>>> local_pnl, fx_pnl, cross_pnl = decompose_fx_pnl(v0, v1, fx0, fx1)
>>> (local_pnl, fx_pnl, cross_pnl)
(Money(FixedArray(1400.00, scale=2), currency='CNY'), Money(FixedArray(200.00, scale=2), currency='CNY'), Money(FixedArray(40.00, scale=2), currency='CNY'))
>>> total_pnl = fx1.convert(v1) - fx0.convert(v0)
>>> bool(local_pnl + fx_pnl + cross_pnl == total_pnl)
True
```

### `covered_interest_parity_forward`

```python
moneta.fx.covered_interest_parity_forward(spot, r_base, r_quote, tenor_years)
```

按抛补利率平价（Covered Interest Parity，单利）计算理论远期汇率：

$$F = S \times \frac{1 + r_{\text{quote}} \times T}{1 + r_{\text{base}} \times T}$$

**参数**

- **spot** : *ExchangeRate* — 即期汇率 $S$。
- **r_base** : *int, float, Decimal, Fraction 或 FixedArray* — 基准货币年化利率（数值类型，`float` 按最短十进制写法精确解析）。
- **r_quote** : *int, float, Decimal, Fraction 或 FixedArray* — 计价货币年化利率（数值类型）。
- **tenor_years** : *int, float, Decimal, Fraction 或 FixedArray* — 远期期限 $T$（以年为单位的非负数值，且须满足 $1 + r \times T > 0$）。

**返回值**

- **forward** : *ExchangeRate* — 远期汇率对象。全程定点计算，商 `(rate * (1 + r_quote * T)).divide(1 + r_base * T)` 的精度按 [`FixedArray`](fixed_array.md#fixedarraydivide) 默认除法规则，买卖价 `bid`、`ask` 按同一比例换算。

**异常**

- **TypeError** — `r_base`、`r_quote` 或 `tenor_years` 不是数值类型（如传入字符串或布尔值）。
- **ValueError** — `tenor_years < 0` 或 `1 + 利率 × 期限 <= 0`。

**示例**

```python
>>> from moneta.fx import ExchangeRate, covered_interest_parity_forward
>>> spot = ExchangeRate("USD", "CNY", "7.2", bid="7.19", ask="7.21")
>>> covered_interest_parity_forward(spot, 0.05, 0.02, 1)  # 7.2 × 1.02 / 1.05 = 6.99428571429
ExchangeRate('USD', 'CNY', '6.99428571429', bid='6.984571428571', ask='7.0040')
>>> covered_interest_parity_forward(spot, 0.05, 0.02, 0)  # T = 0 时等于即期汇率
ExchangeRate('USD', 'CNY', '7.200', bid='7.1900', ask='7.2100')
```

**性能对比**

| 外汇与汇率操作 | 耗时 | 说明 |
|---|---:|---|
| `eng.get_rate("USD", "CNY")`（直盘） | 1.70 µs | `UserDictFXProvider` 字典查找 |
| `ExchangeRate("USD", "CNY", ...)` 构造 | 4.08 µs | 含币种解析与买卖价校验 |
| `eng.get_rate("EUR", "JPY")`（经 `USD` 套算） | 8.72 µs | 两段汇率查询 + 精确乘积合成 |
| `r.convert(m_scalar)`（正向 0-d） | 11.1 µs | 定点乘法一次舍入 |
| `r.convert(m_scalar)`（反向 0-d） | 21.2 µs | 定点除法一次舍入 |
| `r.convert(m_1000)`（正向 1,000 元素） | 22.9 µs | 向量化批量换汇 |
| `r.inverse()` | 26.8 µs | 中间价与买卖价互换取倒数 |
| `decompose_fx_pnl(v0, v1, fx0, fx1)` | 52.5 µs | 守恒三因子汇兑损益分解 |
| `covered_interest_parity_forward(...)` | 96.1 µs | 含 `rate`、`bid`、`ask` 三路平价远期解算 |

## 内部辅助函数、方法与常量（私有 API）

### 模块级类型别名与内部常量

```python
moneta.fx.DateLike = Union[str, date, datetime]
moneta.fx.CurrencyLike = Union[Currency, str]
moneta.fx.Rounding = Union[RoundingMode, str]
moneta.fx._SIDES = ("mid", "bid", "ask")
moneta.fx._SNAPSHOT_RATES: Dict[str, str]
```

- **`DateLike`、`CurrencyLike`、`Rounding`**：模块内部用于类型注解的联合类型别名。
- **`_SIDES`**：[`ExchangeRate.convert`](#exchangerateconvert) 支持的三种价格档位元组 `("mid", "bid", "ask")`。
- **`_SNAPSHOT_RATES`**：供 [`SnapshotFXProvider`](#snapshotfxprovider) 使用的 9 组 `'BASE/QUOTE' -> rate_str` 内置参考汇率字典。

### `ExchangeRate._check_rate` 与 `ExchangeRate._side_rate`

```python
@staticmethod
ExchangeRate._check_rate(value, what)
ExchangeRate._side_rate(side)
```

- **`_check_rate(value, what)`**：将输入值转换为 [`FixedArray`](fixed_array.md)，并校验其不含缺失值（`fa.mask is None`）且所有元素严格大于 0（`not np.any(fa.mantissa <= 0)`）。
- **`_side_rate(side)`**：校验 `side in _SIDES` 并返回对应的 `_rate`、`_bid` 或 `_ask`；若对应价格为 `None` 则抛出 `ValueError(f"汇率 {self.pair} 没有提供 {side} 价")`。

**示例**

```python
>>> from moneta.fx import ExchangeRate
>>> ExchangeRate._check_rate("7.25", "rate")
FixedArray(7.25, scale=2)
>>> r = ExchangeRate("USD", "CNY", "7.20", bid="7.18", ask="7.22")
>>> r._side_rate("bid"), r._side_rate("ask")
(FixedArray(7.18, scale=2), FixedArray(7.22, scale=2))
```

### 数据源内部辅助方法

```python
@staticmethod
FXProvider._identity(base, quote, day)
UserDictFXProvider._add(base, quote, value, day)
FrankfurterFXProvider._fetch(base, quote, day)
```

- **`FXProvider._identity(base, quote, day)`**：当 `base == quote` 时返回 `ExchangeRate(base, quote, 1, as_of_date=day)`，否则返回 `None`。
- **`UserDictFXProvider._add(base, quote, value, day)`**：校验并构造 `(base, quote, day)` 的 `ExchangeRate`，检查正向 `(base.code, quote.code, day)` 与反向 `(quote.code, base.code, day)` 是否已在 `self._rates` 中存在，若重复则抛出 `ValueError`，否则存入 `self._rates`。
- **`FrankfurterFXProvider._fetch(base, quote, day)`**：向 `https://api.frankfurter.dev/v1/{day or 'latest'}` 发起带 `"User-Agent": "Moneta"` 请求头的 HTTP GET 请求并解析 JSON；HTTP `404` / `422` 或发布日期 `published != day` 时返回 `None`，其他网络错误原样抛出。
