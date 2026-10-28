# PositionLot 与 LotMatchingRule — 持仓批次核算与平仓结转规则

`moneta.PositionLot` 表示单笔不可变的现金或证券持仓成本批次（Tax / Accounting Lot），`moneta.LotMatchingRule` 定义先进先出（`FIFO`）、后进先出（`LIFO`）与加权平均成本（`WAC`）三种平仓结转规则，`moneta.lot.match_lots` 则在保持总数量与总成本严格守恒的前提下从多笔同标的批次中扣减指定数量。

每笔批次同时显式记录**计价货币总成本 `cost`** 与**组合基准货币总成本 `cost_base`**（不做任何隐含的 1:1 换汇），支持最大余数法（Hare-Niemeyer）无损拆分（[`split`](#positionlotsplit)）、跨币种单次舍入盯市估值（[`market_value`](#positionlotmarket_value)）、浮动/已实现盈亏（[`profit_loss` / `pnl`](#positionlotprofit_loss-与-positionlotpnl)）以及持有期收益率（[`return_rate`](#positionlotreturn_rate)）核算。

```python
>>> from moneta import Instrument, LotMatchingRule, Money, PositionLot, Price, Quantity, CNY, USD
>>> from moneta.fx import ExchangeRate
>>> from moneta.lot import match_lots
>>> aapl = Instrument("AAPL", "USD")
>>> lot = PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money("1800.00", "USD"), cost_base=Money("12600.00", "CNY"), label="B-01")
>>> lot.asset is aapl, lot.is_cash, lot.base_currency is CNY, lot.holding_days("2024-03-02")
(True, False, True, 60)
>>> fx_now = ExchangeRate("USD", "CNY", "7.20")
>>> lot.market_value(Price("200.00", "USD"), fx=fx_now), lot.pnl(Price("200.00", "USD"), fx=fx_now)
(Money(FixedArray(14400.00, scale=2), currency='CNY'), Money(FixedArray(1800.00, scale=2), currency='CNY'))
>>> part, rest = lot.split(Quantity(3))  # 按 3:7 比例守恒拆分成本
>>> (part.cost, rest.cost, bool(part.cost + rest.cost == lot.cost))
(Money(FixedArray(540.00, scale=2), currency='USD'), Money(FixedArray(1260.00, scale=2), currency='USD'), True)
>>> lots = [PositionLot(aapl, Quantity(10), "2024-01-01", cost=Money("1000", "USD")), PositionLot(aapl, Quantity(5), "2024-02-01", cost=Money("600", "USD"))]
>>> closed, remaining = match_lots(lots, Quantity(12), LotMatchingRule.FIFO)
>>> ([c.amount for c in closed], [r.amount for r in remaining])
([Quantity(FixedArray(10, scale=0), unit='share'), Quantity(FixedArray(2, scale=0), unit='share')], [Quantity(FixedArray(3, scale=0), unit='share')])
```

[← API 参考](index.md)

## 核心机制与设计约束

### 1. 现金与证券双类批次契约及六槽位不可变结构

每个 [`PositionLot`](#positionlotasset-amount-acquired_date-costnone-cost_basenone-label) 实例包含六个只读槽位 `__slots__ = ("_asset", "_amount", "_acquired_date", "_cost", "_cost_base", "_label")`，在构造时根据 `asset` 的类型执行严格的分流安检：

| 维度 | 现金批次（`asset` 为 [`Currency`](currency.md)） | 证券批次（`asset` 为 [`Instrument`](instrument.md)） |
|---|---|---|
| **`is_cash`** | `True` | `False` |
| **持有数量 `amount`** | 必须是币种等于 `asset` 的 0 维正数非缺失 [`Money`](money.md) | 必须是单位与 `asset.base_unit` 满足 [`units_compatible`](_units.md#units_compatible) 的 0 维正数非缺失 [`Quantity`](quantity.md)；若传入通配单位 `"Any"`，构造后自动归一化为 `asset.base_unit` |
| **计价货币成本 `cost`** | **可省略**（现金的计价成本恒等于金额本身 `amount`）；若显式传入，必须与 `amount` 同币种且数值相等 | **必填**，且必须是币种等于 `asset.quote_currency` 的 0 维非负非缺失 [`Money`](money.md)（允许为 `0`，如赠股或零成本权益） |
| **基准货币成本 `cost_base`** | 可选的 0 维非负非缺失 `Money`；省略时默认等于 `cost`（即组合基准货币等于该现金币种） | 可选的 0 维非负非缺失 `Money`；省略时默认等于 `cost`（即组合基准货币等于标的计价货币） |
| **盯市参数要求** | `market_value(price=None, fx=...)` **禁止传 `price`**；当 `asset != base_currency` 时必须传 `fx` | `market_value(price, fx=...)` **必须传 `price`**；当 `asset.quote_currency != base_currency` 时必须传 `fx` |

`PositionLot` 构造后完全**不可变**：重写了 `__setattr__` 与 `__delattr__`，任何修改或删除属性的尝试都会直接抛出 `AttributeError`。

### 2. 双币种成本账本（`cost` vs `cost_base`）与零隐式换汇原则

在多币种组合会计中，单笔持仓同时面临两个货币维度的核算需求：

- **`cost`（计价货币总成本）**：以资产自身的交易/计价货币（[`asset.quote_currency`](instrument.md#instrumentquote_currency) 或现金币种）记录的原始成交总成本（含合约乘数或分摊费用）；
- **`cost_base`（组合基准货币总成本）**：按买入日历史汇率折算为组合报告基准货币（[`base_currency`](#positionlotbase_currency)）的入账总成本。

Moneta 严格遵循**零隐式换汇**原则：

1. 构造时不接受浮点数或汇率作为参数，也不做隐式的 `1:1` 跨币种替用；若组合基准货币（如 `CNY`）与标的计价货币（如 `USD`）不同，调用方应先按买入日汇率算好 `cost_base`（`Money`）后显式传入；
2. 盯市估值（[`market_value`](#positionlotmarket_value)）与盈亏核算（[`profit_loss`](#positionlotprofit_loss-与-positionlotpnl)）时，若计价货币与 `base_currency` 不同，必须显式传入当前的 `ExchangeRate`：既支持正向汇率（`local -> base` 乘以中间价一次舍入），也支持反向汇率（`base -> local` **直接除以中间价一次舍入**，避免先求倒数汇率再相乘导致的二次舍入误差）。

### 3. 最大余数法成本守恒拆分（`split`）

当部分平仓或批次转让需要从单笔批次中拆出数量 $q_{\text{part}}$（$0 < q_{\text{part}} < q_{\text{total}}$）时，[`lot.split(amount)`](#positionlotsplit) 将拆出数量与剩余数量 $q_{\text{rest}} = q_{\text{total}} - q_{\text{part}}$ 组成二元权重向量 `weights = FixedArray([part.data, rest.data])`，分别调用 [`cost.allocate(weights)`](money.md#moneyallocate) 与 [`cost_base.allocate(weights)`](money.md#moneyallocate) 执行最大余数法（Hare-Niemeyer）分配：

$$\text{part.cost} + \text{rest.cost} \equiv \text{lot.cost}, \qquad \text{part.cost\_base} + \text{rest.cost\_base} \equiv \text{lot.cost\_base}$$

即使遇到 `100.00 USD` 拆成 `1 : 2`（`33.33 USD` 与 `66.67 USD`）等除不尽的情形，拆出批次与剩余批次的成本也精确到最小货币原子单位且总和严格守恒。

### 4. 三种平仓结转规则对比（`FIFO` / `LIFO` / `WAC`）

[`match_lots(lots, amount, rule)`](#match_lots) 从同一标的、同一基准货币的有序批次序列 `lots` 中扣减正数标量 `amount`，返回 `(closed, remaining)` 二元组：

| 结转规则 | 枚举成员 | 扣减顺序与拆分机制 | `closed` 返回顺序 | `remaining` 返回顺序 | 典型会计与税务场景 |
|---|---|---|---|---|---|
| **先进先出** | `LotMatchingRule.FIFO` | 从 `lots[0]` 向后依次全额平仓，最后一个被触及的批次调用 [`split`](#positionlotsplit) 拆出所需尾数 | 按买入先后顺序（即扣减顺序） | 保持原 `lots` 顺序（含拆分后的剩余半批） | 国际主流税务默认规则、股票与基金常规清结算 |
| **后进先出** | `LotMatchingRule.LIFO` | 从 `lots[-1]` 向前逆序扣减，最后一个被触及的批次调用 [`split`](#positionlotsplit) 拆分 | **按实际扣减顺序**（最新买入的排在最前） | 保持原 `lots` 顺序（最早买入的在前） | 特定税区税务优化、短期波段头寸优先平仓 |
| **加权平均成本** | `LotMatchingRule.WAC` | 通过 [`_proportional_shares`](#_proportional_shares) 在共同最细 `scale` 上按各批次数量比例把 `amount` 用最大余数法分配到每一批（余数相同时靠前的批次优先），再对每批调用 `split` | 按原 `lots` 顺序（跳过分配为 0 的批次） | 按原 `lots` 顺序 | 移动加权平均成本法核算（平掉部分的总成本天然等于按平均成本计算的成本） |

## API 速查表

| 分类 | 接口 / 属性 / 常量 | 功能说明 |
|---|---|---|
| **平仓规则枚举** | [`LotMatchingRule`](#lotmatchingrule) | 平仓批次结转规则枚举（`FIFO`、`LIFO`、`WAC`） |
| **批次构造与属性** | [`PositionLot(asset, amount, acquired_date, cost=None, cost_base=None, label="")`](#positionlotasset-amount-acquired_date-costnone-cost_basenone-label) | 构造不可变的现金或证券单笔持仓批次 |
| | [`PositionLot.asset`](#positionlotasset) / [`is_cash`](#positionlotis_cash) | 持有的现金 [`Currency`](currency.md) 或证券 [`Instrument`](instrument.md) / 是否为现金批次 |
| | [`PositionLot.amount`](#positionlotamount) | 持有数量（现金为 `Money`，证券为基准单位 `Quantity`） |
| | [`PositionLot.acquired_date`](#positionlotacquired_date) | 归一化后的买入/入账日期（`datetime.date`） |
| | [`PositionLot.cost`](#positionlotcost) / [`cost_base`](#positionlotcost_base) / [`base_currency`](#positionlotbase_currency) | 计价货币总成本 / 组合基准货币总成本 / 组合基准货币 [`Currency`](currency.md) |
| | [`PositionLot.label`](#positionlotlabel) | 交易编号或批次备注字符串 |
| **持有期计算** | [`PositionLot.holding_days`](#positionlotholding_days) | 计算自买入日至指定估值日的日历持有天数（`int`） |
| | [`PositionLot.year_fraction`](#positionlotyear_fraction) | 按指定计日惯例计算自买入日至估值日的年化时间跨度（`float`） |
| **拆分与估值核算** | [`PositionLot.split`](#positionlotsplit) | 按数量比例与最大余数法守恒拆分为 `(拆出批次, 剩余批次)` |
| | [`PositionLot.market_value`](#positionlotmarket_value) | 按给定市价与汇率计算组合基准货币下的单次舍入市值（`Money`） |
| | [`PositionLot.profit_loss` / `PositionLot.pnl`](#positionlotprofit_loss-与-positionlotpnl) | 计算组合基准货币下的浮动或已实现盈亏 `market_value - cost_base`（`Money`） |
| | [`PositionLot.return_rate`](#positionlotreturn_rate) | 计算相对于 `cost_base` 的无量纲收益率比率（[`FixedArray`](fixed_array.md)） |
| | [`repr(lot)` 与不可变性](#reprlot-与不可变性) | 包含资产、数量、日期、双币种成本与备注的可读表示及只读保护 |
| **批量平仓撮合** | [`match_lots`](#match_lots) | 按 `FIFO` / `LIFO` / `WAC` 从同标的批次序列中守恒扣减指定数量 |
| **内部辅助（私有 API）** | [`DateLike`、`Asset`、`Amount`](#datelikeasset-与-amount类型别名) | 模块级类型注解别名 |
| | [`PositionLot._check_scalar`](#positionlot_check_scalar) | 校验数量或成本为 0 维非缺失的正数/非负标量 |
| | [`PositionLot._coerce_amount` 与 `PositionLot._derive`](#positionlot_coerce_amount-与-positionlot_derive) | 校验并归一化拆分/扣减数量 / 派生同元数据的新批次 |
| | [`_convert_at_rate`](#_convert_at_rate) | 按正反向 `ExchangeRate` 将金额单次舍入换算为目标币种（缺失值保持缺失） |
| | [`_proportional_shares`](#_proportional_shares) | WAC 模式下按各批次数量比例与最大余数法分配扣减份额 |

## 平仓规则枚举

### `LotMatchingRule`

```python
class moneta.LotMatchingRule(Enum)
```

平仓批次结转会计规则枚举（同时由顶层 `moneta` 与 `moneta.lot` 导出）。

**成员列表**

- **`LotMatchingRule.FIFO = "FIFO"`** — 先进先出（First-In, First-Out）：优先平掉最早买入的批次。
- **`LotMatchingRule.LIFO = "LIFO"`** — 后进先出（Last-In, First-Out）：优先平掉最近买入的批次。
- **`LotMatchingRule.WAC = "WAC"`** — 加权平均成本（Weighted Average Cost）：按各批次持仓数量比例从所有批次同比例扣减，平掉部分的总成本即加权平均成本。

**示例**

```python
>>> from moneta import LotMatchingRule
>>> [r.value for r in LotMatchingRule]
['FIFO', 'LIFO', 'WAC']
>>> LotMatchingRule("FIFO") is LotMatchingRule.FIFO
True
```

**另请参阅**

- [`match_lots`](#match_lots)

## 批次构造与核心属性

### `PositionLot(asset, amount, acquired_date, cost=None, cost_base=None, label="")`

```python
class moneta.PositionLot(
    asset,
    amount,
    acquired_date,
    cost=None,
    cost_base=None,
    label="",
)
```

构造不可变的单笔现金或证券持仓成本批次（同时由顶层 `moneta` 与 `moneta.lot` 导出）。

**参数**

- **asset** : *Currency 或 Instrument* — 持有的现金币种（[`Currency`](currency.md)）或证券标的（[`Instrument`](instrument.md)）。不接受字符串（如 `"USD"` 或 `"AAPL"`），必须传入对象实例。
- **amount** : *Money 或 Quantity* — 持有数量，必须为 **0 维、非缺失、严格大于 0** 的标量：
  - 当 `asset` 为 `Currency` 时，`amount` 必须是同币种的 [`Money`](money.md)；
  - 当 `asset` 为 `Instrument` 时，`amount` 必须是单位与 `asset.base_unit` 满足 [`units_compatible`](_units.md#units_compatible) 的 [`Quantity`](quantity.md)（若单位为 `"Any"`，构造后自动归一化为 `asset.base_unit`）。
- **acquired_date** : *date, datetime 或 str* — 买入或入账日期，通过 [`to_date(acquired_date, "acquired_date")`](_utils.md#to_date) 解析为 `datetime.date`。
- **cost** : *Money，可选* — 计价货币表示的总成本（0 维、非缺失、$\ge 0$ 的标量）：
  - 现金批次：可省略（默认等于 `amount`）；若显式给出，必须与 `amount` 同币种且数值相等；
  - 证券批次：**必填**，且币种必须等于 `asset.quote_currency`（允许为 `0`，如赠股）。
- **cost_base** : *Money，可选* — 组合基准货币表示的总成本（0 维、非缺失、$\ge 0$ 的标量）。省略时默认等于 `cost`（即基准货币等于计价货币）；跨币种持仓时由调用方按买入日汇率换算后传入。
- **label** : *str，默认 `""`* — 交易流水号或批次备注。

**返回值**

- **out** : *PositionLot* — 不可变的持仓批次实例。

**异常**

- **TypeError** — `asset` 不是 `Currency` 或 `Instrument`；`amount`、`cost`、`cost_base` 类型不匹配或币种/单位不一致；`acquired_date` 类型非法；或 `label` 不是 `str`。
- **ValueError** — `amount` 不是 0 维标量、为缺失值或 $\le 0$；`cost` 或 `cost_base` 不是 0 维标量、为缺失值或 $< 0$；现金批次传入了不等于 `amount` 的 `cost`；或 `acquired_date` 字符串不是合法 ISO 日期。

**机制说明**

1. 按 `isinstance(asset, Currency)` 或 `isinstance(asset, Instrument)` 分支校验 `amount` 与 `cost`（证券批次额外执行 `amount = Quantity(amount, unit=asset.base_unit)` 消除 `"Any"` 通配单位）；
2. 若 `cost_base is None` 则令 `cost_base = cost`，否则校验 `isinstance(cost_base, Money)`；
3. 依次调用 [`_check_scalar`](#positionlot_check_scalar) 校验 `amount > 0`、`cost >= 0`、`cost_base >= 0`；
4. 调用 `object.__setattr__` 写入全部六个只读槽位。

**示例**

构造本币/外币现金批次与跨币种证券批次：

```python
>>> from datetime import date, datetime
>>> from moneta import Instrument, Money, PositionLot, Quantity, CNY, EUR, USD
>>> cash_usd = PositionLot(USD, Money("500.00", "USD"), "2024-01-02")
>>> (cash_usd.is_cash, cash_usd.cost, cash_usd.cost_base, cash_usd.base_currency is USD)
(True, Money(FixedArray(500.00, scale=2), currency='USD'), Money(FixedArray(500.00, scale=2), currency='USD'), True)
>>> cash_eur = PositionLot(EUR, Money("100.00", "EUR"), date(2024, 1, 2), cost_base=Money("108.50", "USD"))
>>> (cash_eur.cost, cash_eur.cost_base, cash_eur.base_currency is USD)
(Money(FixedArray(100.00, scale=2), currency='EUR'), Money(FixedArray(108.50, scale=2), currency='USD'), True)
>>> cl = Instrument("CL", "USD", base_unit="barrel", multiplier=1000)
>>> sec = PositionLot(cl, Quantity("2.5", unit="Any"), datetime(2024, 1, 2, 15, 30), cost=Money("200000", "USD"), cost_base=Money("1420000", "CNY"), label="T-1")
>>> (sec.is_cash, sec.amount, sec.acquired_date, sec.base_currency is CNY, sec.label)
(False, Quantity(FixedArray(2.5, scale=1), unit='barrel'), datetime.date(2024, 1, 2), True, 'T-1')
```

非法参数与量纲冲突拦截：

```python
>>> aapl = Instrument("AAPL", "USD")
>>> PositionLot(aapl, Quantity(10), "2024-01-02")  # 证券批次漏传 cost
Traceback (most recent call last):
    ...
TypeError: 证券批次必须给出计价货币总成本 cost (Money)，收到 NoneType
>>> PositionLot(USD, Money("100", "USD"), "2024-01-02", cost=Money("99", "USD"))
Traceback (most recent call last):
    ...
ValueError: 现金批次的成本就是金额本身，cost 省略或必须等于 amount
>>> PositionLot(aapl, Quantity([10, 20]), "2024-01-02", cost=Money("1000", "USD"))
Traceback (most recent call last):
    ...
ValueError: 持仓数量 必须是标量
```

**另请参阅**

- [`PositionLot.split`](#positionlotsplit)
- [`PositionLot.market_value`](#positionlotmarket_value)
- [`match_lots`](#match_lots)

### `PositionLot.asset`

```python
PositionLot.asset
```

返回批次持有的标的资产：现金批次为 [`Currency`](currency.md) 实例，证券批次为 [`Instrument`](instrument.md) 实例。

**返回值**

- **asset** : *Currency 或 Instrument*

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity, USD
>>> aapl = Instrument("AAPL", "USD")
>>> PositionLot(USD, Money(100, "USD"), "2024-01-02").asset is USD
True
>>> PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money(1000, "USD")).asset is aapl
True
```

### `PositionLot.is_cash`

```python
PositionLot.is_cash
```

判断当前批次是否为现金批次（即 `isinstance(self.asset, Currency)`）。

**返回值**

- **is_cash** : *bool*

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity, USD
>>> aapl = Instrument("AAPL", "USD")
>>> PositionLot(USD, Money(100, "USD"), "2024-01-02").is_cash
True
>>> PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money(1000, "USD")).is_cash
False
```

### `PositionLot.amount`

```python
PositionLot.amount
```

返回批次的持有数量（0 维正数标量）：现金批次为 [`Money`](money.md)，证券批次为单位等于 `asset.base_unit` 的 [`Quantity`](quantity.md)。

**返回值**

- **amount** : *Money 或 Quantity*

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity
>>> cl = Instrument("CL", "USD", base_unit="barrel")
>>> PositionLot(cl, Quantity("5.5", unit="Any"), "2024-01-02", cost=Money(400, "USD")).amount
Quantity(FixedArray(5.5, scale=1), unit='barrel')
```

### `PositionLot.acquired_date`

```python
PositionLot.acquired_date
```

返回批次的买入或入账日期（标准库 `datetime.date` 实例）。

**返回值**

- **acquired_date** : *datetime.date*

**示例**

```python
>>> from moneta import Money, PositionLot, USD
>>> PositionLot(USD, Money(100, "USD"), "2024-06-15").acquired_date
datetime.date(2024, 6, 15)
```

### `PositionLot.cost`

```python
PositionLot.cost
```

返回以计价货币（证券为 `asset.quote_currency`，现金为 `asset` 本身）表示的批次总成本（0 维非负 [`Money`](money.md)）。

**返回值**

- **cost** : *Money*

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity
>>> aapl = Instrument("AAPL", "USD")
>>> PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money("1895.50", "USD")).cost
Money(FixedArray(1895.50, scale=2), currency='USD')
```

### `PositionLot.cost_base`

```python
PositionLot.cost_base
```

返回以组合基准货币表示的批次总成本（0 维非负 [`Money`](money.md)）。构造时未传 `cost_base` 则等于 `self.cost`。

**返回值**

- **cost_base** : *Money*

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity
>>> aapl = Instrument("AAPL", "USD")
>>> lot = PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money("1000", "USD"), cost_base=Money("7150", "CNY"))
>>> lot.cost, lot.cost_base
(Money(FixedArray(1000.00, scale=2), currency='USD'), Money(FixedArray(7150.00, scale=2), currency='CNY'))
```

### `PositionLot.base_currency`

```python
PositionLot.base_currency
```

返回组合基准货币（[`Currency`](currency.md) 单例），等价于 `self.cost_base.currency`。[`market_value`](#positionlotmarket_value) 与 [`profit_loss`](#positionlotprofit_loss-与-positionlotpnl) 的返回金额始终以该币种计价。

**返回值**

- **base_currency** : *Currency*

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity, CNY
>>> aapl = Instrument("AAPL", "USD")
>>> PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money("1000", "USD"), cost_base=Money("7150", "CNY")).base_currency is CNY
True
```

### `PositionLot.label`

```python
PositionLot.label
```

返回批次备注或交易编号字符串（默认 `""`）。拆分批次（[`split`](#positionlotsplit)）时会自动保留原批次的 `label`。

**返回值**

- **label** : *str*

**示例**

```python
>>> from moneta import Money, PositionLot, USD
>>> PositionLot(USD, Money(100, "USD"), "2024-01-02", label="ORDER-88").label
'ORDER-88'
```

## 持有期计算

### `PositionLot.holding_days`

```python
PositionLot.holding_days(as_of_date)
```

计算自买入日 [`acquired_date`](#positionlotacquired_date) 至估值日 `as_of_date` 的日历天数差 `(to_date(as_of_date) - self.acquired_date).days`。

**参数**

- **as_of_date** : *date, datetime 或 str* — 估值日期（必填），通过 [`to_date(as_of_date, "as_of_date")`](_utils.md#to_date) 解析。若早于 `acquired_date` 则返回负数。

**返回值**

- **days** : *int*

**异常**

- **TypeError** — `as_of_date` 类型不是 `date`、`datetime` 或 `str`。
- **ValueError** — `as_of_date` 字符串不是合法的 ISO 日期。

**示例**

```python
>>> from datetime import datetime
>>> from moneta import Money, PositionLot, USD
>>> lot = PositionLot(USD, Money(100, "USD"), "2024-01-01")  # 2024 为闰年（2 月有 29 天）
>>> lot.holding_days("2024-03-01"), lot.holding_days(datetime(2024, 1, 1, 23, 59))
(60, 0)
```

**另请参阅**

- [`PositionLot.year_fraction`](#positionlotyear_fraction)
- [`to_date`](_utils.md#to_date)

### `PositionLot.year_fraction`

```python
PositionLot.year_fraction(as_of_date, convention=DayCountConvention.ACT_365)
```

按指定的金融计日惯例（[`DayCountConvention`](conventions.md#daycountconvention)）计算自 `acquired_date` 至 `as_of_date` 的年化时间跨度 $\tau$，底层委托 [`year_fraction`](conventions.md#year_fraction) 实现。

**参数**

- **as_of_date** : *date, datetime 或 str* — 估值日期（必填）。
- **convention** : *DayCountConvention 或 str，默认 `DayCountConvention.ACT_365`* — 计日惯例；既支持枚举成员，也接受枚举值字符串（如 `"ACT/360"`、`"ACT/365"`、`"30/360"`、`"ACT/ACT"`）或枚举成员名字符串（如 `"ACT_360"`、`"THIRTY_360"`），不区分大小写。

**返回值**

- **tau** : *float* — 年化时间跨度。

**示例**

```python
>>> from moneta import DayCountConvention, Money, PositionLot, USD
>>> lot = PositionLot(USD, Money(100, "USD"), "2024-01-01")
>>> round(lot.year_fraction("2025-01-01", DayCountConvention.ACT_365), 6)  # 366 / 365
1.00274
>>> round(lot.year_fraction("2024-07-01", "act/360"), 6)  # 182 / 360
0.505556
```

**另请参阅**

- [`PositionLot.holding_days`](#positionlotholding_days)
- [`DayCountConvention`](conventions.md#daycountconvention)、[`year_fraction`](conventions.md#year_fraction)

## 批次拆分与估值核算

### `PositionLot.split`

```python
PositionLot.split(amount)
```

从当前批次中拆出数量 `amount`（满足 `0 < amount <= self.amount`），返回 `(拆出的批次, 剩余批次)` 二元组；当 `amount == self.amount`（全额拆出）时返回 `(self, None)`。

**参数**

- **amount** : *Money 或 Quantity* — 待拆出的正数标量数量：
  - 现金批次须传入与 `self.asset` 同币种的 [`Money`](money.md)；
  - 证券批次须传入与 `self.amount.unit` 满足 [`units_compatible`](_units.md#units_compatible) 的 [`Quantity`](quantity.md)（支持 `"Any"` 通配单位）。

**返回值**

- **part, rest** : *Tuple[PositionLot, Optional[PositionLot]]* — 拆出部分与剩余部分；两者均保留原批次的 `asset`、`acquired_date` 与 `label`。

**异常**

- **TypeError** — `amount` 类型或币种/单位与当前批次不匹配（不接受裸数字）。
- **ValueError** — `amount` 不是 0 维标量、为缺失值、$\le 0$ 或严格大于当前持仓数量 `self.amount`。

**机制说明**

1. 调用 [`_coerce_amount(amount)`](#positionlot_coerce_amount-与-positionlot_derive) 与 [`_check_scalar(part, "拆分数量", positive=True)`](#positionlot_check_scalar) 校验数量；
2. 若 `part > self._amount` 抛出 `ValueError`；若 `part == self._amount` 直接返回 `(self, None)`；
3. 计算剩余数量 `rest = self._amount - part`，构造比例权重 `weights = FixedArray([part.data, rest.data])`；
4. 调用 [`self._cost.allocate(weights)`](money.md#moneyallocate) 与 [`self._cost_base.allocate(weights)`](money.md#moneyallocate) 按最大余数法分配 `cost` 与 `cost_base`，确保两部分成本之和恒等于原批次成本。

**示例**

除不尽时的最大余数法守恒分配与全额拆分：

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity
>>> aapl = Instrument("AAPL", "USD")
>>> lot = PositionLot(aapl, Quantity(3), "2024-01-02", cost=Money("100.00", "USD"), cost_base=Money("719.99", "CNY"), label="T-1")
>>> part, rest = lot.split(Quantity(1))
>>> (part.amount, rest.amount, part.label, rest.label)
(Quantity(FixedArray(1, scale=0), unit='share'), Quantity(FixedArray(2, scale=0), unit='share'), 'T-1', 'T-1')
>>> (part.cost, rest.cost, bool(part.cost + rest.cost == lot.cost))
(Money(FixedArray(33.33, scale=2), currency='USD'), Money(FixedArray(66.67, scale=2), currency='USD'), True)
>>> (part.cost_base, rest.cost_base, bool(part.cost_base + rest.cost_base == lot.cost_base))
(Money(FixedArray(240.00, scale=2), currency='CNY'), Money(FixedArray(479.99, scale=2), currency='CNY'), True)
>>> lot.split(Quantity(3)) == (lot, None)
True
```

**性能对比**

| 操作 | 耗时（0-d 标量批次） | 说明 |
|---|---:|---|
| `PositionLot(...)` 构造 | 4.49 µs | 含类型、单位与正数标量校验及日期解析 |
| `lot.split(amount)` | 41.8 µs | 含 `cost` 与 `cost_base` 双重最大余数法守恒分配 |

**另请参阅**

- [`Money.allocate`](money.md#moneyallocate)
- [`match_lots`](#match_lots)

### `PositionLot.market_value`

```python
PositionLot.market_value(price=None, fx=None)
```

计算当前批次在组合基准货币（[`self.base_currency`](#positionlotbase_currency)）下的市值。

**参数**

- **price** : *Price 或 数值/字符串/序列，可选* — 证券批次**必填**（传入 [`Price`](price.md) 或按标的计价货币与基准单位解释的裸数值/张量，支持缺失值 `<NA>`）；现金批次**不接受**（必须为 `None`）。
- **fx** : *ExchangeRate，可选* — 当计价货币（`local.currency`）与组合基准货币（`self.base_currency`）不同时**必填**，为两币种之间任一方向的 `ExchangeRate`；当计价货币就是基准货币时**不接受**（必须为 `None`）。

**返回值**

- **out** : *Money* — 币种为 `self.base_currency` 的市值张量（若 `price` 为多维数组或含 `<NA>`，返回值保持相同形状与缺失掩码）。

**异常**

- **ValueError** — 现金批次传入了 `price`；证券批次未传 `price`；计价货币等于基准货币时多余传入了 `fx`；或计价货币不等于基准货币时漏传 `fx`。
- **TypeError** — `price` 的计价货币或基准单位与证券标的不匹配；或 `fx` 的货币对与 `(local.currency, base_currency)` 不匹配。

**机制说明**

1. **计算计价货币市值 `local`**：现金批次直接取 `local = self._amount`；证券批次调用 [`self._asset.calc_notional(price, self._amount)`](instrument.md#instrumentcalc_notional)（保留 `price × amount` 精确中间积，乘以 `multiplier` 后一次舍入到计价货币精度）；
2. **换算为组合基准货币 `base = self.base_currency`**：
   - 若 `local.currency == base`：确认 `fx is None` 后直接返回 `local`；
   - 若 `local.currency != base`：调用 [`_convert_at_rate(local, fx, base)`](#_convert_at_rate) 按正反向 `fx` 将 `local` 单次舍入换算为 `base`。

**示例**

同币种证券估值、跨币种正反向汇率估值与缺失市价传播：

```python
>>> from decimal import Decimal
>>> from moneta import Instrument, Money, PositionLot, Price, Quantity
>>> from moneta.fx import ExchangeRate
>>> aapl = Instrument("AAPL", "USD")
>>> lot_usd = PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money("1800", "USD"))
>>> lot_usd.market_value(Price("195.50", "USD")), lot_usd.market_value("195.50")
(Money(FixedArray(1955.00, scale=2), currency='USD'), Money(FixedArray(1955.00, scale=2), currency='USD'))
>>> lot_cny = PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money("1000", "USD"), cost_base=Money("7000", "CNY"))
>>> lot_cny.market_value(Decimal("100.01"), fx=ExchangeRate("CNY", "USD", "0.3"))  # 反向汇率直接除一次：1000.10 / 0.3 = 3333.67 CNY
Money(FixedArray(3333.67, scale=2), currency='CNY')
>>> lot_cny.market_value(Price(None, "USD"), fx=ExchangeRate("USD", "CNY", "7.2"))  # 缺失市价保持为 <NA>
Money(FixedArray(<NA>, scale=2), currency='CNY')
```

**另请参阅**

- [`_convert_at_rate`](#_convert_at_rate)
- [`PositionLot.profit_loss`](#positionlotprofit_loss-与-positionlotpnl)
- [`PositionLot.return_rate`](#positionlotreturn_rate)
- [`Instrument.calc_notional`](instrument.md#instrumentcalc_notional)

### `PositionLot.profit_loss` 与 `PositionLot.pnl`

```python
PositionLot.profit_loss(price=None, fx=None)
PositionLot.pnl(price=None, fx=None)
```

计算按给定市价与汇率重估的组合基准货币盈亏（`pnl` 是 `profit_loss` 的方法别名，`PositionLot.pnl is PositionLot.profit_loss`）：

$$\text{PnL} = \text{market\_value}(\text{price}, \text{fx}) - \text{cost\_base}$$

对未平仓批次传入当前市价即为**浮动盈亏（Unrealized PnL）**；对 [`match_lots`](#match_lots) 平出的 `closed` 批次传入卖出成交价即为**已实现盈亏（Realized PnL）**。

**参数**

- **price, fx** — 与 [`market_value`](#positionlotmarket_value) 完全相同。

**返回值**

- **out** : *Money* — 币种为 `self.base_currency` 的盈亏金额。

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity, EUR
>>> from moneta.fx import ExchangeRate
>>> cl = Instrument("CL", "USD", base_unit="barrel", multiplier=1000)
>>> lot = PositionLot(cl, Quantity(2, unit="barrel"), "2024-01-02", cost=Money("160000", "USD"))
>>> lot.profit_loss("81.5"), lot.pnl("79")
(Money(FixedArray(3000.00, scale=2), currency='USD'), Money(FixedArray(-2000.00, scale=2), currency='USD'))
>>> eur_cash = PositionLot(EUR, Money("100", "EUR"), "2024-01-02", cost_base=Money("108.00", "USD"))
>>> eur_cash.pnl(fx=ExchangeRate("EUR", "USD", "1.10"))
Money(FixedArray(2.00, scale=2), currency='USD')
```

**另请参阅**

- [`PositionLot.market_value`](#positionlotmarket_value)
- [`PositionLot.return_rate`](#positionlotreturn_rate)

### `PositionLot.return_rate`

```python
PositionLot.return_rate(price=None, fx=None)
```

计算当前批次在组合基准货币下的无量纲收益率比率：

$$r = \frac{\text{profit\_loss}(\text{price}, \text{fx})}{\text{cost\_base}}$$

**参数**

- **price, fx** — 与 [`market_value`](#positionlotmarket_value) 完全相同。

**返回值**

- **out** : *FixedArray* — 无量纲收益率定点数，精度遵循 [`FixedArray.divide`](fixed_array.md#fixedarraydivide) 的默认规则（默认保留 `self.base_currency.digits + 8` 位小数，放不下 `int64` 时少保留，最后去掉末尾 `0` 且至少保留 `base_currency.digits` 位）。

**异常**

- **ZeroDivisionError** — 当 `self.cost_base` 为 `0`（如零成本赠股批次）时抛出。

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity
>>> from moneta.fx import ExchangeRate
>>> aapl = Instrument("AAPL", "USD")
>>> PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money("1000", "USD")).return_rate(110)  # 整除：去末尾 0 保留 2 位
FixedArray(0.10, scale=2)
>>> lot = PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money("1000", "USD"), cost_base=Money("7000", "CNY"))
>>> lot.return_rate(100, fx=ExchangeRate("USD", "CNY", "7.2"))  # 除不尽：(7200 - 7000) / 7000 保留 2 + 8 = 10 位
FixedArray(0.0285714286, scale=10)
>>> gift = PositionLot(aapl, Quantity(1), "2024-01-02", cost=Money(0, "USD"))
>>> gift.return_rate(100)
Traceback (most recent call last):
    ...
ZeroDivisionError: 除数包含 0
```

**性能对比**

| 估值与核算操作 | 耗时（0-d 标量批次） | 说明 |
|---|---:|---|
| `lot.market_value(px)`（同币种） | 10.6 µs | 调用 `Instrument.calc_notional` 单次舍入 |
| `lot.market_value(px, fx)`（跨币种） | 29.9 µs | 含名义金额计算 + 汇率换算一次舍入 |
| `lot.profit_loss(px, fx)` | 35.3 µs | 跨币种市值减 `cost_base` |
| `lot.return_rate(px, fx)` | 62.3 µs | 跨币种盈亏除以 `cost_base`（保留 `digits + 8` 位） |

**另请参阅**

- [`PositionLot.profit_loss`](#positionlotprofit_loss-与-positionlotpnl)
- [`Money.divide`](money.md#moneydivide)

### `repr(lot)` 与不可变性

```python
repr(lot)
```

返回包含资产代码、持仓数量、买入日期、计价货币成本、基准货币成本以及可选 `label` 的可读文本表示；同时禁止任何属性写入或删除。

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity, USD
>>> aapl = Instrument("AAPL", "USD")
>>> lot = PositionLot(aapl, Quantity(10), "2024-01-02", cost=Money("1895", "USD"), label="T-1")
>>> repr(lot)
"PositionLot(asset='AAPL', amount=10 share, date='2024-01-02', cost=1895.00 USD, cost_base=1895.00 USD, label='T-1')"
>>> repr(PositionLot(USD, Money("100", "USD"), "2024-01-02"))
"PositionLot(asset='USD', amount=100.00 USD, date='2024-01-02', cost=100.00 USD, cost_base=100.00 USD)"
>>> lot._amount = Quantity(5)
Traceback (most recent call last):
    ...
AttributeError: PositionLot 是不可变对象，禁止设置属性 '_amount'
>>> del lot._cost
Traceback (most recent call last):
    ...
AttributeError: PositionLot 是不可变对象，禁止删除属性 '_cost'
```

## 批量平仓撮合函数

### `match_lots`

```python
moneta.lot.match_lots(lots, amount, rule=LotMatchingRule.FIFO)
```

按指定的会计结转规则 `rule`（[`FIFO`、`LIFO` 或 `WAC`](#lotmatchingrule)）从同一标的、同一基准货币的持仓批次序列 `lots` 中扣减数量 `amount`，返回 `(平掉的批次列表, 剩余批次列表)`。

**参数**

- **lots** : *Sequence[PositionLot]* — 按买入顺序排列的非空同标的批次序列。序列中所有批次的 `asset` 与 `base_currency` 必须完全相同。
- **amount** : *Money 或 Quantity* — 待平仓扣减的正数标量数量（0 < `amount` $\le \sum \text{lot.amount}$），类型与单位须与 `lots[0]` 兼容。
- **rule** : *LotMatchingRule，默认 `LotMatchingRule.FIFO`* — 平仓结转规则枚举（不接受裸字符串 `"FIFO"`）。

**返回值**

- **closed, remaining** : *Tuple[List[PositionLot], List[PositionLot]]* —
  - `closed`：被平掉的批次列表（`FIFO` 与 `WAC` 下按原 `lots` 顺序排列，`LIFO` 下按从后往前的实际扣减顺序排列）；
  - `remaining`：扣减后剩余的未平仓批次列表（始终保持原 `lots` 的先后顺序；当全额平仓 `amount == total` 时为空列表 `[]`）。
  - **守恒不变量**：`closed` 各批次数量之和恒等于 `amount`；`closed + remaining` 各批次的 `amount`、`cost` 与 `cost_base` 之和恒等于原 `lots` 的总数量与总成本。

**异常**

- **TypeError** — `rule` 不是 [`LotMatchingRule`](#lotmatchingrule) 实例；`lots` 中含有非 `PositionLot` 元素；或 `amount` 类型/单位不兼容。
- **ValueError** — `lots` 为空序列；`lots` 中混有不同 `asset` 或不同 `base_currency` 的批次；`amount` 不是 0 维正数标量、为缺失值，或超过 `lots` 的持仓总量。

**机制说明**

1. 校验 `isinstance(rule, LotMatchingRule)`、`lots` 非空，且所有元素均为同 `asset`、同 `base_currency` 的 `PositionLot`；
2. 累加全部批次持仓得到 `total`，校验 `0 < target <= total`；
3. 计算每个批次应扣减的份额列表 `shares`：
   - **`FIFO` / `LIFO`**：沿正序 `range(len(lots))` 或逆序 `range(len(lots) - 1, -1, -1)` 贪心扣减，直至 `left` 扣完；
   - **`WAC`**：调用 [`_proportional_shares(lots, target)`](#_proportional_shares)，在 `max(lot.amount.scale, target.scale)` 的最小单位上按各批数量比例用最大余数法分配扣减额；
4. 遍历 `zip(lots, shares)`：未扣减的批次直接追加到 `remaining`，被扣减的批次调用 [`lot.split(share)`](#positionlotsplit) 将拆出部分放入 `closed`、剩余非空部分放入 `remaining`；若为 `LIFO` 则最后执行 `closed.reverse()` 以按实际扣减顺序返回。

**示例**

对比同一组持仓在 `FIFO`、`LIFO` 与 `WAC` 下的平仓结果：

```python
>>> from moneta import Instrument, LotMatchingRule, Money, PositionLot, Quantity
>>> from moneta.lot import match_lots
>>> aapl = Instrument("AAPL", "USD")
>>> l1 = PositionLot(aapl, Quantity(10), "2024-01-01", cost=Money("1000", "USD"))
>>> l2 = PositionLot(aapl, Quantity(5), "2024-02-01", cost=Money("600", "USD"))
>>> l3 = PositionLot(aapl, Quantity(5), "2024-03-01", cost=Money("700", "USD"))
>>> lots = [l1, l2, l3]
>>> closed_fifo, rem_fifo = match_lots(lots, Quantity(12), LotMatchingRule.FIFO)
>>> [(c.amount, c.cost) for c in closed_fifo]
[(Quantity(FixedArray(10, scale=0), unit='share'), Money(FixedArray(1000.00, scale=2), currency='USD')), (Quantity(FixedArray(2, scale=0), unit='share'), Money(FixedArray(240.00, scale=2), currency='USD'))]
>>> [(r.amount, r.cost) for r in rem_fifo]
[(Quantity(FixedArray(3, scale=0), unit='share'), Money(FixedArray(360.00, scale=2), currency='USD')), (Quantity(FixedArray(5, scale=0), unit='share'), Money(FixedArray(700.00, scale=2), currency='USD'))]
>>> closed_lifo, rem_lifo = match_lots(lots, Quantity(7), LotMatchingRule.LIFO)
>>> [(c.amount, c.cost) for c in closed_lifo]
[(Quantity(FixedArray(5, scale=0), unit='share'), Money(FixedArray(700.00, scale=2), currency='USD')), (Quantity(FixedArray(2, scale=0), unit='share'), Money(FixedArray(240.00, scale=2), currency='USD'))]
>>> closed_wac, rem_wac = match_lots(lots, Quantity(4), LotMatchingRule.WAC)
>>> [c.amount for c in closed_wac], sum((c.cost for c in closed_wac), Money(0, "USD"))
([Quantity(FixedArray(2, scale=0), unit='share'), Quantity(FixedArray(1, scale=0), unit='share'), Quantity(FixedArray(1, scale=0), unit='share')], Money(FixedArray(460.00, scale=2), currency='USD'))
```

异常与不变量校验：

```python
>>> match_lots(lots, Quantity(25), LotMatchingRule.FIFO)
Traceback (most recent call last):
    ...
ValueError: 扣减数量 25 share 超过批次总量 20 share
>>> match_lots([], Quantity(1), LotMatchingRule.FIFO)
Traceback (most recent call last):
    ...
ValueError: 没有可扣减的批次
>>> match_lots(lots, Quantity(1), "FIFO")
Traceback (most recent call last):
    ...
TypeError: rule 必须是 LotMatchingRule，收到 'FIFO'
```

**性能对比**

| 平仓结转规则（10 笔批次扣减） | 耗时 | 说明 |
|---|---:|---|
| `match_lots(lots, q, LotMatchingRule.FIFO)` | 138 µs | 仅对最后触及的 1 笔边界批次调用 `split` |
| `match_lots(lots, q, LotMatchingRule.LIFO)` | 139 µs | 逆序扣减，仅拆分 1 笔边界批次 |
| `match_lots(lots, q, LotMatchingRule.WAC)` | 527 µs | 按比例分配并对全部 10 笔批次执行守恒 `split` |

**另请参阅**

- [`LotMatchingRule`](#lotmatchingrule)
- [`PositionLot.split`](#positionlotsplit)
- [`_proportional_shares`](#_proportional_shares)

## 内部辅助函数与方法（私有 API）

### `DateLike`、`Asset` 与 `Amount`（类型别名）

```python
moneta.lot.DateLike = Union[str, date, datetime]
moneta.lot.Asset = Union[Currency, Instrument]
moneta.lot.Amount = Union[Money, Quantity]
```

模块内部用于类型注解的联合类型别名：

- **`DateLike`**：可被 [`to_date`](_utils.md#to_date) 解析的日期输入类型；
- **`Asset`**：批次可持有的两类标的资产（现金 [`Currency`](currency.md) 或证券 [`Instrument`](instrument.md)）；
- **`Amount`**：批次持仓数量类型（现金 [`Money`](money.md) 或证券 [`Quantity`](quantity.md)）。

### `PositionLot._check_scalar`

```python
@staticmethod
PositionLot._check_scalar(value, what, positive)
```

校验 `Money` 或 `Quantity` 实例 `value` 必须是 0 维标量（`value.ndim == 0`）、非缺失（`not value.isna()`），且当 `positive=True` 时严格大于 0（`mantissa > 0`），当 `positive=False` 时非负（`mantissa >= 0`）。

**示例**

```python
>>> from moneta import Money, PositionLot
>>> PositionLot._check_scalar(Money("10.00", "USD"), "持仓数量", positive=True)
>>> PositionLot._check_scalar(Money("0.00", "USD"), "cost", positive=False)
>>> PositionLot._check_scalar(Money("0.00", "USD"), "持仓数量", positive=True)
Traceback (most recent call last):
    ...
ValueError: 持仓数量 必须为正数，收到 0.00 USD
```

### `PositionLot._coerce_amount` 与 `PositionLot._derive`

```python
PositionLot._coerce_amount(amount)
PositionLot._derive(amount, cost, cost_base)
```

- **`_coerce_amount(amount)`**：校验传入 [`split`](#positionlotsplit) 或 [`match_lots`](#match_lots) 的数量 `amount` 与当前批次同类、同币种或单位兼容（证券批次将兼容的 `"Any"` 单位统一转换为 `self._amount.unit`）；
- **`_derive(amount, cost, cost_base)`**：在保持当前批次 `asset`、`acquired_date` 与 `label` 不变的前提下，用拆分后的新数量与新成本构造派生 `PositionLot`（现金批次自动将 `cost` 设为 `None`）。

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity
>>> cl = Instrument("CL", "USD", base_unit="barrel")
>>> lot = PositionLot(cl, Quantity(5, "barrel"), "2024-01-02", cost=Money("400", "USD"))
>>> lot._coerce_amount(Quantity(2, unit="Any"))
Quantity(FixedArray(2, scale=0), unit='barrel')
```

### `_convert_at_rate`

```python
moneta.lot._convert_at_rate(money, rate, target)
```

用 `money.currency` 与目标币种 `target` 之间的 [`ExchangeRate`](fx.md#exchangeratebase_currency-quote_currency-rate--bidnone-asknone-as_of_datenone)（任一方向）将金额 `money` 换算为 `target` 币种，全程**只舍入一次**，且缺失值 `<NA>` 保持为缺失值。同时供 [`PositionLot.market_value`](#positionlotmarket_value) 与 [`Portfolio`](portfolio.md) 复用。

**参数**

- **money** : *Money* — 待换算的源币种金额（标量或数组，支持缺失值 `<NA>`）。
- **rate** : *ExchangeRate* — `money.currency` 与 `target` 之间任一方向的汇率对象。
- **target** : *Currency* — 目标币种实例。

**返回值**

- **out** : *Money* — 换算到 `target` 币种（`scale == target.digits`）的金额。

**异常**

- **ValueError** — `rate is None`（报 `f"需要 {money.currency.code}/{target.code} 的汇率 fx 才能换算为 {target.code}"`）。
- **TypeError** — `{rate.base_currency, rate.quote_currency} != {money.currency, target}`（汇率的货币对与待换算金额及目标币种组成的集合不匹配）。

**机制说明**

1. 若 `rate is None` 立即抛出 `ValueError`；
2. 若 `{rate.base_currency, rate.quote_currency} != {money.currency, target}`，抛出 `TypeError`；
3. 返回 [`rate.convert(money)`](fx.md#exchangerateconvert)，即用中间价（`side="mid"`）、按 `half_up` 舍入：正方向乘、反方向直接除，全程只舍入一次，缺失值 `<NA>` 保持缺失。

**示例**

```python
>>> from moneta import Money, CNY, USD
>>> from moneta.fx import ExchangeRate
>>> from moneta.lot import _convert_at_rate
>>> _convert_at_rate(Money("100.00", "USD"), ExchangeRate("USD", "CNY", "7.20"), CNY)
Money(FixedArray(720.00, scale=2), currency='CNY')
>>> _convert_at_rate(Money("1000.10", "USD"), ExchangeRate("CNY", "USD", "0.3"), CNY)  # 反向汇率直接除一次
Money(FixedArray(3333.67, scale=2), currency='CNY')
>>> _convert_at_rate(Money([None, "100.00"], "USD"), ExchangeRate("USD", "CNY", "7.20"), CNY)
Money(FixedArray([<NA>, 720.00], scale=2), currency='CNY')
>>> _convert_at_rate(Money("100.00", "USD"), None, CNY)
Traceback (most recent call last):
    ...
ValueError: 需要 USD/CNY 的汇率 fx 才能换算为 CNY
>>> _convert_at_rate(Money("100.00", "USD"), ExchangeRate("EUR", "USD", "1.10"), CNY)
Traceback (most recent call last):
    ...
TypeError: 汇率 EUR/USD 与 USD/CNY 不匹配
```

**另请参阅**

- [`ExchangeRate.convert`](fx.md#exchangerateconvert)

- [`PositionLot.market_value`](#positionlotmarket_value)

### `_proportional_shares`

```python
moneta.lot._proportional_shares(lots, target)
```

在 `LotMatchingRule.WAC` 模式下，将总扣减数量 `target` 按 `lots` 中各批次的持仓数量比例分配到共同的最细小数位数 `scale = max([lot.amount.scale for lot in lots] + [target.scale])` 的最小整数单位上，使用最大余数法（Hare-Niemeyer）确保各批扣减份额之和精确等于 `target` 且每批扣减额不超过该批持仓量（余数相同时靠前的批次优先得 1 个最小单位）。

**示例**

```python
>>> from moneta import Instrument, Money, PositionLot, Quantity
>>> from moneta.lot import _proportional_shares
>>> aapl = Instrument("AAPL", "USD")
>>> lots = [PositionLot(aapl, Quantity(1), "2024-01-01", cost=Money(100, "USD")), PositionLot(aapl, Quantity(1), "2024-01-02", cost=Money(200, "USD")), PositionLot(aapl, Quantity(1), "2024-01-03", cost=Money(300, "USD"))]
>>> _proportional_shares(lots, Quantity("1.0"))  # 在 scale=1 上按 1:1:1 分配 1.0 -> 0.4, 0.3, 0.3
[Quantity(FixedArray(0.4, scale=1), unit='share'), Quantity(FixedArray(0.3, scale=1), unit='share'), Quantity(FixedArray(0.3, scale=1), unit='share')]
```

**另请参阅**

- [`match_lots`](#match_lots)
- [`LotMatchingRule`](#lotmatchingrule)
