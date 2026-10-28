# Portfolio — 多币种多资产投资组合管理与核算

`moneta.Portfolio` 是面向多币种现金（[`Currency`](currency.md)）与多品种证券标的（[`Instrument`](instrument.md)）的投资组合容器与核算引擎。
它以不可变的持仓批次 [`PositionLot`](lot.md) 为**唯一事实来源（Single Source of Truth）**，按资产归集持仓，完整支持：

- 现金与证券逐笔入账（[`deposit`](#portfoliodeposit)、[`add_lot`](#portfolioadd_lot)）及历史入账汇率折算；
- 按 `FIFO` / `LIFO` / `WAC` 规则平仓核销批次并计算基准货币已实现盈亏（[`withdraw`](#portfoliowithdraw)）；
- 按估值日汇率与标的市价计算组合总市值（[`total_value`](#portfoliototal_value)）、总成本（[`total_cost`](#portfoliototal_cost)）、未实现盈亏（[`unrealized_pnl`](#portfoliounrealized_pnl)）与总收益率（[`return_rate`](#portfolioreturn_rate)）；
- 将换汇与标的单位价值合并为**单次除法舍入**的目标权重再平衡调仓试算（[`rebalance`](#portfoliorebalance)）。

```python
>>> from decimal import Decimal
>>> from moneta import Instrument, LotMatchingRule, Money, Portfolio, Price, Quantity, CNY, USD
>>> from moneta.fx import ExchangeRate, FXEngine, UserDictFXProvider
>>> aapl = Instrument("AAPL", "USD")
>>> p = Portfolio("CNY")
>>> _ = p.deposit("CNY", 1000, "2024-01-01")
>>> _ = p.deposit(aapl, 100, "2024-01-02", price="100", fx=ExchangeRate("USD", "CNY", "7.0"))
>>> _ = p.deposit(USD, Money(50, "USD"), "2024-01-03", fx=ExchangeRate("USD", "CNY", "7.1"))
>>> p["CNY"], p["AAPL"], p[USD], len(p)
(Money(FixedArray(1000.00, scale=2), currency='CNY'), Quantity(FixedArray(100, scale=0), unit='share'), Money(FixedArray(50.00, scale=2), currency='USD'), 3)
>>> engine = FXEngine(UserDictFXProvider({"USD/CNY": "7.2", "2024-03-01": {"USD/CNY": "7.2"}}))
>>> p.total_cost(), p.total_value({"AAPL": 120}, fx=engine), p.unrealized_pnl({"AAPL": 120}, fx=engine)
(Money(FixedArray(71355.00, scale=2), currency='CNY'), Money(FixedArray(87760.00, scale=2), currency='CNY'), Money(FixedArray(16405.00, scale=2), currency='CNY'))
>>> closed, realized = p.withdraw(aapl, 40, "2024-03-01", price=130, fx=engine, rule=LotMatchingRule.FIFO)
>>> p[aapl], realized  # 40 × 130 × 7.2 - 40 × 100 × 7.0 = 37440 - 28000 = 9440 CNY
(Quantity(FixedArray(60, scale=0), unit='share'), Money(FixedArray(9440.00, scale=2), currency='CNY'))
```

[← API 参考](index.md)

## 核心机制与设计约束

### 1. 批次即唯一事实来源（Single Source of Truth）与资产键语义

`Portfolio` 仅包含两个槽位 `__slots__ = ("_base_currency", "_lots")`，内部以字典 `_lots: Dict[Asset, List[PositionLot]]` 按各资产首次入账顺序存储批次列表，**不冗余维护第二份余额表**：

- **资产键类型（`Asset = Union[Currency, Instrument]`）**：
  - 现金持仓以 [`Currency`](currency.md) 单例为键；
  - 证券持仓以 [`Instrument`](instrument.md) 为键。由于 [`Instrument` 的相等性仅比较 `(symbol, quote_currency)`](instrument.md#身份标识语义相等性与哈希仅看代码与计价货币)，同一标的代码在不同计价币种下的合约（如 `Instrument("A", "USD")` 与 `Instrument("A", "CNY")`）在组合中作为两个独立资产共存。
- **实时汇总与空仓自动清理**：[`position(asset)`](#portfolioposition)（以及 `p[asset]`）实时将该资产名下所有批次的 `amount` 累加；当某资产通过 [`withdraw`](#portfoliowithdraw) 被全额扣完（`remaining` 为空）时，该资产键立即从 `_lots` 中删除。
- **字符串代码智能解析（[`_resolve`](#portfolio_resolve)）**：在查询、`deposit`、`withdraw` 或 `rebalance` 中传入字符串 `code`（去除首尾空白并转大写）时：
  1. **优先匹配已持有资产**：先在当前 `_lots` 中查找 `currency.code == code` 或 `instrument.symbol == code` 的已持有资产。若唯一命中则直接返回该资产（因此若组合持有名为 `"CAD"` 的股票而未持有加元现金，`p["CAD"]` 会准确命中股票 `"CAD"`，不会静默当成加元现金）；若命中多个已持有资产（如同时持有股票 `"CAD"` 与加元现金 `CAD`，或同时持有 `Instrument("A", "USD")` 与 `Instrument("A", "CNY")`），则抛出 `KeyError` 要求调用方显式传入 `Currency` 或 `Instrument` 对象以消除歧义；
  2. **回退匹配已注册币种**：若当前未持有该代码对应的资产，但 `CurrencyRegistry.has(code)` 为 `True`，则解析为该 [`Currency`](currency.md)（未持有的证券标的不支持仅靠字符串凭空构造，必须传入 `Instrument` 实例）。

### 2. 必填基准货币与显式换汇契约（`ExchangeRate` 与 `FXEngine`）

- **基准货币必填**：构造 `Portfolio(base_currency)` 必须显式指定组合记账基准货币（`Currency` 或代码字符串），所有加入组合的 [`PositionLot`](lot.md) 的 `base_currency` 必须与之一致。
- **零隐式换汇**：涉及非基准货币资产的入账折算、平仓盈亏、盯市估值与再平衡，均须通过 `fx` 参数显式提供汇率（不提供时直接抛出 `ValueError`，绝不静默使用历史成本汇率或按 `1:1` 换算）：
  - 当 `fx` 为 [`ExchangeRate`](fx.md#exchangeratebase_currency-quote_currency-rate--bidnone-asknone-as_of_datenone) 时：支持正向或反向货币对，底层统一经由 [`_convert_at_rate`](lot.md#_convert_at_rate) 单次舍入；
  - 当 `fx` 为 [`FXEngine`](fx.md#fxengineprovider-base_anchor_currencyusd)（即具有 `get_rate` 方法的对象）时：[`_rate`](#portfolio_rateportfolio_base_to_localportfolio_to_base-与-portfolio_sum) 自动将 `acquired_date`、`trade_date` 或 `as_of_date` 归一化为 `datetime.date`（或 `None`）后调用 `fx.get_rate(currency, self.base_currency, as_of_date=when)` 获取对应汇率。汇率严格按日期匹配：传入日期时只使用当天的汇率，不传时只使用不带日期的汇率，查不到抛 [`ExchangeRateNotFoundError`](fx.md#exchangeratenotfounderror)；不会沿用前一天的汇率。

### 3. 入账（`deposit`）与出账（`withdraw`）行为矩阵

| 操作与资产类别 | 价格 / 成本参数要求 | 跨币种汇率 `fx` 要求 | 返回值与盈亏核算行为 |
|---|---|---|---|
| **`deposit` — 现金（`Currency`）** | **禁止传 `price` 或 `cost`**（现金的计价成本即金额本身 `amount`） | 未传 `cost_base` 且币种 $\ne$ `base_currency` 时**必传 `fx`**（按 `acquired_date` 折算 `cost_base`） | 返回新建并入账的现金 [`PositionLot`](lot.md) |
| **`deposit` — 证券（`Instrument`）** | **`price` 与 `cost` 二选一必填**：传 `price` 时自动调用 [`calc_notional`](instrument.md#instrumentcalc_notional) 算计价成本（含合约乘数）；传 `cost` 时直接使用该 `Money`（如含买入手续费） | 未传 `cost_base` 且 `quote_currency` $\ne$ `base_currency` 时**必传 `fx`** | 返回新建并入账的证券 [`PositionLot`](lot.md) |
| **`withdraw` — 本币现金** | 不接受 `price` | 不需要 `fx` | 返回 `(closed, Money(0, base_currency))` |
| **`withdraw` — 外币现金** | 不接受 `price` | 传入 `fx` 时按 `trade_date` 汇率计算汇兑已实现盈亏；**省略 `fx` 时不报错，已实现盈亏返回 `None`**（视作直接提现/划转外币） | 返回 `(closed, realized_pnl 或 None)` |
| **`withdraw` — 证券（给出成交价 `price`）** | 传入卖出成交价 `price` | 若 `quote_currency != base_currency` 则**必传 `fx`**（漏传 `fx` 直接抛 `ValueError`，防止盈亏算不完整） | 返回 `(closed, realized_pnl)`，其中 `realized_pnl` 为各平仓批次 `lot.profit_loss(price, rate)` 之和 |
| **`withdraw` — 证券（省略 `price`）** | `price=None`（如证券转托管、非交易过户） | 不需要 `fx` | 返回 `(closed, None)` |

### 4. 单次除法合并的精确再平衡（`rebalance`）

在多币种多资产再平衡计算中，某资产的目标持仓数量理论上等于：

$$Q_{\text{target}} = \frac{V_{\text{total\_base}} \times w_i}{\text{FX}_{\text{local} \to \text{base}} \times (P_i \times M_i)}$$

若先将目标基准货币价值 $V_{\text{total\_base}} \times w_i$ 换算并舍入为本地货币金额，再除以单价 $P_i$ 和合约乘数 $M_i$，会产生多次中间舍入误差。[`Portfolio.rebalance`](#portfoliorebalance) 通过 [`_base_to_local`](#portfolio_rateportfolio_base_to_localportfolio_to_base-与-portfolio_sum) 将“基准货币 → 本地货币”的汇率转换表示为**不舍入的精确有理分式 `(num, den)`**，再把每单位标的价值 `unit_value = price * multiplier`（精确乘法 `rounding=None`）合并进分母 `den.mul_round(unit_value, rounding=None)`，最后**只执行一次 [`FixedArray.divide`](fixed_array.md#fixedarraydivide)** 得到目标数量，从根源上消除了跨币种调仓的级联舍入偏差。

## API 速查表

| 分类 | 接口 / 属性 / 运算符 | 功能说明 |
|---|---|---|
| **构造与核心属性** | [`Portfolio(base_currency)`](#portfoliobase_currency) | 构造指定记账基准货币的多资产投资组合 |
| | [`Portfolio.base_currency`](#portfoliobase_currency-1) | 组合基准货币 [`Currency`](currency.md) 单例 |
| | [`Portfolio.assets`](#portfolioassets) / [`currencies`](#portfoliocurrencies) / [`instruments`](#portfolioinstruments) | 按首次入账顺序返回当前持有的全部资产 / 现金币种 / 证券标的列表 |
| **持仓与批次查询** | [`Portfolio.lots`](#portfoliolots) | 获取指定资产或全部资产的 [`PositionLot`](lot.md) 列表副本 |
| | [`Portfolio.position`](#portfolioposition) / [`p[asset]`](#passetasset-in-pportfoliogetiterp-与-lenp映射与容器协议) | 查询某资产当前汇总持仓量（现金为 `Money`，证券为 `Quantity`，未持有为 `0`） |
| | [`asset in p`、`p.get`、`iter(p)`、`len(p)`](#passetasset-in-pportfoliogetiterp-与-lenp映射与容器协议) | 成员检测、带默认值查询、按 `(asset, position)` 迭代及持有资产种类数 |
| **入账与平仓核销** | [`Portfolio.add_lot`](#portfolioadd_lot) | 直接加入已构造好的同基准货币 [`PositionLot`](lot.md) |
| | [`Portfolio.deposit`](#portfoliodeposit) | 入账一笔现金或证券，自动构造并追加 [`PositionLot`](lot.md) |
| | [`Portfolio.withdraw`](#portfoliowithdraw) | 按 `FIFO` / `LIFO` / `WAC` 核销持仓批次并计算已实现盈亏 |
| **估值与收益核算** | [`Portfolio.total_value`](#portfoliototal_value) | 按基准货币计算组合当前总市值（[`Money`](money.md)） |
| | [`Portfolio.total_cost`](#portfoliototal_cost) | 按基准货币汇总全部批次的历史总成本（[`Money`](money.md)） |
| | [`Portfolio.unrealized_pnl`](#portfoliounrealized_pnl) | 计算基准货币下的组合未实现总盈亏 `total_value - total_cost`（[`Money`](money.md)） |
| | [`Portfolio.return_rate`](#portfolioreturn_rate) | 计算无量纲组合总收益率 `unrealized_pnl / total_cost`（[`FixedArray`](fixed_array.md)） |
| **组合再平衡** | [`Portfolio.rebalance`](#portfoliorebalance) | 按目标权重字典计算各资产需买入/存入（正）或卖出/取出（负）的数量差额 |
| **运算符与文本协议** | [`p1 + p2`、`p1 - p2`、`str(p)`、`repr(p)`](#p1--p2p1---p2strp-与-reprp运算符拦截与文本表示) | 禁止算术加减合并；提供简明持仓摘要与完整批次构造表示 |
| **内部辅助（私有 API）** | [`Asset`、`Position`](#asset-与-position类型别名) | 模块级类型注解别名 |
| | [`Portfolio._resolve`](#portfolio_resolve) | 将资产对象或代码字符串解析为唯一的 `Currency` 或 `Instrument` |
| | [`Portfolio._position_value` 与 `Portfolio._price_of`](#portfolio_position_value-与-portfolio_price_of) | 单一资产基准货币市值计算 / 在价格字典中按对象或代码查找市价 |
| | [`Portfolio._rate`、`_base_to_local`、`_to_base` 与 `_sum`](#portfolio_rateportfolio_base_to_localportfolio_to_base-与-portfolio_sum) | 汇率提取、无损分式逆向换汇、正向基准货币换算与基准货币累加器 |

## 构造与核心属性

### `Portfolio(base_currency)`

```python
class moneta.Portfolio(base_currency)
```

构造一个空的投资组合实例，绑定指定的组合基准（记账）货币。

**参数**

- **base_currency** : *Currency 或 str* — 组合基准货币（必填，如 `"CNY"`、`"USD"` 或 [`CNY`](currency.md)），内部通过 [`get_currency(base_currency)`](currency.md#get_currency) 解析。

**返回值**

- **out** : *Portfolio*

**异常**

- **TypeError** — 未传入 `base_currency` 或类型不是 `Currency` / `str`。
- **KeyError** — `base_currency` 是未在 [`CurrencyRegistry`](currency.md#currencyregistry) 中注册的货币代码。
- **ValueError** — `base_currency` 格式不合法或与注册表已有定义冲突。

**示例**

```python
>>> from moneta import Portfolio, CNY, USD
>>> p = Portfolio("cny")
>>> p.base_currency is CNY and len(p) == 0
True
>>> Portfolio(USD).base_currency is USD
True
>>> Portfolio("XYZ")
Traceback (most recent call last):
    ...
KeyError: "未注册的货币代码: 'XYZ'"
```

**另请参阅**

- [`Portfolio.deposit`](#portfoliodeposit)
- [`Portfolio.add_lot`](#portfolioadd_lot)

### `Portfolio.base_currency`

```python
Portfolio.base_currency
```

返回组合基准货币（[`Currency`](currency.md) 单例）。所有批次的 `cost_base`、组合总成本 [`total_cost`](#portfoliototal_cost)、总市值 [`total_value`](#portfoliototal_value)、未实现盈亏 [`unrealized_pnl`](#portfoliounrealized_pnl) 及 [`withdraw`](#portfoliowithdraw) 的已实现盈亏均以该币种计价。

**返回值**

- **base_currency** : *Currency*

**示例**

```python
>>> from moneta import Portfolio, CNY
>>> Portfolio("CNY").base_currency is CNY
True
```

### `Portfolio.assets`

```python
Portfolio.assets
```

按首次入账顺序返回当前组合持有的全部资产列表（含现金 [`Currency`](currency.md) 与证券 [`Instrument`](instrument.md)）。当某资产被全额平仓后，不再出现在该列表中。

**返回值**

- **assets** : *List[Union[Currency, Instrument]]* — 新列表副本。

**示例**

```python
>>> from moneta import Instrument, Portfolio, CNY, USD
>>> from moneta.fx import ExchangeRate
>>> aapl = Instrument("AAPL", "USD")
>>> p = Portfolio("CNY")
>>> _ = p.deposit("CNY", 1000, "2024-01-01")
>>> _ = p.deposit(aapl, 10, "2024-01-02", price=180, fx=ExchangeRate("USD", "CNY", "7.0"))
>>> _ = p.deposit(USD, 50, "2024-01-03", fx=ExchangeRate("USD", "CNY", "7.1"))
>>> p.assets == [CNY, aapl, USD]
True
```

**另请参阅**

- [`Portfolio.currencies`](#portfoliocurrencies)、[`Portfolio.instruments`](#portfolioinstruments)

### `Portfolio.currencies`

```python
Portfolio.currencies
```

按首次入账顺序返回当前组合中持有现金余额的全部币种列表（`[a for a in self.assets if isinstance(a, Currency)]`）。

**返回值**

- **currencies** : *List[Currency]*

**示例**

```python
>>> p.currencies == [CNY, USD]
True
```

### `Portfolio.instruments`

```python
Portfolio.instruments
```

按首次入账顺序返回当前组合中持有的全部证券标的列表（`[a for a in self.assets if isinstance(a, Instrument)]`）。

**返回值**

- **instruments** : *List[Instrument]*

**示例**

```python
>>> p.instruments == [aapl]
True
```

## 持仓与批次查询

### `Portfolio.lots`

```python
Portfolio.lots(asset=None)
```

返回指定资产（按入账先后顺序）或组合中全部资产（按资产首次入账顺序及各资产内批次顺序）的 [`PositionLot`](lot.md) 列表副本。

**参数**

- **asset** : *Currency, Instrument, str 或 None，默认 `None`* —
  - 传入 `None` 时：返回组合内所有资产的全部批次构成的扁平列表；
  - 传入 `Currency`、`Instrument` 或代码字符串时：通过 [`_resolve(asset)`](#portfolio_resolve) 解析资产并返回其批次列表（若未持有该已注册币种或该 `Instrument`，返回空列表 `[]`）。

**返回值**

- **lots** : *List[PositionLot]* — 批次列表的浅拷贝副本（修改返回的列表不会影响组合内部状态，且每个 `PositionLot` 本身不可变）。

**异常**

- **KeyError** — `asset` 是未持有的非注册币种字符串，或代码字符串在当前持仓中存在歧义。
- **TypeError** — `asset` 既不是 `None` 也不是 `Currency`、`Instrument` 或 `str`。

**示例**

```python
>>> from moneta import EUR
>>> len(p.lots()), len(p.lots(aapl)), len(p.lots("AAPL")), p.lots(EUR)
(3, 1, 1, [])
>>> copy_list = p.lots(aapl)
>>> copy_list.clear()  # 修改返回副本不影响组合内部批次
>>> len(p.lots(aapl))
1
```

**另请参阅**

- [`PositionLot`](lot.md)
- [`Portfolio.position`](#portfolioposition)

### `Portfolio.position`

```python
Portfolio.position(asset)
```

查询指定资产当前的汇总持仓数量：将名下所有 [`PositionLot.amount`](lot.md#positionlotamount) 精确相加；若当前未持有该资产，则返回对应币种的零金额 `Money(0, currency)` 或对应标的基准单位的零数量 `instrument.as_quantity(0)`。

**参数**

- **asset** : *Currency, Instrument 或 str* — 目标资产或代码字符串（通过 [`_resolve`](#portfolio_resolve) 解析）。

**返回值**

- **out** : *Money 或 Quantity* — 现金资产返回同币种 [`Money`](money.md)；证券资产返回单位为 `instrument.base_unit` 的 [`Quantity`](quantity.md)。

**异常**

- **KeyError** — `asset` 是未持有的非币种字符串（如未持有的股票代码 `"MSFT"`），或存在同名歧义。
- **TypeError** — `asset` 类型不是 `Currency`、`Instrument` 或 `str`。

**示例**

```python
>>> cl = Instrument("CL", "USD", base_unit="barrel", multiplier=1000)
>>> p.position(CNY), p.position("aapl"), p.position(USD)
(Money(FixedArray(1000.00, scale=2), currency='CNY'), Quantity(FixedArray(10, scale=0), unit='share'), Money(FixedArray(50.00, scale=2), currency='USD'))
>>> p.position(EUR), p.position(cl)  # 未持有时返回对应币种或对应基准单位的 0
(Money(FixedArray(0.00, scale=2), currency='EUR'), Quantity(FixedArray(0, scale=0), unit='barrel'))
```

**另请参阅**

- [`p[asset]`、`asset in p` 与 `Portfolio.get`](#passetasset-in-pportfoliogetiterp-与-lenp映射与容器协议)

### `p[asset]`、`asset in p`、`Portfolio.get`、`iter(p)` 与 `len(p)`（映射与容器协议）

```python
p[asset]
asset in p
p.get(asset, default=None)
iter(p)
len(p)
```

`Portfolio` 实现了直观的只读映射与容器协议：

- **`p[asset]`**：完全等价于 [`p.position(asset)`](#portfolioposition)（对已注册但未持有的币种返回 `Money(0, ccy)`，对未持有的 `Instrument` 实例返回 `Quantity(0, unit=inst.base_unit)`，对未持有的未知字符串抛出 `KeyError`）。
- **`asset in p`**：判断组合当前是否实际持有该资产（即存在至少一笔未平仓批次）。对未持有的币种、未持有的 `Instrument`、未知代码字符串或任意其他类型（如整数 `1`）均安全返回 `False`，不抛异常。
- **`p.get(asset, default=None)`**：当 `asset in p` 为 `True` 时返回 `p.position(asset)`，否则返回 `default`（注意：对于未持有的已注册币种如 `EUR`，`EUR in p` 为 `False`，因此 `p.get(EUR)` 返回 `None` 而不是 `0.00 EUR`）。
- **`iter(p)`**：按资产首次入账顺序依次产出 `(asset, position)` 二元组。
- **`len(p)`**：返回当前实际持有的资产种类数（即 `len(p.assets)`）。

**示例**

```python
>>> p["cny"], p["AAPL"]
(Money(FixedArray(1000.00, scale=2), currency='CNY'), Quantity(FixedArray(10, scale=0), unit='share'))
>>> ("AAPL" in p, aapl in p, "MSFT" in p, EUR in p, 123 in p)
(True, True, False, False, False)
>>> p.get("AAPL"), p.get("MSFT", "none"), p.get(EUR)
(Quantity(FixedArray(10, scale=0), unit='share'), 'none', None)
>>> list(p)
[(Currency(code='CNY', digits=2, symbol='¥', name='Chinese Yuan'), Money(FixedArray(1000.00, scale=2), currency='CNY')), (Instrument(symbol='AAPL', quote_currency='USD', base_unit='share', multiplier='1', tick_size=None, name='AAPL', asset_class='Equity'), Quantity(FixedArray(10, scale=0), unit='share')), (Currency(code='USD', digits=2, symbol='$', name='US Dollar'), Money(FixedArray(50.00, scale=2), currency='USD'))]
>>> len(p)
3
```

## 入账与平仓核销

### `Portfolio.add_lot`

```python
Portfolio.add_lot(lot)
```

将一个已构造好的 [`PositionLot`](lot.md) 直接追加到组合中对应资产的批次列表末尾。

**参数**

- **lot** : *PositionLot* — 待加入的持仓批次，其 [`lot.base_currency`](lot.md#positionlotbase_currency) 必须等于组合的 [`self.base_currency`](#portfoliobase_currency-1)。

**返回值**

- **None**

**异常**

- **TypeError** — `lot` 不是 `PositionLot` 实例。
- **ValueError** — `lot.base_currency != self.base_currency`。

**示例**

```python
>>> from moneta import Instrument, Money, Portfolio, PositionLot, Quantity
>>> aapl = Instrument("AAPL", "USD")
>>> pf = Portfolio("CNY")
>>> lot = PositionLot(aapl, Quantity(5), "2024-01-02", cost=Money("900", "USD"), cost_base=Money("6300", "CNY"))
>>> pf.add_lot(lot)
>>> pf[aapl]
Quantity(FixedArray(5, scale=0), unit='share')
>>> pf.add_lot(PositionLot(aapl, Quantity(1), "2024-01-03", cost=Money("180", "USD")))  # cost_base 默认为 USD != CNY
Traceback (most recent call last):
    ...
ValueError: 批次的基准货币 USD 与组合基准货币 CNY 不一致
```

**另请参阅**

- [`Portfolio.deposit`](#portfoliodeposit)
- [`PositionLot`](lot.md)

### `Portfolio.deposit`

```python
Portfolio.deposit(
    asset,
    amount,
    acquired_date,
    *,
    price=None,
    cost=None,
    cost_base=None,
    fx=None,
    label="",
)
```

入账一笔现金或证券持仓，自动完成计价成本与基准货币成本的核算、构造 [`PositionLot`](lot.md) 并追加到组合中，返回新建的 `PositionLot`。若校验或汇率折算失败，组合状态保持原样不变。

**参数**

- **asset** : *Currency, Instrument 或 str* — 入账资产（现金可传 `Currency` 或已注册货币代码字符串；证券须传 `Instrument`，若已持有该证券也可传标的代码字符串）。
- **amount** : *Money, Quantity 或 数值/字符串* — 入账正数标量数量（现金可传同币种 `Money` 或数值/字符串；证券可传兼容单位的 `Quantity` 或按 `asset.base_unit` 解释的数值/字符串）。
- **acquired_date** : *date, datetime 或 str* — 买入或入账日期（当 `fx` 为 `FXEngine` 时，同时作为查询入账日历史汇率的日期，须有当天汇率）。
- **price** : *Price 或 数值/字符串，仅关键字参数，可选* — 证券买入成交单价（与 `cost` 二选一必填；现金禁止传入）。传入时计价货币成本通过 [`asset.calc_notional(price, amount)`](instrument.md#instrumentcalc_notional) 单次舍入计算（自动计入合约乘数 `multiplier`）。
- **cost** : *Money，仅关键字参数，可选* — 证券的计价货币总成本（与 `price` 二选一必填，适用于需将买入手续费计入成本基数的场景；现金禁止传入）。
- **cost_base** : *Money，仅关键字参数，可选* — 组合基准货币表示的总成本。省略时：若计价货币就是 `self.base_currency`，自动等于计价成本；否则通过 `fx` 在 `acquired_date` 的汇率将计价成本单次舍入换算为 `self.base_currency`。
- **fx** : *ExchangeRate 或 FXEngine，仅关键字参数，可选* — 当未显式传入 `cost_base` 且计价货币不等于 `self.base_currency` 时必填。
- **label** : *str，仅关键字参数，默认 `""`* — 交易编号或批次备注。

**返回值**

- **lot** : *PositionLot* — 新建并已加入组合的持仓批次。

**异常**

- **ValueError** — 现金入账传入了 `price` 或 `cost`；证券入账同时省略或同时传入了 `price` 与 `cost`；跨币种入账既未传 `cost_base` 也未传 `fx`；或数量/成本/日期校验不合法。
- **TypeError** — 币种、单位或汇率货币对不匹配。
- **KeyError** — `asset` 是未持有的未知代码字符串。

**示例**

现金入账、带合约乘数的期货入账与含手续费成本入账：

```python
>>> from moneta import Instrument, Money, Portfolio, Price, Quantity, EUR
>>> from moneta.fx import ExchangeRate, FXEngine, UserDictFXProvider
>>> aapl = Instrument("AAPL", "USD")
>>> cl = Instrument("CL", "USD", base_unit="barrel", multiplier=1000)
>>> engine = FXEngine(UserDictFXProvider({"USD/CNY": "7.2", "EUR/CNY": "7.8", "2024-01-02": {"USD/CNY": "7.2", "EUR/CNY": "7.8"}}))
>>> pf = Portfolio("CNY")
>>> l_cash = pf.deposit("EUR", 100, "2024-01-02", fx=engine, label="FX-01")
>>> (l_cash.cost, l_cash.cost_base, pf[EUR])
(Money(FixedArray(100.00, scale=2), currency='EUR'), Money(FixedArray(780.00, scale=2), currency='CNY'), Money(FixedArray(100.00, scale=2), currency='EUR'))
>>> l_fut = pf.deposit(cl, 2, "2024-01-02", price="80.25", fx=engine)  # 2 × 80.25 × 1000 = 160500 USD -> 1155600 CNY
>>> (l_fut.cost, l_fut.cost_base)
(Money(FixedArray(160500.00, scale=2), currency='USD'), Money(FixedArray(1155600.00, scale=2), currency='CNY'))
>>> l_fee = pf.deposit(aapl, 10, "2024-01-03", cost=Money("1899.99", "USD"), cost_base=Money("13680.00", "CNY"))  # 含佣金总成本
>>> l_fee.cost, pf[aapl]
(Money(FixedArray(1899.99, scale=2), currency='USD'), Quantity(FixedArray(10, scale=0), unit='share'))
```

**另请参阅**

- [`Portfolio.add_lot`](#portfolioadd_lot)
- [`Portfolio.withdraw`](#portfoliowithdraw)

### `Portfolio.withdraw`

```python
Portfolio.withdraw(
    asset,
    amount,
    trade_date,
    *,
    price=None,
    fx=None,
    rule=LotMatchingRule.FIFO,
)
```

按指定的批次结转规则 `rule`（[`FIFO`、`LIFO` 或 `WAC`](lot.md#lotmatchingrule)）从组合中核销指定资产的数量 `amount`，返回 `(平掉的批次列表, 已实现盈亏)`。当该资产的剩余持仓归零时，自动将其从组合中移除。

**参数**

- **asset** : *Currency, Instrument 或 str* — 待扣减的已持有资产。
- **amount** : *Money, Quantity 或 数值/字符串* — 扣减的正数标量数量（不得超过当前持仓总量 `self.position(asset)`）。
- **trade_date** : *date, datetime 或 str* — 平仓/出账日期（必填，通过 [`to_date(trade_date, "trade_date")`](_utils.md#to_date) 校验；当 `fx` 为 `FXEngine` 时按该日期取汇率，须有当天汇率）。
- **price** : *Price 或 数值/字符串，仅关键字参数，可选* — 证券卖出成交单价。给出时计算已实现盈亏 `realized`；省略（`price=None`）时视为非交易转出，已实现盈亏返回 `None`。现金批次不接受 `price`。
- **fx** : *ExchangeRate 或 FXEngine，仅关键字参数，可选* — 当计价货币不等于 `self.base_currency` 时：
  - 证券给出了 `price`：必须传入 `fx` 以折算基准货币已实现盈亏（漏传 `fx` 会抛出 `ValueError`）；
  - 外币现金：传入 `fx` 时计算汇兑已实现盈亏，省略 `fx` 时已实现盈亏返回 `None`。
- **rule** : *LotMatchingRule，仅关键字参数，默认 `LotMatchingRule.FIFO`* — 批次结转规则枚举。

**返回值**

- **closed, realized** : *Tuple[List[PositionLot], Optional[Money]]* —
  - `closed`：由 [`match_lots`](lot.md#match_lots) 拆出并平掉的 [`PositionLot`](lot.md) 列表；
  - `realized`：以 `self.base_currency` 计价的已实现盈亏 `Money`（证券未传 `price` 或外币现金未传 `fx` 时为 `None`）。

**异常**

- **ValueError** — 组合未持有该资产；`amount` 超过持仓总量或非正；`trade_date` 非法；或跨币种证券给出了 `price` 却漏传 `fx`。
- **TypeError** — `rule` 不是 [`LotMatchingRule`](lot.md#lotmatchingrule) 实例，或数量/价格/汇率类型不匹配。

**示例**

对比 `FIFO` / `LIFO` / `WAC` 平仓已实现盈亏与非交易转出：

```python
>>> from decimal import Decimal
>>> from moneta import Instrument, LotMatchingRule, Money, Portfolio, Quantity, USD
>>> from moneta.fx import ExchangeRate
>>> aapl = Instrument("AAPL", "USD")
>>> def make_pf():
...     pf = Portfolio("CNY")
...     pf.deposit(aapl, 10, "2024-01-01", price=100, fx=ExchangeRate("USD", "CNY", "7.0"))
...     pf.deposit(aapl, 10, "2024-02-01", price=120, fx=ExchangeRate("USD", "CNY", "7.0"))
...     return pf
>>> fx_sell = ExchangeRate("USD", "CNY", "7.2")
>>> # FIFO 卖出 15 股（成交价 $130，汇率 7.2）：收入 15×130×7.2 = 14040 CNY，成本 7000 + 4200 = 11200 CNY -> 盈亏 +2840 CNY
>>> closed_f, pnl_f = make_pf().withdraw(aapl, 15, "2024-03-01", price=130, fx=fx_sell, rule=LotMatchingRule.FIFO)
>>> pnl_f, [c.amount for c in closed_f]
(Money(FixedArray(2840.00, scale=2), currency='CNY'), [Quantity(FixedArray(10, scale=0), unit='share'), Quantity(FixedArray(5, scale=0), unit='share')])
>>> # LIFO 卖出 15 股：成本 8400 + 3500 = 11900 CNY -> 盈亏 +2140 CNY
>>> _, pnl_l = make_pf().withdraw(aapl, 15, "2024-03-01", price=130, fx=fx_sell, rule=LotMatchingRule.LIFO)
>>> pnl_l
Money(FixedArray(2140.00, scale=2), currency='CNY')
>>> # 不给 price 视为转出：realized 为 None
>>> pf_t = make_pf()
>>> _, pnl_transfer = pf_t.withdraw("AAPL", 5, "2024-03-01")
>>> pnl_transfer is None and bool(pf_t[aapl] == Quantity(15))
True
```

**另请参阅**

- [`match_lots`](lot.md#match_lots)
- [`LotMatchingRule`](lot.md#lotmatchingrule)
- [`PositionLot.profit_loss`](lot.md#positionlotprofit_loss-与-positionlotpnl)

## 估值与收益核算

### `Portfolio.total_value`

```python
Portfolio.total_value(prices=None, fx=None, as_of_date=None)
```

计算组合内全部持仓在基准货币（[`self.base_currency`](#portfoliobase_currency-1)）下的总市值。空组合返回 `Money(0, self.base_currency)`。

**参数**

- **prices** : *Mapping[Union[Instrument, str], Any]，可选* — 证券市价字典，键可为 `Instrument` 实例或标的代码字符串（不区分大小写），值可为 [`Price`](price.md) 或按标的计价货币与基准单位解释的裸数值。当组合持有任何证券标的时，每只证券都必须在 `prices` 中提供市价。
- **fx** : *ExchangeRate 或 FXEngine，可选* — 当组合持有计价货币不等于 `self.base_currency` 的现金或证券时必填。
- **as_of_date** : *date, datetime 或 str，可选* — 估值日期；当 `fx` 为 `FXEngine` 时按该日期查询历史汇率（传入日期时须有当天汇率，省略时使用不带日期的汇率）。

**返回值**

- **out** : *Money* — 币种为 `self.base_currency` 的组合总市值。

**异常**

- **ValueError** — 缺少某持有证券的市价，或缺少跨币种资产的汇率 `fx`。
- **TypeError** — 市价的计价货币/基准单位与标的不匹配，或汇率货币对不匹配。

**示例**

```python
>>> from moneta import Portfolio
>>> p.total_value({"AAPL": 120}, fx=engine)  # 1000 CNY + 10 × 120 × 7.2 CNY + 50 × 7.2 CNY = 10000 CNY
Money(FixedArray(10000.00, scale=2), currency='CNY')
>>> Portfolio("USD").total_value()
Money(FixedArray(0.00, scale=2), currency='USD')
```

**另请参阅**

- [`Portfolio.total_cost`](#portfoliototal_cost)
- [`Portfolio.unrealized_pnl`](#portfoliounrealized_pnl)

### `Portfolio.total_cost`

```python
Portfolio.total_cost()
```

汇总组合内全部持仓批次的基准货币历史成本 [`lot.cost_base`](lot.md#positionlotcost_base) 之和。空组合返回 `Money(0, self.base_currency)`。

**返回值**

- **out** : *Money* — 币种为 `self.base_currency` 的组合总成本。

**示例**

```python
>>> p.total_cost()  # 1000 CNY + 10 × 180 × 7.0 CNY + 50 × 7.1 CNY = 13955.00 CNY
Money(FixedArray(13955.00, scale=2), currency='CNY')
```

### `Portfolio.unrealized_pnl`

```python
Portfolio.unrealized_pnl(prices=None, fx=None, as_of_date=None)
```

计算组合在基准货币下的未实现总盈亏（浮动盈亏）：

$$\text{Unrealized PnL} = \text{total\_value}(\text{prices}, \text{fx}, \text{as\_of\_date}) - \text{total\_cost}()$$

**参数**

- **prices, fx, as_of_date** — 与 [`total_value`](#portfoliototal_value) 完全相同。

**返回值**

- **out** : *Money* — 币种为 `self.base_currency` 的未实现盈亏金额。

**示例**

```python
>>> p.unrealized_pnl({"AAPL": 200}, fx=engine)  # (1000 + 14400 + 360) - 13955 = +1805.00 CNY
Money(FixedArray(1805.00, scale=2), currency='CNY')
```

### `Portfolio.return_rate`

```python
Portfolio.return_rate(prices=None, fx=None, as_of_date=None)
```

计算组合在基准货币下的无量纲总收益率比率：

$$R = \frac{\text{unrealized\_pnl}(\text{prices}, \text{fx}, \text{as\_of\_date})}{\text{total\_cost}()}$$

**参数**

- **prices, fx, as_of_date** — 与 [`total_value`](#portfoliototal_value) 完全相同。

**返回值**

- **out** : *FixedArray* — 无量纲收益率定点数，精度遵循 [`FixedArray.divide`](fixed_array.md#fixedarraydivide) 的默认规则（默认保留 `base_currency.digits + 8` 位小数，放不下 `int64` 时少保留，最后去掉末尾 `0` 且至少保留 `base_currency.digits` 位）。

**异常**

- **ZeroDivisionError** — 当 `self.total_cost()` 为 `0`（如空组合或全为零成本赠股）时抛出。

**示例**

```python
>>> p.return_rate({"AAPL": 200}, fx=engine)  # 1805.00 / 13955.00
FixedArray(0.129344321, scale=9)
>>> Portfolio("USD").return_rate()
Traceback (most recent call last):
    ...
ZeroDivisionError: 除数包含 0
```

## 组合再平衡

### `Portfolio.rebalance`

```python
Portfolio.rebalance(target_weights, prices=None, fx=None, as_of_date=None)
```

按目标资产权重字典 `target_weights` 试算组合再平衡所需的调仓数量差额字典 `{asset: delta}`：正数表示需要买入（证券）或存入（现金），负数表示需要卖出（证券）或取出（现金）。该方法为纯计算方法，不会直接修改组合持仓。

**参数**

- **target_weights** : *Mapping[Union[Currency, Instrument, str], Any]* — 目标权重映射：
  - 键可为 [`Currency`](currency.md)、[`Instrument`](instrument.md) 或已持有资产/已注册币种的代码字符串（不允许对同一资产重复给出权重，例如同时传 `"CNY"` 与 `CNY`）；
  - 值必须属于无量纲数值白名单（如 `Decimal("0.5")`、`0.5`、`Fraction(1, 2)`、`FixedArray("0.5")`，拒绝字符串 `"0.5"` 与缺失值），且每个权重均为 **非负 0 维标量**，全部权重之和必须**严格等于 `1`**；
  - 当前已持有但未出现在 `target_weights` 中的资产，其目标权重自动视为 `0`（即全额清仓卖出/取出）。
- **prices** : *Mapping[Union[Instrument, str], Any]，可选* — 证券市价映射（凡是当前已持有或出现在 `target_weights` 中的证券标的，均须提供市价）。
- **fx** : *ExchangeRate 或 FXEngine，可选* — 凡是涉及非基准货币的现金或证券，均须提供汇率。
- **as_of_date** : *date, datetime 或 str，可选* — 估值与取汇率的日期。

**返回值**

- **adjustments** : *Dict[Asset, Union[Money, Quantity]]* — 按“当前已持有资产顺序 + 新增目标资产顺序”排列的调仓差额字典：
  - 现金资产差额为该币种的 [`Money`](money.md)（按 `half_up` 舍入到该币种精度）；
  - 证券资产差额为该标的 `base_unit` 的 [`Quantity`](quantity.md)（精度遵循 [`FixedArray.divide`](fixed_array.md#fixedarraydivide) 的默认规则，需要对齐整股或整手时可对结果继续调用 [`round_to_step`](quantity.md#quantityround_to_step)）。

**异常**

- **ValueError** — 权重为负数、非 0 维标量、含缺失值、同一资产重复指定、权重合计不等于 `1`，或缺少证券市价/汇率。
- **TypeError** — 权重不在数值白名单内（如传入字符串 `"0.5"`），或价格/汇率量纲不匹配。
- **KeyError** — `target_weights` 中包含未持有且非注册币种的字符串键。

**机制说明**

1. 校验并归一化 `target_weights` 中所有键与权重值，验证 $\sum w_i == 1$；
2. 调用 `total = self.total_value(prices, fx, as_of_date)` 计算组合当前总市值；
3. 遍历每个目标资产：
   - 精确计算基准货币下的目标价值尾数 `num = total.data.mul_round(w, rounding=None)`；
   - 调用 [`_base_to_local(num, local_currency, fx, as_of_date)`](#portfolio_rateportfolio_base_to_localportfolio_to_base-与-portfolio_sum) 将其无损表示为本地货币价值的有理分式 `(num, den)`（正向汇率乘入 `num`，反向汇率放入 `den`，全程不舍入）；
   - 若为现金：执行一次 `num.divide(den, scale=local.digits)` 并减去当前持仓 `self.position(asset)`；
   - 若为证券：先精确计算每单位标的价值 `unit_value = price.data.mul_round(asset.multiplier, rounding=None)`，再将其与 `den` 精确相乘后执行**唯一一次除法** `units = num.divide(den.mul_round(unit_value, rounding=None))`，最后减去当前持仓 `self.position(asset)`。

**示例**

跨币种现金与证券（含新开仓期货标的与清仓外币）再平衡试算：

```python
>>> from decimal import Decimal
>>> from moneta import Instrument, Money, Portfolio, Quantity, CNY, USD
>>> from moneta.fx import ExchangeRate
>>> aapl = Instrument("AAPL", "USD")
>>> pf = Portfolio("CNY")
>>> _ = pf.deposit(CNY, 8000, "2024-01-01")
>>> # 目标配置：50% CNY 现金 (4000 CNY)、25% USD 现金 (2000 CNY = 250 USD)、25% AAPL 股票 (2000 CNY = 250 USD = 5 股 @ $50)
>>> adj = pf.rebalance({CNY: Decimal("0.5"), USD: Decimal("0.25"), aapl: Decimal("0.25")}, {aapl: 50}, fx=ExchangeRate("USD", "CNY", "8"))
>>> adj[CNY], adj[USD], adj[aapl]
(Money(FixedArray(-4000.00, scale=2), currency='CNY'), Money(FixedArray(250.00, scale=2), currency='USD'), Quantity(FixedArray(5.0000, scale=4), unit='share'))
```

**性能对比**

| 组合操作（3 类跨币种资产） | 耗时 | 说明 |
|---|---:|---|
| `p.position(asset)` | 0.27 µs | 汇总该资产名下批次数量 |
| `p.total_cost()` | 21.3 µs | 汇总全部批次 `cost_base` |
| `p.deposit(inst, ...)`（含换汇） | 61.5 µs | 含名义金额计算、汇率折算与批次构造 |
| `p.total_value(prices, engine)` | 97.0 µs | 含证券盯市、汇率查询与基准货币折算 |
| `p.unrealized_pnl(prices, engine)` | 125 µs | 总市值减总成本 |
| `p.return_rate(prices, engine)` | 196 µs | 未实现盈亏除以总成本 |
| `p.withdraw(inst, ..., FIFO)` | 272 µs | 含批次匹配拆分与已实现盈亏核算 |
| `p.rebalance(weights, prices, engine)` | 403 µs | 含总市值重估与单次除法跨币种目标仓位解算 |

**另请参阅**

- [`Portfolio.total_value`](#portfoliototal_value)
- [`Quantity.round_to_step`](quantity.md#quantityround_to_step)

## 运算符拦截与文本表示

### `p1 + p2`、`p1 - p2`、`str(p)` 与 `repr(p)`（运算符拦截与文本表示）

```python
p1 + p2  # TypeError
p1 - p2  # TypeError
str(p)
repr(p)
```

- **算术加减拦截（`__add__` / `__sub__`）**：`Portfolio` 显式禁止 `+` 与 `-` 运算符并抛出 `TypeError`。因为每笔 [`PositionLot`](lot.md) 都绑定了独立的买入日期、历史汇率与成本基数，不能脱离具体交易规则随意合并或相减；增减持仓必须通过 [`deposit`](#portfoliodeposit) / [`add_lot`](#portfolioadd_lot) 与 [`withdraw`](#portfoliowithdraw) 进行。
- **`str(p)`**：返回 `"Portfolio(资产1: 数量1, ..., base=基准币种)"` 的简明持仓摘要；空组合返回 `"Portfolio(Empty, base=基准币种)"`。
- **`repr(p)`**：返回包含 `base_currency` 与完整 `lots` 列表的构造表示字符串。

**示例**

```python
>>> from moneta import Portfolio
>>> str(p)
'Portfolio(CNY: 1000.00 CNY, AAPL: 10 share, USD: 50.00 USD, base=CNY)'
>>> str(Portfolio("USD")), repr(Portfolio("USD"))
('Portfolio(Empty, base=USD)', "Portfolio(base_currency='USD', lots=[])")
>>> Portfolio("USD") + Portfolio("USD")
Traceback (most recent call last):
    ...
TypeError: Portfolio 不支持相加：批次的买入日期与成本不能随意合并，请用 deposit / add_lot 逐笔入账
>>> Portfolio("USD") - Portfolio("USD")
Traceback (most recent call last):
    ...
TypeError: Portfolio 不支持相减：减少持仓必须通过 withdraw 按规则核销批次
```

## 内部辅助方法与类型别名（私有 API）

### `Asset` 与 `Position`（类型别名）

```python
moneta.portfolio.Asset = Union[Currency, Instrument]
moneta.portfolio.Position = Union[Money, Quantity]
```

- **`Asset`**：组合持仓字典的键类型（现金 [`Currency`](currency.md) 或证券 [`Instrument`](instrument.md)）。
- **`Position`**：组合持仓汇总数量类型（现金 [`Money`](money.md) 或证券 [`Quantity`](quantity.md)）。

### `Portfolio._resolve`

```python
Portfolio._resolve(asset)
```

将传入的资产标识解析为规范的 `Currency` 或 `Instrument` 实例：

1. 若 `isinstance(asset, (Currency, Instrument))`，原样返回；
2. 若 `isinstance(asset, str)`，转大写并去空白后先在当前 `_lots` 已持有资产中按 `code` 或 `symbol` 匹配（唯一命中则返回，命中多个则抛出 `KeyError` 提示歧义）；若未持有但 `CurrencyRegistry.has(code)` 为真，则返回 `get_currency(code)`；否则抛出 `KeyError`；
3. 其他类型直接抛出 `TypeError`。

**示例**

```python
>>> from moneta import Instrument, Portfolio, CAD, USD
>>> pf = Portfolio("USD")
>>> cad_stock = Instrument("CAD", "USD")  # 股票代码恰好与加元代码同名
>>> _ = pf.deposit(cad_stock, 3, "2024-01-02", price=10)
>>> pf._resolve("cad") is cad_stock  # 优先命中已持有的股票 CAD
True
>>> _ = pf.deposit(CAD, 5, "2024-01-02", fx=ExchangeRate("CAD", "USD", "0.75"))
>>> pf._resolve("CAD")  # 同时持有股票 CAD 与加元现金 CAD：拒绝猜测并报错
Traceback (most recent call last):
    ...
KeyError: "代码 'CAD' 对应多个持仓 [Instrument(symbol='CAD', ...), Currency(code='CAD', ...)]，请传入 Currency 或 Instrument"
```

### `Portfolio._position_value` 与 `Portfolio._price_of`

```python
Portfolio._position_value(asset, prices, fx, as_of_date)
Portfolio._price_of(instrument, prices)
```

- **`_price_of(instrument, prices)`**：在 `prices` 映射中查找证券标的 `instrument` 的市价——优先按 `instrument in prices` 精确查找，未命中时再遍历字符串键按 `key.strip().upper() == instrument.symbol` 不区分大小写匹配；仍未找到则抛出 `ValueError(f"缺少标的 {instrument.symbol} 的市价")`。
- **`_position_value(asset, prices, fx, as_of_date)`**：计算单个持有资产在组合基准货币下的市值——现金直接取持仓额，证券调用 `asset.calc_notional(self._price_of(asset, prices), position)`，最后经由 [`_to_base`](#portfolio_rateportfolio_base_to_localportfolio_to_base-与-portfolio_sum) 折算为基准货币。

**示例**

```python
>>> from moneta import Instrument, Portfolio
>>> aapl = Instrument("AAPL", "USD")
>>> Portfolio._price_of(aapl, {" aapl ": 195})
195
>>> Portfolio._price_of(aapl, {})
Traceback (most recent call last):
    ...
ValueError: 缺少标的 AAPL 的市价
```

### `Portfolio._rate`、`Portfolio._base_to_local`、`Portfolio._to_base` 与 `Portfolio._sum`

```python
Portfolio._rate(fx, currency, as_of_date)
Portfolio._base_to_local(value, currency, fx, as_of_date)
Portfolio._to_base(money, fx, as_of_date)
Portfolio._sum(values)
```

- **`_rate(fx, currency, as_of_date)`**：提取 `currency` 与 `self.base_currency` 之间的 `ExchangeRate`。若 `fx is None` 抛出 `ValueError`；若 `hasattr(fx, "get_rate")`（如 [`FXEngine`](fx.md#fxengineprovider-base_anchor_currencyusd)），则调用 [`to_date(as_of_date, "as_of_date", optional=True)`](_utils.md#to_date) 解析后调用 `fx.get_rate(currency, self.base_currency, as_of_date=when)`；否则直接返回 `fx`。
- **`_base_to_local(value, currency, fx, as_of_date)`**：供 [`rebalance`](#portfoliorebalance) 将基准货币价值 `value`（`FixedArray`）转换为本地货币价值的**精确分子与分母二元组 `(num, den)`**（不执行任何除法舍入）：同币种返回 `(value, FixedArray(1))`；正向汇率 `(base_currency, currency)` 返回 `(value.mul_round(rate.rate, rounding=None), FixedArray(1))`；反向汇率 `(currency, base_currency)` 返回 `(value, rate.rate)`。
- **`_to_base(money, fx, as_of_date)`**：若 `money.currency == self.base_currency` 原样返回 `money`，否则调用 [`_convert_at_rate(money, self._rate(fx, money.currency, as_of_date), self.base_currency)`](lot.md#_convert_at_rate) 单次舍入换算为基准货币。
- **`_sum(values)`**：以 `Money(0, self.base_currency)` 为初值，将可迭代对象 `values` 中的基准货币 `Money` 依次相加。

**示例**

```python
>>> from moneta import Portfolio, USD
>>> from moneta.fx import ExchangeRate
>>> pf = Portfolio("CNY")
>>> pf._base_to_local(FixedArray("720.00"), USD, ExchangeRate("USD", "CNY", "7.20"), None)
(FixedArray(720.00, scale=2), FixedArray(7.20, scale=2))
>>> pf._base_to_local(FixedArray("720.00"), USD, ExchangeRate("CNY", "USD", "0.125"), None)
(FixedArray(90.00000, scale=5), FixedArray(1, scale=0))
```
