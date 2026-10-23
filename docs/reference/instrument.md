# Instrument — 金融标的物合约定义与名义金额计算

`moneta.Instrument` 表示除法定现金货币（[`Currency`](currency.md)）之外的独立金融标的物不可变参考合约（如股票、期货合约、大宗商品、加密资产、期权）。
它集中保存标的代码（`symbol`）、计价/结算货币（`quote_currency`）、交易基准单位（`base_unit`）、合约乘数（`multiplier`）、最小变动价位（
`tick_size`）及资产全称与大类，
并提供按本标的量纲构造持仓数量（[`as_quantity`](#instrumentas_quantity)）、市场报价（[`as_price`](#instrumentas_price)）以及执行
**单次舍入**名义金额估值（[`calc_notional`](#instrumentcalc_notional)）的工厂与核算方法。

```python
>>> from decimal import Decimal
>>> from moneta import Instrument, Price, Quantity, USD
>>> cl = Instrument("cl2412", "usd", base_unit="contract", multiplier=1000, tick_size=Decimal("0.01"), name="WTI Crude Dec24", asset_class="Commodity")
>>> cl.symbol, cl.quote_currency is USD, cl.base_unit, cl.multiplier, cl.tick_size
('CL2412', True, 'contract', FixedArray(1000, scale=0), FixedArray(0.01, scale=2))
>>> cl.as_quantity([2, 5])
Quantity(FixedArray([2, 5], scale=0), unit='contract')
>>> cl.as_price("80.23").round_to_step()
Price(FixedArray(80.23, scale=2), quote_currency='USD', base_unit='contract')
>>> cl.calc_notional("80.25", 3)  # 80.25 × 3 × 1000 = 240750.00 USD
Money(FixedArray(240750.00, scale=2), currency='USD')
```

[← API 参考](index.md)

## 核心机制与设计约束

### 不可变合约实体与七槽位结构

每个 [`Instrument`](#instrumentsymbol-quote_currency-base_unitshare-multiplier1-tick_sizenone-name-asset_classequity)
实例包含七个只读槽位（
`__slots__ = ("_symbol", "_quote_currency", "_base_unit", "_multiplier", "_tick_size", "_name", "_asset_class")`
），在构造时一次性完成全部校验与归一化：

- **`symbol`**：资产代码。必须为 `str`，构造时自动去除首尾空白并转为大写（`symbol.strip().upper()`），去除空白后不能为空字符串。
- **`quote_currency`**：计价与结算货币（必填，无默认币种）。通过 [`get_currency`](currency.md#get_currency) 解析为 [
  `Currency`](currency.md) 实例。
- **`base_unit`**：报价与持仓的基准单位（默认 `"share"`）。通过 [`check_unit`](_units.md#check_unit) 去除首尾空白并校验非空，且
  **严禁使用通配单位 `"Any"`**（[`WILDCARD_UNIT`](_units.md#wildcard_unit)）——具体金融合约必须绑定明确的物理或合约单位（如
  `"share"`、`"barrel"`、`"contract"`、`"BTC"`）。
- **`multiplier` 与 `tick_size`**：合约乘数（每 1 个 `base_unit` 对应的标的数量，默认 `1`）与最小变动价位（默认 `None`
  表示不限制）。均通过 [`Instrument._positive_scalar`](#instrument_positive_scalar) 校验：必须属于数值白名单（拒绝 `str`、
  `bool` 及带单位对象）、为 **0 维非缺失正数标量**，并完整保留输入自身的十进制精度 `scale`（不会静默取整）。
- **`name` 与 `asset_class`**：资产全称与资产大类。必须为 `str`，去除首尾空白后若为空字符串，则 `name` 自动回退为大写
  `symbol`，`asset_class` 自动回退为 `"Equity"`。

`Instrument` 在构造后完全 **不可变**：重写了 `__setattr__` 与 `__delattr__`，任何试图修改或删除属性的操作均直接抛出
`AttributeError`。

### 身份标识语义（相等性与哈希仅看代码与计价货币）

在投资组合（`Portfolio`）与持仓批次（[`PositionLot`](lot.md)）管理中，`Instrument` 充当非现金资产的 **唯一身份键（Identity Key）**
。同一标的代码在同一计价货币下代表同一交易品种，而全称、乘数或 `tick_size` 属于合约参考数据（Reference Data）：

| 操作                                     | 行为与语义                                                                                                                                                                                    |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `i1 == i2`、`i1 != i2`                   | **仅比较 `(symbol, quote_currency)`**；`base_unit`、`multiplier`、`tick_size`、`name`、`asset_class` 不参与相等性判断。与非 `Instrument` 对象（如字符串 `"AAPL"`）比较返回 `False` / `True`。 |
| `hash(inst)`                             | 基于 `hash((inst.symbol, inst.quote_currency.code))` 计算，与 `==` 严格一致，可直接作为持仓字典 `dict` 的键或放入 `set`。                                                                     |
| `copy.copy(inst)`、`copy.deepcopy(inst)` | 直接返回 `inst` 自身，不创建新实例。                                                                                                                                                          |
| `pickle.loads(pickle.dumps(inst))`       | 通过 `__reduce__` 完整还原全部七个字段（含 `multiplier` 与 `tick_size` 的精确 `scale`）。                                                                                                     |
| `str(inst)`、`repr(inst)`                | `str` 返回 `"SYMBOL (Name, CCY/unit)"` 简明摘要；`repr` 返回包含全部七个参数的完整构造形式。                                                                                                  |

### 按标的构造数量、价格与一次舍入名义金额（Notional）

`Instrument` 将合约元数据与 [`Quantity`](quantity.md)、[`Price`](price.md)、[`Money`](money.md) 的量纲体系连接起来：

| 方法                                                                                  | 输入类型                              | 返回类型                  | 核心语义与精度规则                                                                                                                                          |
|---------------------------------------------------------------------------------------|---------------------------------------|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`inst.as_quantity(values, scale=None)`](#instrumentas_quantity)                      | 数值、字符串、序列或 `Quantity`       | [`Quantity`](quantity.md) | 自动绑定 `unit=inst.base_unit`；若传入 `Quantity` 则校验单位兼容（或将 `"Any"` 具体化为 `inst.base_unit`）                                                  |
| [`inst.as_price(values, scale=None)`](#instrumentas_price)                            | 数值、字符串、序列或 `Price`          | [`Price`](price.md)       | 裸数据自动绑定 `quote_currency`、`base_unit` 与 `tick_size`；已是 `Price` 时校验币种与单位兼容（`scale=None` 时零拷贝原样返回）                             |
| [`inst.calc_notional(price, quantity, rounding="half_up")`](#instrumentcalc_notional) | `Price` 或裸数据、`Quantity` 或裸数据 | [`Money`](money.md)       | 计算 $\text{price} \times \text{quantity} \times \text{multiplier}$，全程保留精确中间乘积，**仅在最后一步按 `rounding` 舍入一次**到 `quote_currency.digits` |

**为什么需要 `calc_notional` 的单次舍入机制？**
若先计算 `p * q` 得到 `Money`（此时已按币种精度舍入了一次），再乘以含有小数的 `multiplier`（如迷你合约乘数 `0.5` 或 `0.1`
），会发生 **二次舍入误差（Double Rounding）**。例如当价格为 `0.333`、数量为 `3`、合约乘数为 `0.5` 时：

- 真实精确乘积为 $0.333 \times 3 \times 0.5 = 0.4995$，按 `half_up` 舍入到美分应得 **`0.50 USD`**；
- `calc_notional` 先用 `p.data.mul_round(q.data, rounding=None)` 算出精确中间积 `0.999`，再调用
  `exact.mul_round(self._multiplier, scale=self._quote_currency.digits, rounding=rounding)` 一次性舍入为 `0.50 USD`
  ，彻底消除中间舍入偏差。

## API 速查表

| 分类                     | 接口 / 运算符                                                                                                                                                                                                            | 功能说明                                                                   |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| **构造与核心属性**       | [`Instrument(symbol, quote_currency, base_unit="share", multiplier=1, tick_size=None, name="", asset_class="Equity")`](#instrumentsymbol-quote_currency-base_unitshare-multiplier1-tick_sizenone-name-asset_classequity) | 构造不可变的金融标的物参考数据合约                                         |
|                          | [`Instrument.symbol`](#instrumentsymbol)                                                                                                                                                                                 | 标准化后的大写资产代码（如 `'AAPL'`、`'CL2412'`）                          |
|                          | [`Instrument.quote_currency`](#instrumentquote_currency)                                                                                                                                                                 | 计价与结算货币 [`Currency`](currency.md) 单例                              |
|                          | [`Instrument.base_unit`](#instrumentbase_unit)                                                                                                                                                                           | 报价与持仓基准单位字符串（默认 `'share'`，禁止 `'Any'`）                   |
|                          | [`Instrument.multiplier`](#instrumentmultiplier)                                                                                                                                                                         | 合约乘数（0 维正数 [`FixedArray`](fixed_array.md)，默认 `1`）              |
|                          | [`Instrument.tick_size`](#instrumenttick_size)                                                                                                                                                                           | 最小变动价位（0 维正数 [`FixedArray`](fixed_array.md)，未设置时为 `None`） |
|                          | [`Instrument.name`](#instrumentname)                                                                                                                                                                                     | 资产全称（未传或空白时默认等于 `symbol`）                                  |
|                          | [`Instrument.asset_class`](#instrumentasset_class)                                                                                                                                                                       | 资产大类（未传或空白时默认为 `'Equity'`）                                  |
| **数量、价格与名义金额** | [`Instrument.as_quantity`](#instrumentas_quantity)                                                                                                                                                                       | 以本标的 `base_unit` 构造持仓数量 [`Quantity`](quantity.md)                |
|                          | [`Instrument.as_price`](#instrumentas_price)                                                                                                                                                                             | 以本标的计价货币、基准单位与 `tick_size` 构造或校验 [`Price`](price.md)    |
|                          | [`Instrument.calc_notional`](#instrumentcalc_notional)                                                                                                                                                                   | 计算 `价格 × 数量 × 合约乘数` 并单次舍入为计价货币 [`Money`](money.md)     |
| **比较、序列化与展示**   | [`i1 == i2`、`i1 != i2`、`hash(inst)`](#i1--i2i1--i2-与-hashinst相等与哈希)                                                                                                                                              | 基于 `(symbol, quote_currency)` 的相等判断与哈希支持                       |
|                          | [`copy`、`deepcopy`、`pickle`](#copydeepcopy-与-pickle拷贝与序列化)                                                                                                                                                      | 浅/深拷贝返回自身，`pickle` 完整还原合约元数据                             |
|                          | [`str(inst)`、`repr(inst)`](#strinst-与-reprinst文本表示)                                                                                                                                                                | 简明可读摘要与完整构造表达式展示                                           |
| **内部辅助（私有 API）** | [`Instrument._positive_scalar`](#instrument_positive_scalar)                                                                                                                                                             | 校验并转换数值白名单内的 0 维非缺失正数标量                                |

## 构造与核心属性

### `Instrument(symbol, quote_currency, base_unit="share", multiplier=1, tick_size=None, name="", asset_class="Equity")`

```python
class moneta.Instrument(symbol, quote_currency, base_unit="share", multiplier=1, tick_size=None, name="", asset_class="Equity")
```

构造金融标的物不可变参考数据合约。

**参数**

- **symbol** : *str* — 资产代码（如 `"AAPL"`、`"BTC"`、`"CL2412"`）。首尾空白会被自动去除并统一转为大写，不能为空字符串。
- **quote_currency** : *Currency 或 str* — 计价/结算货币（如 `"USD"`、`CNY`），通过 [
  `get_currency`](currency.md#get_currency) 解析。
- **base_unit** : *str，默认 `"share"`* — 报价与持仓的基准单位（如 `"share"`、`"barrel"`、`"contract"`
  ）。自动去除首尾空白，不能为空，且不能是通配单位 `"Any"`（[`WILDCARD_UNIT`](_units.md#wildcard_unit)）。
- **multiplier** : *int, float, Decimal, Fraction 或 FixedArray，默认 `1`* — 合约乘数（每 1 个 `base_unit`
  对应的标的数量）。必须为数值白名单内的 0 维非缺失正数标量，不接受字符串或布尔值。
- **tick_size** : *int, float, Decimal, Fraction, FixedArray 或 None，默认 `None`* — 最小变动价位（如 `Decimal("0.01")`
  ）。传入非 `None` 时必须为数值白名单内的 0 维非缺失正数标量；默认为 `None`（不限制）。
- **name** : *str，默认 `""`* — 资产全称（如 `"Apple Inc."`）。首尾空白会被去除；若为空或全空白，则默认设为标准化后的
  `symbol`。
- **asset_class** : *str，默认 `"Equity"`* — 资产大类（如 `"Equity"`、`"Commodity"`、`"Crypto"`、`"Option"`
  ）。首尾空白会被去除；若为空或全空白，则默认设为 `"Equity"`。

**返回值**

- **out** : *Instrument* — 不可变的金融标的物实例。

**异常**

- **TypeError** — `symbol`、`base_unit`、`name` 或 `asset_class` 不是 `str`；`quote_currency` 类型非法；或 `multiplier` /
  `tick_size` 不在数值白名单内（如传入了字符串 `"1"`、`bool` 或 `Money`）。
- **KeyError** — `quote_currency` 是未在 [`CurrencyRegistry`](currency.md#currencyregistry) 中注册的货币代码字符串。
- **ValueError** — `symbol` 或 `base_unit` 去除空白后为空字符串；`base_unit` 为通配单位 `"Any"`；`multiplier` 或
  `tick_size` 不是 0 维标量、含有缺失值（如 `None`、`float("nan")`）或 `<= 0`；或 `quote_currency` 与注册表定义冲突。
- **AttributeError** — 构造后尝试对任何属性赋值（`setattr`）或删除属性（`delattr`）。

**机制说明**

构造器内部依次执行以下校验与绑定：

1. 检查 `isinstance(symbol, str)`，计算 `code = symbol.strip().upper()` 并校验非空；
2. 调用 [`check_unit(base_unit)`](_units.md#check_unit) 规范化单位字符串，并检查 `unit == WILDCARD_UNIT`（若为 `"Any"` 则抛出
   `ValueError`）；
3. 校验 `name` 与 `asset_class` 均为 `str`；
4. 通过 `object.__setattr__` 依次写入 `_symbol`、`get_currency(quote_currency)`、`_base_unit`、[
   `_positive_scalar(multiplier, "multiplier")`](#instrument_positive_scalar)、`_tick_size`、`name.strip() or code` 以及
   `asset_class.strip() or "Equity"`。

**示例**

默认参数归一化与完整合约构造：

```python
>>> from decimal import Decimal
>>> from fractions import Fraction
>>> from moneta import Instrument, CNY, USD
>>> aapl = Instrument("  aapl ", "usd")
>>> (aapl.symbol, aapl.quote_currency is USD, aapl.base_unit, aapl.multiplier, aapl.tick_size, aapl.name, aapl.asset_class)
('AAPL', True, 'share', FixedArray(1, scale=0), None, 'AAPL', 'Equity')
>>> oil = Instrument("cl2412", CNY, base_unit=" barrel ", multiplier=1000, tick_size=Decimal("0.5"), name=" Crude Oil ", asset_class="Commodity")
>>> (oil.symbol, oil.quote_currency is CNY, oil.base_unit, oil.multiplier, oil.tick_size, oil.name, oil.asset_class)
('CL2412', True, 'barrel', FixedArray(1000, scale=0), FixedArray(0.5, scale=1), 'Crude Oil', 'Commodity')
>>> Instrument("BTC", "USD", base_unit="BTC", multiplier=Fraction(1, 4), name="   ", asset_class="  ").multiplier
FixedArray(0.25, scale=2)
```

不可变性保护与非法参数拦截：

```python
>>> aapl.symbol = "MSFT"
Traceback (most recent call last):
    ...
AttributeError: Instrument 是不可变对象，禁止设置属性 'symbol'
>>> del aapl._symbol
Traceback (most recent call last):
    ...
AttributeError: Instrument 是不可变对象，禁止删除属性 '_symbol'
>>> Instrument("   ", "USD")
Traceback (most recent call last):
    ...
ValueError: 标的代码不能为空
>>> Instrument("AAPL", "USD", base_unit="Any")
Traceback (most recent call last):
    ...
ValueError: 标的的基准单位必须是具体单位，不能是通配单位 'Any'
>>> Instrument("AAPL", "USD", multiplier=0)
Traceback (most recent call last):
    ...
ValueError: multiplier 必须为正数
>>> Instrument("AAPL", "USD", tick_size="0.01")
Traceback (most recent call last):
    ...
TypeError: tick_size 只接受数值（int、float、Decimal、Fraction、FixedArray 或数值数组），收到 str
```

**另请参阅**

- [`Instrument.as_quantity`](#instrumentas_quantity)、[`Instrument.as_price`](#instrumentas_price)、[
  `Instrument.calc_notional`](#instrumentcalc_notional)
- [`Instrument._positive_scalar`](#instrument_positive_scalar)

### Instrument.symbol

```python
Instrument.symbol
```

返回标准化后的大写资产代码字符串（如 `'AAPL'`、`'CL2412'`）。

**返回值**

- **symbol** : *str*

**示例**

```python
>>> from moneta import Instrument
>>> Instrument(" tsla ", "USD").symbol
'TSLA'
```

### Instrument.quote_currency

```python
Instrument.quote_currency
```

返回标的的计价/结算货币 [`Currency`](currency.md) 单例。

**返回值**

- **quote_currency** : *Currency*

**示例**

```python
>>> from moneta import Instrument, HKD
>>> Instrument("0700", "hkd").quote_currency is HKD
True
```

### Instrument.base_unit

```python
Instrument.base_unit
```

返回标的的报价与持仓基准单位字符串（默认 `'share'`，恒为非 `"Any"` 的具体单位）。

**返回值**

- **base_unit** : *str*

**示例**

```python
>>> from moneta import Instrument
>>> Instrument("AAPL", "USD").base_unit, Instrument("CL", "USD", base_unit="barrel").base_unit
('share', 'barrel')
```

### Instrument.multiplier

```python
Instrument.multiplier
```

返回合约乘数（每 1 个 `base_unit` 对应的标的数量），类型为 0 维正数 [`FixedArray`](fixed_array.md)，完整保留构造时的十进制小数位数
`scale`。

**返回值**

- **multiplier** : *FixedArray*

**示例**

```python
>>> from decimal import Decimal
>>> from moneta import Instrument
>>> Instrument("ES", "USD", base_unit="contract", multiplier=50).multiplier
FixedArray(50, scale=0)
>>> Instrument("MINI", "USD", multiplier=Decimal("0.10")).multiplier
FixedArray(0.10, scale=2)
```

### Instrument.tick_size

```python
Instrument.tick_size
```

返回最小变动价位（0 维正数 [`FixedArray`](fixed_array.md)）；若构造时未设置（`tick_size=None`），则返回 `None`。调用 [
`as_price`](#instrumentas_price) 从数值构造 [`Price`](price.md) 时会自动将该 `tick_size` 注入生成的 `Price` 实例。

**返回值**

- **tick_size** : *FixedArray 或 None*

**示例**

```python
>>> from decimal import Decimal
>>> from moneta import Instrument
>>> Instrument("AAPL", "USD").tick_size is None
True
>>> Instrument("AAPL", "USD", tick_size=Decimal("0.01")).tick_size
FixedArray(0.01, scale=2)
```

### Instrument.name

```python
Instrument.name
```

返回资产全称字符串；若构造时未传入或传入空白字符串，则等于 `self.symbol`。

**返回值**

- **name** : *str*

**示例**

```python
>>> from moneta import Instrument
>>> Instrument("aapl", "USD").name, Instrument("aapl", "USD", name="Apple Inc.").name
('AAPL', 'Apple Inc.')
```

### Instrument.asset_class

```python
Instrument.asset_class
```

返回资产大类字符串（默认 `'Equity'`）。

**返回值**

- **asset_class** : *str*

**示例**

```python
>>> from moneta import Instrument
>>> Instrument("AAPL", "USD").asset_class, Instrument("BTC", "USD", asset_class="Crypto").asset_class
('Equity', 'Crypto')
```

## 按标的构造数量、价格与名义金额

### Instrument.as_quantity

```python
Instrument.as_quantity(values, scale=None)
```

以本标的的基准单位（[`self.base_unit`](#instrumentbase_unit)）构造持仓数量 [`Quantity`](quantity.md)。

**参数**

- **values** : *int, float, str, Decimal, Fraction, FixedArray, Quantity, Sequence 或 np.ndarray* — 数量标量或数组（支持缺失值
  `<NA>`）。
- **scale** : *int，可选* — 目标小数位数。默认 `None` 时保留输入自身精度；显式指定时，浮点数静默按 `half_up`
  舍入，精确输入若丢精度则发出 [`PrecisionLossWarning`](fixed_array.md#precisionlosswarning) 后按 `half_up` 舍入。

**返回值**

- **out** : *Quantity* — `unit == self.base_unit` 的 [`Quantity`](quantity.md) 实例。

**异常**

- **TypeError** — `values` 是单位与 `self.base_unit` 不兼容的 `Quantity`，或包含非法类型（如 `bool`、`Money`）。
- **ValueError** — `scale` 不是非负整数。

**警告**

- **[`PrecisionLossWarning`](fixed_array.md#precisionlosswarning)** — 显式指定 `scale` 且非浮点的精确输入在多出的小数位上含有非零数字时发出。

**机制说明**

直接委托调用 [`Quantity(values, unit=self._base_unit, scale=scale)`](quantity.md#quantityvalues-unitshare-scalenone)。

**示例**

```python
>>> import warnings
>>> from decimal import Decimal
>>> from moneta import Instrument, PrecisionLossWarning
>>> cl = Instrument("CL", "USD", base_unit="barrel")
>>> q = cl.as_quantity(["1.5", 2])
>>> q, q.unit, q.mantissa.tolist()
(Quantity(FixedArray([1.5, 2.0], scale=1), unit='barrel'), 'barrel', [15, 20])
>>> cl.as_quantity(Decimal("3.20"), scale=1)
Quantity(FixedArray(3.2, scale=1), unit='barrel')
>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     q2 = cl.as_quantity(Decimal("3.25"), scale=1)
...     (q2, len(w), issubclass(w[0].category, PrecisionLossWarning))
(Quantity(FixedArray(3.3, scale=1), unit='barrel'), 1, True)
```

**另请参阅**

- [`Quantity(values, unit="share", scale=None)`](quantity.md#quantityvalues-unitshare-scalenone)
- [`Instrument.as_price`](#instrumentas_price)

### Instrument.as_price

```python
Instrument.as_price(values, scale=None)
```

以本标的的计价货币（[`quote_currency`](#instrumentquote_currency)）、基准单位（[`base_unit`](#instrumentbase_unit)
）与最小变动价位（[`tick_size`](#instrumenttick_size)）构造市场报价 [`Price`](price.md)；若 `values` 已是 `Price`
，则校验其计价货币与单位是否与本标的兼容。

**参数**

- **values** : *int, float, str, Decimal, Fraction, FixedArray, Price, Sequence 或 np.ndarray* — 报价数值、序列或已有的
  `Price` 实例。
- **scale** : *int，可选* — 目标小数位数。默认 `None` 时保留输入自身精度。

**返回值**

- **out** : *Price*

**异常**

- **TypeError** — `values` 是 `Price` 但 `quote_currency != self.quote_currency`，或 `values.base_unit` 既不等于
  `self.base_unit` 也不是通配单位 `"Any"`；或 `values` 是 `Money` / `Quantity` 等不兼容类型。
- **ValueError** — `scale` 不是非负整数。

**警告**

- **[`PrecisionLossWarning`](fixed_array.md#precisionlosswarning)** — 显式指定 `scale` 且非浮点的精确输入在多出的小数位上含有非零数字时发出。

**机制说明**

1. **当 `isinstance(values, Price)` 时**：
    - 校验 `values.quote_currency == self._quote_currency`，否则抛出 `TypeError`；
   - 调用 [`units_compatible(values.base_unit, self._base_unit)`](_units.md#units_compatible) 校验单位兼容，否则抛出
     `TypeError`；
    - 若 `scale is None`， **直接原样返回 `values`**（零拷贝透传，保留 `values` 原有的 `base_unit` 与 `tick_size`）；否则调用
      `Price(values, self._quote_currency, values.base_unit, scale=scale)` 调整精度。
2. **当 `values` 不是 `Price` 时**：
    - 调用 [
      `Price(values, self._quote_currency, base_unit=self._base_unit, tick_size=self._tick_size, scale=scale)`](price.md#pricevalues-quote_currency-base_unitshare-tick_sizenone-scalenone)
      ，自动绑定本标的的 `tick_size`。

**示例**

从数值构造（自动带入合约 `tick_size`）与已有 `Price` 校验透传：

```python
>>> from decimal import Decimal
>>> from moneta import Instrument, Price, CNY
>>> cl = Instrument("CL", "CNY", base_unit="barrel", tick_size=Decimal("0.5"))
>>> p = cl.as_price("80.25")
>>> (p.quote_currency is CNY, p.base_unit, p.tick_size, p.round_to_step())
(True, 'barrel', FixedArray(0.5, scale=1), Price(FixedArray(80.50, scale=2), quote_currency='CNY', base_unit='barrel'))
>>> aapl = Instrument("AAPL", "USD")
>>> aapl.as_price(189.505, scale=2)
Price(FixedArray(189.51, scale=2), quote_currency='USD', base_unit='share')
>>> given = Price("189.5", "USD")
>>> aapl.as_price(given) is given  # scale=None 时零拷贝原样返回
True
>>> aapl.as_price(Price("1.5", "USD", base_unit="Any")).base_unit  # 通配单位兼容
'Any'
>>> aapl.as_price(Price("189.5", "CNY"))
Traceback (most recent call last):
    ...
TypeError: 价格计价货币 CNY 与标的 AAPL 的计价货币 USD 不一致
>>> aapl.as_price(Price("189.5", "USD", base_unit="barrel"))
Traceback (most recent call last):
    ...
TypeError: 价格单位 'barrel' 与标的 AAPL 的基准单位 'share' 不一致
```

**另请参阅**

- [`Price(values, quote_currency, ...)`](price.md#pricevalues-quote_currency-base_unitshare-tick_sizenone-scalenone)
- [`Instrument.calc_notional`](#instrumentcalc_notional)

### Instrument.calc_notional

```python
Instrument.calc_notional(price, quantity, rounding="half_up")
```

计算持仓或交易的名义金额（Notional Value）：

$$\text{Notional} = \text{price} \times \text{quantity} \times \text{multiplier}$$

全程保留 `price × quantity` 的完整精确中间位，仅在最后乘以合约乘数 `multiplier` 时按 `rounding` **只舍入一次**到计价货币的法定精度
`self.quote_currency.digits`，返回 [`Money`](money.md)。

**参数**

- **price** : *Price，或 int, float, str, Decimal, Fraction, FixedArray, Sequence, np.ndarray* — 市场价格或成交价。若是
  `Price`，其计价货币须等于 `self.quote_currency` 且 `base_unit` 须等于 `self.base_unit` 或 `"Any"`；若是裸数值或字符串，则按
  `self.quote_currency` 与 `self.base_unit` 解释。
- **quantity** : *Quantity，或 int, float, str, Decimal, Fraction, FixedArray, Sequence, np.ndarray* — 持仓或交易数量。若是
  `Quantity`，其 `unit` 须等于 `self.base_unit` 或 `"Any"`；若是裸数值或字符串，则按 `self.base_unit` 解释。支持与 `price`
  广播及缺失值 `<NA>` 传播。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 最终舍入到计价货币精度时采用的舍入模式。

**返回值**

- **out** : *Money* — 币种为 `self.quote_currency`、`scale == self.quote_currency.digits` 的金额张量。

**异常**

- **TypeError** — `price` 的计价货币或单位与本标的不一致；`quantity` 的单位与本标的 `base_unit` 不一致；或将 `Money`
  等无关带单位对象传入 `price` / `quantity`。
- **ValueError** — `price` 与 `quantity` 形状无法广播，或 `rounding` 不是合法的舍入模式。
- **OverflowError** — 最终金额超出 `int64` 表示范围。

**机制说明**

1. 调用 `p = self.as_price(price)` 解析并校验价格；
2. 若 `quantity` 已是 `Quantity` 则直接使用，否则调用 `self.as_quantity(quantity)`，随后通过 [
   `units_compatible(q.unit, self._base_unit)`](_units.md#units_compatible) 校验单位兼容；
3. 调用 [`exact = p.data.mul_round(q.data, rounding=None)`](fixed_array.md#fixedarraymul_round)
   精确计算 $\text{price} \times \text{quantity}$（`scale = p.scale + q.scale`，无任何中间舍入）；
4. 调用 `total = exact.mul_round(self._multiplier, scale=self._quote_currency.digits, rounding=rounding)`
   乘以合约乘数并一次性舍入到币种精度；
5. 调用 [`Money._wrap(total, self._quote_currency)`](money.md#money_wrap) 零拷贝封装为 `Money`。

**示例**

带合约乘数的期货名义价值与裸数值便捷调用：

```python
>>> from decimal import Decimal
>>> from moneta import Instrument, Money, Price, Quantity
>>> cl = Instrument("CL2412", "USD", base_unit="contract", multiplier=1000)
>>> cl.calc_notional(Price("80.25", "USD", base_unit="contract"), Quantity(3, unit="contract"))
Money(FixedArray(240750.00, scale=2), currency='USD')
>>> cl.calc_notional("80.25", 3)  # 裸数字/字符串自动按 USD 与 contract 解释
Money(FixedArray(240750.00, scale=2), currency='USD')
```

单次舍入消除二次舍入偏差（对比 `0.333 × 3 × 0.5 = 0.4995`）：

```python
>>> inst = Instrument("AAPL", "USD", multiplier=Decimal("0.5"))
>>> inst.calc_notional(Decimal("0.333"), 3)  # 0.4995 一次 half_up -> 0.50 USD
Money(FixedArray(0.50, scale=2), currency='USD')
>>> inst.calc_notional(Decimal("0.333"), 3, rounding="down")
Money(FixedArray(0.49, scale=2), currency='USD')
>>> Instrument("NK", "JPY", multiplier=Decimal("0.1")).calc_notional("12.34", 5)  # 6.17 -> 6 JPY
Money(FixedArray(6, scale=0), currency='JPY')
```

数组广播、缺失值 `<NA>` 传播与量纲安检：

```python
>>> aapl = Instrument("AAPL", "USD", multiplier=100)
>>> got = aapl.calc_notional(Price([None, "1.50"], "USD"), Quantity([2, 3]))
>>> got, got.isna().tolist(), got.units.tolist()
(Money(FixedArray([<NA>, 450.00], scale=2), currency='USD'), [True, False], [0, 45000])
>>> aapl.calc_notional("189.50", Quantity(10, unit="barrel"))
Traceback (most recent call last):
    ...
TypeError: 数量单位 'barrel' 与标的 AAPL 的基准单位 'share' 不一致
>>> aapl.calc_notional(Money("189.50", "USD"), 10)
Traceback (most recent call last):
    ...
TypeError: Money 不能隐式转成数组：要 float 请用 to_float()，要整数尾数请用 .mantissa
```

**另请参阅**

- [`Instrument.as_price`](#instrumentas_price)、[`Instrument.as_quantity`](#instrumentas_quantity)
- [`FixedArray.mul_round`](fixed_array.md#fixedarraymul_round)

## 相等、哈希、序列化与文本协议

### `i1 == i2`、`i1 != i2` 与 `hash(inst)`（相等与哈希）

```python
i1 == i2
i1 != i2
hash(inst)
```

按 `(symbol, quote_currency)` 判断两个 `Instrument` 是否代表同一标的，并支持作为持仓字典键或集合元素。

**参数**

- **other** : *Any* — 另一个比较对象。

**返回值**

- **out** : *bool 或 int* — `==` / `!=` 返回 `bool`；`hash(inst)` 返回 `int`。

**机制说明**

- `__eq__` 首先检查 `isinstance(other, Instrument)`，若不是则返回 `NotImplemented`（因此 `inst == "AAPL"` 返回 `False`）；
- 当且仅当 `self._symbol == other._symbol and self._quote_currency == other._quote_currency` 时返回 `True`；参考数据字段
  `base_unit`、`multiplier`、`tick_size`、`name`、`asset_class` 不参与比较；
- `__hash__` 返回 `hash((self._symbol, self._quote_currency.code))`。

**示例**

```python
>>> from decimal import Decimal
>>> from moneta import Instrument, USD
>>> a = Instrument("aapl", "USD", name="Apple Inc.", multiplier=1)
>>> b = Instrument("AAPL", USD, name="Apple", multiplier=100, tick_size=Decimal("0.01"))
>>> a == b and hash(a) == hash(b) and len({a, b}) == 1
True
>>> {a: 100}[b]  # 可直接作为持仓字典的键
100
>>> a != Instrument("AAPL", "CNY") and a != Instrument("MSFT", "USD") and (a == "AAPL") is False
True
```

### copy、deepcopy 与 pickle（拷贝与序列化）

```python
copy.copy(inst)
copy.deepcopy(inst)
pickle.dumps(inst) / pickle.loads(data)
```

浅拷贝与深拷贝直接返回 `self`；`pickle` 反序列化完整还原全部七个合约属性。

**返回值**

- **out** : *Instrument*

**机制说明**

- `__copy__` 与 `__deepcopy__` 均直接返回 `self`；
- `__reduce__` 返回
  `(Instrument, (self._symbol, self._quote_currency, self._base_unit, self._multiplier, self._tick_size, self._name, self._asset_class))`
  ，保证反序列化后不仅身份相等，`multiplier` 与 `tick_size` 的 `FixedArray` 精度 `scale` 也完全一致。

**示例**

```python
>>> import copy, pickle
>>> from decimal import Decimal
>>> from moneta import Instrument
>>> inst = Instrument("CL2412", "USD", base_unit="barrel", multiplier=1000, tick_size=Decimal("0.01"), name="Crude", asset_class="Commodity")
>>> copy.copy(inst) is inst and copy.deepcopy(inst) is inst
True
>>> restored = pickle.loads(pickle.dumps(inst))
>>> restored == inst and repr(restored) == repr(inst) and bool(restored.tick_size == inst.tick_size)
True
```

### `str(inst)` 与 `repr(inst)`（文本表示）

```python
str(inst)
repr(inst)
```

`str(inst)` 返回 `"{symbol} ({name}, {quote_currency}/{base_unit})"` 简明描述；`repr(inst)` 返回带全部七个字段的构造表达式字符串。

**返回值**

- **out** : *str*

**示例**

```python
>>> from decimal import Decimal
>>> from moneta import Instrument
>>> inst = Instrument("CL2412", "USD", base_unit="barrel", multiplier=1000, tick_size=Decimal("0.01"), name="Crude")
>>> str(inst)
'CL2412 (Crude, USD/barrel)'
>>> repr(inst)
"Instrument(symbol='CL2412', quote_currency='USD', base_unit='barrel', multiplier='1000', tick_size='0.01', name='Crude', asset_class='Equity')"
>>> repr(Instrument("AAPL", "USD"))
"Instrument(symbol='AAPL', quote_currency='USD', base_unit='share', multiplier='1', tick_size=None, name='AAPL', asset_class='Equity')"
```

## 内部辅助方法（私有 API）

### Instrument._positive_scalar

```python
@staticmethod
Instrument._positive_scalar(value, what)
```

校验 `value` 属于无量纲数值白名单、非缺失、为 0 维标量且严格大于 0，返回保留输入自身精度的 [`FixedArray`](fixed_array.md)。

**参数**

- **value** : *Any* — 待校验的数值（如 `multiplier` 或 `tick_size`）。
- **what** : *str* — 字段名称（如 `"multiplier"` 或 `"tick_size"`），用于拼接错误提示。

**返回值**

- **out** : *FixedArray* — 0 维正数 `FixedArray`（不含缺失值，`scale` 保留原值，浮点数按最短十进制写法解析）。

**异常**

- **TypeError** — `value` 不在数值白名单内（如字符串、`bool`、`Money`，由 [
  `UnitArray._as_grid`](unit_array.md#unitarray_as_grid-与-unitarray_fill_value) 抛出）。
- **ValueError** — `value` 是缺失值（`None`、`NaN`）、不是 0 维标量（`fa.ndim != 0`），或 `fa.mantissa <= 0`。

**机制说明**

1. 调用 [`fa = UnitArray._as_grid(value, f"{what} ")`](unit_array.md#unitarray_as_grid-与-unitarray_fill_value)
   将输入转换为无缺失值的 `FixedArray`；
2. 检查 `fa.ndim != 0`（拒绝 1 维及以上数组）与 `fa.mantissa <= 0`（拒绝 `0` 与负数）。

**示例**

```python
>>> from decimal import Decimal
>>> from moneta import Instrument
>>> Instrument._positive_scalar(Decimal("0.25"), "multiplier")
FixedArray(0.25, scale=2)
>>> Instrument._positive_scalar([1, 2], "multiplier")
Traceback (most recent call last):
    ...
ValueError: multiplier 必须是标量
>>> Instrument._positive_scalar(None, "multiplier")
Traceback (most recent call last):
    ...
ValueError: multiplier 不能是缺失值
```

**另请参阅**

- [`UnitArray._as_grid`](unit_array.md#unitarray_as_grid-与-unitarray_fill_value)
