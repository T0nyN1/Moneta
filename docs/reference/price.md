# Price — 单位标的报价与 Tick 报价网格

`moneta.Price` 表示单位标的物的金融报价（强度量 / Intensive Quantity），复合量纲为 **`[计价货币 / 标的单位]`**（如
`USD/share`、`USD/barrel`、`CNY/g`），继承自 [`UnitArray`](unit_array.md)。
与 [`Quantity`](quantity.md) 一样，`Price` 不強制绑定固定小数位数（加减与乘以无量纲系数默认完全精确）；当与 [
`Quantity`](quantity.md) 相乘时，标的单位相消并按计价币种精度自动舍入为成交金额 [`Money`](money.md)。
此外，`Price` 支持可选的最小变动价位元数据 `tick_size`，可通过统一的 [`round_to_step`](#priceround_to_step) 与 [
`is_on_step`](#priceis_on_step) 对齐或检验报价网格。

```python
>> > from decimal import Decimal
>> > from moneta import Price, Quantity, Money
>> > p = Price(["189.50", "190.25"], "USD", base_unit="share", tick_size=Decimal("0.05"))
>> > p.quote_currency.code, p.base_unit, p.tick_size
('USD', 'share', FixedArray(0.05, scale=2))
>> > p * Quantity([10, 20], unit="share")  # Price * Quantity -> Money (成交金额)
Money(FixedArray([1895.00, 3805.00], scale=2), currency='USD')
>> > ask, bid = Price("101.25", "USD"), Price("101.20", "USD")
>> > ask - bid  # 同量纲价差 (Spread)
Price(FixedArray(0.05, scale=2), quote_currency='USD', base_unit='share')
>> > Price("10.23", "USD", tick_size=Decimal("0.05")).round_to_step()
Price(FixedArray(10.25, scale=2), quote_currency='USD', base_unit='share')
```

[← API 参考](index.md)

## 核心机制与设计约束

### 复合量纲 `[quote_currency / base_unit]` 与双向通配

每个 `Price` 实例携带两个量纲维度：

- **`quote_currency`**：计价货币（[`Currency`](currency.md) 单例，构造时通过 [`get_currency`](currency.md#get_currency)
  解析）；
- **`base_unit`**：计价标的基准单位（区分大小写的非空字符串，默认 `"share"`，通过 [`check_unit`](_units.md#check_unit)
  去除首尾空白）。

**通配单位 `"Any"`（[`WILDCARD_UNIT`](_units.md#wildcard_unit)）双向兼容规则：**

1. **两个 `Price` 之间（加减、比率除法、比较）**：当且仅当 `quote_currency` 相同，且 `base_unit` 相同或任一方为 `"Any"`（见 [
   `units_compatible`](_units.md#units_compatible)）时视为量纲兼容。加减运算后的 `base_unit` 通过 [
   `_resolve_unit`](#price_resolve_unit) 自动收敛为非 `"Any"` 的具体单位。
2. **`Price` 与 `Quantity` 相乘（计算成交金额）**：只要 `price.base_unit == qty.unit`，或其中任一方为 `"Any"`，即可相乘消去标的单位，得到以
   `price.quote_currency` 计价的 [`Money`](money.md)。

### 精度传播与量纲代数规则

| 运算                                                       | 另一操作数类型                       | 返回类型                       | 结果精度与舍入规则                                                                             |
|------------------------------------------------------------|--------------------------------------|--------------------------------|------------------------------------------------------------------------------------------------|
| `p1 + p2`、`p1 - p2`                                       | 同计价币种、兼容单位的 `Price`       | `Price`                        | `max(p1.scale, p2.scale)`，精确整数加减                                                        |
| `+p`、`-p`、`abs(p)`、`p.sum()`                            | —                                    | `Price`                        | `p.scale`，精确（继承自 [`UnitArray`](unit_array.md)）                                         |
| `p * qty`、`qty * p`                                       | 兼容单位的 [`Quantity`](quantity.md) | [`Money`](money.md)            | 固定为 `quote_currency.digits`，默认 `half_up`（`multiply` 可指定 `rounding`，不接受 `scale`） |
| `p * k`、`k * p`                                           | 无量纲数值                           | `Price`                        | `p.scale + k.scale`，**精确相乘不舍入**                                                        |
| `p.multiply(k, scale=None, rounding="half_up")`            | 无量纲数值                           | `Price`                        | `scale=None` 时精确；指定 `scale` 时按 `rounding` 一次性舍入到 `scale`                         |
| `p1 / p2`、`p1.divide(p2, scale=None, rounding="half_up")` | 同计价币种、兼容单位的 `Price`       | [`FixedArray`](fixed_array.md) | 量纲相消得无量纲比率，默认 `max(p1.scale, p2.scale) + 8`，可用 `scale` 指定                    |
| `p / k`、`p.divide(k, scale=None, rounding="half_up")`     | 无量纲数值                           | `Price`                        | 默认 `max(p.scale, k.scale) + 8`，可用 `scale` 指定并按 `rounding` 舍入                        |
| `p.round_to_step(step=None, rounding="half_up")`           | 正数步长或 `None`                    | `Price`                        | `step=None` 时对齐到自带 `tick_size`（无 `tick_size` 则原样返回）；传入 `step` 时对齐到 `step` |

### `tick_size` 元数据语义与传播规则

`tick_size`（最小变动价位，如股票的 `0.01`、外汇的 `0.0001` 或可广播的逐元素步长数组）在 `Price` 中作为 **可选正数元数据**（
`FixedArray` 或 `None`）管理，遵循以下四条约定：

1. **构造时不强制对齐网格**：`Price("1.234", "USD", tick_size=Decimal("0.01"))`
   不会在构造时自动舍入或报错，便于精确承载均价、理论定价或中间点差；需要对齐或检查网格时显式调用 [
   `p.round_to_step()`](#priceround_to_step) 或 [`p.is_on_step()`](#priceis_on_step)。
2. **不参与相等性比较**：`p1 == p2` 只比较量纲（`quote_currency`、`base_unit`）与数值（`_data`），即使两个 `Price` 的
   `tick_size` 不同（或一方为 `None`），只要价格数值与量纲相同即判定相等。
3. **一元运算、切片求和与无量纲乘除保留原 `tick_size`**：`-p`、`abs(p)`、`p[key]`、`p.sum()`、`p * 2`、`p / 2`、
   `p.round_to_step(...)` 均自动沿用原 `p.tick_size`。
4. **两个 `Price` 加减时相同则保留、不同则清空**：`p1 + p2` 与 `p1 - p2` 通过 [`_resolve_tick`](#price_resolve_tick)
   判断——仅当两侧均设置了 `tick_size` 且形状与数值完全相同时（如 `Decimal("0.01")` 与 `Decimal("0.010")`）结果保留该
   `tick_size`，否则结果的 `tick_size` 设为 `None`。

## API 速查表

| 分类                   | 接口 / 运算符                                                                                                                                        | 功能说明                                                                                  |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **构造与核心属性**     | [`Price(values, quote_currency, base_unit="share", tick_size=None, scale=None)`](#pricevalues-quote_currency-base_unitshare-tick_sizenone-scalenone) | 构造带计价货币、基准单位与可选 `tick_size` 的标价张量                                     |
|                        | [`Price.quote_currency`](#pricequote_currency)                                                                                                       | 计价货币 [`Currency`](currency.md) 单例                                                   |
|                        | [`Price.base_unit`](#pricebase_unit)                                                                                                                 | 计价标的基准单位字符串（默认 `"share"`）                                                  |
|                        | [`Price.tick_size`](#pricetick_size)                                                                                                                 | 最小变动价位 `FixedArray`（未设置时为 `None`）                                            |
|                        | [继承自 `UnitArray` 的数组属性](#继承自-unitarray-的数组属性)                                                                                        | `data`、`mantissa`、`scale`、`shape`、`ndim`、`size`、`dtype`、`T`                        |
| **报价网格（Tick）**   | [`Price.round_to_step`](#priceround_to_step)                                                                                                         | 按自带 `tick_size`（默认）或显式 `step` 对齐到步长网格                                    |
|                        | [`Price.is_on_step`](#priceis_on_step)                                                                                                               | 检查价格是否严格落在自带 `tick_size`（默认）或显式 `step` 网格上                          |
| **算术与跨量纲运算**   | [`p1 + p2`、`p1 - p2`](#p1--p2-与-p1---p2同量纲价格加减与点差)                                                                                       | 同计价币种、兼容单位的价格精确加减（计算点差或涨跌额）                                    |
|                        | [`+p`、`-p`、`abs(p)`](#p-p-与-absp正负号与绝对值)                                                                                                   | 一元正负号与逐元素绝对值（保留 `tick_size`）                                              |
|                        | [`p * other`、`other * p`](#p--other-与-other--p成交金额计算与精确缩放)                                                                              | 乘以 [`Quantity`](quantity.md) 得成交金额 [`Money`](money.md)；乘以无量纲数得精确 `Price` |
|                        | [`Price.multiply`](#pricemultiply)                                                                                                                   | 支持自定义 `rounding`（乘 `Quantity` 或数值）与 `scale`（仅乘数值）的乘法                 |
|                        | [`p / other`](#p--other真除法运算符)                                                                                                                 | 除以同类 `Price` 得无量纲比率 [`FixedArray`](fixed_array.md)；除以无量纲数得 `Price`      |
|                        | [`Price.divide`](#pricedivide)                                                                                                                       | 支持自定义 `scale` 与 `rounding` 的价格除法                                               |
| **规约、比较与导出**   | [`Price.sum`、`Price.is_zero`](#pricesum-与-priceis_zero)                                                                                            | 沿轴精确求和与零值判定（继承自 [`UnitArray`](unit_array.md)）                             |
|                        | [比较、索引与形状变换](#比较索引与形状变换)                                                                                                          | `==`、`!=`、`<`、`<=`、`>`、`>=`、`p[key]`、`reshape`、`transpose`、`len`、`bool`、`iter` |
|                        | [`Price.to_pandas`](#priceto_pandas)                                                                                                                 | 导出为 `pandas.Series` / `DataFrame` 并自动注入 `quote_currency` 与 `base_unit`           |
|                        | [`Price.to_polars`、`to_float`、`to_decimal`](#priceto_polarspriceto_float-与-priceto_decimal)                                                       | 导出为 `polars.Series`、`float64` 或 `Decimal`                                            |
|                        | [`str(p)`、`repr(p)`](#strp-与-reprp文本表示)                                                                                                        | `"数值 币种/单位"` 简明文本与完整构造表达式展示                                           |
| **内部钩子与私有 API** | [`Price._as_tick`](#price_as_tick)                                                                                                                   | 校验 `tick_size` / `step` 在数值白名单内且严格为正数                                      |
|                        | [`Price._create_like`](#price_create_like)                                                                                                           | `UnitArray` 钩子：用新 `FixedArray` 与可选单位/`tick_size` 零拷贝构造 `Price`             |
|                        | [`Price._check_compatible`](#price_check_compatible)                                                                                                 | `UnitArray` 钩子：校验另一对象是否为同币种、兼容单位的 `Price`                            |
|                        | [`Price._resolve_unit`](#price_resolve_unit)                                                                                                         | 合并两个兼容单位（优先取非 `"Any"` 的具体单位）                                           |
|                        | [`Price._resolve_tick`](#price_resolve_tick)                                                                                                         | 合并两个 `Price` 的 `tick_size`（相等则保留，否则返回 `None`）                            |
|                        | [`_KEEP` 与 `_RATIO_EXTRA_DIGITS`](#_keep-与-_ratio_extra_digits)                                                                                    | `_create_like` 默认沿用原 `tick` 的哨兵对象与除法额外保留位数常量（`8`）                  |

## 构造与核心属性

### `Price(values, quote_currency, base_unit="share", tick_size=None, scale=None)`

```python
class moneta.Price(values, quote_currency, base_unit="share", tick_size=None, scale=None)
```

构造表示 `[计价货币 / 标的单位]` 的金融标价张量。

**参数**

- **values** : *int, float, str, Decimal, Fraction, FixedArray, Price, Sequence 或 np.ndarray* — 报价数值或矩阵。
- **quote_currency** : *Currency 或 str* — 计价货币（如 `"USD"`、`USD`），通过 [`get_currency`](currency.md#get_currency)
  解析。
- **base_unit** : *str，默认 `"share"`* — 计价标的基准单位（如 `"share"`、`"barrel"`、`"oz"` 或通配单位 `"Any"`
  ）。自动去除首尾空白，区分大小写，不能为空。
- **tick_size** : *数值或数值数组，可选* — 最小变动价位（如 `Decimal("0.01")` 或逐元素步长数组），必须严格为正数且属于数值白名单（不接受字符串）。默认
  `None`。
- **scale** : *int，可选* — 目标小数位数。默认 `None` 时保留输入自身的精度；指定时，浮点数按最短十进制写法 `half_up`
  静默舍入，精确输入若在多出的小数位上含有非零数字（缺失位置与末尾多余的 `0` 除外）则先发出 [
  `PrecisionLossWarning`](fixed_array.md#precisionlosswarning) 再按 `half_up`
  舍入，不足时补零。

**返回值**

- **out** : *Price*

**异常**

- **TypeError** — `base_unit` 不是 `str`；`tick_size` 不是合法数值（如字符串）；`values` 是量纲不匹配的 `Price`；或 `values`
  是 `bool` 等非法类型。
- **KeyError** — `quote_currency` 是未注册的货币代码字符串。
- **ValueError** — `base_unit` 为空字符串；`tick_size <= 0` 或不是标量；`scale` 不是非负整数；或 `quote_currency` 与注册表定义冲突。
- **OverflowError** — 底层尾数超出 `int64` 范围。

**警告**

- **[`PrecisionLossWarning`](fixed_array.md#precisionlosswarning)** — 显式指定 `scale` 且非浮点的精确输入在超出 `scale`
  的小数位上含有非零数字时发出。

**机制说明**

1. 依次通过 [`get_currency(quote_currency)`](currency.md#get_currency)、[`check_unit(base_unit)`](_units.md#check_unit)
   与 [`self._as_tick(tick_size)`](#price_as_tick) 解析并校验计价币种、基准单位与正数标量 `tick_size`；
2. 若 `values` 已是 `Price`：校验 `values._quote_currency == currency` 且 [
   `units_compatible(values._base_unit, unit)`](_units.md#units_compatible)，否则抛 `TypeError`；取 `data = values._data`
   ，且若本次未显式传入 `tick_size` 则自动继承 `values._tick_size`；否则取 `data = values`；
3. 直接调用 `self._data = FixedArray(data, scale=scale, on_excess="warn")`：指定 `scale` 时浮点数静默舍入，精确输入丢精度时由
   `FixedArray` 发出 [`PrecisionLossWarning`](fixed_array.md#precisionlosswarning) 后按 `half_up` 舍入。

**示例**

基本构造、指定 `scale` 与从已有 `Price` 构造：

```python
>>> import warnings
>>> from decimal import Decimal
>>> from moneta import Price, USD, PrecisionLossWarning
>>> p = Price("189.50", "usd", base_unit=" barrel ", tick_size=Decimal("0.01"))
>>> (p.quote_currency is USD, p.base_unit, p.scale, p.tick_size)
(True, 'barrel', 2, FixedArray(0.01, scale=2))
>>> Price(1.005, "USD", scale=2)     # float 指定 scale 静默按最短写法 half_up 舍入
Price(FixedArray(1.01, scale=2), quote_currency='USD', base_unit='share')
>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     p2 = Price("1.235", "USD", scale=2)
...     (p2, len(w), issubclass(w[0].category, PrecisionLossWarning))
(Price(FixedArray(1.24, scale=2), quote_currency='USD', base_unit='share'), 1, True)
>>> Price(p, "USD", base_unit="barrel").tick_size  # 从已有 Price 构造默认继承 tick_size
FixedArray(0.01, scale=2)
```

**性能对比**

|         N | `Price(float_arr, "USD")` | `Price(FixedArray, "USD")`（零拷贝包装） |
|----------:|--------------------------:|-----------------------------------------:|
|     1,000 |                   55.7 µs |                                   650 ns |
| 1,000,000 |                   36.4 ms |                                   650 ns |

**另请参阅**

- [`Price.quote_currency`](#pricequote_currency)、[`Price.base_unit`](#pricebase_unit)、[
  `Price.tick_size`](#pricetick_size)
- [`Price.round_to_step`](#priceround_to_step)

### Price.quote_currency

```python
Price.quote_currency
```

返回计价货币 [`Currency`](currency.md) 单例（分子维度）。

**返回值**

- **quote_currency** : *Currency*

**示例**

```python
>>> from moneta import Price, USD
>>> Price("100.00", "usd").quote_currency is USD
True
```

### Price.base_unit

```python
Price.base_unit
```

返回计价标的基准单位字符串（分母维度，默认 `"share"`）。

**返回值**

- **base_unit** : *str*

**示例**

```python
>>> Price("80.00", "USD", base_unit="barrel").base_unit
'barrel'
```

### Price.tick_size

```python
Price.tick_size
```

返回最小变动价位 [`FixedArray`](fixed_array.md)；若构造时未设置则返回 `None`。

**返回值**

- **tick_size** : *FixedArray 或 None*

**示例**

```python
>>> from decimal import Decimal
>>> Price("10.00", "USD", tick_size=Decimal("0.05")).tick_size
FixedArray(0.05, scale=2)
>>> Price("10.00", "USD").tick_size is None
True
```

### 继承自 UnitArray 的数组属性

`Price` 继承自 [`UnitArray`](unit_array.md)，直接提供 [`data`](unit_array.md#unitarraydata)、[
`mantissa`](unit_array.md#unitarraymantissa)、[`scale`](unit_array.md#unitarrayscale)、[
`shape`](unit_array.md#unitarrayshape)、[`ndim`](unit_array.md#unitarrayndim)、[`size`](unit_array.md#unitarraysize)、[
`dtype`](unit_array.md#unitarraydtype) 与转置视图 [`T`](unit_array.md#unitarrayt)。

**示例**

```python
>>> p = Price([["10.5", "11.0"], ["12.0", "12.5"]], "USD", base_unit="barrel")
>>> (p.scale, p.shape, p.ndim, p.size)
(1, (2, 2), 2, 4)
>>> p.T
Price(FixedArray([[10.5, 12.0],
            [11.0, 12.5]], scale=1), quote_currency='USD', base_unit='barrel')
```

## 报价网格（Tick 对齐与检验）

### Price.round_to_step

```python
Price.round_to_step(step=None, rounding="half_up")
```

将价格对齐到步长网格：当 `step=None`（默认）时使用实例自带的 [`tick_size`](#pricetick_size)（若 `tick_size` 也为 `None`
则原样返回 `self`）；当显式传入 `step` 时按该正数步长对齐（且不会改动结果实例自带的 `tick_size` 属性）。

**参数**

- **step** : *数值或数值 array_like，可选* — 显式指定的正数步长（可广播，不接受字符串）。默认 `None` 时取 `self.tick_size`
  。注意第一个位置参数是 `step`，若只传舍入模式请使用关键字参数 `p.round_to_step(rounding="down")`。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式（即使 `tick_size` 与 `step` 均为 `None`，也会先校验
  `rounding` 合法性）。

**返回值**

- **out** : *Price* — 同计价货币、同 `base_unit`、保留原 `tick_size` 的对齐后价格。

**异常**

- **TypeError** — `step` 不是数值白名单类型（例如误将舍入模式字符串传给了第一个位置参数 `step`）。
- **ValueError** — `step <= 0` 或 `rounding` 无效。

**机制说明**

先通过 [`check_rounding(rounding)`](_rounding.md#check_rounding) 校验舍入模式，再取
`grid = self._tick_size if step is None else self._as_tick(step, "step")`。若 `grid is None` 直接返回 `self`；否则返回
`self._create_like(self._data.round_to_step(grid, rounding=mode))`。

**示例**

使用自带 `tick_size`（含逐元素 `tick_size`）与显式覆盖 `step`：

```python
>>> from decimal import Decimal
>>> p = Price(["1.234", "1.235", "-1.235"], "USD", tick_size=Decimal("0.01"))
>>> p.round_to_step()
Price(FixedArray([1.230, 1.240, -1.240], scale=3), quote_currency='USD', base_unit='share')
>>> p.round_to_step(rounding="down")
Price(FixedArray([1.230, 1.230, -1.230], scale=3), quote_currency='USD', base_unit='share')
>>> p.round_to_step(Decimal("0.05"))  # 临时按 0.05 对齐，但结果的 tick_size 仍保持 0.01
Price(FixedArray([1.250, 1.250, -1.250], scale=3), quote_currency='USD', base_unit='share')
>>> p.round_to_step(Decimal("0.05")).tick_size
FixedArray(0.01, scale=2)
>> > Price(["10.23", "10.23"], "USD").round_to_step([Decimal("0.05"), Decimal("0.10")])  # 逐元素步长通过 step= 传入
Price(FixedArray([10.25, 10.20], scale=2), quote_currency='USD', base_unit='share')
```

**性能对比**

|         N | `p.round_to_step()` (`Price`) | `np.round(x / 0.05) * 0.05` (`NumPy float64`) |
|----------:|------------------------------:|----------------------------------------------:|
|  0-d 标量 |                       24.7 µs |                                       1.23 µs |
|     1,000 |                       34.7 µs |                                       2.34 µs |
| 1,000,000 |                       12.2 ms |                                       1.86 ms |

**另请参阅**

- [`Price.is_on_step`](#priceis_on_step)
- [`Price.tick_size`](#pricetick_size)
- [`FixedArray.round_to_step`](fixed_array.md#fixedarrayround_to_step)

### Price.is_on_step

```python
Price.is_on_step(step=None)
```

检查价格是否严格落在步长网格上：当 `step=None`（默认）时检查是否落在自带的 `tick_size` 网格上（若未设置 `tick_size`，0 维返回
`True`，高维返回全 `True` 布尔数组）；当显式传入 `step` 时检查是否为 `step` 的整数倍。

**参数**

- **step** : *数值或数值 array_like，可选* — 正数步长。默认 `None` 时取 `self.tick_size`。

**返回值**

- **out** : *bool 或 numpy.ndarray（bool）*

**异常**

- **TypeError** — `step` 不是数值白名单类型。
- **ValueError** — `step <= 0`。

**示例**

```python
>>> from decimal import Decimal
>>> p = Price(["10.20", "10.23"], "USD", tick_size=Decimal("0.05"))
>>> p.is_on_step()
array([ True, False])
>>> p.is_on_step(Decimal("0.01"))
array([ True,  True])
>>> Price("1.234", "USD").is_on_step()  # 未设置 tick_size 时恒为 True
True
```

**另请参阅**

- [`Price.round_to_step`](#priceround_to_step)

## 算术与跨量纲运算

### p1 + p2 与 p1 - p2（同量纲价格加减与点差）

```python
p1 + p2
p1 - p2
```

同计价货币且 `base_unit` 兼容（相同或任一方为 `"Any"`）的两个 `Price` 逐元素精确相加或相减（如计算买卖盘点差 `ask - bid`
或价格变动）。

**参数**

- **other** : *Price* — 同计价货币、兼容单位的 `Price`，可广播。

**返回值**

- **out** : *Price* — `scale = max(p1.scale, p2.scale)`，`base_unit` 取具体单位（[`_resolve_unit`](#price_resolve_unit)），
  `tick_size` 在两侧相同时保留、否则为 `None`（[`_resolve_tick`](#price_resolve_tick)）。

**异常**

- **TypeError** — `other` 的 `quote_currency` 或 `base_unit` 不匹配，或 `other` 不是 `Price`（如裸数字、`Money`、
  `Quantity`）。
- **ValueError** — 形状无法广播。
- **OverflowError** — 结果超出 `int64` 范围。

**示例**

```python
>>> from decimal import Decimal
>>> ask = Price("101.25", "USD", tick_size=Decimal("0.05"))
>>> bid = Price("101.20", "USD", tick_size=Decimal("0.05"))
>>> spread = ask - bid
>>> spread, spread.tick_size
(Price(FixedArray(0.05, scale=2), quote_currency='USD', base_unit='share'), FixedArray(0.05, scale=2))
>>> Price("10.00", "USD", base_unit="Any") + Price("2.50", "USD", base_unit="barrel")
Price(FixedArray(12.50, scale=2), quote_currency='USD', base_unit='barrel')
>>> Price("10.00", "USD") - Price("10.00", "CNY")
Traceback (most recent call last):
    ...
TypeError: 价格量纲不匹配: USD/share vs CNY/share
```

**性能对比**

|         N | `p1 + p2` (`Price`) | `x + y` (`NumPy float64`) |
|----------:|--------------------:|--------------------------:|
|  0-d 标量 |             7.72 µs |                   43.9 ns |
|     1,000 |             9.26 µs |                    494 ns |
| 1,000,000 |             3.12 ms |                    742 µs |

**另请参阅**

- [`Price._resolve_unit`](#price_resolve_unit)、[`Price._resolve_tick`](#price_resolve_tick)

### `+p`、`-p` 与 `abs(p)`（正负号与绝对值）

```python
+p
-p
abs(p)
```

直接继承自 [`UnitArray`](unit_array.md#a-与--a一元正负号)，保留原 `quote_currency`、`base_unit`、`tick_size` 与 `scale`。

**示例**

```python
>>> from decimal import Decimal
>>> p = Price("-1.50", "USD", tick_size=Decimal("0.05"))
>>> abs(p), abs(p).tick_size
(Price(FixedArray(1.50, scale=2), quote_currency='USD', base_unit='share'), FixedArray(0.05, scale=2))
```

### p * other 与 other * p（成交金额计算与精确缩放）

```python
p * other
other * p
```

按另一操作数类型自动分派两种乘法语义：

1. **`Price * Quantity` 或 `Quantity * Price` → [`Money`](money.md)（成交金额）**：当 `other` 为 [`Quantity`](quantity.md)
   且单位兼容（`other.unit == p.base_unit` 或任一方为 `"Any"`）时，标的单位相消，乘积按 `half_up` 舍入到
   `p.quote_currency.digits`，返回对应币种的 `Money`；
2. **`Price * 无量纲数值` 或 `无量纲数值 * Price` → `Price`（价格缩放）**：当 `other` 为无量纲数值或数值数组时，
   **默认精确相乘不舍入**（`scale = p.scale + factor.scale`），返回保留原 `quote_currency`、`base_unit` 与 `tick_size` 的
   `Price`。

**参数**

- **other** : *Quantity，或 int, float, Decimal, Fraction, FixedArray, 数值 array_like* — 乘数（禁止 `Price * Price` 或
  `Price * Money`）。

**返回值**

- **out** : *Money 或 Price*

**异常**

- **TypeError** — `other` 是单位不匹配的 `Quantity`；或 `other` 是 `Price`、`Money`、字符串、`bool` 等非法操作数。
- **OverflowError** — 结果超出 `int64` 范围。

**机制说明**

`__mul__` 与 `__rmul__` 均直接调用 [`self.multiply(other)`](#pricemultiply)（若 `other` 既不是 `Quantity` 也不是合法无量纲数值，则先返回
`NotImplemented`）。

**示例**

计算成交金额（`Price * Quantity -> Money`）与双向 `"Any"` 通配：

```python
>> > from moneta import Price, Quantity
>> > Price("189.50", "USD") * Quantity(10, "share")
Money(FixedArray(1895.00, scale=2), currency='USD')
>> > Quantity(3, "barrel") * Price("80.25", "USD", base_unit="Any")
Money(FixedArray(240.75, scale=2), currency='USD')
```

无量纲精确缩放（`Price * 数值 -> Price`）：

```python
>>> from decimal import Decimal
>>> Price("1.50", "USD") * Decimal("1.1")
Price(FixedArray(1.650, scale=3), quote_currency='USD', base_unit='share')
>>> np.array([1, 2]) * Price(["10.00", "20.00"], "USD")
Price(FixedArray([10.00, 40.00], scale=2), quote_currency='USD', base_unit='share')
```

**性能对比**

|         N | `p * qty` (`Price * Quantity -> Money`) | `p * 2` (`Price` 精确缩放) | `np.round(p * q, 2)` (`NumPy float64`) |
|----------:|----------------------------------------:|---------------------------:|---------------------------------------:|
|  0-d 标量 |                                 5.66 µs |                    7.62 µs |                                2.58 µs |
|     1,000 |                                 6.79 µs |                    8.48 µs |                                2.74 µs |
| 1,000,000 |                                  1.2 ms |                     883 µs |                                1.91 ms |

**另请参阅**

- [`Price.multiply`](#pricemultiply) — 支持自定义 `rounding`（如成交金额向上取整）与目标 `scale`（价格缩放时）。

### Price.multiply

```python
Price.multiply(other, scale=None, rounding="half_up")
```

通用价格乘法：乘数为 [`Quantity`](quantity.md) 时按 `rounding` 舍入到计价币种精度返回 [`Money`](money.md)；乘数为无量纲数值时，不指定
`scale` 则精确相乘，指定 `scale` 则按 `rounding` 一次性舍入到 `scale` 返回 `Price`。

**参数**

- **other** : *Quantity 或无量纲数值 array_like* — 乘数。
- **scale** : *int，可选* — 仅当 `other` 为无量纲数值（结果为 `Price`）时可用：指定结果 `Price` 的小数位数。当 `other` 为
  `Quantity`（结果为 `Money`）时必须为 `None`。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式。

**返回值**

- **out** : *Money 或 Price*

**异常**

- **TypeError** — `other` 是单位不匹配的 `Quantity`；或 `other` 是 `Money`、`Price`（报 `"价格只能乘以 Quantity 或数值，收到 ..."`
  ）；或 `other` 是字符串/`bool`。
- **ValueError** — 当 `other` 为 `Quantity` 时传入了非 `None` 的 `scale`（报 `"结果为 Money 时精度固定为币种精度，不能指定 scale"`
  ）；或 `scale` / `rounding` 取值无效。

**示例**

```python
>> > from moneta import Price, Quantity
>> > Price("0.001", "USD").multiply(Quantity("0.4"), rounding="up")
Money(FixedArray(0.01, scale=2), currency='USD')
>> > Price("1.37", "USD").multiply(3, scale=1, rounding="half_up")
Price(FixedArray(4.1, scale=1), quote_currency='USD', base_unit='share')
>> > Price("1.00", "USD").multiply(Quantity(3), scale=4)
Traceback(most
recent
call
last):
...
ValueError: 结果为
Money
时精度固定为币种精度，不能指定
scale
```

**另请参阅**

- [`p * other`](#p--other-与-other--p成交金额计算与精确缩放)

### p / other（真除法运算符）

```python
p / other
```

将价格除以同量纲 `Price`（量纲相消返回无量纲比率 [`FixedArray`](fixed_array.md)）或除以无量纲数值（返回保留原单位与
`tick_size` 的 `Price`），默认保留 `max(两侧 scale) + 8` 位小数并按 `half_up` 舍入。

**参数**

- **other** : *同计价币种且单位兼容的 Price，或无量纲数值 array_like* — 除数。

**返回值**

- **out** : *FixedArray 或 Price*

**异常**

- **ZeroDivisionError** — 除数含有 0。
- **TypeError** — 除以量纲不匹配的 `Price`、除以 `Money` / `Quantity` / 字符串，或用数值除以 `Price`（如 `1 / p`）。

**示例**

```python
>>> Price("189.50", "USD") / Price("100.00", "USD")
FixedArray(1.8950000000, scale=10)
>>> Price("189.50", "USD", base_unit="barrel") / 2
Price(FixedArray(94.7500000000, scale=10), quote_currency='USD', base_unit='barrel')
```

**性能对比**

|         N | `p / 2` (`Price`) | `p1 / p2` (`FixedArray` 比率) | `x / y` (`NumPy float64`) |
|----------:|------------------:|------------------------------:|--------------------------:|
|  0-d 标量 |           30.7 µs |                       27.9 µs |                   43.9 ns |
|     1,000 |           51.2 µs |                         46 µs |                    490 ns |
| 1,000,000 |           20.6 ms |                       17.9 ms |                    772 µs |

**另请参阅**

- [`Price.divide`](#pricedivide)

### Price.divide

```python
Price.divide(other, scale=None, rounding="half_up")
```

价格除法，支持显式指定目标小数位数 `scale` 与舍入模式 `rounding`。

**参数**

- **other** : *兼容的 Price，或无量纲数值 array_like* — 除数。
- **scale** : *int，可选* — 目标小数位数；默认 `None` 时取 `max(self.scale, other.scale) + 8`。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式。

**返回值**

- **out** : *FixedArray 或 Price*

**异常**

- **ZeroDivisionError** — 除数含有 0。
- **TypeError** — `other` 是量纲不匹配的 `Price`；或 `other` 是 `Money` / `Quantity`（报 `"价格只能除以数值或同类 Price，收到 ..."`
  ）；或 `other` 是非数值类型。
- **ValueError** — `scale` 或 `rounding` 无效。

**示例**

```python
>> > Price("1.00", "USD").divide(Price("3.00", "USD"), scale=3, rounding="up")
FixedArray(0.334, scale=3)
>> > Price("10.00", "USD").divide(3, scale=2)
Price(FixedArray(3.33, scale=2), quote_currency='USD', base_unit='share')
>> > Price("10.00", "USD").divide(Quantity(2))
Traceback(most
recent
call
last):
...
TypeError: 价格只能除以数值或同类
Price，收到
Quantity
```

**另请参阅**

- [`p / other`](#p--other真除法运算符)
- [`_RATIO_EXTRA_DIGITS`](#_keep-与-_ratio_extra_digits)

## 规约、比较、导出与展示

### Price.sum 与 Price.is_zero

```python
Price.sum(axis=None, keepdims=False, skipna=True, out=None, dtype=None)
Price.is_zero()
```

直接继承自 [`UnitArray.sum`](unit_array.md#unitarraysum) 与 [`UnitArray.is_zero`](unit_array.md#unitarrayis_zero)，保留原
`quote_currency`、`base_unit`、`tick_size` 与 `scale`。

**示例**

```python
>>> p = Price(["10.00", "20.00"], "USD")
>>> p.sum()
Price(FixedArray(30.00, scale=2), quote_currency='USD', base_unit='share')
>>> bool(Price("0.00", "USD").is_zero())
True
```

### 比较、索引与形状变换

`Price` 继承 [`UnitArray`](unit_array.md) 的量纲安检比较、切片索引与容器协议：

- **相等与大小比较**：先通过 [`_check_compatible`](#price_check_compatible) 校验 `quote_currency` 相同且 `base_unit` 兼容（含
  `"Any"` 通配），再比较数值； **`tick_size` 不影响比较结果**。跨币种或跨单位时，`==` / `!=` 返回 `False` / `True`，而 `<` /
  `<=` / `>` / `>=` 抛出 `TypeError`。
- **索引与变形（`p[key]`、`reshape`、`transpose`、`iter`）**：全部保留原 `quote_currency`、`base_unit` 与 `tick_size`。

**示例**

```python
>>> from decimal import Decimal
>>> bool(Price("1.50", "USD", tick_size=Decimal("0.01")) == Price("1.5", "USD", tick_size=Decimal("0.05")))
True
>>> bool(Price("1.00", "USD", base_unit="Any") < Price("2.00", "USD", base_unit="barrel"))
True
>>> Price("1.00", "USD") < Price("2.00", "CNY")
Traceback (most recent call last):
    ...
TypeError: 价格量纲不匹配: USD/share vs CNY/share
```

### Price.to_pandas

```python
Price.to_pandas(index=None, name=None, as_float=True)
```

将 1 维或 2 维 `Price` 导出为 `pandas.Series` 或 `pandas.DataFrame`，并在返回对象的 `.attrs` 中自动写入
`{"quote_currency": self.quote_currency.code, "base_unit": self.base_unit}`。

**参数**

- **index** : *array-like 或 Index，可选* — 行索引。
- **name** : *str，可选* — 1 维 `Series` 的名称。
- **as_float** : *bool，默认 `True`* — 为 `True` 时导出为 `float64` 列；为 `False` 时导出为无损 `Decimal` 对象列。

**返回值**

- **out** : *pandas.Series 或 pandas.DataFrame*

**示例**

```python
>>> s = Price(["1.5", "2.0"], "USD", base_unit="barrel").to_pandas(name="px")
>>> s.tolist(), s.attrs
([1.5, 2.0], {'quote_currency': 'USD', 'base_unit': 'barrel'})
```

**另请参阅**

- [`UnitArray.to_pandas`](unit_array.md#unitarrayto_pandas)

### Price.to_polars、Price.to_float 与 Price.to_decimal

```python
Price.to_polars(name="value", as_float=True)
Price.to_float()
Price.to_decimal()
```

直接继承自 [`UnitArray`](unit_array.md)，剥离量纲标签后导出为 `polars.Series`、`float` / `float64` 数组或无损 `Decimal` /
`object` 数组。

**示例**

```python
>>> p = Price("1.50", "USD")
>>> p.to_float(), p.to_decimal()
(1.5, Decimal('1.50'))
```

### `str(p)` 与 `repr(p)`（文本表示）

```python
str(p)
repr(p)
```

`str(p)` 返回 `"{数值} {计价货币代码}/{基准单位}"`；`repr(p)` 返回包含底层 `FixedArray`、`quote_currency` 与 `base_unit`
的完整构造表达式。

**示例**

```python
>>> p = Price("1.5", "USD", base_unit="barrel")
>>> str(p)
'1.5 USD/barrel'
>>> repr(Price("1.5", "USD"))
"Price(FixedArray(1.5, scale=1), quote_currency='USD', base_unit='share')"
```

## 内部钩子与辅助函数（私有 API）

### Price._as_tick

```python
@classmethod
Price._as_tick(tick_size, what="tick_size")
```

校验 `tick_size`（或 `step`）属于无量纲数值白名单且所有元素严格为正数，返回对应的 [`FixedArray`](fixed_array.md)。

**参数**

- **tick_size** : *Any* — 待校验的步长值或步长数组。
- **what** : *str，默认 `"tick_size"`* — 错误消息中的参数名称（如 `"tick_size"` 或 `"step"`）。

**返回值**

- **out** : *FixedArray*

**异常**

- **TypeError** — `tick_size` 不在数值白名单内（由 [`UnitArray._as_factor`](unit_array.md#unitarray_as_factor) 抛出）。
- **ValueError** — `np.any(tick.mantissa <= 0)`。

**示例**

```python
>>> from decimal import Decimal
>>> Price._as_tick(Decimal("0.05"))
FixedArray(0.05, scale=2)
>>> Price._as_tick(0)
Traceback (most recent call last):
    ...
ValueError: tick_size 必须为正数
```

### Price._create_like

```python
Price._create_like(new_data, base_unit=None, tick_size=_KEEP)
```

实现 [`UnitArray._create_like`](unit_array.md#unitarray_create_like) 抽象钩子：跳过 `__init__` 零拷贝构造同
`quote_currency` 的新 `Price` 实例。

**参数**

- **new_data** : *FixedArray* — 新底层定点数张量。
- **base_unit** : *str，可选* — 默认 `None` 时沿用 `self._base_unit`。
- **tick_size** : *FixedArray、None 或 `_KEEP`* — 默认哨兵 `_KEEP` 时沿用 `self._tick_size`；传入 `None` 时显式清空结果的
  `tick_size`。

**返回值**

- **out** : *Price*

**示例**

```python
>>> from decimal import Decimal
>>> p = Price("10.00", "USD", tick_size=Decimal("0.05"))
>>> p._create_like(FixedArray("20.00")).tick_size
FixedArray(0.05, scale=2)
>>> p._create_like(FixedArray("20.00"), tick_size=None).tick_size is None
True
```

### Price._check_compatible

```python
Price._check_compatible(other)
```

实现 [`UnitArray._check_compatible`](unit_array.md#unitarray_check_compatible) 抽象钩子：校验 `other` 是否为相同
`quote_currency` 且 `base_unit` 满足 [`units_compatible`](_units.md#units_compatible)（含 `"Any"`）的 `Price`。

**示例**

```python
>> > p = Price("10.00", "USD", base_unit="barrel")
>> > p._check_compatible(Price("12.00", "USD", base_unit="Any"))
>> > p._check_compatible(Price("12.00", "USD", base_unit="share"))
Traceback(most
recent
call
last):
...
TypeError: 价格量纲不匹配: USD / barrel
vs
USD / share
```

### Price._resolve_unit

```python
Price._resolve_unit(other_unit)
```

合并两个兼容的基准单位字符串：若 `self._base_unit == "Any"` 则返回 `other_unit`，否则返回 `self._base_unit`。

**示例**

```python
>>> Price("1.00", "USD", base_unit="Any")._resolve_unit("barrel")
'barrel'
>>> Price("1.00", "USD", base_unit="barrel")._resolve_unit("Any")
'barrel'
```

### Price._resolve_tick

```python
Price._resolve_tick(other)
```

计算两个 `Price` 加减运算后的 `tick_size`：仅当 `self._tick_size` 与 `other._tick_size` 均不为 `None`、且形状相同并满足
`np.all(a == b)` 时保留并返回 `self._tick_size`；否则返回 `None`。

**示例**

```python
>>> from decimal import Decimal
>>> p1 = Price("10.00", "USD", tick_size=Decimal("0.01"))
>>> p1._resolve_tick(Price("12.00", "USD", tick_size=Decimal("0.010")))
FixedArray(0.01, scale=2)
>>> p1._resolve_tick(Price("12.00", "USD", tick_size=Decimal("0.05"))) is None
True
```

### _KEEP 与 _RATIO_EXTRA_DIGITS

```python
moneta.price._KEEP = object()
moneta.price._RATIO_EXTRA_DIGITS = 8
```

- **`_KEEP`**：模块内部唯一哨兵对象，用作 [`Price._create_like`](#price_create_like) 的 `tick_size` 默认参数值，以区分“沿用原实例的
  `self._tick_size`”（`tick_size is _KEEP`）与“显式将新实例的 `tick_size` 设为 `None`”（`tick_size is None`）。
- **`_RATIO_EXTRA_DIGITS = 8`**：当 [`Price.divide`](#pricedivide) 或 [`p / other`](#p--other真除法运算符) 未显式传入
  `scale` 时，默认保留的小数位数为 `max(两侧 scale) + 8`。

**示例**

```python
>>> from moneta.price import _KEEP, _RATIO_EXTRA_DIGITS
>>> _KEEP is not None and _RATIO_EXTRA_DIGITS == 8
True
```
