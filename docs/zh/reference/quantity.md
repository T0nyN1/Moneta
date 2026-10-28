# Quantity — 标的物份额与物理数量张量

`moneta.Quantity` 表示股票股数、期货合约手数、大宗商品物理度量（如桶 `"barrel"`、盎司 `"oz"`）或数字资产持仓数量，继承自 [
`UnitArray`](unit_array.md)。
与固定精度的 [`Money`](money.md) 不同，`Quantity` **不绑定固定小数位数**：加减与无量纲乘法默认完全精确（乘法的 `scale`
为两侧之和），只在你显式调用 [`multiply(..., scale=...)`](#quantitymultiply)、[`rescale`](#quantityrescale) 或 [
`round_to_step`](#quantityround_to_step)（如按整股、每手 100 股取整）时才执行舍入。
模块同时提供通配单位常量 [`WILDCARD_UNIT = "Any"`](#wildcard_unit)，可与任意具体单位运算并自动收敛为该具体单位。

```python
>>> from decimal import Decimal
>>> from moneta import Quantity, Price, Money
>>> from moneta.quantity import WILDCARD_UNIT
>>> q = Quantity([150, 240, 380], unit="share")
>>> q.unit, q.scale, q.mantissa.tolist()
('share', 0, [150, 240, 380])
>>> q * Decimal("1.5")  # 乘法默认精确，scale = 0 + 1 = 1
Quantity(FixedArray([225.0, 360.0, 570.0], scale=1), unit='share')
>>> q.round_to_step(100, rounding="down")  # 按每手 100 股向下取整
Quantity(FixedArray([100, 200, 300], scale=0), unit='share')
>>> Quantity(10, "share") * Price("189.50", "USD", base_unit="share")
Money(FixedArray(1895.00, scale=2), currency='USD')
```

[← API 参考](index.md)

## 核心机制与设计约束

### 弹性精度与显式取整原则

不同类型的标的物对数量精度有着截然不同的要求：A 股买单要求 100 股整手（步长 `100`），美股整股要求 `scale=0`，美股碎股或公募基金份额常见
`scale=2~4`，而比特币（`BTC`）可达 8 位小数。
因此，`Quantity` 遵循 **默认精确、按需取整**的设计：

1. **构造时自动推断或按显式 `scale` 对齐**：
    - `scale=None`（默认）时，保留输入数值自身的十进制小数位数（如 `100` 为 `0`，`"1.50"` 为 `2`，`Fraction(1, 8)` 为 `3`）；
   - 显式传入 `scale` 时：内部委托 `FixedArray(data, scale=scale, on_excess="warn")`，浮点数（`float` 及浮点数组）按最短十进制写法以
     `half_up` 静默舍入到 `scale`；精确输入（`str`、
     `Decimal`、`Fraction`、`FixedArray` 等）若在超出 `scale` 的小数位上含有非零数字（缺失位置与末尾多余的 `0` 除外），由
     `FixedArray` 先发出 [
     `PrecisionLossWarning`](fixed_array.md#precisionlosswarning) 警告，再按 `half_up` 舍入到 `scale`；位数不足时补尾随零。
2. **算术运算的 `scale` 传播**：

| 运算                                                       | 另一操作数类型                 | 返回类型                       | 结果 `scale` 与舍入规则                                                     |
|------------------------------------------------------------|--------------------------------|--------------------------------|-----------------------------------------------------------------------------|
| `q1 + q2`、`q1 - q2`                                       | 同单位或 `"Any"` 的 `Quantity` | `Quantity`                     | `max(q1.scale, q2.scale)`，精确整数加减                                     |
| `+q`、`-q`、`abs(q)`、`q.sum()`                            | —                              | `Quantity`                     | `q.scale`，精确（继承自 [`UnitArray`](unit_array.md)）                      |
| `q * k`、`k * q`                                           | 无量纲数值                     | `Quantity`                     | `q.scale + k.scale`，**精确相乘不舍入**                                     |
| `q.multiply(k, scale=None, rounding="half_up")`            | 无量纲数值                     | `Quantity`                     | `scale=None` 时精确；指定 `scale` 时按 `rounding` 一次性舍入到 `scale`      |
| `q * price`、`price * q`                                   | 单位匹配的 [`Price`](price.md) | [`Money`](money.md)            | 委托 [`Price`](price.md) 计算，按计价币种精度得到 `Money`                   |
| `q / k`、`q.divide(k, scale=None, rounding="half_up")`     | 无量纲数值                     | `Quantity`                     | 默认 `max(q.scale, k.scale) + 8`（放不下 `int64` 时少保留，最后去末尾 0），可用 `scale` 指定并按 `rounding` 舍入 |
| `q1 / q2`、`q1.divide(q2, scale=None, rounding="half_up")` | 同单位或 `"Any"` 的 `Quantity` | [`FixedArray`](fixed_array.md) | 单位相消得无量纲比率，默认 `max(q1.scale, q2.scale) + 8`（放不下 `int64` 时少保留，最后去末尾 0），可用 `scale` 指定 |
| `q.rescale(scale, rounding="half_up")`                     | —                              | `Quantity`                     | 缩小 `scale` 时按 `rounding` 舍入（如取整股），扩大时补零                   |
| `q.round_to_step(step, rounding="half_up")`                | 正数步长                       | `Quantity`                     | `max(q.scale, step.scale)`，按 `rounding` 对齐到步长网格（如整手 100 股）   |

### 单位规则与通配单位 `"Any"`

- **大小写敏感的非空字符串**：`unit` 在构造时自动去除首尾空白（[`check_unit`](_units.md#check_unit)），不能为空字符串，且严格区分大小写（例如
  `"BTC"` 与 `"btc"` 是两个不同的单位）。默认单位为 `"share"`。
- **通配单位 [`WILDCARD_UNIT = "Any"`](#wildcard_unit)**：
    - 当两个 `Quantity` 进行加减、比率除法或比较（`==`、`!=`、`<`、`<=`、`>`、`>=`）时，只要其中一方的单位为 `"Any"`，即视为量纲兼容；
    - 加减运算后的结果单位通过 [`_resolve_unit`](#quantity_resolve_unit) 优先收敛为 **非 `"Any"` 的具体单位**（例如
      `Quantity(3, "Any") + Quantity(2, "barrel")` 得到单位为 `"barrel"` 的 `Quantity`；仅当两侧均为 `"Any"` 时结果才保持
      `"Any"`）。

## API 速查表

| 分类                   | 接口 / 运算符 / 常量                                                                                       | 功能说明                                                                                     |
|------------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| **构造与常量**         | [`Quantity(values, unit="share", scale=None)`](#quantityvalues-unitshare-scalenone)                        | 构造带标的物单位的数量/份额定点数张量                                                        |
|                        | [`WILDCARD_UNIT`](#wildcard_unit)                                                                          | 通配单位常量 `"Any"`，可与任意单位兼容运算                                                   |
| **核心属性**           | [`Quantity.unit`](#quantityunit)                                                                           | 标的物单位字符串（默认 `"share"`）                                                           |
|                        | [继承自 `UnitArray` 的数组属性](#继承自-unitarray-的数组属性)                                              | `data`、`mantissa`、`scale`、`shape`、`ndim`、`size`、`dtype`、`T`                           |
| **算术与跨量纲运算**   | [`q1 + q2`、`q1 - q2`](#q1--q2-与-q1---q2同单位或通配单位加减法)                                           | 同单位（或含 `"Any"`）数量精确加减，结果收敛至具体单位                                       |
|                        | [`+q`、`-q`、`abs(q)`](#q-q-与-absq正负号与绝对值)                                                         | 一元正负号与逐元素绝对值                                                                     |
|                        | [`q * other`、`other * q`](#q--other-与-other--q精确缩放与量纲相消乘法)                                    | 乘以无量纲数得精确 `Quantity`；乘以 [`Price`](price.md) 得 [`Money`](money.md)               |
|                        | [`Quantity.multiply`](#quantitymultiply)                                                                   | 支持可选目标 `scale` 与舍入模式 `rounding` 的无量纲乘法                                      |
|                        | [`q / other`](#q--other真除法运算符)                                                                       | 除以无量纲数得 `Quantity`；除以同单位 `Quantity` 得无量纲比率 [`FixedArray`](fixed_array.md) |
|                        | [`Quantity.divide`](#quantitydivide)                                                                       | 支持自定义 `scale` 与 `rounding` 的数量除法                                                  |
| **显式取整与网格对齐** | [`Quantity.rescale`](#quantityrescale)                                                                     | 调整小数位数（如碎股舍入到整股 `rescale(0)`）                                                |
|                        | [`Quantity.round_to_step`](#quantityround_to_step)                                                         | 对齐到正数步长网格（如按每手 100 股取整）                                                    |
|                        | [`Quantity.is_on_step`](#quantityis_on_step)                                                               | 逐元素判断数量是否严格落在步长网格上                                                         |
| **规约、比较与导出**   | [`Quantity.sum`、`Quantity.is_zero`](#quantitysum-与-quantityis_zero)                                      | 沿轴精确求和与零值判定（继承自 [`UnitArray`](unit_array.md)）                                |
|                        | [比较、索引与形状变换](#比较索引与形状变换)                                                                | `==`、`!=`、`<`、`<=`、`>`、`>=`、`q[key]`、`reshape`、`transpose`、`len`、`bool`、`iter`    |
|                        | [`Quantity.to_pandas`](#quantityto_pandas)                                                                 | 导出为 `pandas.Series` / `DataFrame` 并自动注入 `attrs["unit"]`                              |
|                        | [`Quantity.to_polars`、`to_float`、`to_decimal`](#quantityto_polarsquantityto_float-与-quantityto_decimal) | 导出为 `polars.Series`、`float64` 或 `Decimal`                                               |
|                        | [`str(q)`、`repr(q)`](#strq-与-reprq文本表示)                                                              | 带单位的简明文本与完整构造表达式展示                                                         |
| **内部钩子与私有 API** | [`Quantity._create_like`](#quantity_create_like)                                                           | `UnitArray` 钩子：用新 `FixedArray` 与可选新单位构造 `Quantity`                              |
|                        | [`Quantity._check_compatible`](#quantity_check_compatible)                                                 | `UnitArray` 钩子：校验另一对象是否为同单位或 `"Any"` 的 `Quantity`                           |
|                        | [`Quantity._resolve_unit`](#quantity_resolve_unit)                                                         | 合并两个兼容 `Quantity` 的单位（优先取非 `"Any"` 的具体单位）                                |

## 构造、常量与核心属性

### `Quantity(values, unit="share", scale=None)`

```python
class moneta.Quantity(values, unit="share", scale=None)
```

从数值、字符串、`Decimal`、`Fraction`、`FixedArray`、序列、NumPy 数组或已有 `Quantity` 构造标的物数量张量。

**参数**

- **values** : *int, float, str, Decimal, Fraction, FixedArray, Quantity, Sequence 或 np.ndarray* — 数量数值或数组。
- **unit** : *str，默认 `"share"`* — 标的物单位（如 `"share"`、`"barrel"`、`"contract"`、`"BTC"` 或 `"Any"`
  ）。首尾空白会被自动去除，区分大小写，不能为空。
- **scale** : *int，可选* — 目标小数位数（非负整数）。默认 `None` 时保留 `values` 自身的推断精度；显式指定时，浮点数按最短十进制写法
  `half_up` 静默舍入，精确输入若在超出 `scale` 的小数位上含有非零数字（缺失位置与末尾多余的 `0` 除外）则先发出 [
  `PrecisionLossWarning`](fixed_array.md#precisionlosswarning) 再按 `half_up` 舍入，位数不足时补零。

**返回值**

- **out** : *Quantity*

**异常**

- **TypeError** — `unit` 不是 `str`；或 `values` 是不兼容单位的 `Quantity`（当两侧均不为 `"Any"` 且单位不同时）；或 `values`
  是 `bool` 等非法类型。
- **ValueError** — `unit` 去除首尾空白后为空字符串；或 `scale` 不是非负整数（如负数、`float`、`bool`）；或输入包含 `Inf`
  、无限小数 `Fraction` 等。
- **OverflowError** — 底层整数尾数超出 `int64` 范围。

**警告**

- **[`PrecisionLossWarning`](fixed_array.md#precisionlosswarning)** — 当显式指定 `scale` 且非浮点的精确输入（`str`、
  `Decimal`、`Fraction`、`FixedArray` 等）在超出 `scale` 的小数位上含有非零数字时发出。

**机制说明**

1. 先调用 [`check_unit(unit)`](_units.md#check_unit) 校验 `isinstance(unit, str)` 并执行 `.strip()`；
2. 若 `values` 是 `Quantity`：调用 [`units_compatible(values._unit, unit)`](_units.md#units_compatible) 校验单位兼容，不兼容则抛
   `TypeError`，兼容则取 `data = values._data`（可将 `"Any"` 数量窄化为具体单位）；否则取 `data = values`；
3. 直接调用 `self._data = FixedArray(data, scale=scale, on_excess="warn")`：指定 `scale` 时浮点数静默舍入，精确输入丢精度时由
   `FixedArray` 发出 [`PrecisionLossWarning`](fixed_array.md#precisionlosswarning) 后按 `half_up` 舍入。

**示例**

自动推断精度与显式指定 `scale`：

```python
>>> import warnings
>>> from decimal import Decimal
>>> from fractions import Fraction
>>> from moneta import Quantity, PrecisionLossWarning
>>> Quantity(100)
Quantity(FixedArray(100, scale=0), unit='share')
>>> Quantity("1.50", unit="  barrel ")
Quantity(FixedArray(1.50, scale=2), unit='barrel')
>>> Quantity(Fraction(1, 8), unit="BTC")
Quantity(FixedArray(0.125, scale=3), unit='BTC')
>>> Quantity(1.5, scale=0)  # float 指定 scale 时静默按最短写法 half_up 舍入
Quantity(FixedArray(2, scale=0), unit='share')
>>> Quantity("1.5", scale=4)  # 位数不足时自动补零
Quantity(FixedArray(1.5000, scale=4), unit='share')
>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     q = Quantity(Decimal("1.25"), scale=1)
...     (q, len(w), issubclass(w[0].category, PrecisionLossWarning))
(Quantity(FixedArray(1.3, scale=1), unit='share'), 1, True)
```

从已有 `Quantity` 构造与通配单位具体化：

```python
>>> q_any = Quantity(5, unit="Any")
>>> Quantity(q_any, unit="barrel")
Quantity(FixedArray(5, scale=0), unit='barrel')
>>> Quantity(Quantity(5, unit="barrel"), unit="share")
Traceback (most recent call last):
    ...
TypeError: 单位不匹配: barrel 与 share
```

**性能对比**

|         N | `Quantity(float_arr, "share")` | `Quantity(FixedArray)`（零拷贝包装） |
|----------:|-------------------------------:|-------------------------------------:|
|     1,000 |                        54.1 µs |                               420 ns |
| 1,000,000 |                        34.8 ms |                               420 ns |

**另请参阅**

- [`WILDCARD_UNIT`](#wildcard_unit)
- [`Quantity.unit`](#quantityunit)
- [`Quantity.rescale`](#quantityrescale)

### WILDCARD_UNIT

```python
moneta.quantity.WILDCARD_UNIT = "Any"
```

通配单位字符串常量 `"Any"`（定义于 [`moneta._units.WILDCARD_UNIT`](_units.md#wildcard_unit) 并由 `moneta.quantity`
重导出）。携带 `"Any"` 单位的 `Quantity` 可以与任意具体单位的 `Quantity` 进行加减、除法和比较，也可与任意 `base_unit` 的 [
`Price`](price.md) 相乘；在二元加减运算中，结果单位会自动收敛为另一方的具体单位。

**示例**

```python
>>> from moneta import Quantity, Price
>>> from moneta.quantity import WILDCARD_UNIT
>>> a = Quantity(3, unit=WILDCARD_UNIT)
>>> b = Quantity(2, unit="barrel")
>>> a + b
Quantity(FixedArray(5, scale=0), unit='barrel')
>>> bool(Quantity(5, unit=WILDCARD_UNIT) == Quantity(5, unit="barrel"))
True
>>> Price("80.00", "USD", base_unit="barrel") * a
Money(FixedArray(240.00, scale=2), currency='USD')
```

**另请参阅**

- [`moneta._units.WILDCARD_UNIT`](_units.md#wildcard_unit)
- [`Quantity._check_compatible`](#quantity_check_compatible)
- [`Quantity._resolve_unit`](#quantity_resolve_unit)

### Quantity.unit

```python
Quantity.unit
```

返回当前数量张量绑定的标的物单位字符串（默认 `"share"`）。

**返回值**

- **unit** : *str*

**机制说明**

直接读取只读槽位 `self._unit`。单位区分大小写，`"BTC"` 与 `"btc"` 不相等。

**示例**

```python
>>> Quantity(100).unit, Quantity("2.5", unit="BTC").unit
('share', 'BTC')
>>> Quantity(1, unit="BTC") == Quantity(1, unit="btc")
False
```

**另请参阅**

- [`check_unit`](_units.md#check_unit)
- [`WILDCARD_UNIT`](#wildcard_unit)

### 继承自 UnitArray 的数组属性

`Quantity` 继承自 [`UnitArray`](unit_array.md)，直接提供以下底层数据与形状属性：

| 属性                                                                                                                                                                | 类型                              | 说明                                            |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|-------------------------------------------------|
| [`q.data`](unit_array.md#unitarraydata)                                                                                                                             | [`FixedArray`](fixed_array.md)    | 底层定点数张量实例                              |
| [`q.mantissa`](unit_array.md#unitarraymantissa)                                                                                                                     | `ndarray(int64)`                  | 底层只读 `int64` 整数尾数视图                   |
| [`q.scale`](unit_array.md#unitarrayscale)                                                                                                                           | `int`                             | 当前数量张量的小数位数                          |
| [`q.shape`](unit_array.md#unitarrayshape) / [`ndim`](unit_array.md#unitarrayndim) / [`size`](unit_array.md#unitarraysize) / [`dtype`](unit_array.md#unitarraydtype) | `tuple` / `int` / `int` / `dtype` | 张量形状、维度数、元素总数、尾数类型（`int64`） |
| [`q.T`](unit_array.md#unitarrayt)                                                                                                                                   | `Quantity`                        | 反转全部轴并保持原单位的转置视图                |

**示例**

```python
>>> q = Quantity([["1.5", "2.0"], ["3.0", "4.5"]], unit="barrel")
>>> (q.scale, q.shape, q.ndim, q.size, q.dtype)
(1, (2, 2), 2, 4, dtype('int64'))
>>> q.T
Quantity(FixedArray([[1.5, 3.0],
                     [2.0, 4.5]], scale=1), unit='barrel')
```

## 算术与跨量纲运算

### q1 + q2 与 q1 - q2（同单位或通配单位加减法）

```python
q1 + q2
q1 - q2
```

两个单位相同（或至少一方为单位 `"Any"`）的 `Quantity` 逐元素精确相加或相减。

**参数**

- **other** : *Quantity* — 同单位或通配单位 `"Any"` 的 `Quantity`，形状可广播。

**返回值**

- **out** : *Quantity* — `scale = max(q1.scale, q2.scale)`，单位为 [`self._resolve_unit(other)`](#quantity_resolve_unit)
  （优先取非 `"Any"` 的具体单位）。

**异常**

- **TypeError** — `other` 是单位不匹配的 `Quantity`（如 `"share"` 与 `"barrel"`），或 `other` 不是 `Quantity`（如裸数字、
  `Money`、[`Price`](price.md)）。
- **ValueError** — 形状无法广播。
- **OverflowError** — 对齐 `scale` 或加减结果超出 `int64` 范围。

**机制说明**

若 `not isinstance(other, Quantity)` 则返回 `NotImplemented`；否则调用 `self._check_compatible(other)`，委托底层
`self._data + other._data`（自动对齐 `scale` 后精确加减），再用 `self._create_like(..., self._resolve_unit(other))` 包装。

**示例**

```python
>>> Quantity("1.5") + Quantity(2)
Quantity(FixedArray(3.5, scale=1), unit='share')
>>> Quantity(1) - Quantity("0.001")
Quantity(FixedArray(0.999, scale=3), unit='share')
>>> Quantity(3, unit="Any") + Quantity(2, unit="barrel")
Quantity(FixedArray(5, scale=0), unit='barrel')
>>> Quantity(1, unit="share") + Quantity(1, unit="barrel")
Traceback (most recent call last):
    ...
TypeError: 单位不匹配: share 与 barrel
```

**性能对比**

|         N | `q1 + q2` (`Quantity`) | `x + y` (`NumPy float64`) |
|----------:|-----------------------:|--------------------------:|
|  0-d 标量 |                 4.7 µs |                   43.3 ns |
|     1,000 |                5.92 µs |                    475 ns |
| 1,000,000 |                3.22 ms |                    715 µs |

**另请参阅**

- [`Quantity._resolve_unit`](#quantity_resolve_unit)
- [`Quantity.sum`](#quantitysum-与-quantityis_zero)

### `+q`、`-q` 与 `abs(q)`（正负号与绝对值）

```python
+q
-q
abs(q)
```

一元正号返回自身；一元负号与 `abs(q)` 逐元素取反或取绝对值，保持原单位与原 `scale`。

**返回值**

- **out** : *Quantity*

**机制说明**

直接继承自 [`UnitArray` 的 `+a`、`-a`](unit_array.md#a-与--a一元正负号) 与 [`abs(a)`](unit_array.md#absa绝对值)。

**示例**

```python
>>> q = Quantity([-10, 25], unit="barrel")
>>> -q, abs(q)
(Quantity(FixedArray([10, -25], scale=0), unit='barrel'), Quantity(FixedArray([10, 25], scale=0), unit='barrel'))
```

### q * other 与 other * q（精确缩放与量纲相消乘法）

```python
q * other
other * q
```

按另一操作数类型支持两种乘法语义：

1. **无量纲精确缩放（`q * 数值` 或 `数值 * q`）**：将数量乘以无量纲标量或数值数组， **默认完全精确、不舍入**，结果
   `scale = q.scale + factor.scale`，保持原单位；
2. **量纲相消得金额（`q * price` 或 `price * q`）**：当另一方为 [`Price`](price.md) 且单位匹配（`q.unit == price.base_unit`
   或 `q.unit == "Any"`）时，自动交由 [`Price`](price.md) 计算，返回以 `price.quote_currency` 计价的 [`Money`](money.md)。

**参数**

- **other** : *int, float, Decimal, Fraction, FixedArray, 数值 array_like 或 Price* — 乘数（不支持字符串、`bool`、
  `Quantity` 或 `Money`）。

**返回值**

- **out** : *Quantity 或 Money*

**异常**

- **TypeError** — `other` 是 `Quantity`、`Money`、字符串、`bool`，或 `other` 是 `base_unit` 不匹配的 [`Price`](price.md)。
- **OverflowError** — 乘积尾数超出 `int64` 范围。

**机制说明**

`__mul__` 与 `__rmul__` 首先检查 [
`FixedArray._is_numeric_operand(other)`](fixed_array.md#fixedarray_is_numeric_operand)：

- 若 `other` 是合法的无量纲数值，调用 [`self.multiply(other)`](#quantitymultiply)（`scale=None`，即调用
  `self._data.mul_round(factor, rounding=None)` 精确相乘）；
- 若 `other` 不是无量纲数值（例如 `other` 是 [`Price`](price.md)），`Quantity.__mul__` 返回 `NotImplemented`，Python 随即调用
  `Price.__rmul__(q)` 完成量纲相消（`[share] * [USD/share] -> [USD]`）。

**示例**

无量纲精确乘法（`scale` 相加）：

```python
>>> from decimal import Decimal
>>> Quantity(3) * 0.5
Quantity(FixedArray(1.5, scale=1), unit='share')
>>> Quantity("1.5") * Decimal("0.33")
Quantity(FixedArray(0.495, scale=3), unit='share')
>>> np.array([2, 3]) * Quantity(["1.25", "2.00"], unit="barrel")
Quantity(FixedArray([2.50, 6.00], scale=2), unit='barrel')
```

与 [`Price`](price.md) 相乘消去标的单位得到 [`Money`](money.md)：

```python
>>> from moneta import Price
>>> Quantity(10, unit="share") * Price("189.50", "USD", base_unit="share")
Money(FixedArray(1895.00, scale=2), currency='USD')
>>> Quantity(1, unit="barrel") * Price("189.50", "USD", base_unit="share")
Traceback (most recent call last):
    ...
TypeError: 标的单位不匹配: 数量单位 'barrel' 与标价基准单位 'share' 不一致
```

**性能对比**

|         N | `q * factor`（精确，`Quantity`） | `q.multiply(factor, scale=0)`（含舍入） | `x * r` (`NumPy float64`) |
|----------:|---------------------------------:|----------------------------------------:|--------------------------:|
|  0-d 标量 |                          5.52 µs |                                 19.4 µs |                   43.7 ns |
|     1,000 |                          6.62 µs |                                 29.9 µs |                    476 ns |
| 1,000,000 |                          1.24 ms |                                 11.6 ms |                    659 µs |

**另请参阅**

- [`Quantity.multiply`](#quantitymultiply) — 在乘法同时直接舍入到指定 `scale`。

### Quantity.multiply

```python
Quantity.multiply(other, scale=None, rounding="half_up")
```

将数量乘以无量纲数值：不指定 `scale` 时精确相乘（`scale` 为两侧之和）；指定 `scale` 时按 `rounding` 一次性舍入到目标小数位数。

**参数**

- **other** : *int, float, Decimal, Fraction, FixedArray 或数值 array_like* — 无量纲乘数。
- **scale** : *int，可选* — 结果的目标小数位数。默认 `None` 时精确相乘不舍入。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式（即使 `scale=None` 也会校验模式名合法性）。

**返回值**

- **out** : *Quantity* — 同单位的数量张量。

**异常**

- **TypeError** — `other` 不是数值白名单类型（如字符串、`Quantity`、`Money`）。
- **ValueError** — `scale` 不是非负整数，或 `rounding` 不是合法的舍入模式名。
- **OverflowError** — 结果超出 `int64` 范围。

**机制说明**

先通过 [`self._as_factor(other, "乘数")`](unit_array.md#unitarray_as_factor) 与 [
`check_rounding(rounding)`](_rounding.md#check_rounding) 校验参数。若 `scale is None`，调用
`self._data.mul_round(factor, rounding=None)`；否则调用 `self._data.mul_round(factor, scale=scale, rounding=mode)`（支持
Base $2^{21}$ 分块防中间溢出慢路径，只舍入一次）。

**示例**

```python
>>> from decimal import Decimal
>>> Quantity("1.37").multiply(3)  # scale=None：精确相乘
Quantity(FixedArray(4.11, scale=2), unit='share')
>>> Quantity("1.37").multiply(3, scale=1)  # half_up 舍入到 1 位小数
Quantity(FixedArray(4.1, scale=1), unit='share')
>>> Quantity("1.37").multiply(3, scale=0, rounding="down")  # 向零截断到整股
Quantity(FixedArray(4, scale=0), unit='share')
>>> Quantity(3).multiply(Decimal("0.5"), scale=0, rounding="half_even")
Quantity(FixedArray(2, scale=0), unit='share')
```

**另请参阅**

- [`q * other`](#q--other-与-other--q精确缩放与量纲相消乘法)
- [`Quantity.rescale`](#quantityrescale)
- [`FixedArray.mul_round`](fixed_array.md#fixedarraymul_round)

### q / other（真除法运算符）

```python
q / other
```

将数量除以无量纲数值（返回同单位 `Quantity`）或除以兼容单位的 `Quantity`（单位相消返回无量纲比率 [`FixedArray`](fixed_array.md)）。默认按 [`FixedArray.divide`](fixed_array.md#fixedarraydivide) 的自动精度规则保留 `max(两侧 scale) + 8` 位小数（放不下 `int64` 时少保留几位）、按 `half_up` 舍入并去掉末尾的 `0`（至少保留 `max(两侧 scale)` 位）。

**参数**

- **other** : *数值 array_like，或同单位/通配单位的 Quantity* — 除数。

**返回值**

- **out** : *Quantity 或 FixedArray*

**异常**

- **ZeroDivisionError** — 除数含有 0。
- **TypeError** — 除以不同单位的 `Quantity`、除以 `Money` / [`Price`](price.md) / 字符串，或用数值除以 `Quantity`（如
  `1 / q`）。

**机制说明**

直接调用 [`self.divide(other)`](#quantitydivide)。

**示例**

```python
>>> Quantity(10, "share") / 4  # 整除：2.50000000 去掉末尾 0 -> scale=1
Quantity(FixedArray(2.5, scale=1), unit='share')
>>> Quantity(1, "barrel") / Quantity(3, "barrel")  # 除不尽：保留 0 + 8 = 8 位
FixedArray(0.33333333, scale=8)
>>> Quantity(1, "Any") / Quantity(4, "barrel")  # 0.25000000 去掉末尾 0 -> scale=2
FixedArray(0.25, scale=2)
```

**性能对比**

|         N | `q / 4` (`Quantity`) | `q1 / q2` (`FixedArray` 比率) | `x / y` (`NumPy float64`) |
|----------:|---------------------:|------------------------------:|--------------------------:|
|  0-d 标量 |              30.1 µs |                       26.7 µs |                   43.2 ns |
|     1,000 |              49.8 µs |                       44.7 µs |                    492 ns |
| 1,000,000 |              18.2 ms |                         17 ms |                    771 µs |

**另请参阅**

- [`Quantity.divide`](#quantitydivide)

### Quantity.divide

```python
Quantity.divide(other, scale=None, rounding="half_up")
```

数量除法，支持显式指定目标小数位数 `scale` 与舍入模式 `rounding`。

**参数**

- **other** : *数值 array_like，或同单位/通配单位的 Quantity* — 除数。
- **scale** : *int，可选* — 目标小数位数。默认 `None` 时遵循 [`FixedArray.divide`](fixed_array.md#fixedarraydivide) 的自动精度规则（保留 `max(两侧 scale) + 8` 位，放不下 `int64` 时少保留，最后去掉末尾 `0`）。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式。

**返回值**

- **out** : *Quantity 或 FixedArray* — 当 `other` 为无量纲数值时返回同单位 `Quantity`；当 `other` 为兼容 `Quantity`
  时返回无量纲比率 `FixedArray`。

**异常**

- **ZeroDivisionError** — 除数含有 0。
- **TypeError** — `other` 是单位不匹配的 `Quantity`；或 `other` 是其他 `UnitArray` 子类（报
  `"数量只能除以数值或同单位的 Quantity，收到 ..."`）；或 `other` 是字符串/`bool` 等非数值对象。
- **ValueError** — `scale` 不是非负整数或 `rounding` 无效。

**机制说明**

1. 若 `isinstance(other, Quantity)`：调用 `self._check_compatible(other)` 校验单位兼容，直接返回 [`self._data.divide(other._data, scale=scale, rounding=mode)`](fixed_array.md#fixedarraydivide)（类型为 `FixedArray`）。
2. 若 `isinstance(other, UnitArray)`（如 `Money`、[`Price`](price.md)）：直接抛出 `TypeError`。
3. 否则通过 `self._as_factor(other, "除数")` 转为 `FixedArray`，调用 `self._data.divide(divisor, scale=scale, rounding=mode)` 并返回包装后的同单位 `Quantity`。

**示例**

```python
>>> Quantity(1, "share").divide(3, scale=2, rounding="up")
Quantity(FixedArray(0.34, scale=2), unit='share')
>>> Quantity("1", "share").divide(Quantity(3, "share"), scale=3, rounding="floor")
FixedArray(0.333, scale=3)
>>> Quantity(1, "share").divide(Money("1.00", "USD"))
Traceback (most recent call last):
    ...
TypeError: 数量只能除以数值或同单位的 Quantity，收到 Money
```

**另请参阅**

- [`q / other`](#q--other真除法运算符)
- [`FixedArray.divide`](fixed_array.md#fixedarraydivide)

## 显式取整与网格对齐

### Quantity.rescale

```python
Quantity.rescale(scale, rounding="half_up")
```

调整数量张量的小数位数 `scale`，保持原单位不变：当新 `scale` 小于原 `scale` 时按 `rounding` 舍入（例如将碎股取整到整股
`scale=0`）；当新 `scale` 大于原 `scale` 时在末尾精确补零。

**参数**

- **scale** : *int* — 目标小数位数（非负整数）。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 缩小 `scale` 时使用的舍入模式。

**返回值**

- **out** : *Quantity* — 同单位、新 `scale` 的数量张量。

**异常**

- **ValueError** — `scale` 不是非负整数，或 `rounding` 无效。
- **OverflowError** — 扩大 `scale` 后尾数超出 `int64` 范围。

**机制说明**

直接执行 `self._create_like(self._data.rescale(scale, rounding))`。

**示例**

```python
>>> Quantity("2.5", "share").rescale(0)  # 默认 half_up -> 3
Quantity(FixedArray(3, scale=0), unit='share')
>>> Quantity("2.5", "share").rescale(0, "half_even")  # 银行家舍入 -> 2
Quantity(FixedArray(2, scale=0), unit='share')
>>> Quantity("2.5", "barrel").rescale(0, "down")  # 向零截断并保留单位 barrel
Quantity(FixedArray(2, scale=0), unit='barrel')
>>> Quantity(2, "BTC").rescale(4)
Quantity(FixedArray(2.0000, scale=4), unit='BTC')
```

**性能对比**

|         N | `q.rescale(0)` (`Quantity`) | `np.round(x)` (`NumPy float64`) |
|----------:|----------------------------:|--------------------------------:|
|  0-d 标量 |                     11.4 µs |                         1.12 µs |
|     1,000 |                     28.3 µs |                          847 ns |
| 1,000,000 |                     10.6 ms |                          664 µs |

**另请参阅**

- [`Quantity.round_to_step`](#quantityround_to_step)
- [`FixedArray.rescale`](fixed_array.md#fixedarrayrescale)

### Quantity.round_to_step

```python
Quantity.round_to_step(step, rounding="half_up")
```

将数量对齐到正数步长 `step` 的整数倍网格（例如 A 股买单按每手 `100` 股向下取整 `round_to_step(100, "down")`，或加密货币按最小下单步长
`0.001` 取整）。

**参数**

- **step** : *int, float, Decimal, Fraction, FixedArray 或数值 array_like* — 正数步长（支持广播，无精度上限限制）。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式。

**返回值**

- **out** : *Quantity* — 同单位、`scale = max(self.scale, step.scale)` 的数量张量。

**异常**

- **TypeError** — `step` 不是数值白名单类型（如字符串）。
- **ValueError** — `step <= 0` 或 `rounding` 无效。
- **OverflowError** — 结果超出 `int64` 范围。

**机制说明**

通过 [`self._as_factor(step, "步长")`](unit_array.md#unitarray_as_factor) 将步长转为 `FixedArray`，再委托 [
`self._data.round_to_step(..., rounding=rounding)`](fixed_array.md#fixedarrayround_to_step) 计算并包装为同单位
`Quantity`。注意与 `Money.round_to_step` 不同，`Quantity` 没有固定币种精度限制，因此 `step` 可以拥有任意合法的小数位数。

**示例**

```python
>>> from decimal import Decimal
>>> Quantity(250, "share").round_to_step(100, "down")  # 250 股按整手 100 股向下截断 -> 200 股
Quantity(FixedArray(200, scale=0), unit='share')
>>> Quantity(250, "share").round_to_step(100)  # 默认 half_up -> 300 股
Quantity(FixedArray(300, scale=0), unit='share')
>>> Quantity("0.123456", "BTC").round_to_step(Decimal("0.001"), "floor")
Quantity(FixedArray(0.123000, scale=6), unit='BTC')
```

**性能对比**

|         N | `q.round_to_step(100, "down")` (`Quantity`) | `(x // 100) * 100` (`NumPy float64`) |
|----------:|--------------------------------------------:|-------------------------------------:|
|  0-d 标量 |                                     24.2 µs |                              89.9 ns |
|     1,000 |                                     31.1 µs |                              10.7 µs |
| 1,000,000 |                                     5.78 ms |                              19.5 ms |

**另请参阅**

- [`Quantity.is_on_step`](#quantityis_on_step)
- [`Quantity.rescale`](#quantityrescale)
- [`FixedArray.round_to_step`](fixed_array.md#fixedarrayround_to_step)

### Quantity.is_on_step

```python
Quantity.is_on_step(step)
```

判断数量是否严格落在正数步长 `step` 的整数倍网格上（例如检查持仓或下单数量是否为整手）。

**参数**

- **step** : *int, float, Decimal, Fraction, FixedArray 或数值 array_like* — 正数步长。

**返回值**

- **out** : *bool 或 numpy.ndarray（bool）* — 0 维标量返回 Python `bool`，高维张量返回同形状的 `bool` 数组。

**异常**

- **TypeError** — `step` 不是数值白名单类型。
- **ValueError** — `step <= 0`。

**机制说明**

调用 `self._data.is_on_step(self._as_factor(step, "步长"))`。

**示例**

```python
>>> Quantity(300, "share").is_on_step(100), Quantity(250, "share").is_on_step(100)
(True, False)
>>> Quantity([100, 150, 200], "share").is_on_step(100)
array([ True, False,  True])
```

**另请参阅**

- [`Quantity.round_to_step`](#quantityround_to_step)
- [`FixedArray.is_on_step`](fixed_array.md#fixedarrayis_on_step)

## 规约、比较、导出与展示

### Quantity.sum 与 Quantity.is_zero

```python
Quantity.sum(axis=None, keepdims=False, skipna=True, out=None, dtype=None)
Quantity.is_zero()
```

直接继承自 [`UnitArray.sum`](unit_array.md#unitarraysum) 与 [`UnitArray.is_zero`](unit_array.md#unitarrayis_zero)：

- `q.sum(...)`（及 `np.sum(q, ...)`）沿指定轴精确求和，保持原单位与原 `scale`；
- `q.is_zero()` 检查底层尾数是否为 0（切勿用 `q == 0`，带单位对象与裸数字 `0` 比较恒为 `False`）。

**示例**

```python
>>> q = Quantity([100, -100, 50], unit="share")
>>> q.sum()
Quantity(FixedArray(50, scale=0), unit='share')
>>> (q - Quantity([100, -100, 0], unit="share")).is_zero()
array([ True,  True, False])
```

### 比较、索引与形状变换

`Quantity` 直接继承 [`UnitArray`](unit_array.md) 的比较、索引切片、形状变换与容器协议：

- **相等比较（[`q1 == q2`、`q1 != q2`](unit_array.md#a--b-与-a--b相等性比较)）**：同单位（或一方为 `"Any"`）的 `Quantity` 对齐
  `scale` 后逐元素判等；与不同单位的 `Quantity`、`Money`、[`Price`](price.md) 或无量纲数字比较时返回 `False` / `True`
  （不抛异常）。
- **大小比较（[`q1 < q2`、`q1 <= q2`、`q1 > q2`、`q1 >= q2`](unit_array.md#a--ba--ba--ba--b大小比较)）**：同单位（或一方为
  `"Any"`）的 `Quantity` 对齐 `scale` 后逐元素比较大小；与不同单位的 `Quantity`、其他 `UnitArray` 或无量纲数字比大小时
  **立即抛出 `TypeError`**。
- **索引与变形（[`q[key]`](unit_array.md#akey索引与切片)、[`reshape`](unit_array.md#unitarrayreshape)、[
  `transpose`](unit_array.md#unitarraytranspose)、[`len`](unit_array.md#lenaboola-与-itera容器与真值协议)、[
  `bool`](unit_array.md#lenaboola-与-itera容器与真值协议)、[`iter`](unit_array.md#lenaboola-与-itera容器与真值协议)）**
  ：所有切片、单元素索引与迭代产出的元素均保持原单位。

**示例**

```python
>>> q = Quantity(["10.5", "20.0", "5.25"], unit="barrel")
>>> q[q >= Quantity(10, unit="barrel")]
Quantity(FixedArray([10.50, 20.00], scale=2), unit='barrel')
>>> Quantity(1, unit="share") < Quantity(2, unit="barrel")
Traceback (most recent call last):
    ...
TypeError: 单位不匹配: share 与 barrel
```

### Quantity.to_pandas

```python
Quantity.to_pandas(index=None, name=None, as_float=True)
```

将 1 维或 2 维 `Quantity` 导出为 `pandas.Series` 或 `pandas.DataFrame`，并在返回对象的 `.attrs["unit"]` 中自动写入单位字符串（如
`"share"`、`"barrel"`）。

**参数**

- **index** : *array-like 或 Index，可选* — 行索引。
- **name** : *str，可选* — 1 维 `Series` 的名称。
- **as_float** : *bool，默认 `True`* — 为 `True` 时导出为 `float64` 列；为 `False` 时导出为无损 `Decimal` 对象列。

**返回值**

- **out** : *pandas.Series 或 pandas.DataFrame*（携带 `out.attrs["unit"] == self.unit`）。

**示例**

```python
>>> s = Quantity(["1.5", "2.0"], unit="barrel").to_pandas(name="qty")
>>> s.tolist(), s.attrs["unit"], s.name
([1.5, 2.0], 'barrel', 'qty')
```

**另请参阅**

- [`UnitArray.to_pandas`](unit_array.md#unitarrayto_pandas)

### Quantity.to_polars、Quantity.to_float 与 Quantity.to_decimal

```python
Quantity.to_polars(name="value", as_float=True)
Quantity.to_float()
Quantity.to_decimal()
```

直接继承自 [`UnitArray`](unit_array.md)，剥离单位标签后导出为 `polars.Series`、`float` / `float64` 数组或无损 `Decimal` /
`object` 数组。

**示例**

```python
>>> q = Quantity("1.50", unit="barrel")
>>> q.to_float(), q.to_decimal()
(1.5, Decimal('1.50'))
```

**另请参阅**

- [`UnitArray.to_float`](unit_array.md#unitarrayto_float)、[`UnitArray.to_decimal`](unit_array.md#unitarrayto_decimal)、[
  `UnitArray.to_polars`](unit_array.md#unitarrayto_polars)

### `str(q)` 与 `repr(q)`（文本表示）

```python
str(q)
repr(q)
```

`str(q)` 返回 `"{数值} {单位}"`；`repr(q)` 返回包含底层 `FixedArray` 与 `unit` 的完整构造表达式。

**返回值**

- **out** : *str*

**示例**

```python
>>> q = Quantity("1.5", unit="barrel")
>>> str(q)
'1.5 barrel'
>>> repr(Quantity(3))
"Quantity(FixedArray(3, scale=0), unit='share')"
```

## 内部钩子与辅助函数（私有 API）

### Quantity._create_like

```python
Quantity._create_like(new_data, unit=None)
```

实现 [`UnitArray._create_like`](unit_array.md#unitarray_create_like) 抽象钩子：跳过 `__init__` 零拷贝构造携带 `new_data`
与指定单位（默认 `self._unit`）的新 `Quantity` 实例。

**参数**

- **new_data** : *FixedArray* — 新的底层定点数张量。
- **unit** : *str，可选* — 结果单位；默认 `None` 时保持 `self._unit`。

**返回值**

- **out** : *Quantity*

**示例**

```python
>>> q = Quantity(10, unit="barrel")
>>> q._create_like(FixedArray("2.5"))
Quantity(FixedArray(2.5, scale=1), unit='barrel')
>>> q._create_like(FixedArray(5), unit="share")
Quantity(FixedArray(5, scale=0), unit='share')
```

**另请参阅**

- [`UnitArray._create_like`](unit_array.md#unitarray_create_like)

### Quantity._check_compatible

```python
Quantity._check_compatible(other)
```

实现 [`UnitArray._check_compatible`](unit_array.md#unitarray_check_compatible) 抽象钩子：通过 [
`units_compatible(self._unit, other._unit)`](_units.md#units_compatible) 校验 `other` 是否为相同单位或通配单位 `"Any"` 的
`Quantity`。

**参数**

- **other** : *Any*

**异常**

- **TypeError** — `other` 不是 `Quantity`，或两侧单位均不为 `"Any"` 且 `self._unit != other._unit`。

**示例**

```python
>>> q = Quantity(1, unit="barrel")
>>> q._check_compatible(Quantity(2, unit="barrel"))
>>> q._check_compatible(Quantity(2, unit="Any"))
>>> q._check_compatible(Quantity(2, unit="share"))
Traceback (most recent call last):
    ...
TypeError: 单位不匹配: barrel 与 share
```

**另请参阅**

- [`Quantity._resolve_unit`](#quantity_resolve_unit)
- [`UnitArray._check_compatible`](unit_array.md#unitarray_check_compatible)

### Quantity._resolve_unit

```python
Quantity._resolve_unit(other)
```

计算两个兼容 `Quantity` 运算后的结果单位：若 `self._unit == "Any"` 则取 `other._unit`，否则取 `self._unit`。

**参数**

- **other** : *Quantity* — 已通过兼容性校验的另一 `Quantity`。

**返回值**

- **out** : *str*

**示例**

```python
>>> Quantity(1, "Any")._resolve_unit(Quantity(2, "barrel"))
'barrel'
>>> Quantity(1, "barrel")._resolve_unit(Quantity(2, "Any"))
'barrel'
>>> Quantity(1, "Any")._resolve_unit(Quantity(2, "Any"))
'Any'
```

**另请参阅**

- [`WILDCARD_UNIT`](#wildcard_unit)
- [`q1 + q2` 与 `q1 - q2`](#q1--q2-与-q1---q2同单位或通配单位加减法)
