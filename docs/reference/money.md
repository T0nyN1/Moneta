# Money — 货币金额张量与守恒分配

`moneta.Money` 是绑定单一币种（[`Currency`](currency.md)）的广延资金量张量，继承自 [`UnitArray`](unit_array.md)。
它的底层由 [`FixedArray`](fixed_array.md) 承载，且小数位数 `scale` **恒等于所属币种的法定最小精度 `currency.digits`**
（例如美元与人民币恒为 `2`、日元恒为 `0`），不存在更高精度的“中间金额”。
除提供同币种加减、无量纲缩放、矩阵内积、沿轴求和/均值外，还内置了步长网格一次舍入、带余守恒整除（`divmod`
）、最大余数法（Hare-Niemeyer）守恒分配（[`allocate`](#moneyallocate)）以及单档/分档累进交易费计算（[
`calc_tiered_fee`](#moneycalc_tiered_fee)）。

```python
>> > from decimal import Decimal
>> > from moneta import Money, USD, JPY
>> > m = Money(["100.50", "200.25", "-50.00"], "USD")
>> > m.currency, m.scale, m.units.tolist()
(Currency(code='USD', digits=2, symbol='$', name='US Dollar'), 2, [10050, 20025, -5000])
>> > m.sum()
Money(FixedArray(250.75, scale=2), currency='USD')
>> > Money("100.00", "USD").allocate([1, 1, 1])  # 最大余数法守恒分配，零头不丢
[Money(FixedArray(33.34, scale=2), currency='USD'), Money(FixedArray(33.33, scale=2), currency='USD'),
 Money(FixedArray(33.33, scale=2), currency='USD')]
>> > divmod(Money("10.00", "USD"), 3)  # 均分 3 份：(每份金额, 剩余零头)
(Money(FixedArray(3.33, scale=2), currency='USD'), Money(FixedArray(0.01, scale=2), currency='USD'))
```

[← API 参考](index.md)

## 核心机制与设计约束

### 币种精度不变式（`scale == currency.digits`）

每个 `Money` 实例的底层 `self._data`（[`FixedArray`](fixed_array.md)）都严格满足不变式：

$$\text{m.scale} \equiv \text{m.currency.digits}, \qquad \text{真实金额} = \text{m.units} \times 10^{-\text{m.currency.digits}}$$

- **构造不接受 `scale` 参数**：金额的小数位数完全由所属 [`Currency.digits`](currency.md#currencydigits)
  决定。内部构造入口 [`Money._wrap`](#money_wrap) 会强制断言 `data.scale == currency.digits`，任何偏离都会抛出
  `ValueError`。
- **浮点数静默舍入，精确输入超精度告警后舍入**：构造器内部统一调用
  `FixedArray(values, scale=curr.digits, on_excess="warn")`：
    - 当输入为 `float` 或浮点数组（`dtype.kind == "f"`）时，浮点数本身是近似值，直接按最短十进制写法以 `half_up` 静默舍入到
      `currency.digits`；
  - 当输入为精确表示（`str`、`Decimal`、`Fraction`、`FixedArray` 或含这些元素的序列/`object` 数组）且在多出的小数位上存在非零数字（缺失位置与末尾多余的
    `0` 除外）时，由 `FixedArray` 发出 [`PrecisionLossWarning`](fixed_array.md#precisionlosswarning) 警告（告警位置自动指向用户调用代码），再按
    `half_up` 舍入到
      `currency.digits`。

### 量纲代数与精度传播规则

| 运算                                            | 另一操作数类型                 | 返回类型                       | 结果精度与舍入规则                                                               |
|-------------------------------------------------|--------------------------------|--------------------------------|----------------------------------------------------------------------------------|
| `m1 + m2`、`m1 - m2`                            | 同币种 `Money`                 | `Money`                        | `currency.digits`，精确整数加减                                                  |
| `+m`、`-m`、`abs(m)`、`m.sum()`                 | —                              | `Money`                        | `currency.digits`，精确（继承自 [`UnitArray`](unit_array.md)）                   |
| `m * k`、`k * m`                                | 无量纲数值                     | `Money`                        | 按 `half_up` 舍入到 `currency.digits`（禁止 `Money * Money`）                    |
| `m.multiply(k, rounding, step)`                 | 无量纲数值                     | `Money`                        | 按 `rounding` 舍入到 `currency.digits`；给出 `step` 时**一次性**舍入到步长网格   |
| `m @ w`、`w @ m`、`m.matmul(w, rounding)`       | 无量纲权重                     | `Money`                        | 矩阵内积后按 `rounding`（运算符默认 `half_up`）一次性舍入到 `currency.digits`    |
| `m / k`、`m.divide(k, rounding, step)`          | 无量纲数值                     | `Money`                        | 按 `rounding` 舍入到 `currency.digits` 或步长网格（不可指定 `scale`）            |
| `m1 / m2`、`m1.divide(m2, rounding, scale)`     | 同币种 `Money`                 | [`FixedArray`](fixed_array.md) | 量纲相消为无量纲比率，默认保留 `currency.digits + 8` 位小数，可通过 `scale` 指定 |
| `m / price`、`m.divide(price, rounding, scale)` | 同计价币种 [`Price`](price.md) | [`Quantity`](quantity.md)      | 金额除以单价得数量，默认保留 `currency.digits + 8` 位小数，可通过 `scale` 指定   |
| `m / qty`、`m.divide(qty, rounding, scale)`     | [`Quantity`](quantity.md)      | [`Price`](price.md)            | 金额除以数量得单价，默认保留 `currency.digits + 8` 位小数，可通过 `scale` 指定   |
| `m.mean(axis, keepdims, rounding, skipna)`      | —                              | `Money`                        | 沿轴求算术均值（默认忽略 `<NA>`），按 `rounding` 舍入到 `currency.digits`        |
| `m.round_to_step(step, rounding)`               | 正数步长                       | `Money`                        | 对齐到步长网格（`step` 精度不得超过 `currency.digits`）                          |
| `divmod(m1, m2)`                                | 同币种 `Money`                 | `(int \| ndarray, Money)`      | 向下整除求可拆出的整数份数与剩余 `Money` 零头                                    |
| `divmod(m, n)`                                  | 正整数 `n`                     | `(Money, Money)`               | 向下整除均分为 `n` 份，返回 `(每份金额, 剩余零头金额)`                           |
| `m.allocate(ratios)`                            | 一维非负数值序列               | ` list[Money]`                 | 最大余数法按比例守恒分配，各份之和严格等于 `m`                                   |
| `m.calc_tiered_fee(rate, ...)`                  | 单一费率或分档表               | `Money`                        | 各档费用精确累加后只按 `rounding` 舍入一次，再逐元素应用 `min_fee` / `max_fee`   |

### 操作数白名单与一步到位步长舍入

- **操作数白名单**：所有接受无量纲系数的方法（`multiply`、`divide`、`matmul`、`round_to_step`、`is_on_step`、`allocate`、
  `calc_tiered_fee`）内部统一通过继承自基类的 [`UnitArray._as_factor`](unit_array.md#unitarray_as_factor) 校验，只接受
  `int`、`float`、`Decimal`、`Fraction`、
  `FixedArray` 以及整数/浮点 `ndarray` 或序列， **拒绝字符串与布尔值**（运算符遇到未知类型返回 `NotImplemented`，方法直接抛
  `TypeError`）。
- **一步到位步长舍入（消除二次舍入偏差）**：当在 [`multiply`](#moneymultiply) 或 [`divide`](#moneydivide) 中指定 `step`（例如
  `step=Decimal("0.10")`）时，如果先四舍五入到币种精度（`0.01`）再四舍五入到 `0.10`，像 `$1.00 \times 0.0499 = 0.0499$` 会先变成
  `0.05` 再被错误进位成 `0.10`。`Money` 通过 [`_fused_round_step`](#money_fused_round_step) 直接在 `step.scale + 1`
  的精度上求向下取整商与非精确标志，一次性裁决相对于 `step` 半步长边界的位置，保证全程只舍入一次。

## API 速查表

| 分类                   | 接口 / 运算符                                                                                  | 功能说明                                                                                  |
|------------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **构造与工厂方法**     | [`Money(values, currency)`](#moneyvalues-currency)                                             | 从主货币单位数值、字符串、`Decimal`、`FixedArray` 或同币种 `Money` 构造                   |
|                        | [`Money.from_units`](#moneyfrom_units)                                                         | 从最小原子单位（如“分”）的整数或整数数组构造（自动复制外部数组）                          |
|                        | [`Money.zeros`](#moneyzeros-与-moneyones) / [`Money.ones`](#moneyzeros-与-moneyones)           | 构造指定形状的全 0 或全 1 主货币单位金额张量                                              |
| **核心属性**           | [`Money.units`](#moneyunits)                                                                   | 以最小原子单位表示的只读 `int64` 整数矩阵（等价于 `mantissa`）                            |
|                        | [`Money.currency`](#moneycurrency)                                                             | 绑定的 [`Currency`](currency.md) 币种单例                                                 |
|                        | [继承自 `UnitArray` 的数组属性](#继承自-unitarray-的数组属性)                                  | `data`、`mantissa`、`scale`、`shape`、`ndim`、`size`、`dtype`、`T`                        |
| **算术与矩阵运算**     | [`m1 + m2`、`m1 - m2`](#m1--m2-与-m1---m2同币种加减法)                                         | 同币种金额逐元素精确加减                                                                  |
|                        | [`+m`、`-m`、`abs(m)`](#m-m-与-absm正负号与绝对值)                                             | 一元正负号与逐元素绝对值                                                                  |
|                        | [`m * other`、`other * m`](#m--other-与-other--m无量纲乘法)                                    | 金额乘以无量纲倍数（默认 `half_up` 舍入到币种精度）                                       |
|                        | [`Money.multiply`](#moneymultiply)                                                             | 指定舍入模式与可选对齐步长 `step` 的无量纲乘法                                            |
|                        | [`m / other`](#m--other真除法运算符)                                                           | 金额除以无量纲数、同币种 `Money`、[`Price`](price.md) 或 [`Quantity`](quantity.md)        |
|                        | [`Money.divide`](#moneydivide)                                                                 | 支持自定义 `rounding`、`step` 或目标 `scale` 的通用除法                                   |
|                        | [`m @ other`、`other @ m`、`Money.matmul`](#m--otherother--m-与-moneymatmul矩阵内积)           | 金额张量与无量纲权重矩阵的内积乘法                                                        |
| **规约与步长网格**     | [`Money.sum`](#moneysum)                                                                       | 沿轴精确求和（支持 `np.sum(m)` 转发）                                                     |
|                        | [`Money.mean`](#moneymean)                                                                     | 沿轴计算算术平均金额并按指定模式舍入（支持 `np.mean(m)` 转发）                            |
|                        | [`Money.round_to_step`](#moneyround_to_step)                                                   | 将金额对齐到正数步长网格（步长精度不超过币种精度）                                        |
|                        | [`Money.is_on_step`](#moneyis_on_step)                                                         | 逐元素判断金额是否严格落在步长格点上                                                      |
|                        | [`Money.is_zero`](#moneyis_zero)                                                               | 逐元素或标量判断金额是否为 0                                                              |
| **守恒分配与费用**     | [`divmod(m, other)`](#divmodm-other守恒带余整除)                                               | 按单份金额求整数份数与零头，或按正整数份数均分金额与零头                                  |
|                        | [`Money.allocate`](#moneyallocate)                                                             | 按权重使用最大余数法（Hare-Niemeyer）100% 守恒分配标量金额                                |
|                        | [`Money.calc_tiered_fee`](#moneycalc_tiered_fee)                                               | 单一费率或分档累进交易费/印花税计算（支持最低收费与封顶）                                 |
| **比较、导出与格式化** | [比较、索引与形状变换](#比较索引与形状变换)                                                    | `==`、`!=`、`<`、`<=`、`>`、`>=`、`m[key]`、`reshape`、`transpose`、`len`、`bool`、`iter` |
|                        | [`Money.to_pandas`](#moneyto_pandas)                                                           | 导出为 `pandas.Series` / `DataFrame` 并自动注入 `attrs["currency"]`                       |
|                        | [`Money.to_polars`、`to_float`、`to_decimal`](#moneyto_polarsmoneyto_float-与-moneyto_decimal) | 导出为 `polars.Series`、`float64` 或 `Decimal`                                            |
|                        | [`Money.format`](#moneyformat)                                                                 | 调用 `MonetaryFormatter` 生成千分位/货币符号/会计括号格式字符串                           |
|                        | [`str(m)`、`repr(m)`](#strm-与-reprm文本表示)                                                  | 带币种代码的简明文本与完整构造表达式展示                                                  |
| **内部钩子与私有 API** | [`Money._create_like`](#money_create_like)                                                     | `UnitArray` 钩子：用新 `FixedArray` 零拷贝包装同币种 `Money`                              |
|                        | [`Money._check_compatible`](#money_check_compatible)                                           | `UnitArray` 钩子：校验另一对象是否为同币种 `Money`                                        |
|                        | [`Money._check_compatible_currency`](#money_check_compatible_currency)                         | 校验当前实例币种是否与目标 `Currency` 一致                                                |
|                        | [`Money._wrap`](#money_wrap)                                                                   | 内部零拷贝构造器（强制校验 `data.scale == currency.digits`）                              |
|                        | [`Money._fill_value`](#money_fill_value)                                                       | `UnitArray` 钩子：将 `fillna` 数值填充值按 `Money` 构造规则换算到币种精度                 |
|                        | [`Money._as_step`](#money_as_step)                                                             | 校验正数步长且精度不超过币种精度                                                          |
|                        | [`Money._fused_round_step`](#money_fused_round_step)                                           | 由 `step.scale + 1` 精度的下界与非精确标志一次性舍入到步长网格                            |
|                        | [`_RATIO_EXTRA_DIGITS`](#_ratio_extra_digits)                                                  | 除法结果非 `Money` 时相对币种精度默认额外保留的小数位数（`8`）                            |

## 构造、工厂方法与核心属性

### `Money(values, currency)`

```python
class moneta.Money(values, currency)
```

从以主货币单位（如“元”、“美元”）表示的数值、字符串、`Decimal`、`Fraction`、`FixedArray`、序列、NumPy 数组或同币种 `Money`
构造货币金额张量。

**参数**

- **values** : *int, float, str, Decimal, Fraction, FixedArray, Money, Sequence, np.ndarray 或 pandas / polars Series* —
  以主货币单位表示的输入数值。
- **currency** : *Currency 或 str* — 所属币种，支持传入 [`Currency`](currency.md) 实例或已在 [
  `CurrencyRegistry`](currency.md#currencyregistry) 注册的货币代码字符串（忽略大小写与首尾空白，如 `"USD"`、`"cny"`）。

**返回值**

- **out** : *Money* — `scale` 恒等于 `currency.digits` 的金额张量。

**异常**

- **TypeError** — `values` 是不支持的类型（如 `bool`），或 `values` 是不同币种的 `Money`；或传入了多余的 `scale=`
  关键字参数。
- **KeyError** — `currency` 为未注册的货币代码字符串。
- **ValueError** — `currency` 是与注册表同代码定义不一致的伪造 `Currency` 实例；或字符串/`Fraction` 无法解析为有限十进制数。
- **OverflowError** — 转换到原子单位后的整数尾数超出 `int64` 范围。

**警告**

- **[`PrecisionLossWarning`](fixed_array.md#precisionlosswarning)** — 非浮点数的精确输入在超出 `currency.digits`
  的小数位上含有非零数字（缺失位置除外）时触发。

**机制说明**

构造器将数值解析与精度对齐全部委托给底层 [`FixedArray`](fixed_array.md)：

1. 先调用 [`get_currency(currency)`](currency.md#get_currency) 解析并校验币种单例 `curr`；
2. 若 `values` 已是 `Money`：调用 [`values._check_compatible_currency(curr)`](#money_check_compatible_currency)
   校验币种一致后直接复用 `values._data`；
3. 否则直接执行 `self._data = FixedArray(values, scale=curr.digits, on_excess="warn")`：
    - 浮点数按最短十进制写法 `half_up` 静默舍入到 `curr.digits`；
    - 精确输入（`str`、`Decimal`、`Fraction`、`FixedArray` 及包含它们的序列）若多出的位只是末尾 `0`（如
      `Money("1.500", "USD")`）或缺失位置，直接无损对齐；若多出的位含有非零数字，由 `FixedArray` 发出 [
      `PrecisionLossWarning`](fixed_array.md#precisionlosswarning)（告警栈帧自动跳过 `moneta` 内部指向用户代码）后按
      `half_up` 舍入。

**示例**

从精确输入与浮点数构造：

```python
>>> import warnings
>>> from decimal import Decimal
>>> from fractions import Fraction
>>> from moneta import Money, USD, JPY, PrecisionLossWarning
>>> Money("12.34", "USD")
Money(FixedArray(12.34, scale=2), currency='USD')
>>> Money([1, Decimal("2.5"), Fraction(1, 4)], USD)
Money(FixedArray([1.00, 2.50, 0.25], scale=2), currency='USD')
>>> Money(1.005, "USD")          # float 按最短写法静默 half_up 舍入
Money(FixedArray(1.01, scale=2), currency='USD')
>>> Money([0.4, -2.5], JPY)      # JPY 的 digits=0，直接舍入到整数日元
Money(FixedArray([0, -3], scale=0), currency='JPY')
```

精确输入超出币种精度时发出 `PrecisionLossWarning` 后按 `half_up` 舍入（末尾多余的 `0` 不触发告警）：

```python
>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     m_zero = Money("1.500", "USD")
...     m_round = Money("1.005", "USD")
...     (m_zero, m_round, len(w), issubclass(w[0].category, PrecisionLossWarning))
(Money(FixedArray(1.50, scale=2), currency='USD'), Money(FixedArray(1.01, scale=2), currency='USD'), 1, True)
>>> with warnings.catch_warnings():
...     warnings.simplefilter("error", PrecisionLossWarning)
...     Money("0.001", "USD")
Traceback (most recent call last):
    ...
moneta.fixed_array.PrecisionLossWarning: 输入的小数位数 (3) 超出指定 scale (2)，已按 half_up 舍入
```

从已有 `Money` 构造（同币种复用，跨币种报错）：

```python
>>> m = Money("15.00", "USD")
>>> bool(Money(m, "usd") == m)
True
>>> Money(m, "CNY")
Traceback (most recent call last):
    ...
TypeError: 币种不匹配: USD 与 CNY
```

**性能对比**

从浮点数或字符串构造 `Money` 需要解析十进制小数位；若已有以最小原子单位（如“分”）表示的整数数组，使用 [
`Money.from_units`](#moneyfrom_units) 可跳过解析直接构造（快约 30–50 倍）：

|         N | `Money(float_arr, "USD")` | `Money.from_units(int_arr, "USD")` | NumPy `np.round(x * 100).astype(np.int64)` |
|----------:|--------------------------:|-----------------------------------:|-------------------------------------------:|
|     1,000 |                   57.9 µs |                            2.07 µs |                                    2.06 µs |
| 1,000,000 |                     35 ms |                             641 µs |                                    1.98 ms |

**另请参阅**

- [`Money.from_units`](#moneyfrom_units)
- [`PrecisionLossWarning`](fixed_array.md#precisionlosswarning)

### Money.from_units

```python
@classmethod
Money.from_units(units, currency)
```

由以币种最小原子单位（如美元的“美分”、人民币的“分”、日元的“元”）表示的整数或整数数组直接构造 `Money`。

**参数**

- **units** : *int, np.integer, Sequence[int] 或整数 np.ndarray* —
  原子单位整数值（满足 $\text{金额} = \text{units} \times 10^{-\text{currency.digits}}$）。
- **currency** : *Currency 或 str* — 所属币种。

**返回值**

- **out** : *Money*

**异常**

- **TypeError** — `units` 含浮点数、`bool` 或其他非整数类型。
- **OverflowError** — `units` 超出有符号 `int64` 范围。

**机制说明**

先通过 [`FixedArray._as_int64(units)`](fixed_array.md#fixedarray_as_int64) 校验整数类型并转为 `int64` 数组 `m`。若传入的
`units` 本身是 `np.ndarray` 且与 `m` 共享内存（`np.may_share_memory(m, units)`），会主动执行 `m = m.copy()`
复制一份，确保调用方事后原地修改外部数组不会破坏 `Money` 的不可变性；随后调用 [
`FixedArray._from_mantissa(m, curr.digits)`](fixed_array.md#fixedarray_from_mantissa) 与 [`Money._wrap`](#money_wrap)
封装返回。

**示例**

```python
>>> arr = np.array([100, 250], dtype=np.int64)
>>> m = Money.from_units(arr, "USD")
>>> m
Money(FixedArray([1.00, 2.50], scale=2), currency='USD')
>>> arr[0] = 99999  # 修改原数组不影响已构造的 Money
>>> m.units.tolist()
[100, 250]
>>> Money.from_units(500, "JPY")
Money(FixedArray(500, scale=0), currency='JPY')
>>> Money.from_units(1.5, "USD")
Traceback (most recent call last):
    ...
TypeError: 只接受整数，收到 float64
```

**另请参阅**

- [`Money.units`](#moneyunits)
- [`Money(values, currency)`](#moneyvalues-currency)

### Money.zeros 与 Money.ones

```python
@classmethod
Money.zeros(shape, currency)

@classmethod
Money.ones(shape, currency)
```

快速构造指定形状的全 `0` 金额张量，或各元素均为 `1` 主货币单位（如 `1.00 USD`、`1 JPY`）的金额张量。

**参数**

- **shape** : *int 或 tuple of int* — 目标张量形状（`()` 表示 0 维标量）。
- **currency** : *Currency 或 str* — 所属币种。

**返回值**

- **out** : *Money*

**机制说明**

- `Money.zeros(shape, currency)` 调用 `cls.from_units(np.zeros(shape, dtype=np.int64), currency)`；
- `Money.ones(shape, currency)` 将每个元素的原子单位设为 `10 ** curr.digits`（即恰好等于 `1` 个主货币单位，而非 `1`
  个最小原子单位），调用 `cls.from_units(np.full(shape, 10 ** curr.digits, dtype=np.int64), curr)`。

**示例**

```python
>>> Money.zeros(3, "USD")
Money(FixedArray([0.00, 0.00, 0.00], scale=2), currency='USD')
>>> Money.ones((2, 2), "USD")
Money(FixedArray([[1.00, 1.00],
            [1.00, 1.00]], scale=2), currency='USD')
>>> Money.ones((), "JPY")
Money(FixedArray(1, scale=0), currency='JPY')
```

**另请参阅**

- [`Money.from_units`](#moneyfrom_units)
- [`Money.is_zero`](#moneyis_zero)

### Money.units

```python
Money.units
```

以币种最小原子单位表示的只读 `int64` 整数尾数数组（等价于 [`self.mantissa`](unit_array.md#unitarraymantissa)）。

**返回值**

- **units** : *numpy.ndarray（int64）* — 0 维时为 0 维 `int64` 数组，1 维及以上为对应形状的只读视图。

**机制说明**

直接返回 `self.mantissa`。对于 `USD` / `CNY`（`digits=2`），`units` 即为“美分”或“分”的整数个数；对于 `JPY`（`digits=0`），
`units` 即为整数日元。

**示例**

```python
>>> m = Money(["1.50", "-2.05"], "USD")
>>> m.units
array([ 150, -205])
>>> m.units.flags.writeable
False
```

**另请参阅**

- [`Money.from_units`](#moneyfrom_units)
- [`UnitArray.mantissa`](unit_array.md#unitarraymantissa)

### Money.currency

```python
Money.currency
```

当前金额张量绑定的 [`Currency`](currency.md) 币种对象。

**返回值**

- **currency** : *Currency*

**机制说明**

直接返回只读槽位 `self._currency`。若币种代码已在全局 [`CurrencyRegistry`](currency.md#currencyregistry) 中注册，该属性与预置币种常量（如
`USD`、`CNY`）满足 `is` 同一性。

**示例**

```python
>>> from moneta import Money, USD
>>> m = Money("10.00", "usd")
>>> m.currency is USD
True
```

**另请参阅**

- [`Currency`](currency.md#currencycode-digits-symbol-name)
- [`get_currency`](currency.md#get_currency)

### 继承自 UnitArray 的数组属性

`Money` 继承自 [`UnitArray`](unit_array.md)，直接提供以下底层数据与形状属性：

| 属性                                                                                                                                                                | 类型                              | 说明                                                              |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|-------------------------------------------------------------------|
| [`m.data`](unit_array.md#unitarraydata)                                                                                                                             | [`FixedArray`](fixed_array.md)    | 底层定点数张量实例（`m.data.scale == m.currency.digits`）         |
| [`m.mantissa`](unit_array.md#unitarraymantissa)                                                                                                                     | `ndarray(int64)`                  | 底层只读 `int64` 尾数视图（与 [`m.units`](#moneyunits) 完全相同） |
| [`m.scale`](unit_array.md#unitarrayscale)                                                                                                                           | `int`                             | 小数位数，恒等于 `m.currency.digits`                              |
| [`m.shape`](unit_array.md#unitarrayshape) / [`ndim`](unit_array.md#unitarrayndim) / [`size`](unit_array.md#unitarraysize) / [`dtype`](unit_array.md#unitarraydtype) | `tuple` / `int` / `int` / `dtype` | 张量形状、维度数、元素总数、尾数类型（`int64`）                   |
| [`m.T`](unit_array.md#unitarrayt)                                                                                                                                   | `Money`                           | 反转全部轴并保持原币种的转置视图                                  |

**示例**

```python
>>> m = Money([["1.50", "2.00"], ["3.00", "4.00"]], "USD")
>>> (m.scale, m.shape, m.ndim, m.size, m.dtype)
(2, (2, 2), 2, 4, dtype('int64'))
>>> m.T
Money(FixedArray([[1.50, 3.00],
            [2.00, 4.00]], scale=2), currency='USD')
```

## 算术、缩放与矩阵运算

### m1 + m2 与 m1 - m2（同币种加减法）

```python
m1 + m2
m1 - m2
```

同币种金额按 NumPy 广播规则逐元素精确相加或相减。

**参数**

- **other** : *Money* — 必须是相同币种的 `Money` 实例，形状可广播。

**返回值**

- **out** : *Money* — 同币种、广播后形状的精确结果。

**异常**

- **TypeError** — `other` 是不同币种的 `Money`（报 `"币种不匹配: ..."`），或 `other` 不是 `Money`（如裸数字、`FixedArray`、[
  `Price`](price.md)、[`Quantity`](quantity.md)）。注意：内置 `sum([m1, m2])` 默认从整数 `0` 开始累加（即 `0 + m1`），会因 `0`
  不是 `Money` 而报 `TypeError`；请使用张量方法 [`m.sum()`](#moneysum) 或指定起始零金额
  `sum(parts, Money.zeros((), "USD"))`。
- **ValueError** — 形状无法广播。
- **OverflowError** — 结果超出 `int64` 范围。

**机制说明**

若 `not isinstance(other, Money)` 则返回 `NotImplemented`；否则调用 `self._check_compatible(other)` 校验币种相同。由于同币种
`Money` 的底层 `scale` 必然相等，底层 `self._data + other._data` 直接走同 `scale` 整数加减快路径并执行符号位溢出检测。

**示例**

```python
>>> Money("1.50", "USD") + Money("0.25", "USD")
Money(FixedArray(1.75, scale=2), currency='USD')
>>> Money(["10.00", "20.00"], "USD") - Money("1.50", "USD")
Money(FixedArray([8.50, 18.50], scale=2), currency='USD')
>>> Money("1.00", "USD") + Money("1.00", "CNY")
Traceback (most recent call last):
    ...
TypeError: 币种不匹配: USD 与 CNY
>>> Money("1.00", "USD") + 1
Traceback (most recent call last):
    ...
TypeError: unsupported operand type(s) for +: 'Money' and 'int'
```

**性能对比**

|         N | `m1 + m2` (`Money`) | `x + y` (`NumPy float64`) |
|----------:|--------------------:|--------------------------:|
|  0-d 标量 |             4.93 µs |                     43 ns |
|     1,000 |             6.08 µs |                    458 ns |
| 1,000,000 |             2.56 ms |                    319 µs |

**另请参阅**

- [`Money.sum`](#moneysum)
- [`FixedArray` 加减法](fixed_array.md#a--b-与-a---b加减法)

### `+m`、`-m` 与 `abs(m)`（正负号与绝对值）

```python
+m
-m
abs(m)
```

一元正号返回自身；一元负号与 `abs(m)` 逐元素取反或取绝对值，保持原币种。

**返回值**

- **out** : *Money*

**异常**

- **OverflowError** — `m.units` 含有 `-2**63` 时调用 `-m` 或 `abs(m)`。

**机制说明**

直接继承自 [`UnitArray` 的 `+a`、`-a`](unit_array.md#a-与--a一元正负号) 与 [`abs(a)`](unit_array.md#absa绝对值)。

**示例**

```python
>>> m = Money(["-1.50", "2.00"], "USD")
>>> -m
Money(FixedArray([1.50, -2.00], scale=2), currency='USD')
>>> abs(m)
Money(FixedArray([1.50, 2.00], scale=2), currency='USD')
```

**另请参阅**

- [`UnitArray` 保单位算术运算](unit_array.md#a-与--a一元正负号)

### m * other 与 other * m（无量纲乘法）

```python
m * other
other * m
```

将金额乘以无量纲标量或数值数组，结果按 `half_up` 舍入到币种精度 `currency.digits`。

**参数**

- **other** : *int, float, Decimal, Fraction, FixedArray 或数值 array_like* — 无量纲乘数（不支持字符串、`bool` 或
  `Money`）。

**返回值**

- **out** : *Money* — 同币种、广播后形状的金额张量。

**异常**

- **TypeError** — `other` 是 `Money`（显式报错 `"禁止两笔金额直接相乘 (Money * Money)：货币量纲平方无物理与金融意义。"`），或
  `other` 是字符串/`bool` 等非数值类型。
- **OverflowError** — 舍入后的结果超出 `int64` 范围。

**机制说明**

- `__mul__` 先拦截 `isinstance(other, Money)` 并抛出明确的量纲错误；随后检查 [
  `FixedArray._is_numeric_operand(other)`](fixed_array.md#fixedarray_is_numeric_operand)
  ，若不是合法的无量纲数值（例如右操作数是 [`Price`](price.md)、[`Quantity`](quantity.md) 或自定义对象），返回
  `NotImplemented` 以便对方接管 `__rmul__`；若是合法数值，则转发调用 `self.multiply(other)`（默认 `rounding="half_up"`）。
- `__rmul__` 支持左操作数为标量或 `ndarray`（由于 `Money.__array_ufunc__ = None`，`np.array([1, 2]) * m` 会直接转发到
  `m.__rmul__` 并返回 `Money`）。

**示例**

```python
>>> from decimal import Decimal
>>> m = Money("10.00", "USD")
>>> m * Decimal("0.333")
Money(FixedArray(3.33, scale=2), currency='USD')
>>> np.array([1, 2]) * Money(["1.50", "2.00"], "USD")
Money(FixedArray([1.50, 4.00], scale=2), currency='USD')
>>> m * m
Traceback (most recent call last):
    ...
TypeError: 禁止两笔金额直接相乘 (Money * Money)：货币量纲平方无物理与金融意义。
```

**性能对比**

|         N | `m * rate` (`Money`) | `np.round(x * r, 2)` (`NumPy float64`) |
|----------:|---------------------:|---------------------------------------:|
|  0-d 标量 |              20.3 µs |                                2.46 µs |
|     1,000 |                30 µs |                                2.67 µs |
| 1,000,000 |              9.59 ms |                                1.01 ms |

**另请参阅**

- [`Money.multiply`](#moneymultiply) — 支持指定舍入模式 `rounding` 与步长网格 `step`。

### Money.multiply

```python
Money.multiply(other, rounding="half_up", step=None)
```

将金额乘以无量纲标量或权重数组，按指定舍入模式舍入到币种精度；若给出 `step`，则 **一次性**直接舍入到步长网格。

**参数**

- **other** : *int, float, Decimal, Fraction, FixedArray 或数值 array_like* — 无量纲乘数。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式（见 [`RoundingMode`](fixed_array.md#roundingmode)）。
- **step** : *int, float, Decimal, Fraction, FixedArray 或数值 array_like，可选* — 步长网格（如 `Decimal("0.05")`
  ），必须为正数且有效小数位数不超过 `currency.digits`。

**返回值**

- **out** : *Money*

**异常**

- **TypeError** — `other` 或 `step` 不是数值白名单类型（如传入了字符串）。
- **ValueError** — `rounding` 无效；或 `step <= 0`；或 `step` 的有效小数位数超过 `currency.digits`（如对 `USD` 传入
  `Decimal("0.005")`）。
- **OverflowError** — 结果超出 `int64` 范围。

**机制说明**

- 当 `step is None` 时，直接委托 [
  `self._data.mul_round(factor, scale=self._currency.digits, rounding=mode)`](fixed_array.md#fixedarraymul_round)
  ，即使中间乘积超出 `int64` 也会自动走 Base $2^{21}$ 分块长乘慢路径，只舍入一次。
- 当给定 `step` 时，为避免“先舍入到 `currency.digits` 再舍入到 `step`”带来的二次舍入误差，先在精度 `t = step_fa.scale + 1`
  上分别用 `"floor"` 和 `"ceil"` 计算乘积下界 `floor` 与非精确掩码 `inexact = (floor != ceil)`，再交由 [
  `_fused_round_step`](#money_fused_round_step) 一次性裁决进位。

**示例**

不同舍入模式与一步到位步长舍入（对比二次舍入差异）：

```python
>>> from decimal import Decimal
>>> from moneta import Money, RoundingMode
>>> Money("0.25", "USD").multiply(Decimal("0.1"), rounding=RoundingMode.HALF_EVEN)
Money(FixedArray(0.02, scale=2), currency='USD')
>>> # 1.00 * 0.0499 = 0.0499：若先舍入到 0.01 会变 0.05 再进位到 0.10；一步到位舍入到 0.10 网格严格得 0.00
>>> Money("1.00", "USD").multiply(Decimal("0.0499"), step=Decimal("0.10"))
Money(FixedArray(0.00, scale=2), currency='USD')
>>> Money("1.00", "USD").multiply(Decimal("0.05"), step=Decimal("0.10"), rounding="half_up")
Money(FixedArray(0.10, scale=2), currency='USD')
>>> Money("1.00", "USD").multiply(Decimal("0.05"), step=Decimal("0.10"), rounding="half_even")
Money(FixedArray(0.00, scale=2), currency='USD')
```

**另请参阅**

- [`m * other`](#m--other-与-other--m无量纲乘法)
- [`Money.round_to_step`](#moneyround_to_step)
- [`Money._fused_round_step`](#money_fused_round_step)

### m / other（真除法运算符）

```python
m / other
```

按除数类型自动分派量纲代数：除以无量纲数得 `Money`，除以同币种 `Money` 得无量纲比率 [`FixedArray`](fixed_array.md)，除以 [
`Price`](price.md) 得 [`Quantity`](quantity.md)，除以 [`Quantity`](quantity.md) 得 [`Price`](price.md)。默认均采用
`half_up` 舍入。

**参数**

- **other** : *数值 array_like、同币种 Money、同计价币种 Price 或 Quantity* — 除数。

**返回值**

- **out** : *Money、FixedArray、Quantity 或 Price* — 当结果为 `Money` 时，`scale` 为 `currency.digits`；当结果为
  `FixedArray`、[`Quantity`](quantity.md) 或 [`Price`](price.md) 时，默认保留 `currency.digits + 8` 位小数（如 `USD` 为
  `10` 位，`JPY` 为 `8` 位）。

**异常**

- **ZeroDivisionError** — 除数含有 0。
- **TypeError** — 除以不同币种的 `Money`、不同计价币种的 [`Price`](price.md)、非数值对象，或将无量纲数除以 `Money`（如
  `1 / m`）。

**机制说明**

直接调用 [`self.divide(other)`](#moneydivide)。

**示例**

```python
>> > from moneta import Money, Price, Quantity
>> > Money("10.00", "USD") / 3  # Money / 无量纲数 -> Money (scale=2)
Money(FixedArray(3.33, scale=2), currency='USD')
>> > Money("1.00", "USD") / Money("3.00", "USD")  # Money / Money -> FixedArray (scale = 2 + 8 = 10)
FixedArray(0.3333333333, scale=10)
>> > Money("1895.00", "USD") / Price("189.50", "USD", base_unit="share")  # Money / Price -> Quantity
Quantity(FixedArray(10.0000000000, scale=10), unit='share')
>> > Money("100.00", "USD") / Quantity(4, unit="barrel")  # Money / Quantity -> Price
Price(FixedArray(25.0000000000, scale=10), quote_currency='USD', base_unit='barrel')
```

**性能对比**

|         N | `m / 3` (`Money`) | `m1 / m2` (`FixedArray` 比率) | `np.round(x / y, 2)` (`NumPy float64`) |
|----------:|------------------:|------------------------------:|---------------------------------------:|
|  0-d 标量 |           18.5 µs |                         28 µs |                                2.47 µs |
|     1,000 |           28.2 µs |                       45.8 µs |                                3.13 µs |
| 1,000,000 |           8.31 ms |                       13.5 ms |                                 749 µs |

**另请参阅**

- [`Money.divide`](#moneydivide)
- [`divmod(m, other)`](#divmodm-other守恒带余整除)

### Money.divide

```python
Money.divide(other, rounding="half_up", step=None, scale=None)
```

通用金额除法，支持指定舍入模式 `rounding`、金额步长网格 `step`（仅当结果为 `Money` 时）或目标小数位数 `scale`（仅当结果不是
`Money` 时）。

**参数**

- **other** : *数值 array_like、Money、Price 或 Quantity* — 除数。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式。
- **step** : *数值 array_like，可选* — 仅当 `other` 为无量纲数值（结果为 `Money`）时可用：将商一次性舍入到该正数步长网格。
- **scale** : *int，可选* — 仅当 `other` 为 `Money`、[`Price`](price.md) 或 [`Quantity`](quantity.md)（结果不是 `Money`
  ）时可用：指定返回的比率、数量或单价的小数位数；默认 `None` 时取 `self.currency.digits + 8`。

**返回值**

- **out** : *Money、FixedArray、Quantity 或 Price*

**异常**

- **ZeroDivisionError** — 除数含有 0。
- **TypeError** — 币种不匹配或除数类型非法。
- **ValueError** — 当结果为 `Money` 时传入了 `scale`（报 `"结果为 Money 时精度固定为币种精度，不能指定 scale"`）；或当结果不是
  `Money` 时传入了 `step`（报 `"step 只适用于结果为 Money 的除法"`）；或 `step` / `scale` / `rounding` 取值非法。

**机制说明**

1. **量纲消去分支（`other` 为 `Money` / [`Price`](price.md) / [`Quantity`](quantity.md)）**：校验 `step is None`，计算目标精度
   `target = self._currency.digits + 8 if scale is None else FixedArray._check_scale(scale)`，调用
   `self._data.divide(other._data, scale=target, rounding=mode)` 并封装为 `FixedArray`、[`Quantity`](quantity.md) 或 [
   `Price`](price.md)。
2. **无量纲缩放分支（`other` 为数值）**：校验 `scale is None`。若 `step is None` 则直接调用
   `self._data.divide(divisor, scale=self._currency.digits, rounding=mode)`；若给出了 `step`，则在 `t = step_fa.scale + 1`
   精度上求 `"floor"` 与 `"ceil"` 并通过 [`_fused_round_step`](#money_fused_round_step) 一次性舍入到步长网格。

**示例**

```python
>> > from decimal import Decimal
>> > from moneta import Money, Quantity
>> > Money("-10.00", "USD").divide(3, rounding="floor")
Money(FixedArray(-3.34, scale=2), currency='USD')
>> > Money("1.00", "USD").divide(Decimal("20.04"), step=Decimal("0.10"))  # 0.0499... 一步舍入到 0.10 得 0.00
Money(FixedArray(0.00, scale=2), currency='USD')
>> > Money("1.00", "USD").divide(Money("3.00", "USD"), scale=4, rounding="up")
FixedArray(0.3334, scale=4)
>> > Money("100.00", "USD").divide(Quantity(3, "share"), scale=2)
Price(FixedArray(33.33, scale=2), quote_currency='USD', base_unit='share')
```

互斥参数校验：

```python
>>> Money("10.00", "USD").divide(3, scale=4)
Traceback (most recent call last):
    ...
ValueError: 结果为 Money 时精度固定为币种精度，不能指定 scale
>>> Money("10.00", "USD").divide(Money("3.00", "USD"), step=Decimal("0.05"))
Traceback (most recent call last):
    ...
ValueError: step 只适用于结果为 Money 的除法
```

**另请参阅**

- [`m / other`](#m--other真除法运算符)
- [`Money.multiply`](#moneymultiply)
- [`_RATIO_EXTRA_DIGITS`](#_ratio_extra_digits)

### m @ other、other @ m 与 Money.matmul（矩阵内积）

```python
m @ other
other @ m
Money.matmul(other, rounding="half_up")
```

计算金额向量/矩阵与无量纲权重矩阵的内积，所有乘积项先精确累加，最后按指定模式一次性舍入到币种精度。

**参数**

- **other** : *FixedArray、ndarray 或数值序列* — 无量纲权重向量或矩阵（不支持 `Money` @ `Money` 或字符串）。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式（`@` 与反向 `@` 固定使用 `"half_up"`；需自定义舍入模式时调用
  `m.matmul(other, rounding=...)`）。

**返回值**

- **out** : *Money* — 同币种的矩阵乘法结果。

**异常**

- **TypeError** — `other` 不是合法的无量纲数值操作数（如传入 `Money` 或字符串）。
- **ValueError** — 维度不满足矩阵乘法规则（如 0 维标量或内维不匹配）。
- **OverflowError** — 最终结果超出 `int64` 范围。

**机制说明**

委托底层 [`FixedArray.matmul(weights, scale=self._currency.digits, rounding=rounding)`](fixed_array.md#fixedarraymatmul)
：内积求和在未舍入的精确精度上完成（若最坏上界超出 `int64` 则自动切换至 Base $2^{21}$
分块矩阵乘法），只在内积汇总后执行一次舍入，避免逐项相乘先舍入导致的累积偏差。反向运算符 `__rmatmul__` 同样支持
`weights @ m`。

**示例**

```python
>>> from decimal import Decimal
>>> m = Money(["10.00", "20.00", "30.00"], "USD")
>>> m @ FixedArray(["0.2", "0.3", "0.5"])
Money(FixedArray(23.00, scale=2), currency='USD')
>>> m.matmul([Decimal("0.333")] * 3, rounding="down")
Money(FixedArray(19.98, scale=2), currency='USD')
>>> np.array([[1, 1, 0], [0, 1, 1]]) @ m
Money(FixedArray([30.00, 50.00], scale=2), currency='USD')
```

**另请参阅**

- [`FixedArray.matmul`](fixed_array.md#fixedarraymatmul)
- [`Money.multiply`](#moneymultiply)

## 规约统计与步长网格

### Money.sum

```python
Money.sum(axis=None, keepdims=False, skipna=True, out=None, dtype=None)
```

沿指定轴对金额张量精确求和，结果保持原币种。调用 `np.sum(m, axis=..., keepdims=...)` 也会自动转发到此方法。

**参数**

- **axis** : *int 或 tuple of int，可选* — 求和轴；默认 `None` 时对全部元素求和。
- **keepdims** : *bool，默认 `False`* — 是否保留长度为 1 的规约轴。
- **skipna** : *bool，默认 `True`* — 是否忽略缺失值 `<NA>`。为 `True` 时缺失位置按 `0` 处理；为 `False` 时若求和切片中含有
  `<NA>`，该切片结果为 `<NA>`。
- **out / dtype** : *None* — 兼容 NumPy 签名预留，必须为 `None`。

**返回值**

- **out** : *Money*

**机制说明**

直接继承自 [`UnitArray.sum`](unit_array.md#unitarraysum)，底层调用 [`FixedArray.sum`](fixed_array.md#fixedarraysum)（含
Base $2^{21}$ 分块防中间溢出慢路径与 `skipna` 缺失值处理）。

**示例**

```python
>>> m = Money([["1.50", "2.00"], ["3.00", "4.25"]], "USD")
>>> m.sum()
Money(FixedArray(10.75, scale=2), currency='USD')
>>> np.sum(m, axis=0)
Money(FixedArray([4.50, 6.25], scale=2), currency='USD')
>>> m_na = Money(["1.50", None, "2.50"], "USD")
>>> m_na.sum(), m_na.sum(skipna=False)
(Money(FixedArray(4.00, scale=2), currency='USD'), Money(FixedArray(<NA>, scale=2), currency='USD'))
```

**另请参阅**

- [`Money.mean`](#moneymean)
- [`UnitArray.sum`](unit_array.md#unitarraysum)

### Money.mean

```python
Money.mean(axis=None, keepdims=False, rounding="half_up", skipna=True, out=None, dtype=None)
```

沿指定轴计算算术平均金额，并按 `rounding` 舍入到币种精度。`np.mean(m, axis=..., keepdims=...)` 也会自动转发到此方法。

**参数**

- **axis** : *int 或 tuple of int，可选* — 求均值的轴；默认 `None` 时对全部元素求均值。
- **keepdims** : *bool，默认 `False`* — 是否保留规约轴。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式。
- **skipna** : *bool，默认 `True`* — 是否忽略缺失值 `<NA>`。为 `True` 时仅对有效非缺失元素求均值（若某切片全部为 `<NA>`
  则该切片结果仍为 `<NA>`）；为 `False` 时含 `<NA>` 的切片结果为 `<NA>`。
- **out / dtype** : *None* — 兼容 `np.mean` 签名预留，必须为 `None`。

**返回值**

- **out** : *Money*

**异常**

- **ValueError** — 在空数组（`size == 0`）上求均值，或 `rounding` 无效。
- **TypeError** — 传入了非 `None` 的 `out` 或 `dtype`。

**机制说明**

内部调用
`self._create_like(self._data.mean(axis=axis, keepdims=keepdims, rounding=rounding, skipna=skipna, out=out, dtype=dtype))`
。底层先沿轴对有效元素精确求和，再除以有效元素个数 `count` 并按 `rounding` 一次性舍入到 `self._data.scale`（即
`currency.digits`）。

**示例**

```python
>>> Money(["1.00", "2.00", "4.00"], "USD").mean()
Money(FixedArray(2.33, scale=2), currency='USD')
>>> Money(["0.01", "0.02"], "USD").mean(rounding="floor")
Money(FixedArray(0.01, scale=2), currency='USD')
>>> np.mean(Money(["1.00", "2.00"], "USD"))
Money(FixedArray(1.50, scale=2), currency='USD')
>> > m_na = Money(["1.00", None, "2.00"], "USD")
>> > m_na.mean(), m_na.mean(skipna=False)
(Money(FixedArray(1.50, scale=2), currency='USD'), Money(FixedArray( < NA >, scale=2), currency='USD'))
```

**性能对比**

|         N | `m.mean()` (`Money`) | `np.round(np.mean(x), 2)` (`NumPy float64`) |
|----------:|---------------------:|--------------------------------------------:|
|     1,000 |              13.2 µs |                                     4.71 µs |
| 1,000,000 |              2.15 ms |                                      188 µs |

**另请参阅**

- [`Money.sum`](#moneysum)
- [`FixedArray.mean`](fixed_array.md#fixedarraymean)

### Money.round_to_step

```python
Money.round_to_step(step, rounding="half_up")
```

将金额离散对齐到指定的正数步长网格（例如瑞士法郎/加元现金结算的 `0.05` 网格，或整百元网格 `100`）。

**参数**

- **step** : *int, float, Decimal, Fraction, FixedArray 或数值 array_like* — 步长网格，必须为正数，且有效小数位数不能超过
  `currency.digits`。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式。

**返回值**

- **out** : *Money*

**异常**

- **TypeError** — `step` 不是数值白名单类型（如字符串）。
- **ValueError** — `step <= 0`，或 `step` 的有效精度超过 `currency.digits`（如对 `USD` 传入 `0.005`）。

**机制说明**

先通过 [`self._as_step(step)`](#money_as_step) 校验步长为正数且有效小数位不超过 `currency.digits`（若 `step` 带有多余尾随零如
`FixedArray("0.050")`，会被无损缩减到 `digits`），随后调用 [
`self._data.round_to_step(step_fa, rounding=rounding)`](fixed_array.md#fixedarrayround_to_step) 并包装为同币种 `Money`。

**示例**

```python
>>> from decimal import Decimal
>>> Money("10.23", "USD").round_to_step(Decimal("0.05"))
Money(FixedArray(10.25, scale=2), currency='USD')
>>> Money("10.23", "USD").round_to_step(Decimal("0.05"), rounding="down")
Money(FixedArray(10.20, scale=2), currency='USD')
>>> Money("345.00", "USD").round_to_step(200)
Money(FixedArray(400.00, scale=2), currency='USD')
>>> Money("1.00", "USD").round_to_step(Decimal("0.003"))
Traceback (most recent call last):
    ...
ValueError: 步长 0.003 的精度超过 USD 的精度 (2)
```

**性能对比**

|         N | `m.round_to_step(Decimal("0.05"))` | `np.round(x / 0.05) * 0.05` (`NumPy float64`) |
|----------:|-----------------------------------:|----------------------------------------------:|
|  0-d 标量 |                              31 µs |                                        1.2 µs |
|     1,000 |                            41.8 µs |                                        2.3 µs |
| 1,000,000 |                            9.25 ms |                                        643 µs |

**另请参阅**

- [`Money.is_on_step`](#moneyis_on_step)
- [`Money.multiply`](#moneymultiply)
- [`FixedArray.round_to_step`](fixed_array.md#fixedarrayround_to_step)

### Money.is_on_step

```python
Money.is_on_step(step)
```

检查金额是否严格位于指定的正数步长格点上（即是否为 `step` 的整数倍）。

**参数**

- **step** : *int, float, Decimal, Fraction, FixedArray 或数值 array_like* — 正数步长（允许传入比 `currency.digits`
  更细的步长，此时若不能整除会返回 `False`）。

**返回值**

- **out** : *bool 或 numpy.ndarray（bool）* — 0 维标量返回 Python `bool`，高维张量返回同形状的 `bool` 数组。

**异常**

- **TypeError** — `step` 不是数值白名单类型。
- **ValueError** — `step <= 0`。

**机制说明**

先通过 [`_as_grid(step, "步长")`](unit_array.md#unitarray_as_grid-与-unitarray_fill_value) 转为不含缺失值的 `FixedArray`
，再委托 [
`self._data.is_on_step(...)`](fixed_array.md#fixedarrayis_on_step) 对齐 `scale` 后检查尾数余数是否为 0。

**示例**

```python
>>> from decimal import Decimal
>>> Money("10.25", "USD").is_on_step(Decimal("0.05"))
True
>>> Money("1.00", "USD").is_on_step(Decimal("0.003"))
False
>>> Money(["1.00", "1.05"], "USD").is_on_step(Decimal("0.10"))
array([ True, False])
```

**另请参阅**

- [`Money.round_to_step`](#moneyround_to_step)
- [`FixedArray.is_on_step`](fixed_array.md#fixedarrayis_on_step)

### Money.is_zero

```python
Money.is_zero()
```

判断金额是否为 0。0 维标量返回 `np.bool_`，高维张量返回同形状的布尔数组。

**机制说明**

直接继承自 [`UnitArray.is_zero`](unit_array.md#unitarrayis_zero)，等价于 `self.units == 0`。切勿使用 `m == 0` 判断零金额（
`Money` 与无量纲整数 `0` 比较恒返回 `False`）。

**示例**

```python
>>> bool(Money("0.00", "USD").is_zero())
True
>>> Money(["0.00", "1.50"], "USD").is_zero()
array([ True, False])
```

**另请参阅**

- [`UnitArray.is_zero`](unit_array.md#unitarrayis_zero)

## 守恒整除、比例分配与阶梯费用

### `divmod(m, other)`（守恒带余整除）

```python
divmod(m, other)
```

资金守恒带余除法，按 `other` 的类型支持两种互补模式：

1. **`divmod(m, other_money)`（按单份金额拆份数）**：计算总金额 `m` 能拆出多少整份同币种金额 `other_money`，返回
   `(整数份数, 剩余零头 Money)`；
2. **`divmod(m, n)`（按正整数份数均分）**：将总金额 `m` 均分为正整数 `n` 份，返回 `(每份 Money, 剩余零头 Money)`。

两种模式均遵循向下取整（与 Python 内置 `divmod` 一致，当除数为正时零头恒 $\ge 0$），且严格满足守恒恒等式：

$$\text{商} \times \text{other} + \text{零头} \equiv m$$

**参数**

- **other** : *同币种 Money，或正整数（int / np.integer / 整数 ndarray）* — 单份金额或均分份数（支持广播）。

**返回值**

- **out** : *tuple*
    - 当 `other` 为 `Money` 时：返回 `(count, rest)`，其中 `count` 为 `int`（0 维时）或 `ndarray(int64)`，`rest` 为同币种
      `Money`；
    - 当 `other` 为正整数时：返回 `(per_part, rest)`，两者均为同币种 `Money`。

**异常**

- **ZeroDivisionError** — `other` 为 0 金额。
- **ValueError** — 整数均分份数 `other <= 0`。
- **TypeError** — `other` 是不同币种的 `Money`；或 `other` 是非整数的数值（如 `3.0`、`Decimal("3")`、`FixedArray("3")`，报错提示
  `"均分份数必须是整数...；按比例分配请使用 allocate"`）；或 `other` 是 `bool`/字符串/其他类型。

**机制说明**

- 当 `isinstance(other, Money)` 时：调用 `self._check_compatible(other)` 后委托 `divmod(self._data, other._data)`
  ，将余数包装为同币种 `Money`。
- 当 `other` 为整数或整数数组时：先排除 `bool` 并校验 `parts > 0`，随后直接在原子单位尾数上执行
  `per_part, rest = np.divmod(self.mantissa, parts)`，将商和余数分别通过 `FixedArray._from_mantissa(..., digits)` 包装为
  `Money`。由于直接在最小原子单位整数上做 `divmod`，1 分钱都不会多出或丢失。

**示例**

模式 1：除以同币种 `Money`（求可买份数与剩余资金）：

```python
>>> count, rest = divmod(Money("10.00", "USD"), Money("3.00", "USD"))
>>> count, rest
(FixedArray(3, scale=0), Money(FixedArray(1.00, scale=2), currency='USD'))
```

模式 2：除以正整数 `n`（均分为 `n` 份与剩余零头）：

```python
>>> per_part, rest = divmod(Money("10.00", "USD"), 3)
>>> per_part, rest
(Money(FixedArray(3.33, scale=2), currency='USD'), Money(FixedArray(0.01, scale=2), currency='USD'))
>>> bool(per_part * 3 + rest == Money("10.00", "USD"))
True
>>> divmod(Money("-10.00", "USD"), 3)  # 向下取整，零头保持非负 (+0.02 USD)
(Money(FixedArray(-3.34, scale=2), currency='USD'), Money(FixedArray(0.02, scale=2), currency='USD'))
>>> divmod(Money("10.00", "USD"), 3.0)
Traceback (most recent call last):
    ...
TypeError: 均分份数必须是整数，收到 float；按比例分配请使用 allocate
```

**性能对比**

|         N | `divmod(m, 3)` (`Money`) | `np.divmod(units, 3)` (`NumPy int64`) |
|----------:|-------------------------:|--------------------------------------:|
|  0-d 标量 |                  6.14 µs |                                771 ns |
|     1,000 |                  7.77 µs |                               2.27 µs |
| 1,000,000 |                  1.32 ms |                                1.3 ms |

**另请参阅**

- [`Money.allocate`](#moneyallocate) — 将剩余零头按最大余数法直接摊入各份。
- [`FixedArray` 整除与取余](fixed_array.md#a--ba--b-与-divmoda-b整除与取余)

### Money.allocate

```python
Money.allocate(ratios)
```

使用**最大余数法（Hare-Niemeyer / Hamilton 方法）**按权重比例对标量金额进行守恒分配，保证分配出的各份金额之和 **100%
严格等于原金额**，且每一份与理论精确值的误差严格小于 1 个最小原子单位。

**参数**

- **ratios** : *Sequence 或 1-d array_like（int, float, Decimal, Fraction, FixedArray）* — 一维非负分配权重序列。若传入空序列
  `[]` 则返回 `[]`；非空时权重总和不能为 0。不支持字符串或无限循环小数 `Fraction(1, 3)`。

**返回值**

- **out** : *list of Money* — 长度与 `ratios` 相同的同币种 0 维 `Money` 列表，满足
  `sum(out, Money.zeros((), self.currency)) == self`。

**异常**

- **ValueError** — `self.ndim != 0`（仅支持标量金额）；或 `ratios` 不是 1 维；或含有负数权重；或权重总和为 0；或含有非有限小数的
  `Fraction`。
- **TypeError** — `ratios` 含字符串、`bool` 等非数值元素。

**机制说明**

1. 先调用 [`self._as_factor(ratios, "分配权重")`](unit_array.md#unitarray_as_factor) 将整组权重统一对齐到共同 `scale` 的
   `FixedArray`
   ，取其整数尾数列表 `weights = [int(w) for w in weights_fa.mantissa]`（由于所有权重同乘 $10^{\text{scale}}$
   不改变相对比例，浮点数、小数 `Decimal` 与有限 `Fraction` 都会被无损化为精确整数比！）。
2. 设待分配的原子单位总数为 $T = \text{int (self.mantissa)}$，总权重为 $W = \sum w_i$。对每个权重计算向下整除商与正余数：
   $$q_i, r_i = \text{divmod} (T \cdot w_i, W), \qquad 0 \le r_i < W$$
3. 剩余待分配的最小原子单位个数为 $R = T - \sum q_i$（必有 $0 \le R < \text{len} (weights)$）。按余数 $r_i$
   **从大到小稳定排序**（余数相同时保持原下标先后顺序，排在前面的优先），为前 $R$ 份各加 `1` 个最小原子单位。
4. 对于负金额（$T < 0$），由于 Python `divmod` 向负无穷取整（例如 `-100` 均分 3
   份时 $q_i = -34, r_i = 2$，$\sum q_i = -102$，$R = -100 - (-102) = 2$），前两份各加 `1` 变为 `-33`，最后一份保持 `-34`，得到
   `[-0.33, -0.33, -0.34]`，同样严格守恒。

**示例**

```python
>>> from decimal import Decimal
>>> Money("100.00", "USD").allocate([1, 1, 1])
[Money(FixedArray(33.34, scale=2), currency='USD'), Money(FixedArray(33.33, scale=2), currency='USD'), Money(FixedArray(33.33, scale=2), currency='USD')]
>>> Money("1.00", "USD").allocate([Decimal("0.1"), Decimal("0.2")])
[Money(FixedArray(0.33, scale=2), currency='USD'), Money(FixedArray(0.67, scale=2), currency='USD')]
>>> Money("-1.00", "USD").allocate([1, 1, 1])
[Money(FixedArray(-0.33, scale=2), currency='USD'), Money(FixedArray(-0.33, scale=2), currency='USD'), Money(FixedArray(-0.34, scale=2), currency='USD')]
>>> Money("1.00", "USD").allocate([1, 0, 1])  # 权重为 0 的项严格分得 0.00
[Money(FixedArray(0.50, scale=2), currency='USD'), Money(FixedArray(0.00, scale=2), currency='USD'), Money(FixedArray(0.50, scale=2), currency='USD')]
```

**另请参阅**

- [`divmod(m, other)`](#divmodm-other守恒带余整除)

### Money.calc_tiered_fee

```python
Money.calc_tiered_fee(rate, min_fee=None, max_fee=None, rounding="half_up")
```

计算基于金额的交易佣金、印花税或过户费：支持 **单一比例费率**或 **超额累进分档费率表**，并支持逐元素生效的同币种最低收费（
`min_fee`）与封顶收费（`max_fee`）。

**参数**

- **rate** : *数值 array_like，或 Sequence[Tuple[Optional[Money | 数值], 数值]]*
    - **单一费率模式**：传入单个数值（如 `Decimal("0.0003")`）或可与当前金额张量广播的数值数组；
    - **分档累进模式**：传入形如 `[(上限阈值_1, 费率_1), (上限阈值_2, 费率_2), ..., (None, 尾档费率)]` 的二元组序列。每个非空阈值表示该档的
      **累计金额上限**（可传同币种 0 维 `Money` 或主单位数值），必须为正数且严格递增； **只有且必须最后一档的阈值为 `None`**
      （表示超出上一档上限的部分不封顶）。
- **min_fee** : *Money，可选* — 同币种的最低收费下限（可与金额广播）。
- **max_fee** : *Money，可选* — 同币种的最高收费上限（可与金额广播）。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 费率乘积的舍入模式。

**返回值**

- **out** : *Money* — 同币种的费用张量。

**异常**

- **TypeError** — 费率或阈值不是合法数值（如传入了字符串），或阈值/`min_fee`/`max_fee` 的币种与当前金额不一致。
- **ValueError** — 分档阈值不是标量、非正数、未严格递增、非最后一档使用了 `None`、最后一档未使用 `None`；或存在元素使得
  `min_fee > max_fee`。

**机制说明**

1. **分档累进只舍入一次**：在分档模式下，设各档区间为 $[L_i, U_i)$（其中 $L_0 = 0$，最后一档 $U_k = +\infty$
   ）。每个元素落入第 $i$ 档的应税金额为 $\max (\min (m, U_i), L_i) - L_i$（因此 **小于等于 0 的负金额或零金额在分档模式下的应税额恒为
   0**）。各档应税金额乘以对应费率后，先在未舍入的精确 [`FixedArray`](fixed_array.md) 上累加 `exact = exact + part`
   ，待全部档位汇总完毕后，才调用 `exact.rescale(digits, mode)` **一次性舍入**到币种精度，杜绝逐档舍入造成的分位误差。
2. **单一费率模式**：直接计算 `self._data.mul_round(rate_factor, scale=digits, rounding=mode)`。
3. **最低与封顶钳制**：舍入完成后，在原子单位尾数上依次执行 `np.maximum(fee_m, min_fee.mantissa)` 与
   `np.minimum(fee_m, max_fee.mantissa)`（支持标量或与金额同形状的逐行上下限）。

**示例**

单一费率与逐元素最低/封顶收费：

```python
>>> from decimal import Decimal
>>> m = Money(["10.00", "1000.00", "100000.00"], "USD")
>>> m.calc_tiered_fee(Decimal("0.001"), min_fee=Money("5.00", "USD"), max_fee=Money("50.00", "USD"))
Money(FixedArray([5.00, 5.00, 50.00], scale=2), currency='USD')
```

超额累进分档费率（0~1000 收 1%，1000~5000 收 0.5%，5000 以上收 0.1%）：

```python
>>> tiers = [
...     (Decimal("1000"), Decimal("0.01")),
...     (Decimal("5000"), Decimal("0.005")),
...     (None, Decimal("0.001")),
... ]
>>> Money(["500.00", "6000.00", "-200.00"], "USD").calc_tiered_fee(tiers)
Money(FixedArray([5.00, 31.00, 2.00], scale=2), currency='USD')
```

各档费用先精确累加再一次性舍入（`0.004 + 0.004 = 0.008 -> 0.01`）：

```python
>>> small_tiers = [(Decimal("0.40"), Decimal("0.01")), (None, Decimal("0.01"))]
>>> Money("0.80", "USD").calc_tiered_fee(small_tiers)
Money(FixedArray(0.01, scale=2), currency='USD')
```

**性能对比**

|         N | 单一费率 (`Money`) | 单一费率 (`NumPy float64`) | 三档累进 (`Money`) | 三档累进 (`NumPy float64`) |
|----------:|-------------------:|---------------------------:|-------------------:|---------------------------:|
|  0-d 标量 |            24.3 µs |                    2.47 µs |            79.2 µs |                     7.4 µs |
|     1,000 |            34.2 µs |                    2.85 µs |             108 µs |                    12.7 µs |
| 1,000,000 |            7.98 ms |                     855 µs |            18.8 ms |                    3.72 ms |

## 比较、索引、导出与格式化

### 比较、索引与形状变换

`Money` 直接继承 [`UnitArray`](unit_array.md) 的量纲安检比较、切片索引、形状变换与容器协议：

- **相等比较（[`m1 == m2`、`m1 != m2`](unit_array.md#a--b-与-a--b相等性比较)）**：同币种 `Money` 逐元素判等；与不同币种
  `Money`、[`Price`](price.md)、[`Quantity`](quantity.md) 或无量纲数字（如 `0`）比较时 **不抛异常**，直接返回 `False` /
  `True`。
- **大小比较（[`m1 < m2`、`m1 <= m2`、`m1 > m2`、`m1 >= m2`](unit_array.md#a--ba--ba--ba--b大小比较)）**：同币种 `Money`
  逐元素比大小；与不同币种 `Money`、其他 `UnitArray` 或无量纲数字比大小时 **立即抛出 `TypeError`**。
- **索引与变形（[`m[key]`](unit_array.md#akey索引与切片)、[`reshape`](unit_array.md#unitarrayreshape)、[
  `transpose`](unit_array.md#unitarraytranspose)、[`len`](unit_array.md#lenaboola-与-itera容器与真值协议)、[
  `bool`](unit_array.md#lenaboola-与-itera容器与真值协议)、[`iter`](unit_array.md#lenaboola-与-itera容器与真值协议)）**
  ：所有切片、单元素索引与迭代产出的元素均保持原 `Money` 类型与原币种。

**示例**

```python
>>> m = Money(["10.00", "25.00", "5.00"], "USD")
>>> m[m > Money("8.00", "USD")]
Money(FixedArray([10.00, 25.00], scale=2), currency='USD')
>>> m == Money(["10.00", "25.00", "5.00"], "CNY")
False
>>> m < Money("50.00", "CNY")
Traceback (most recent call last):
    ...
TypeError: 币种不匹配: USD 与 CNY
```

### Money.to_pandas

```python
Money.to_pandas(index=None, name=None, as_float=True)
```

将 1 维或 2 维 `Money` 导出为 `pandas.Series` 或 `pandas.DataFrame`，并在返回对象的 `.attrs["currency"]` 中自动写入大写币种代码（如
`"USD"`）。

**参数**

- **index** : *array-like 或 Index，可选* — 行索引。
- **name** : *str，可选* — 1 维 `Series` 的名称。
- **as_float** : *bool，默认 `True`* — 为 `True` 时导出为 `float64`，为 `False` 时导出为无损 `Decimal` 对象列。

**返回值**

- **out** : *pandas.Series 或 pandas.DataFrame*（携带 `out.attrs["currency"] == self.currency.code`）。

**机制说明**

在基类 [`UnitArray.to_pandas`](unit_array.md#unitarrayto_pandas) 的基础上，重写注入了
`res.attrs["currency"] = self.currency.code`，使下游 Pandas 分析流程能够读取币种元数据。导出的 `Series` 也可直接传回
`Money(s, "USD")` 重建 `Money` 张量。

**示例**

```python
>>> s = Money(["1.50", "2.00"], "USD").to_pandas(index=["a", "b"], name="amt")
>>> s.to_dict(), s.attrs["currency"]
({'a': 1.5, 'b': 2.0}, 'USD')
>>> Money(s, "USD")  # 直接从 pandas.Series 构造 Money
Money(FixedArray([1.50, 2.00], scale=2), currency='USD')
```

**另请参阅**

- [`Money(values, currency)`](#moneyvalues-currency)
- [`UnitArray.to_pandas`](unit_array.md#unitarrayto_pandas)

### Money.to_polars、Money.to_float 与 Money.to_decimal

```python
Money.to_polars(name="value", as_float=True)
Money.to_float()
Money.to_decimal()
```

直接继承自 [`UnitArray`](unit_array.md)，剥离币种标签后导出为 `polars.Series`、`float` / `float64` 数组或无损 `Decimal` /
`object` 数组。

**示例**

```python
>>> m = Money("1.50", "USD")
>>> m.to_float(), m.to_decimal()
(1.5, Decimal('1.50'))
```

**另请参阅**

- [`UnitArray.to_float`](unit_array.md#unitarrayto_float)、[`UnitArray.to_decimal`](unit_array.md#unitarrayto_decimal)、[
  `UnitArray.to_polars`](unit_array.md#unitarrayto_polars)

### Money.format

```python
Money.format(locale="en_US", accounting=False)
```

调用 `MonetaryFormatter.format` 将金额格式化为带货币符号、千分位分隔符或会计负数括号的展示字符串。

**参数**

- **locale** : *str，默认 `"en_US"`* — 区域格式约定。
- **accounting** : *bool，默认 `False`* — 为 `True` 时负数采用会计括号表示（如 `($1,234.50)`）。

**返回值**

- **out** : *str*

**示例**

```python
>>> "1,234.50" in Money("1234.50", "USD").format()
True
```

**另请参阅**

- [`str(m)` 与 `repr(m)`](#strm-与-reprm文本表示)

### `str(m)` 与 `repr(m)`（文本表示）

```python
str(m)
repr(m)
```

`str(m)` 返回 `"{数值} {币种代码}"`；`repr(m)` 返回包含底层 `FixedArray` 与 `currency` 代码的完整构造表达式。

**返回值**

- **out** : *str*

**示例**

```python
>>> m = Money("1.5", "USD")
>>> str(m)
'1.50 USD'
>>> repr(m)
"Money(FixedArray(1.50, scale=2), currency='USD')"
```

## 内部钩子与辅助方法（私有 API）

### Money._create_like

```python
Money._create_like(new_data)
```

实现 [`UnitArray._create_like`](unit_array.md#unitarray_create_like) 抽象钩子：用新的底层 `FixedArray` 构建同币种的
`Money` 实例。

**参数**

- **new_data** : *FixedArray* — 底层定点数张量，其 `scale` 必须恰好等于 `self._currency.digits`。

**返回值**

- **out** : *Money*

**异常**

- **ValueError** — `new_data.scale != self._currency.digits`（由 [`_wrap`](#money_wrap) 抛出内部不变式错误）。

**机制说明**

直接转发调用 `self._wrap(new_data, self._currency)`。

**示例**

```python
>>> m = Money("10.00", "USD")
>>> m._create_like(FixedArray("25.50"))
Money(FixedArray(25.50, scale=2), currency='USD')
>>> m._create_like(FixedArray("25.5"))
Traceback (most recent call last):
    ...
ValueError: 内部错误：金额精度 1 与 USD 的精度 2 不一致
```

**另请参阅**

- [`Money._wrap`](#money_wrap)
- [`UnitArray._create_like`](unit_array.md#unitarray_create_like)

### Money._check_compatible

```python
Money._check_compatible(other)
```

实现 [`UnitArray._check_compatible`](unit_array.md#unitarray_check_compatible) 抽象钩子：校验 `other` 是否为相同币种的
`Money`。

**参数**

- **other** : *Any*

**异常**

- **TypeError** — `other` 不是 `Money` 实例，或 `other._currency != self._currency`。

**示例**

```python
>> > m = Money("1.00", "USD")
>> > m._check_compatible(Money("2.00", "USD"))
>> > m._check_compatible(Money("2.00", "CNY"))
Traceback(most
recent
call
last):
...
TypeError: 币种不匹配: USD
与
CNY
```

**另请参阅**

- [`Money._check_compatible_currency`](#money_check_compatible_currency)
- [`UnitArray._check_compatible`](unit_array.md#unitarray_check_compatible)

### Money._check_compatible_currency

```python
Money._check_compatible_currency(currency)
```

校验当前 `Money` 实例的币种是否与目标 [`Currency`](currency.md) 对象相同。

**参数**

- **currency** : *Currency* — 目标币种对象。

**异常**

- **TypeError** — `self._currency != currency`。

**机制说明**

供 `Money(existing_money, currency)` 构造路径调用，防止在传入已有 `Money` 实例时静默更换币种。

**示例**

```python
>>> from moneta import Money, USD, CNY
>>> Money("1.00", USD)._check_compatible_currency(USD)
>>> Money("1.00", USD)._check_compatible_currency(CNY)
Traceback (most recent call last):
    ...
TypeError: 币种不匹配: USD 与 CNY
```

**另请参阅**

- [`Money._check_compatible`](#money_check_compatible)

### Money._wrap

```python
@classmethod
Money._wrap(data, currency)
```

内部零拷贝构造器：校验 `data.scale == currency.digits` 后，跳过 `__init__` 直接绑定 `_data` 与 `_currency`。

**参数**

- **data** : *FixedArray* — 已对齐到币种精度的底层定点数张量。
- **currency** : *Currency* — 已解析的 `Currency` 实例。

**返回值**

- **out** : *Money*

**异常**

- **ValueError** — `data.scale != currency.digits`（用于捕获内部方法漏掉舍入或精度对齐的 bug）。

**示例**

```python
>>> from moneta import Money, USD
>>> Money._wrap(FixedArray("12.34"), USD)
Money(FixedArray(12.34, scale=2), currency='USD')
>>> Money._wrap(FixedArray("12.345"), USD)
Traceback (most recent call last):
    ...
ValueError: 内部错误：金额精度 3 与 USD 的精度 2 不一致
```

**另请参阅**

- [`Money._create_like`](#money_create_like)
- [`Money.from_units`](#moneyfrom_units)

### Money._fill_value

```python
Money._fill_value(value)
```

覆盖 [`UnitArray._fill_value`](unit_array.md#unitarray_as_grid-与-unitarray_fill_value) 钩子：先调用 [
`self._as_factor(value, "填充值")`](unit_array.md#unitarray_as_factor) 校验 `value` 在无量纲数值白名单内，再通过
`Money(value, self._currency)._data` 按 `Money` 的构造规则换算到当前币种精度（`float`
静默舍入，精确输入超出币种精度时发出 [`PrecisionLossWarning`](fixed_array.md#precisionlosswarning) 后按 `half_up` 舍入）。

**示例**

```python
>> > m = Money(["1.00", None], "USD")
>> > m._fill_value(2.5)
FixedArray(2.50, scale=2)
>> > m.fillna("2.5")
Traceback (most recent call last):
    ...
TypeError: 填充值只接受数值（int、float、Decimal、Fraction、FixedArray
或数值数组），收到
str
```

**另请参阅**

- [`UnitArray.fillna`](unit_array.md#unitarrayfillna)
- [`Money._as_step`](#money_as_step)

### Money._as_step

```python
Money._as_step(step)
```

将步长参数 `step` 转换为 `FixedArray`，并校验其为正数且有效精度不超过当前币种精度 `self._currency.digits`。

**参数**

- **step** : *Any* — 步长输入。

**返回值**

- **out** : *FixedArray* — `scale <= self._currency.digits` 的正数步长 `FixedArray`。

**异常**

- **TypeError** — `step` 不是数值白名单类型。
- **ValueError** — `step <= 0`，或 `step` 在去除尾随零后的有效小数位数仍超过 `self._currency.digits`。

**机制说明**

先调用 `self._as_factor(step, "步长")`，检查 `np.any(step_fa.mantissa <= 0)`。若 `step_fa.scale > digits`，尝试向零截断
`reduced = step_fa.rescale(digits, "down")`：若 `np.all(reduced == step_fa)`（说明多出的位数全是末尾 0，如
`FixedArray("0.050")`），则采用 `reduced`；否则抛出 `ValueError`。

**示例**

```python
>>> m = Money("1.00", "USD")
>>> m._as_step(FixedArray("0.050"))  # 末尾多余的 0 被自动归约到 scale=2
FixedArray(0.05, scale=2)
>>> m._as_step(0)
Traceback (most recent call last):
    ...
ValueError: 步长 step 必须为正数
```

**另请参阅**

- [`Money.round_to_step`](#moneyround_to_step)
- [`Money._fused_round_step`](#money_fused_round_step)

### Money._fused_round_step

```python
Money._fused_round_step(floor, inexact, step, mode)
```

由精确结果在 `10^-(step.scale + 1)` 网格上的向下取整值 `floor` 以及非精确布尔掩码 `inexact`，一次性将真实值 $v$ 按舍入模式
`mode` 对齐到步长 `step` 网格。

**参数**

- **floor** : *FixedArray* — 真实结果 $v$ 在精度 `t = step.scale + 1` 下按 `"floor"`
  舍入得到的下界（满足 $\text{floor} \le v < \text{floor} + 10^{-t}$）。
- **inexact** : *bool 或 ndarray (bool)* — 标记 $v$ 是否严格大于 `floor`（即 `floor != ceil`）。
- **step** : *FixedArray* — 已通过 [`_as_step`](#money_as_step) 校验的正数步长。
- **mode** : *str* — 已规范化的舍入模式名称。

**返回值**

- **out** : *FixedArray* — 已对齐到 `step` 网格且 `scale == self._currency.digits` 的 `FixedArray`。

**机制说明**

为什么只需 `t = step.scale + 1` 的下界 `floor` 与 1 比特 `inexact` 就能对任意实数商/乘积做 **精确的一次步长舍入**？

- 设 `unit = step.mantissa * 10`（即以 $10^{-t}$ 为单位表示的步长）。因为 `unit` 是 10
  的倍数（必为偶数），所有步长格点 $k \cdot \text{step}$ 都是 `unit`
  的整数倍，且所有半步长中点 $(k + 0.5) \cdot \text{step}$ 也都严格落在 `unit / 2`（整数个 $10^{-t}$）上！
- 因此，在相邻两个 $10^{-t}$ 格点 $(\text{floor}, \text{floor} + 10^{-t})$ 的开区间内部，
  **绝不可能包含任何步长格点或半步长边界**。
- 对 `floor.mantissa` 除以 `unit` 得商 `q` 与余数 `r`（$0 \le r < \text{unit}$）：
    - $v$ 存在非零小数余量当且仅当 `(r != 0) | inexact`；
    - $v$ 严格超过半步长当且仅当 `(r > unit - r) | ((r == unit - r) & inexact)`；
    - $v$ 恰好落在半步长中点当且仅当 `(r == unit - r) & ~inexact`。
- 将上述布尔条件送入 [`decide_bump(..., floor_base=True)`](_rounding.md#decide_bump)，即可一步求得精确步数
  `q + bump`，再乘以 `step` 并 `rescale` 到币种精度。

**示例**

```python
>> > m = Money("1.00", "USD")
>> > step = m._as_step(FixedArray("0.10"))  # step.scale == 2, t = 3
>> > m._fused_round_step(FixedArray("0.049"), np.array(True), step, "half_up")  # v = 0.0499 > 0.049 (inexact=True)
FixedArray(0.00, scale=2)
>> > m._fused_round_step(FixedArray("0.050"), np.array(False), step, "half_up")  # v = 0.0500 == 0.050 (inexact=False)
FixedArray(0.10, scale=2)
```

**另请参阅**

- [`Money.multiply`](#moneymultiply)
- [`Money.divide`](#moneydivide)
- [`decide_bump`](_rounding.md#decide_bump)

### _RATIO_EXTRA_DIGITS

```python
moneta.money._RATIO_EXTRA_DIGITS = 8
```

模块级整数常量（值为 `8`）：当 [`Money.divide`](#moneydivide) 或 [`m / other`](#m--other真除法运算符) 的结果不是 `Money`（即
`Money / Money -> FixedArray`、`Money / Price -> Quantity`、`Money / Quantity -> Price`）且未显式传入 `scale` 时，默认保留的小数位数为
`self.currency.digits + _RATIO_EXTRA_DIGITS`（例如 `USD` 默认保留 `2 + 8 = 10` 位，`JPY` 默认保留 `0 + 8 = 8` 位）。

**示例**

```python
>>> from moneta.money import _RATIO_EXTRA_DIGITS
>>> _RATIO_EXTRA_DIGITS
8
```

**另请参阅**

- [`Money.divide`](#moneydivide)
