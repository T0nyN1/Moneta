# tvm — 货币时间价值（现值、终值、净现值与贴现因子）

`moneta.tvm` 模块提供货币时间价值（Time Value of Money, TVM）核心计算函数：现值（[`pv`](#pv)）、终值（[`fv`](#fv)）、净现值（[`npv`](#npv)）与无量纲贴现因子（[`discount_factors`](#discount_factors)）。

```python
>>> from moneta import CashFlow, Money, discount_factors, fv, npv, pv
>>> pv(Money("110.00", "USD"), 0.10, 1), fv(Money("100.00", "USD"), 0.10, 1)
(Money(FixedArray(100.00, scale=2), currency='USD'), Money(FixedArray(110.00, scale=2), currency='USD'))
>>> pv(Money("1000000000.00", "USD"), 0.07, 30)  # 18 位有效数字因子，大额计算精确到分
Money(FixedArray(131367117.15, scale=2), currency='USD')
>>> npv(0.10, [Money("-1000", "USD"), Money("550", "USD"), Money("605", "USD")])
Money(FixedArray(0.00, scale=2), currency='USD')
>>> dated = [CashFlow(Money("-1000", "USD"), "2026-01-01"), CashFlow(Money("1100", "USD"), "2026-07-01")]
>>> npv(0.10, dated, "2026-01-01", convention="ACT/365")
Money(FixedArray(49.22, scale=2), currency='USD')
>>> discount_factors(0.05, [0, 1, 2])
array([1.        , 0.95238095, 0.90702948])
```

[← API 参考](index.md)

## 核心机制与设计约束

### 1. 三种复利公式及利率 $r$、期限 $t$ 的取值要求

所有 TVM 函数均通过关键字参数 `compounding` 支持三种标准复利/计息方式（默认 `"annual"`），先在 `float64` 中计算增长因子 $g(r, t)$（贴现因子即为 $1 / g(r, t)$）：

| `compounding` 取值 | 计息方式 | 增长因子 $g(r, t)$ 公式 | 对利率 $r$ 与期限 $t$ 的定义域要求 |
|---|---|---|---|
| **`"annual"`**（默认） | 按期复利（Periodic / Annual Compounding） | $g = (1 + r)^t$ | 要求 **$r > -1$**（若存在 $r \le -1$ 抛 `ValueError("annual 复利要求 rate > -1")`）；$t$ 可为任意有限实数（含小数或负数） |
| **`"continuous"`** | 连续复利（Continuous Compounding） | $g = e^{r \cdot t}$ | $r$ 与 $t$ 可为任意有限实数 |
| **`"simple"`** | 单利（Simple Interest） | $g = 1 + r \cdot t$ | 要求 **$1 + r \cdot t > 0$**（若存在 $1 + r \cdot t \le 0$ 抛 `ValueError("simple 计息要求 1 + rate × periods > 0")`） |

此外，计算出的增长因子 $g$ 必须满足 **$0 < g < +\infty$**；若因利率或期数过大导致浮点数上溢为 `inf` 或下溢为 `0.0`，[`_growth`](#_real_array_check_compounding_growth-与-_factor) 会立即抛出 `ValueError("增长因子超出浮点数范围（利率或期数过大）")`。

### 2. 利率与期数接受的输入形式及广播规则

- **输入形式（[`_real_array`](#_real_array_check_compounding_growth-与-_factor)）**：`rate`/`rates` 与 `periods`/`times` 参数统一支持：
  1. Python 实数标量（`int`、`float`、`Decimal`、`Fraction`）或 NumPy 数值标量（如 `np.float64`、`np.int64`）；
  2. 单个数值字符串（如 `"0.05"`、`" 0.05 "`；无法解析为有限浮点数时抛 `ValueError`）；
  3. 不含缺失值 `<NA>` 的 [`FixedArray`](fixed_array.md)（含缺失值时抛 `ValueError`）；
  4. 元素类型为整数或浮点数的序列或 NumPy 数组（`arr.dtype.kind in "iuf"`）。
  - **拒绝的类型**：布尔值（`bool`、`np.bool_`）以及元素为字符串、`Fraction`、`Decimal` 或 `None` 的序列（如 `["0.1"]`、`[Fraction(1, 10)]`）均会抛出 `TypeError`；包含 `NaN` 或 `±inf` 的输入会抛出 `ValueError("... 必须是有限数")`。
- **NumPy 广播对齐规则**：
  - 在 [`pv(future_value, rate, periods)`](#pv) 与 [`fv(present_value, rate, periods)`](#fv) 中，金额张量、`rate` 与 `periods` 三者按标准 NumPy 广播规则对齐（例如标量金额配一维期数数组，或 `(2, 1)` 金额矩阵配 `(2,)` 期数向量）；若输入金额含有 `<NA>` 缺失值，对应输出位置保持为 `<NA>`；
  - 在 [`npv(rate, cash_flows, ...)`](#npv) 中，`rate` 可为标量或实数数组，与各笔现金流的 `Money` 张量按广播规则对齐；
  - 在 [`discount_factors(rates, times)`](#discount_factors) 中，`rates` 与 `times` 按广播规则对齐：当**两个参数都是标量**时返回 0 维标量 `np.float64`，当任一参数为数组/序列时返回广播形状的 `np.ndarray`（`dtype=float64`）。

### 3. 18 位有效数字因子与单次舍入机制

在将浮点增长因子 $g$（或贴现因子 $1/g$）作用到定点数金额 [`Money`](money.md) 上时，若简单将因子固定截断到 8 位小数，对大额本金（例如 10 亿美元贴现 30 年）会产生数美元的截断误差。

`moneta.tvm` 通过 [`_factor(f)`](#_real_array_check_compounding_growth-与-_factor) 动态提取因子数组中的最大绝对值 $\text{peak} = \max |f_i|$：

- 若 $\text{peak} \ge 10^{18}$，抛出 `OverflowError`；
- 否则取小数位数：
  $$\text{scale} = \max\bigl(0,\ 18 - (\lfloor \log_{10}(\text{peak}) \rfloor + 1)\bigr)$$
  将因子转为保留 **18 位有效数字**的 [`FixedArray`](fixed_array.md)（其尾数严格在 `int64` 范围内，且同一数组内较小因子的绝对误差不超过最大因子的最低有效位），再与金额执行精确乘法并按 `half_up` **只舍入一次**到币种精度 `currency.digits`。

### 4. `npv` 的两种用法、保护位累加与金额上限

[`npv(rate, cash_flows, as_of_date=None, *, convention=DayCountConvention.ACT_365, compounding="annual")`](#npv) 根据是否传入估值日 `as_of_date` 严格区分两种模式：

| 模式 | `as_of_date` 参数 | `cash_flows` 元素类型要求 | 第 $i$ 笔现金流的贴现期数/年数 $t_i$ |
|---|---|---|---|
| **按期等距贴现** | `None`（省略） | 必须全部是 [`Money`](money.md)（若混入 `CashFlow` 抛 `ValueError`） | 第 $i$ 笔（从 $0$ 开始）贴现 $t_i = i$ 期（第 $0$ 笔不贴现） |
| **按日历日期贴现** | 传入 `date`、`datetime` 或 ISO 字符串 | 必须全部是 [`CashFlow`](cashflow.md)（若混入 `Money` 抛 `ValueError`） | $t_i = \text{year\_fraction}(\text{as\_of\_date},\ \text{cf}_i.\text{pay\_date},\ \text{convention})$（早于 `as_of_date` 的现金流 $t_i < 0$，自动向后复利至估值日） |

- **只舍入一次（`digits + 4` 保护位累加）**：若对每笔现金流先舍入到 `currency.digits` 再相加，多笔微小尾差会同向累积（例如 1,000 笔 `$0.01` 在 `rate=0.5` 下逐笔 `pv` 求和得 `$0.02`，而真实和为 `$0.03`）。`npv` 先将每笔现金流现值按 `guard = currency.digits + 4` 位小数（`_NPV_GUARD_DIGITS = 4`）计算并累加，最后调用 `total.rescale(currency.digits)` 按 `half_up` **只舍入一次**。
- **`npv` 的金额上限**：由于各笔现值在 `scale = currency.digits + 4` 的 `FixedArray`（`int64` 尾数，上限 $2^{63} - 1 \approx 9.22 \times 10^{18}$）中累加，单笔现值及累加过程中的部分和绝对值不得超过：
  $$V_{\max} = \frac{2^{63} - 1}{10^{\text{digits} + 4}} \approx 9.22 \times 10^{14 - \text{digits}}$$
  对于 2 位小数的币种（如 `USD`、`CNY`、`EUR`，`guard = 6`），`npv` 支持的现值金额上限约为 **$9.22 \times 10^{12}$（约 9.22 万亿）**；对于 0 位小数的 `JPY`（`guard = 4`）约为 $9.22 \times 10^{14}$。超出该范围时会抛出 `OverflowError`。

## API 速查表

| 分类 | 接口 / 常量 | 功能说明 |
|---|---|---|
| **货币时间价值函数** | [`pv`](#pv) | 计算未来金额 `future_value` 贴现 `periods` 期的现值（`Money`） |
| | [`fv`](#fv) | 计算当前金额 `present_value` 复利 `periods` 期的终值（`Money`） |
| | [`npv`](#npv) | 计算按期 `Money` 序列或带日期 [`CashFlow`](cashflow.md) 序列的净现值（`Money`） |
| | [`discount_factors`](#discount_factors) | 计算给定零息利率与期限的无量纲贴现因子（`np.float64` 或 `np.ndarray`） |
| **内部辅助（私有 API）** | [`_COMPOUNDINGS` 与 `_NPV_GUARD_DIGITS`](#_compoundings-与-_npv_guard_digits) | 支持的复利方式元组与 `npv` 中间累加保护位数常量 |
| | [`_real_array`、`_check_compounding`、`_growth` 与 `_factor`](#_real_array_check_compounding_growth-与-_factor) | 实数数组转换、复利方式校验、浮点增长因子计算与 18 位有效数字定点因子转换 |

## 现值、终值、净现值与贴现因子

### `pv`

```python
moneta.pv(future_value, rate, periods, *, compounding="annual")
```

计算未来金额 `future_value` 贴现 `periods` 期后的现值（$\text{future\_value} / g(r, t)$，按 `half_up` 单次舍入到币种精度；同时由顶层 `moneta` 与 `moneta.tvm` 导出）。

**参数**

- **future_value** : *Money* — 未来金额（标量或多维数组，支持缺失值 `<NA>`）。
- **rate** : *实数, 数值字符串, FixedArray 或 实数数组* — 每期利率（如 `0.05` 表示 5%），不能含缺失值或非有限数。
- **periods** : *实数, 数值字符串, FixedArray 或 实数数组* — 贴现期数（可为小数或负数；负期数等价于向后复利），与 `future_value`、`rate` 按广播规则对齐。
- **compounding** : *`{"annual", "continuous", "simple"}`，仅关键字参数，默认 `"annual"`* — 复利方式。

**返回值**

- **out** : *Money* — 与 `future_value` 同币种、形状为 `(future_value, rate, periods)` 广播形状的现值金额。

**异常**

- **TypeError** — `future_value` 不是 [`Money`](money.md)；或 `rate`、`periods` 传入了布尔值、非数值类型或元素非实数的序列。
- **ValueError** — `rate` 或 `periods` 含 `NaN`/`inf`、含 `<NA>` 缺失值或无法解析的字符串；`compounding` 不在 `("annual", "continuous", "simple")` 中；`compounding="annual"` 时存在 `rate <= -1`；`compounding="simple"` 时存在 `1 + rate * periods <= 0`；增长因子超出 `float64` 正数范围；或广播形状不兼容。
- **OverflowError** — 贴现因子 `1 / g >= 1e18`，或贴现后的金额超出 [`Money`](money.md) 的 `int64` 尾数范围。

**示例**

```python
>>> from moneta import Money, pv
>>> pv(Money("110.00", "USD"), 0.10, 1)
Money(FixedArray(100.00, scale=2), currency='USD')
>>> pv(Money("110.52", "USD"), 0.05, 2, compounding="continuous")
Money(FixedArray(100.00, scale=2), currency='USD')
>>> pv(Money("100.00", "USD"), 0.10, [0, 1, 2])  # 期数向量广播
Money(FixedArray([100.00, 90.91, 82.64], scale=2), currency='USD')
>>> pv(Money("100.00", "USD"), -1.0, 1)
Traceback (most recent call last):
    ...
ValueError: annual 复利要求 rate > -1
```

**另请参阅**

- [`fv`](#fv)
- [`npv`](#npv)
- [`discount_factors`](#discount_factors)

### `fv`

```python
moneta.fv(present_value, rate, periods, *, compounding="annual")
```

计算当前金额 `present_value` 经过 `periods` 期复利后的终值（$\text{present\_value} \times g(r, t)$，按 `half_up` 单次舍入到币种精度；同时由顶层 `moneta` 与 `moneta.tvm` 导出）。

**参数**

- **present_value** : *Money* — 当前金额（标量或多维数组，支持缺失值 `<NA>`）。
- **rate, periods, compounding** — 与 [`pv`](#pv) 完全相同。

**返回值**

- **out** : *Money* — 与 `present_value` 同币种、形状为 `(present_value, rate, periods)` 广播形状的终值金额。

**异常**

- **TypeError** — `present_value` 不是 [`Money`](money.md)；或 `rate`、`periods` 类型非法（同 [`pv`](#pv)）。
- **ValueError** — 利率、期数、复利方式或定义域校验失败（同 [`pv`](#pv)）。
- **OverflowError** — 增长因子 `g >= 1e18`，或终值金额超出 `int64` 尾数范围。

**示例**

```python
>>> from moneta import Money, fv
>>> fv(Money("100.00", "USD"), 0.10, 2)
Money(FixedArray(121.00, scale=2), currency='USD')
>>> fv(Money("100.00", "USD"), 0.10, 2.5, compounding="simple")
Money(FixedArray(125.00, scale=2), currency='USD')
>>> fv(Money(["100.00", "200.00"], "USD"), [0.10, 0.20], 1)
Money(FixedArray([110.00, 240.00], scale=2), currency='USD')
>>> fv(Money("1.00", "USD"), 1.0, 70)  # 2^70 > 1e18 超出 18 位定点因子范围
Traceback (most recent call last):
    ...
OverflowError: 增长因子 1.18059e+21 超出定点数范围（利率或期数过大）
```

**另请参阅**

- [`pv`](#pv)

### `npv`

```python
moneta.npv(
    rate,
    cash_flows,
    as_of_date=None,
    *,
    convention=DayCountConvention.ACT_365,
    compounding="annual",
)
```

计算现金流序列的净现值（各笔现金流现值在 `currency.digits + 4` 精度下累加，最后按 `half_up` **只舍入一次**到币种精度；同时由顶层 `moneta` 与 `moneta.tvm` 导出）。

**参数**

- **rate** : *实数, 数值字符串, FixedArray 或 实数数组* — 每期利率（当给出 `as_of_date` 时为年化利率）。
- **cash_flows** : *Iterable[Union[Money, CashFlow]]* — 非空的同币种现金流可迭代对象（支持列表、元组、生成器等，但不能直接传入单个 `Money`、`CashFlow` 或 `str`）：
  - **省略 `as_of_date`（`as_of_date=None`）**：元素必须全部是 [`Money`](money.md)，第 $i$ 笔（$i = 0, 1, \dots$）按 $i$ 期贴现；
  - **给出 `as_of_date`**：元素必须全部是 [`CashFlow`](cashflow.md)，每笔按 [`year_fraction(as_of_date, cf.pay_date, convention)`](conventions.md#year_fraction) 计算的年数贴现。
- **as_of_date** : *date, datetime, str 或 None，默认 `None`* — 估值基准日（给出时通过 [`to_date(as_of_date, "as_of_date")`](_utils.md#to_date) 解析）。
- **convention** : *DayCountConvention 或 str，仅关键字参数，默认 `DayCountConvention.ACT_365`* — 计日惯例（始终通过 [`_parse_convention`](conventions.md#_convention_keys-与-_parse_convention) 校验；在给出 `as_of_date` 时用于计算年化跨度）。
- **compounding** : *`{"annual", "continuous", "simple"}`，仅关键字参数，默认 `"annual"`* — 复利方式。

**返回值**

- **out** : *Money* — 净现值金额（币种与现金流一致；当 `rate` 或现金流金额为数组时返回广播后的 `Money` 张量）。

**异常**

- **TypeError** —
  - `cash_flows` 不是可迭代序列（或直接传入了单个 `Money`、`CashFlow`、`str`）；
  - `as_of_date=None` 时 `cash_flows` 含有既非 `Money` 也非 `CashFlow` 的元素，或给出 `as_of_date` 时含有既非 `CashFlow` 也非 `Money` 的元素；
  - 各笔现金流的币种不一致（报 `f"币种不匹配: {currency.code} 与 {amount.currency.code}"`）；
  - `rate`、`as_of_date` 或 `convention` 类型非法。
- **ValueError** —
  - `cash_flows` 为空序列；
  - `as_of_date=None` 时 `cash_flows` 中含有 `CashFlow`（报 `"cash_flows 含 CashFlow 时必须给出 as_of_date；不带日期的现金流请用 Money"`）；
  - 给出 `as_of_date` 时 `cash_flows` 中含有 `Money`（报 `"给出 as_of_date 时 cash_flows 元素必须是带日期的 CashFlow"`）；
  - `rate`、`as_of_date`、`convention` 或 `compounding` 取值非法或超出定义域。
- **OverflowError** — 贴现因子超出 $10^{18}$，或单笔现值/累加现值超出 `scale = currency.digits + 4` 下的 `int64` 范围（对 2 位小数币种约为 $9.22 \times 10^{12}$）。

**示例**

等距 `Money` 序列、带日期 `CashFlow` 序列、多利率向量化广播与单次舍入精度：

```python
>>> from moneta import CashFlow, Money, generate_cash_flow_schedule, npv
>>> # 1. 不带日期的按期现金流：第 0、1、2 期
>>> npv(0.10, [Money("-1000", "USD"), Money("550", "USD"), Money("605", "USD")])
Money(FixedArray(0.00, scale=2), currency='USD')
>>> # 2. 带日期的现金流排程贴现（结合 generate_cash_flow_schedule）
>>> schedule = generate_cash_flow_schedule("2026-01-01", "2027-01-01", "semi-annual", principal=Money("1000", "USD"), coupon_rate="0.05")
>>> npv(0.05, schedule, "2026-01-01", convention="ACT/ACT")
Money(FixedArray(1000.59, scale=2), currency='USD')
>>> # 3. 多折现率同时估值（广播）
>>> npv([0.10, 0.00], [Money("-100", "USD"), Money("110", "USD")])
Money(FixedArray([0.00, 10.00], scale=2), currency='USD')
>>> # 4. 保护位累加只舍入一次（逐笔 pv 舍入再相加会因截断得 0.02 USD，精确值约 0.0300 USD）
>>> npv(0.50, [Money("0.01", "USD")] * 1000)
Money(FixedArray(0.03, scale=2), currency='USD')
```

模式混用与币种不一致时的异常拦截：

```python
>>> npv(0.10, [CashFlow(Money("100", "USD"), "2026-01-01")])
Traceback (most recent call last):
    ...
ValueError: cash_flows 含 CashFlow 时必须给出 as_of_date；不带日期的现金流请用 Money
>>> npv(0.10, [Money("100", "USD")], "2026-01-01")
Traceback (most recent call last):
    ...
ValueError: 给出 as_of_date 时 cash_flows 元素必须是带日期的 CashFlow
>>> npv(0.10, [Money("100", "USD"), Money("100", "EUR")])
Traceback (most recent call last):
    ...
TypeError: 币种不匹配: USD 与 EUR
```

**另请参阅**

- [`pv`](#pv)
- [`CashFlow`](cashflow.md#cashflowamount-pay_date-label)、[`generate_cash_flow_schedule`](cashflow.md#generate_cash_flow_schedule)
- [`DayCountConvention`](conventions.md#daycountconvention)、[`year_fraction`](conventions.md#year_fraction)

### `discount_factors`

```python
moneta.discount_factors(rates, times, *, compounding="annual")
```

计算无量纲贴现因子 $d(r, t) = 1 / g(r, t)$（同时由顶层 `moneta` 与 `moneta.tvm` 导出）。

**参数**

- **rates** : *实数, 数值字符串, FixedArray 或 实数数组* — 零息利率。
- **times** : *实数, 数值字符串, FixedArray 或 实数数组* — 期限（期数或年数），与 `rates` 按广播规则对齐。
- **compounding** : *`{"annual", "continuous", "simple"}`，仅关键字参数，默认 `"annual"`* — 复利方式。

**返回值**

- **out** : *np.float64 或 np.ndarray* — `float64` 精度的贴现因子：当 `rates` 与 `times` **都是标量**时返回 0 维标量 `np.float64`；当任一为数组/序列时返回广播后的 `np.ndarray`（`dtype=np.float64`）。

**异常**

- **TypeError** — `rates` 或 `times` 类型不是实数、数值字符串、`FixedArray` 或实数数组。
- **ValueError** — `rates` 或 `times` 非有限或含缺失值；`compounding` 非法；违反复利方式定义域（如 `annual` 下 `rates <= -1`，或 `simple` 下 `1 + rates * times <= 0`）；增长因子超出 `float64` 范围；或广播形状不兼容。

**示例**

```python
>>> import numpy as np
>>> from moneta import discount_factors
>>> d_scalar = discount_factors(0.05, 1)
>>> isinstance(d_scalar, np.float64), round(float(d_scalar), 8)
(True, 0.95238095)
>>> discount_factors([[0.01], [0.02]], [1, 2, 3]).shape  # (2, 1) 与 (3,) 广播为 (2, 3)
(2, 3)
```

**性能对比**

| TVM 计算操作 | 耗时 | 说明 |
|---|---:|---|
| `discount_factors(0.05, 10)`（0-d 标量） | 11.2 µs | 纯 `float64` 贴现因子计算 |
| `discount_factors(rates_1000, 10)`（1,000 元素） | 46.5 µs | 向量化 `float64` 贴现因子数组 |
| `fv(m, 0.05, 10)` / `pv(m, 0.05, 10)`（0-d 标量） | 77.0 µs | 含 18 位有效数字定点因子转换与单次舍入 |
| `pv(m_1000, 0.05, 10)`（1,000 元素） | 156 µs | 1,000 笔金额向量化贴现 |
| `npv(0.05, flows_20)`（20 笔现金流） | 1.48 ms | 20 笔现金流 `digits + 4` 保护位累加与单次舍入 |

## 内部辅助函数与常量（私有 API）

### `_COMPOUNDINGS` 与 `_NPV_GUARD_DIGITS`

```python
moneta.tvm._COMPOUNDINGS = ("annual", "continuous", "simple")
moneta.tvm._NPV_GUARD_DIGITS = 4
```

- **`_COMPOUNDINGS`**：受支持的三种复利方式名称常量元组。
- **`_NPV_GUARD_DIGITS`**：[`npv`](#npv) 在累加各笔现金流现值时在币种精度 `currency.digits` 之上额外保留的小数保护位数（`4` 位）。

### `_real_array`、`_check_compounding`、`_growth` 与 `_factor`

```python
moneta.tvm._real_array(value, name)
moneta.tvm._check_compounding(compounding)
moneta.tvm._growth(r, t, compounding)
moneta.tvm._factor(f)
```

- **`_real_array(value, name)`**：将利率或期数参数转换为 `dtype=np.float64` 的 `np.ndarray`（标量输入转为 0 维数组），校验不含缺失值、非布尔值、数值类型合法且所有元素满足 `np.isfinite`。
- **`_check_compounding(compounding)`**：校验 `compounding` 是属于 `_COMPOUNDINGS` 的字符串，否则抛出 `ValueError`。
- **`_growth(r, t, compounding)`**：在 `np.errstate(over="ignore", under="ignore")` 下计算增长因子数组 `g`，校验 `annual`（`r > -1`）与 `simple`（`1 + r * t > 0`）的定义域，并确认结果满足 `np.all(np.isfinite(g) & (g > 0))`。
- **`_factor(f)`**：按 `f` 中绝对值最大者 `peak = float(np.max(np.abs(f)))` 计算 `scale = max(0, 18 - (math.floor(math.log10(peak)) + 1))`，将浮点因子数组 `f` 转换为保留 18 位有效数字的 [`FixedArray`](fixed_array.md)；当 `peak >= 1e18` 时抛出 `OverflowError`。

**示例**

```python
>>> import numpy as np
>>> from moneta.tvm import _factor, _growth, _real_array
>>> _real_array("0.05", "rate")
array(0.05)
>>> _factor(_growth(np.asarray(0.10), np.asarray(2.0), "annual"))
FixedArray(1.21000000000000020, scale=17)
```
