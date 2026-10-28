# cashflow — 现金流定义与定期付息排程生成

`moneta.cashflow` 模块提供单笔带日期的不可变现金流对象（[`CashFlow`](#cashflowamount-pay_date-label)）与定期付息/本金归还排程生成器（[`generate_cash_flow_schedule`](#generate_cash_flow_schedule)）。

模块在日期推算与票息计算上遵循两项核心设计契约：

- **无月末漂移的锚定推算**：第 $k$ 期付息日始终由起息日直接推算（$\text{start\_date} + k \times \text{step}$ 个月），绝不逐期累加；当起息日处于当月月末（如 `1月31日`、`2月29日` 或 `11月30日`）时，所有付息日均自动锁定为各月月末；到期日 `end_date` 必须严格落在付息日序列上，不支持不足一期的零碎付息期（Stub Period）。
- **无溢出且严格等价于无限精度单次舍入的票息计算**：每期票息 $\text{principal} \times \text{coupon\_rate} / n$ 先将乘积按 `digits + 1` 位向零截断（`rounding="down"`），再除以每年付息次数 $n \in \{1, 2, 4, 12\}$ 按 `half_up` 舍入到币种精度 `digits`——由于 `half_up` 的半数平局点（$0.5\text{ ULP}$）乘以整数 $n$ 后恒落在 `digits + 1` 位（$0.1\text{ ULP}$）网格上，向零截断绝不会越过任何平局点，既避免了高小数位利率（如 18 位浮点小数）乘积超出 `int64`，又保证结果与对无限精度有理数直接做一次 `half_up` 舍入完全一致。

```python
>>> from moneta import CashFlow, Money, generate_cash_flow_schedule
>>> cf = CashFlow(Money("100.00", "USD"), "2026-06-01", "Coupon")
>>> cf.amount, cf.currency.code, cf.pay_date, cf.label
(Money(FixedArray(100.00, scale=2), currency='USD'), 'USD', datetime.date(2026, 6, 1), 'Coupon')
>>> str(cf)
'CashFlow(100.00 USD on 2026-06-01 [Coupon])'
>>> flows = generate_cash_flow_schedule(
...     "2026-01-31", "2026-04-30", "monthly",
...     principal=Money("1000.00", "USD"), coupon_rate="0.06",
... )
>>> [(str(f.pay_date), f.amount, f.label) for f in flows]
[('2026-02-28', Money(FixedArray(5.00, scale=2), currency='USD'), 'Coupon'), ('2026-03-31', Money(FixedArray(5.00, scale=2), currency='USD'), 'Coupon'), ('2026-04-30', Money(FixedArray(5.00, scale=2), currency='USD'), 'Coupon'), ('2026-04-30', Money(FixedArray(1000.00, scale=2), currency='USD'), 'Principal')]
```

[← API 参考](index.md)

## 核心机制与设计约束

### 1. 不可变单笔现金流与向量化金额支持（`CashFlow`）

[`CashFlow(amount, pay_date, label="")`](#cashflowamount-pay_date-label) 仅包含三个只读槽位 `__slots__ = ("_amount", "_pay_date", "_label")`：

- **金额与方向语义**：`amount` 必须是 [`Money`](money.md) 实例（正数表示现金流入，负数表示现金流出）。由于 `Money` 本身支持任意维度的定点数张量与缺失值掩码 `<NA>`，单个 `CashFlow` 即可同时承载多个情景或多个账户在同一支付日的现金流向量；
- **日期归一化**：`pay_date` 通过 [`to_date(pay_date, "pay_date")`](_utils.md#to_date) 统一解析为标准库 `datetime.date`（不接受 `None`）；
- **不可变性**：构造后禁止修改或删除任何属性（重写了 `__setattr__` 与 `__delattr__`，违者抛出 `AttributeError`）。

### 2. 起息日锚定推算与月末锁定规则（消除月末漂移）

在固定收益付息日生成中，若采用“上一期付息日 + 1 个周期”的递推写法，从 `2024-01-31` 按月递推会先变成 `2024-02-29`，再变成 `2024-03-29`、`2024-04-29`，导致后续所有付息日永久漂移到 29 日。

[`generate_cash_flow_schedule`](#generate_cash_flow_schedule) 通过 [`_add_months(start, k * step, end_of_month)`](#_parse_frequency_add_months-与-_coupon_amount) 始终以起息日 `start` 为唯一基准推算第 $k$ 期（$k = 1, 2, \dots$）付息日：

| 起息日 `start_date` 特征 | `end_of_month` 标志 | 第 $k$ 期付息日推算规则 | 示例（`2024` 闰年按月付息） |
|---|---|---|---|
| **恰为当月最后一天**（如 `1月31日`、`2月29日`、`4月30日`） | `True` | 目标月份始终取**当月最后一天**（月末对月末） | `2024-01-31` → `02-29`、`03-31`、`04-30`、`05-31` |
| **非当月最后一天**（如 `1月30日`、`1月15日`） | `False` | 取 `min(start.day, 目标月天数)`：短月临时钳制到月末，下个长月立即恢复为 `start.day` | `2024-01-30` → `02-29`、`03-30`、`04-30` |

同时，循环生成直到 `dates[-1] >= end` 后，严格校验 `dates[-1] == end`：若到期日 `end_date` 未能恰好落在付息日序列上（例如起息日与到期日不足一期，或相差非整数个周期），直接抛出 `ValueError` 并提示到期日位于哪两个日期之间（早于第一个付息日时为起息日与第一个付息日）。

### 3. `digits + 1` 中间截断的精确票息定理（`_coupon_amount`）

当通过 `principal` 与年化票息率 `coupon_rate` 计算每期票息 $\text{coupon} = \text{principal} \times \text{rate} / n$（其中 $n \in \{1, 2, 4, 12\}$ 为每年付息次数）时：

1. 若先用无损乘法 `principal * rate` 再除以 $n$，当 `coupon_rate` 是高精度的 `float`（如 `0.07 / 12`，最短十进制表示有 18 位小数）时，本金 `1000.00` 与它的乘积尾数约为 $10^5 \times 5.8 \times 10^{15} \approx 5.8 \times 10^{20}$，超出 `int64` 范围（约 $9.2 \times 10^{18}$）；
2. [`_coupon_amount`](#_parse_frequency_add_months-与-_coupon_amount) 先执行：
   $$\text{product} = \text{principal.data.mul\_round}(\text{rate},\ \text{scale}=\text{digits} + 1,\ \text{rounding}=\text{"down"})$$
   再执行：
   $$\text{product.divide}(n,\ \text{scale}=\text{digits},\ \text{rounding}=\text{"half\_up"})$$
3. **无偏差数学证明**：以最小货币单位 $\text{ULP} = 10^{-\text{digits}}$ 为单位，记本金与年利率的精确乘积为实数 $X \ge 0$。除以整数 $n$ 并按 `half_up` 舍入到整数 ULP 的全部判定边界（即整数半点 $m + \frac{1}{2}$）对应于 $X$ 轴上的点：
   $$X^* = n \cdot \left(m + \frac{1}{2}\right) = \frac{5n(2m+1)}{10}$$
   由于 $n \in \{1, 2, 4, 12\}$ 为正整数，$5n(2m+1)$ 恒为整数，即所有平局边界点 $X^*$ 在 `digits + 1` 位（$\frac{1}{10}\text{ ULP}$）网格上都是**格点**。因此，将 $X$ 向零截断到 $\frac{1}{10}\text{ ULP}$ 网格得到 $\lfloor 10X \rfloor / 10$ 时，满足：
   $$X \ge X^* \iff \frac{\lfloor 10X \rfloor}{10} \ge X^*$$
   $X < 0$ 时同理：向零截断与 `half_up`（平局远离零）都按绝对值对称，对 $|X|$ 应用上述结论即可。从而保证两步计算的结果与对有理数精确值 $\text{principal} \times \text{rate} / n$ 直接做一次 `half_up` 舍入**100% 恒等**。

## API 速查表

| 分类 | 接口 / 属性 / 常量 | 功能说明 |
|---|---|---|
| **单笔现金流** | [`CashFlow(amount, pay_date, label="")`](#cashflowamount-pay_date-label) | 构造不可变的单笔带日期现金流对象（支持标量或向量化 `Money`） |
| | [`CashFlow.amount`](#cashflowamount) / [`currency`](#cashflowcurrency) | 现金流金额（[`Money`](money.md)） / 现金流币种（[`Currency`](currency.md)） |
| | [`CashFlow.pay_date`](#cashflowpay_date) / [`label`](#cashflowlabel) | 支付日期（`datetime.date`） / 现金流类型标签（`str`） |
| | [`str(cf)`、`repr(cf)` 与不可变性](#strcfreprcf-与不可变性) | 可读文本摘要、精确构造表示与只读属性保护 |
| **付息排程生成** | [`generate_cash_flow_schedule`](#generate_cash_flow_schedule) | 按付息频率生成定期票息（`"Coupon"`）与到期本金（`"Principal"`）现金流序列 |
| **内部辅助（私有 API）** | [`DateLike`、`_FREQ_NAMES`](#datelike-与-_freq_names类型别名与频率映射表) | 日期联合类型别名与频率名称到每年付息次数的映射字典 |
| | [`_parse_frequency`、`_add_months` 与 `_coupon_amount`](#_parse_frequency_add_months-与-_coupon_amount) | 付息频率解析、无漂移月份推算与单次舍入等价的票息计算函数 |

## 单笔现金流（`CashFlow`）

### `CashFlow(amount, pay_date, label="")`

```python
class moneta.CashFlow(amount, pay_date, label="")
```

构造单笔带支付日期的不可变现金流对象（同时由顶层 `moneta` 与 `moneta.cashflow` 导出）。

**参数**

- **amount** : *Money* — 现金流金额（正数表示流入，负数表示流出），支持 0 维标量或多维数组（含缺失值 `<NA>`）。不接受裸数字，必须传入 [`Money`](money.md) 实例。
- **pay_date** : *date, datetime 或 str* — 支付日期（必填），通过 [`to_date(pay_date, "pay_date")`](_utils.md#to_date) 解析为 `datetime.date`。
- **label** : *str，默认 `""`* — 现金流标签，如 `"Coupon"`、`"Principal"`、`"Fee"`。

**返回值**

- **out** : *CashFlow* — 不可变的现金流实例。

**异常**

- **TypeError** — `amount` 不是 `Money` 实例；`pay_date` 不是 `date`、`datetime` 或 `str`；或 `label` 不是 `str`。
- **ValueError** — `pay_date` 字符串不是合法的 ISO-8601 日期格式。

**示例**

```python
>>> from datetime import datetime
>>> from moneta import CashFlow, Money
>>> cf = CashFlow(Money("100.00", "USD"), datetime(2026, 6, 1, 9, 30), "Coupon")
>>> cf.amount, cf.currency.code, cf.pay_date, cf.label
(Money(FixedArray(100.00, scale=2), currency='USD'), 'USD', datetime.date(2026, 6, 1), 'Coupon')
>>> cf_vec = CashFlow(Money(["100.00", None, "-50.00"], "USD"), "2026-06-01")
>>> cf_vec.amount, cf_vec.label
(Money(FixedArray([100.00, <NA>, -50.00], scale=2), currency='USD'), '')
>>> CashFlow(100, "2026-06-01")
Traceback (most recent call last):
    ...
TypeError: amount 必须是 Money，收到 int
```

**另请参阅**

- [`generate_cash_flow_schedule`](#generate_cash_flow_schedule)

### `CashFlow.amount`

```python
CashFlow.amount
```

返回现金流金额（[`Money`](money.md) 实例）。

**返回值**

- **amount** : *Money*

**示例**

```python
>>> from moneta import CashFlow, Money
>>> CashFlow(Money("-250.00", "CNY"), "2026-06-01", "Fee").amount
Money(FixedArray(-250.00, scale=2), currency='CNY')
```

### `CashFlow.currency`

```python
CashFlow.currency
```

返回现金流币种（[`Currency`](currency.md) 单例），等价于 `self.amount.currency`。

**返回值**

- **currency** : *Currency*

**示例**

```python
>>> from moneta import CashFlow, Money, USD
>>> CashFlow(Money("100.00", "USD"), "2026-06-01").currency is USD
True
```

### `CashFlow.pay_date`

```python
CashFlow.pay_date
```

返回归一化后的支付日期（标准库 `datetime.date` 实例）。

**返回值**

- **pay_date** : *datetime.date*

**示例**

```python
>>> from moneta import CashFlow, Money
>>> CashFlow(Money("100.00", "USD"), "2026-06-01").pay_date
datetime.date(2026, 6, 1)
```

### `CashFlow.label`

```python
CashFlow.label
```

返回现金流标签字符串（默认 `""`）。

**返回值**

- **label** : *str*

**示例**

```python
>>> from moneta import CashFlow, Money
>>> CashFlow(Money("100.00", "USD"), "2026-06-01", "Principal").label
'Principal'
```

### `str(cf)`、`repr(cf)` 与不可变性

```python
str(cf)
repr(cf)
```

- **`str(cf)`**：返回 `f"CashFlow({self._amount} on {self._pay_date} [{self._label}])"` 的易读文本。
- **`repr(cf)`**：返回 `f"CashFlow(amount={self._amount!r}, pay_date='{self._pay_date}', label='{self._label}')"` 的完整构造表示。
- **不可变性**：任何对属性的赋值或删除操作都会直接抛出 `AttributeError`。

**示例**

```python
>>> from moneta import CashFlow, Money
>>> cf = CashFlow(Money("100.00", "USD"), "2026-06-01", "Coupon")
>>> str(cf)
'CashFlow(100.00 USD on 2026-06-01 [Coupon])'
>>> repr(cf)
"CashFlow(amount=Money(FixedArray(100.00, scale=2), currency='USD'), pay_date='2026-06-01', label='Coupon')"
>>> cf._amount = Money("200.00", "USD")
Traceback (most recent call last):
    ...
AttributeError: CashFlow 是不可变对象，禁止设置属性 '_amount'
>>> del cf._label
Traceback (most recent call last):
    ...
AttributeError: CashFlow 是不可变对象，禁止删除属性 '_label'
```

## 定期付息排程生成

### `generate_cash_flow_schedule`

```python
moneta.generate_cash_flow_schedule(
    start_date,
    end_date,
    freq,
    *,
    coupon=None,
    principal=None,
    coupon_rate=None,
)
```

生成定期付息排程（同时由顶层 `moneta` 与 `moneta.cashflow` 导出）：在每个付息日生成一笔标签为 `"Coupon"` 的票息现金流，并在到期日 `end_date` 另加一笔标签为 `"Principal"` 的本金现金流。

**参数**

- **start_date** : *date, datetime 或 str* — 起息日（计息起始日，当天本身不产生票息现金流）。
- **end_date** : *date, datetime 或 str* — 到期日；必须严格晚于 `start_date`，且必须**正好落在付息日序列上**（不支持不足一期的零碎付息期）。
- **freq** : *str 或 int* — 付息频率（必填）：
  - 字符串（不区分大小写，忽略首尾空白）：`"monthly"`（月付，12 次/年）、`"quarterly"`（季付，4 次/年）、`"semi-annual"` 或 `"semiannual"`（半年付，2 次/年）、`"annual"` 或 `"annually"`（年付，1 次/年）；
  - 整数每年付息次数：`12`、`4`、`2` 或 `1`（不接受布尔值 `True`）。
- **coupon** : *Money，仅关键字参数，可选* — 直接指定的每期固定票息金额（不能与 `coupon_rate` 同时给出）。
- **principal** : *Money，仅关键字参数，可选* — 到期归还的本金金额（支持标量或向量化 `Money`）。若同时给出了 `coupon`，两者的币种必须一致。
- **coupon_rate** : *数值或字符串，仅关键字参数，可选* — 年化票息率（不能是缺失值 `<NA>` / `NaN`）。必须与 `principal` 同时给出，且不能与 `coupon` 同时给出；每期票息按 $\text{principal} \times \text{coupon\_rate} / n$ 单次 `half_up` 舍入到币种精度。

> **注**：`coupon`、`principal`、`coupon_rate` 三个关键字参数中至少需要给出一种有效现金流（可只生成票息流、只生成到期本金流，或同时生成定期票息与到期本金）。

**返回值**

- **flows** : *List[CashFlow]* — 按时间顺序排列的 [`CashFlow`](#cashflowamount-pay_date-label) 列表：若存在票息，前 $K$ 项为各付息日的 `"Coupon"` 现金流；若给出了 `principal`，最后一项为到期日 `end_date` 的 `"Principal"` 现金流。

**异常**

- **TypeError** — `start_date`/`end_date` 类型非法；`coupon` 或 `principal` 不是 [`Money`](money.md) 实例；或 `coupon_rate` 类型不被 [`FixedArray`](fixed_array.md) 接受（如传入 `bool`）。
- **ValueError** —
  - `end_date <= start_date`；
  - `freq` 不是受支持的付息频率名称或次数；
  - `coupon_rate` 给出时未给 `principal`，或同时给出了 `coupon` 与 `coupon_rate`，或 `coupon_rate` 为缺失值，或 `coupon_rate` 是无法解析的字符串（如 `"abc"`，报 `无法解析数值`）；
  - `coupon`、`principal`、`coupon_rate` 全部省略；
  - `coupon.currency != principal.currency`；
  - `end_date` 不在从 `start_date` 按 `freq` 推算出的付息日序列上。

**机制说明**

1. 调用 [`to_date`](_utils.md#to_date) 解析 `start` 与 `end` 并验证 `start < end`，调用 [`_parse_frequency(freq)`](#_parse_frequency_add_months-与-_coupon_amount) 获取每年付息次数 $n \in \{1, 2, 4, 12\}$；
2. 校验 `coupon`、`principal` 与 `coupon_rate` 的组合合法性；当给出 `coupon_rate` 时，调用 [`_coupon_amount(principal, FixedArray(coupon_rate), n)`](#_parse_frequency_add_months-与-_coupon_amount) 算出每期票息 `coupon`；
3. 设每期月数步长 `step = 12 // n`，判断 `start` 是否为当月最后一天 `end_of_month`；从 $k = 1$ 开始循环调用 `_add_months(start, k * step, end_of_month)` 生成付息日序列 `dates`，直至 `dates[-1] >= end`，并验证 `dates[-1] == end`；
4. 若 `coupon is not None` 则为每个日期生成 `CashFlow(coupon, d, "Coupon")`，若 `principal is not None` 则在末尾追加 `CashFlow(principal, end, "Principal")`。

**示例**

按年票息率生成半年付息排程、月末不漂移排程与纯本金排程：

```python
>>> from moneta import Money, generate_cash_flow_schedule
>>> flows = generate_cash_flow_schedule(
...     "2026-01-01", "2027-01-01", "semi-annual",
...     principal=Money("1000.00", "USD"), coupon_rate="0.05",
... )
>>> [(str(cf.pay_date), cf.amount, cf.label) for cf in flows]
[('2026-07-01', Money(FixedArray(25.00, scale=2), currency='USD'), 'Coupon'), ('2027-01-01', Money(FixedArray(25.00, scale=2), currency='USD'), 'Coupon'), ('2027-01-01', Money(FixedArray(1000.00, scale=2), currency='USD'), 'Principal')]
>>> # 起息日为月末（2024-01-31）：所有付息日严格对齐各月月末，不会因 2 月 29 日而漂移到 29 日
>>> eom_flows = generate_cash_flow_schedule("2024-01-31", "2024-05-31", 12, coupon=Money("10.00", "USD"))
>>> [str(cf.pay_date) for cf in eom_flows]
['2024-02-29', '2024-03-31', '2024-04-30', '2024-05-31']
>>> # 起息日非月末（2024-01-30）：2 月钳制到 29 日，3 月立即恢复到 30 日
>>> mid_flows = generate_cash_flow_schedule("2024-01-30", "2024-04-30", "monthly", coupon=Money("10.00", "USD"))
>>> [str(cf.pay_date) for cf in mid_flows]
['2024-02-29', '2024-03-30', '2024-04-30']
```

到期日不在付息序列上或参数冲突时的异常拦截：

```python
>>> generate_cash_flow_schedule("2024-01-01", "2024-05-01", "quarterly", coupon=Money("10", "USD"))
Traceback (most recent call last):
    ...
ValueError: end_date (2024-05-01) 不在付息日序列上（位于付息日 2024-04-01 与付息日 2024-07-01 之间），不支持不足一期的付息
>>> generate_cash_flow_schedule("2024-01-01", "2024-06-01", "annual", coupon=Money("10", "USD"))
Traceback (most recent call last):
    ...
ValueError: end_date (2024-06-01) 不在付息日序列上（位于起息日 2024-01-01 与付息日 2025-01-01 之间），不支持不足一期的付息
>>> generate_cash_flow_schedule("2024-01-01", "2025-01-01", "weekly", coupon=Money("10", "USD"))
Traceback (most recent call last):
    ...
ValueError: 不支持的付息频率 'weekly'：可选 ['monthly', 'quarterly', 'semi-annual', 'semiannual', 'annual', 'annually'] 或 1、2、4、12
```

**性能对比**

| 现金流与排程生成操作 | 耗时 | 说明 |
|---|---:|---|
| `CashFlow(amount, pay_date, label)` | 0.69 µs | 含类型校验与 ISO 日期解析 |
| `generate_cash_flow_schedule`（1 年半年付，给定 `coupon`） | 8.53 µs | 2 期票息 + 1 期本金 |
| `generate_cash_flow_schedule`（1 年半年付，给定 `coupon_rate`） | 60.0 µs | 含票息率定点数解析、单次舍入票息计算与排程构造 |
| `generate_cash_flow_schedule`（1 年月付，1,000 维向量本金） | 87.6 µs | 1,000 笔本金向量同步计算 12 期票息 |
| `generate_cash_flow_schedule`（10 年季付，41 笔现金流） | 96.6 µs | 40 期无漂移付息日推算 + 1 期到期本金 |

**另请参阅**

- [`CashFlow`](#cashflowamount-pay_date-label)
- [`_coupon_amount`](#_parse_frequency_add_months-与-_coupon_amount)

## 内部辅助函数与常量（私有 API）

### `DateLike` 与 `_FREQ_NAMES`（类型别名与频率映射表）

```python
moneta.cashflow.DateLike = Union[str, date, datetime]
moneta.cashflow._FREQ_NAMES = {
    "monthly": 12,
    "quarterly": 4,
    "semi-annual": 2,
    "semiannual": 2,
    "annual": 1,
    "annually": 1,
}
```

- **`DateLike`**：可被 [`to_date`](_utils.md#to_date) 解析的日期输入联合类型别名。
- **`_FREQ_NAMES`**：受支持的付息频率英文名称（小写）到每年付息次数（`12`、`4`、`2`、`1`）的映射字典。

### `_parse_frequency`、`_add_months` 与 `_coupon_amount`

```python
moneta.cashflow._parse_frequency(freq)
moneta.cashflow._add_months(d, months, end_of_month)
moneta.cashflow._coupon_amount(principal, rate, n)
```

- **`_parse_frequency(freq)`**：将付息频率参数 `freq`（不区分大小写的字符串或整数 `1`、`2`、`4`、`12`，排除 `bool`）转换为每年付息次数整数；不支持的值抛出 `ValueError`。
- **`_add_months(d, months, end_of_month)`**：计算日期 `d` 加上 `months` 个月后的日期：通过 `divmod(d.month - 1 + months, 12)` 算出目标年月，并查询 `calendar.monthrange(year, month)[1]` 获取该月总天数 `last`；当 `end_of_month=True` 时始终取 `last`，否则取 `min(d.day, last)`。
- **`_coupon_amount(principal, rate, n)`**：计算每期票息 `principal × rate / n`，先执行 `product = principal.data.mul_round(rate, scale=digits + 1, rounding="down")` 将乘积按 `digits + 1` 位向零截断，再调用 `Money._wrap(product.divide(n, scale=digits), principal.currency)` 按 `half_up` 舍入到币种精度。

**示例**

```python
>>> from datetime import date
>>> from moneta import FixedArray, Money
>>> from moneta.cashflow import _add_months, _coupon_amount, _parse_frequency
>>> _parse_frequency(" Semi-Annual "), _parse_frequency(4)
(2, 4)
>>> _add_months(date(2024, 1, 31), 1, end_of_month=True), _add_months(date(2024, 1, 30), 2, end_of_month=False)
(datetime.date(2024, 2, 29), datetime.date(2024, 3, 30))
>>> _coupon_amount(Money("1000.00", "USD"), FixedArray("0.07"), 12)  # 70 / 12 = 5.8333... -> 5.83 USD
Money(FixedArray(5.83, scale=2), currency='USD')
```
