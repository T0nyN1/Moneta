# conventions — 金融计日惯例与年化时间跨度计算

`moneta.conventions` 模块提供标准金融市场计日惯例枚举（[`DayCountConvention`](#daycountconvention)）与年化时间跨度计算函数（[`year_fraction`](#year_fraction)），供持仓批次持有期折算（[`PositionLot.year_fraction`](lot.md#positionlotyear_fraction)）与带日期净现值贴现（[`npv`](tvm.md#npv)）统一调用。

```python
>>> from datetime import date
>>> from moneta import DayCountConvention, year_fraction
>>> [(m.name, m.value) for m in DayCountConvention]
[('ACT_360', 'ACT/360'), ('ACT_365', 'ACT/365'), ('THIRTY_360', '30/360'), ('ACT_ACT', 'ACT/ACT')]
>>> year_fraction("2026-01-01", "2026-07-01", DayCountConvention.ACT_360)  # 181 / 360
0.5027777777777778
>>> year_fraction("2024-01-01", "2025-01-01", "ACT/365")  # ACT/365 Fixed：闰年仍除以 365 (366 / 365)
1.0027397260273974
>>> year_fraction("2026-01-01", "2026-07-01", "30/360")  # 180 / 360
0.5
>>> year_fraction("2024-01-01", "2025-01-01", "act/act")  # ACT/ACT ISDA：闰年整年 366 / 366
1.0
```

[← API 参考](index.md)

## 核心机制与设计约束

### 1. 四种计日惯例的标准定义与精确计算公式

`DayCountConvention` 涵盖货币市场、债券与衍生品定价中最常用的四种标准计日惯例（设起始日 $d_1 = (Y_1, M_1, D_1)$，终止日 $d_2 = (Y_2, M_2, D_2)$，且 $d_1 \le d_2$）：

| 枚举成员 | 枚举值 (`value`) | 对应国际标准名称 | 年化时间跨度 $\tau(d_1, d_2)$ 计算规则 | 典型适用市场与品种 |
|---|---|---|---|---|
| `DayCountConvention.ACT_360` | `"ACT/360"` | **Actual/360** | $\dfrac{(d_2 - d_1).\text{days}}{360}$ | 美元 SOFR、欧元 EURIBOR 等货币市场与短期拆借 |
| `DayCountConvention.ACT_365` | `"ACT/365"` | **ACT/365 Fixed** | $\dfrac{(d_2 - d_1).\text{days}}{365}$（**不论平年或闰年，分母恒定为 365**） | 英镑 SONIA、港币 HIBOR、人民币银行间回购（如 FR007）与银行间债券及常规年化折算 |
| `DayCountConvention.THIRTY_360` | `"30/360"` | **30/360 Bond Basis**（ISDA 2006 4.16(f)） | 令 $D_1' = \min(D_1, 30)$；若 $D_1' = 30$ 则 $D_2' = \min(D_2, 30)$，否则 $D_2' = D_2$；$\tau = \dfrac{360(Y_2 - Y_1) + 30(M_2 - M_1) + (D_2' - D_1')}{360}$（**无 2 月月末特殊规则**） | 以 ISDA 30/360 计息的利率互换固定端（美国公司债/机构债常用的 30/360 US 含 2 月月末规则，结果可能不同） |
| `DayCountConvention.ACT_ACT` | `"ACT/ACT"` | **ACT/ACT ISDA** | 按自然年分段，各段实际天数除以所在年份的实际总天数（平年 $365$、闰年 $366$），先以有理数 `Fraction` 精确求和再单次转为 `float` | ISDA 定义中的 Actual/Actual (ISDA)，见于部分利率互换与跨年精确计息 |

### 2. `30/360`（ISDA Bond Basis）的月末调整细节与无 2 月规则

`DayCountConvention.THIRTY_360` 严格遵循 **ISDA 2006 Definitions Section 4.16(f)（30/360 Bond Basis）**，只包含两条日调整规则：

1. 若起始日为 31 日（$D_1 = 31$），则将 $D_1$ 改为 $30$；
2. 若调整后的 $D_1 = 30$（即原始 $D_1 \in \{30, 31\}$）且终止日为 31 日（$D_2 = 31$），则将 $D_2$ 改为 $30$；若 $D_1 < 30$，则 $D_2 = 31$ **保持为 31 不变**。

注意该标准**不包含** US 30/360（SIFMA）中的 2 月月末特殊调整规则：2 月 28 日或闰年 2 月 29 日均保留其真实日号 $28$ 或 $29$，不会被视作 $30$ 日。例如：

- `2024-02-29` 至 `2024-08-31`：因 $D_1 = 29 < 30$，终止日的 $D_2 = 31$ 不调整，计日天数为 $30 \times (8 - 2) + (31 - 29) = 182$ 天，$\tau = 182 / 360$；
- `2026-01-31` 至 `2026-02-28`：$D_1 = 31 \to 30$，而 2 月末 $D_2 = 28$ 不变，计日天数为 $30 \times (2 - 1) + (28 - 30) = 28$ 天，$\tau = 28 / 360$；
- `2026-01-15` 至 `2026-03-31`：因 $D_1 = 15 < 30$，$D_2 = 31$ 保留，计日天数为 $30 \times (3 - 1) + (31 - 15) = 76$ 天，$\tau = 76 / 360$。

### 3. 反对称性（日期颠倒返回负值）与灵活的字符串写法

- **严格反对称性**：当起始日 `start_date` 晚于终止日 `end_date`（$d_1 > d_2$）时，`year_fraction` 先交换两日期按上述规则计算正跨度，再冠以负号返回，严格保证对任意日期对与任意计日惯例均有：
  $$\text{year\_fraction}(d_2, d_1, \text{conv}) = -\text{year\_fraction}(d_1, d_2, \text{conv})$$
  这一性质使 [`npv`](tvm.md#npv) 在遇到早于估值日 `as_of_date` 的历史现金流时，能够自然得到负数时间跨度 $\tau < 0$ 并自动向后复利折算到估值日。
- **双形式字符串解析**：`convention` 参数除了接受 `DayCountConvention` 枚举成员外，还接受**枚举值字符串**（`"ACT/360"`、`"ACT/365"`、`"30/360"`、`"ACT/ACT"`）或**枚举成员名字符串**（`"ACT_360"`、`"ACT_365"`、`"THIRTY_360"`、`"ACT_ACT"`），不区分大小写且自动忽略首尾空白字符（如 `"act/360"`、`"thirty_360"`、`"  ACT/ACT\t"`）。

## API 速查表

| 分类 | 接口 / 常量 | 功能说明 |
|---|---|---|
| **计日惯例枚举** | [`DayCountConvention`](#daycountconvention) | 四种金融计日惯例枚举（`ACT_360`、`ACT_365`、`THIRTY_360`、`ACT_ACT`） |
| **年化跨度计算** | [`year_fraction`](#year_fraction) | 计算两个日期之间在指定计日惯例下的年化时间跨度 $\tau$（`float`） |
| **内部辅助（私有 API）** | [`_CONVENTION_KEYS` 与 `_parse_convention`](#_convention_keys-与-_parse_convention) | 计日惯例字符串查找字典与惯例参数校验解析函数 |

## 计日惯例枚举与时间跨度计算

### `DayCountConvention`

```python
class moneta.DayCountConvention(Enum)
```

金融计日惯例枚举（同时由顶层 `moneta` 与 `moneta.conventions` 导出）。

**成员列表**

- **`DayCountConvention.ACT_360 = "ACT/360"`** — Actual/360：实际日历天数除以 360。
- **`DayCountConvention.ACT_365 = "ACT/365"`** — ACT/365 Fixed：实际日历天数除以固定的 365（即使跨越闰年也除以 365）。
- **`DayCountConvention.THIRTY_360 = "30/360"`** — 30/360 Bond Basis（ISDA 2006 4.16(f)）：每月按 30 天、每年按 360 天折算，无 2 月月末规则。
- **`DayCountConvention.ACT_ACT = "ACT/ACT"`** — ACT/ACT ISDA：按自然年分段，平年部分除以 365，闰年部分除以 366。

**示例**

```python
>>> from moneta import DayCountConvention
>>> DayCountConvention.ACT_365.value, DayCountConvention("30/360") is DayCountConvention.THIRTY_360
('ACT/365', True)
```

**另请参阅**

- [`year_fraction`](#year_fraction)
- [`PositionLot.year_fraction`](lot.md#positionlotyear_fraction)
- [`npv`](tvm.md#npv)

### `year_fraction`

```python
moneta.year_fraction(
    start_date,
    end_date,
    convention=DayCountConvention.ACT_365,
)
```

计算两个日期之间在指定计日惯例下的年化时间跨度 $\tau$（同时由顶层 `moneta` 与 `moneta.conventions` 导出）。

**参数**

- **start_date** : *date, datetime 或 str* — 起始日期，通过 [`to_date(start_date, "start_date")`](_utils.md#to_date) 解析。
- **end_date** : *date, datetime 或 str* — 终止日期，通过 [`to_date(end_date, "end_date")`](_utils.md#to_date) 解析；当 `end_date` 早于 `start_date` 时返回负值。
- **convention** : *DayCountConvention 或 str，默认 `DayCountConvention.ACT_365`* — 计日惯例；传入字符串时既可用枚举值（如 `"ACT/360"`、`"30/360"`），也可用枚举成员名（如 `"ACT_360"`、`"THIRTY_360"`），不区分大小写且忽略首尾空白。

**返回值**

- **tau** : *float* — 年化时间跨度 $\tau$（当 `start_date == end_date` 时返回 `0.0`）。

**异常**

- **TypeError** —
  - `start_date` 或 `end_date` 的类型不是 `date`、`datetime` 或 `str`（如传入 `None` 或整数，报 `start_date/end_date 必须是 date、datetime 或 ISO 字符串，收到 ...`）；
  - `convention` 的类型不是 `DayCountConvention` 或 `str`（如传入 `None`、整数或布尔值，报 `f"convention 必须是 DayCountConvention 或 str，收到 {type(convention).__name__}"`）。
- **ValueError** —
  - `start_date` 或 `end_date` 是字符串但不是合法的 ISO-8601 日期（如 `"2024-13-01"`，报 `start_date/end_date 不是合法的 ISO 日期: ...`）；
  - `convention` 是字符串但不在受支持的枚举值或成员名集合中（如 `"ACT/365F"`、`"30E/360"`，报 `不支持的计日惯例 ...`）。

**示例**

四种惯例对比、`30/360` 无 2 月月末规则、字符串多种写法与日期颠倒返回负值：

```python
>>> from fractions import Fraction
>>> from moneta import DayCountConvention, year_fraction
>>> # 1. ACT/365 Fixed vs ACT/ACT ISDA（2024 为闰年，全年 366 天）
>>> year_fraction("2024-01-01", "2025-01-01", "ACT/365") == 366 / 365
True
>>> year_fraction("2024-01-01", "2025-01-01", "ACT/ACT") == 1.0
True
>>> # 2. ACT/ACT ISDA 跨平年与闰年分段：2023 下半年 (184/365) + 2024 上半年 (182/366)
>>> year_fraction("2023-07-01", "2024-07-01", DayCountConvention.ACT_ACT) == float(Fraction(184, 365) + Fraction(182, 366))
True
>>> # 3. 30/360 Bond Basis：D1=31 调整为 30，无 2 月月末规则
>>> year_fraction("2026-01-31", "2026-02-28", "30/360") == 28 / 360
True
>>> year_fraction("2024-02-29", "2024-08-31", "thirty_360") == 182 / 360
True
>>> # 4. 大小写与成员名/值字符串写法等价，且日期颠倒返回负值
>>> year_fraction("2024-07-01", "2024-01-01", " act_360 ") == -182 / 360
True
```

`ValueError` 与 `TypeError` 异常触发边界：

```python
>>> year_fraction("2024-01-01", "2024-07-01", "ACT/365F")
Traceback (most recent call last):
    ...
ValueError: 不支持的计日惯例 'ACT/365F'：可选 ['ACT/360', 'ACT/365', '30/360', 'ACT/ACT'] 或 ['ACT_360', 'ACT_365', 'THIRTY_360', 'ACT_ACT']
>>> year_fraction("2024-01-01", "2024-13-01")
Traceback (most recent call last):
    ...
ValueError: end_date 不是合法的 ISO 日期: '2024-13-01'
>>> year_fraction("2024-01-01", "2024-07-01", None)
Traceback (most recent call last):
    ...
TypeError: convention 必须是 DayCountConvention 或 str，收到 NoneType
>>> year_fraction(None, "2024-07-01")
Traceback (most recent call last):
    ...
TypeError: start_date 必须是 date、datetime 或 ISO 字符串，收到 NoneType
```

**性能对比**

| 计日惯例计算操作 | 耗时 | 说明 |
|---|---:|---|
| `year_fraction(..., ACT_360)` | 1.70 µs | ISO 日期解析 + 实际天数除以 360 |
| `year_fraction(..., ACT_365)` | 2.78 µs | ISO 日期解析 + 实际天数除以 365 |
| `year_fraction(..., THIRTY_360)` | 4.02 µs | 含 `D1`/`D2` 30 日钳制与 30/360 公式折算 |
| `year_fraction(..., ACT_ACT)`（跨年） | 13.4 µs | 含闰年判定与 `Fraction` 分段精确累加 |

**另请参阅**

- [`DayCountConvention`](#daycountconvention)
- [`to_date`](_utils.md#to_date)、[`is_leap_year`](_utils.md#is_leap_year)

## 内部辅助函数与常量（私有 API）

### `_CONVENTION_KEYS` 与 `_parse_convention`

```python
moneta.conventions._CONVENTION_KEYS: Dict[str, DayCountConvention]
moneta.conventions._parse_convention(convention)
```

- **`_CONVENTION_KEYS`**：将每个 `DayCountConvention` 成员的 `value`（如 `"30/360"`）与 `name`（如 `"THIRTY_360"`）同时映射到该枚举成员的字典。
- **`_parse_convention(convention)`**：若输入已是 `DayCountConvention` 则直接返回；若为 `str` 则执行 `convention.strip().upper()` 后在 `_CONVENTION_KEYS` 中查找（未命中抛 `ValueError`）；其他类型直接抛出 `TypeError`。同时供 [`tvm.npv`](tvm.md#npv) 复用。

**示例**

```python
>>> from moneta.conventions import DayCountConvention, _parse_convention
>>> _parse_convention(" 30/360 ") is DayCountConvention.THIRTY_360
True
>>> _parse_convention("thirty_360") is DayCountConvention.THIRTY_360
True
```
