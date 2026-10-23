# _utils — 内部通用数学、日期与调用栈工具模块

`moneta._utils` 提供无状态的内部通用技术工具函数，涵盖 `int64` 极值绝对值提取（[`max_abs`](#max_abs)）、多格式日期解析（[
`to_date`](#to_date)）、公历闰年判定（[`is_leap_year`](#is_leap_year)）以及跨包内部栈帧的动态告警层级计算（[
`user_stacklevel`](#user_stacklevel)），支撑 [`FixedArray`](fixed_array.md)、`conventions`、`cashflow` 与 `lot` 等模块。

```python
>> > from datetime import date, datetime
>> > import numpy as np
>> > from moneta._utils import is_leap_year, max_abs, to_date, user_stacklevel
>> > max_abs(np.array([-(1 << 63), 100], dtype=np.int64)) == 1 << 63
True
>> > to_date("2024-02-29"), to_date(datetime(2024, 2, 29, 15, 30))
(datetime.date(2024, 2, 29), datetime.date(2024, 2, 29))
>> > is_leap_year(2024), is_leap_year(1900), is_leap_year(2000)
(True, False, True)
>> > user_stacklevel() >= 1
True
```

[← API 参考](index.md)

## 核心机制与设计约束

### 1. `-2**63` 无溢出极值提取（`max_abs`）

在有符号 64 位整数（`int64`，范围 $[-2^{63}, 2^{63}-1]$）中，最小负数 $-2^{63}$ 的绝对值 $+2^{63}$ 无法用 `int64` 表示。若直接调用
`np.abs(x).max()`，当数组中含有 `-2**63` 时，`np.abs` 会在 `int64` 内发生二进制补码回绕（仍得到负数 `-2**63`
），导致乘法、矩阵内积与求和的快路径上界估算严重失真。

[`max_abs(x)`](#max_abs) 先调用 `x.min()` 与 `x.max()` 在 `int64` 内提取最小值与最大值，随即转换为任意精度的 Python `int`
计算 `max(-int(x.min()), int(x.max()))`，仅用两次归约即可在 Python `int` 域内无溢出地返回真实最大绝对值，并对空数组（
`x.size == 0`）安全返回 `0`。

### 2. 多源日期归一化与语义化报错（`to_date` 与 `is_leap_year`）

在持仓批次（`PositionLot`）、计日惯例（`year_fraction`）与现金流排程（`CashFlow`）中，所有涉及日期的参数均通过 [
`to_date(value, what="日期")`](#to_date) 统一归一化为标准库的 `datetime.date` 对象：

- 若传入 `datetime.datetime`（注意 `issubclass(datetime, date)` 为 `True`，故必须先判 `datetime`），自动调用 `.date()`
  剥离时间部分；
- 若传入 `datetime.date`，原样返回；
- 若传入 `str`，先 `.strip()` 去除首尾空白再按 `date.fromisoformat` 解析 ISO-8601 字符串（`"YYYY-MM-DD"`）；
- 可选的 `what` 参数允许调用方注入具体的字段名（如 `"acquired_date"`、`"as_of_date"`），使抛出的 `TypeError` 或 `ValueError`
  直接指出报错参数名称。

### 3. 穿透内部调用链的精准告警定位（`user_stacklevel`）

当精确输入（如 `"1.005"` 或 `Decimal("1.005")`）传入固定精度的 [`Money`](money.md) 或显式指定 `scale` 的 [
`Price`](price.md) / [`Quantity`](quantity.md) / [`FixedArray(..., on_excess="warn")`](fixed_array.md) 时，底层的 [
`FixedArray._quantize_precision`](fixed_array.md#fixedarray_quantize_precision) 会发出 [
`PrecisionLossWarning`](fixed_array.md#precisionlosswarning)。
由于调用路径可能经过 `Money.__init__` → `FixedArray.__init__` → `_quantize_precision`，也可能经过 `Money.fillna` →
`UnitArray.fillna` → `Money._fill_value` 等不同深度的内部栈帧，硬编码的固定 `stacklevel` 无法同时适配所有入口。
[`user_stacklevel()`](#user_stacklevel) 通过 `sys._getframe(1)` 自底向上遍历调用栈，自动跳过所有 `f_globals["__name__"]`
顶层包名为 `"moneta"` 的内部栈帧，使告警始终精准指向用户业务代码中的调用行。

## API 速查表

| 分类               | 接口                                  | 功能说明                                                                         |
|--------------------|---------------------------------------|----------------------------------------------------------------------------------|
| **数值极值工具**   | [`max_abs`](#max_abs)                 | 计算整数数组绝对值的最大值（返回 Python `int`，`-2**63` 不溢出，空数组返回 `0`） |
| **日期与日历工具** | [`to_date`](#to_date)                 | 将 `date`、`datetime` 或 ISO 日期字符串解析为标准 `datetime.date` 对象           |
|                    | [`is_leap_year`](#is_leap_year)       | 判断公历年份是否为闰年（四年一闰、百年不闰、四百年再闰）                         |
| **调用栈工具**     | [`user_stacklevel`](#user_stacklevel) | 计算跳过 `moneta` 包内部调用帧的 `warnings.warn` 栈层级 `stacklevel`             |

## 数值极值工具

### `max_abs`

```python
moneta._utils.max_abs(x)
```

计算整数 NumPy 数组 `x` 中所有元素绝对值的最大值，返回标准 Python `int`。

**参数**

- **x** : *np.ndarray* — 输入的整数数组（支持 0 维标量数组、任意多维数组以及空数组）。

**返回值**

- **out** : *int* — 当 `x.size > 0` 时返回 $\max_{i} |x_i|$（类型为 Python `int`；即使包含 `-2**63` 也精确返回
  `2**63 = 9223372036854775808`）；当 `x.size == 0` 时返回 `0`。

**机制说明**

直接执行 `max(-int(x.min()), int(x.max())) if x.size else 0`。供 [
`FixedArray._mul_pow10`](fixed_array.md#fixedarray_mul_pow10)、[
`FixedArray._divmod_round`](fixed_array.md#fixedarray_divmod_round)、[
`FixedArray.mul_round`](fixed_array.md#fixedarraymul_round)、[`FixedArray.matmul`](fixed_array.md#fixedarraymatmul)、[
`FixedArray.sum`](fixed_array.md#fixedarraysum) 与 [`FixedArray.mean`](fixed_array.md#fixedarraymean) 进行 $O (1)$
快慢路径分流判定。

**示例**

```python
>> > import numpy as np
>> > from moneta._utils import max_abs
>> > max_abs(np.array([-(1 << 63), 5], dtype=np.int64)) == 1 << 63
True
>> > max_abs(np.array([[-9, 2], [7, -3]], dtype=np.int64))
9
>> > max_abs(np.array([], dtype=np.int64)), max_abs(np.zeros((0, 3), dtype=np.int64))
(0, 0)
```

**另请参阅**

- [`FixedArray._mul_pow10`](fixed_array.md#fixedarray_mul_pow10)
- [`FixedArray.mul_round`](fixed_array.md#fixedarraymul_round)
- [`FixedArray.sum`](fixed_array.md#fixedarraysum)

## 日期与日历工具

### `to_date`

```python
moneta._utils.to_date(value, what="日期")
```

将 `datetime.date`、`datetime.datetime` 或 ISO-8601 格式的日期字符串（`"YYYY-MM-DD"`）解析并归一化为标准 `datetime.date`
对象。

**参数**

- **value** : *date, datetime 或 str* — 待解析的日期值；若为 `str`，会自动去除首尾空白字符。
- **what** : *str，默认 `"日期"`* — 用于异常提示信息的参数名称标签（如 `"acquired_date"`、`"as_of_date"`）。

**返回值**

- **out** : *datetime.date* — 标准日期对象（当输入为 `datetime` 时返回 `value.date()`，保证 `type(out) is date`）。

**异常**

- **TypeError** — `value` 不是 `date`、`datetime` 或 `str`（报
  `f"{what} 必须是 date、datetime 或 ISO 字符串，收到 {type(value).__name__}"`）。
- **ValueError** — `value` 是字符串但不是合法的 ISO 日期格式或日历日期越界（报
  `f"{what} 不是合法的 ISO 日期: {value!r}"`）。

**示例**

```python
>> > from datetime import date, datetime
>> > from moneta._utils import to_date
>> > to_date(date(2024, 1, 2)), to_date(datetime(2024, 1, 2, 15, 30)), to_date("  2024-01-02\t")
(datetime.date(2024, 1, 2), datetime.date(2024, 1, 2), datetime.date(2024, 1, 2))
>> > to_date("2024/01/02", "acquired_date")
Traceback(most
recent
call
last):
...
ValueError: acquired_date
不是合法的
ISO
日期: '2024/01/02'
>> > to_date(20240102, "as_of_date")
Traceback(most
recent
call
last):
...
TypeError: as_of_date
必须是
date、datetime
或
ISO
字符串，收到
int
```

**另请参阅**

- [`is_leap_year`](#is_leap_year)

### `is_leap_year`

```python
moneta._utils.is_leap_year(year)
```

判断公历年份 `year` 是否为闰年。

**参数**

- **year** : *int* — 公历年份整数。

**返回值**

- **out** : *bool* — 当 `(year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)` 时返回 `True`，否则返回 `False`。

**示例**

```python
>> > from moneta._utils import is_leap_year
>> > [is_leap_year(y) for y in (1900, 2000, 2023, 2024, 2100)]
[False, True, False, True, False]
```

**另请参阅**

- [`to_date`](#to_date)

## 调用栈工具

### `user_stacklevel`

```python
moneta._utils.user_stacklevel()
```

计算供 `warnings.warn(..., stacklevel=user_stacklevel())` 使用的动态栈层级：从调用栈自底向上回溯，跳过所有属于 `moneta`
包内部的调用帧，返回指向用户业务代码的第一帧层级。

**返回值**

- **level** : *int* — 传给 `warnings.warn` 的 `stacklevel` 整数（$\ge 1$）。当由外部用户代码直接调用时返回 `1`
  ；当中间经过 $k$ 层 `moneta.*` 模块内部函数调用时返回 $k + 1$。

**机制说明**

从 `sys._getframe(1)` 与 `level = 1` 开始循环检查 `frame.f_globals.get("__name__", "").split(".")[0] == "moneta"`
：只要当前帧仍处于 `moneta` 包内部，就继续向调用方回退 `frame = frame.f_back` 并将 `level` 加 `1`，直到遇到第一个非
`moneta` 的外部调用帧（或栈顶 `None`）为止。

**示例**

```python
>> > import warnings
>> > from moneta import Money, PrecisionLossWarning
>> > from moneta._utils import user_stacklevel
>> > user_stacklevel()
1
>> > with warnings.catch_warnings(record=True) as record:
    ...
warnings.simplefilter("always")
...
_ = Money("1.005", "USD")
>> > len(record) == 1 and issubclass(record[0].category, PrecisionLossWarning)
True
```

**另请参阅**

- [`PrecisionLossWarning`](fixed_array.md#precisionlosswarning)
- [`FixedArray._quantize_precision`](fixed_array.md#fixedarray_quantize_precision)
