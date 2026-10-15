# FixedArray — 定点数张量引擎

`moneta.FixedArray` 是基于 `int64` 尾数的定点数 N 维数组，`moneta.RoundingMode` 是配套的舍入模式枚举。
它们是 Moneta 全部精确运算的数值内核：加减乘、求和、矩阵乘法全程走整数运算，
不会产生 IEEE 754 浮点累积误差；舍入只在你明确要求的地方、按你指定的规则发生一次；
结果超出 `int64` 时一定报错，不会静默回绕。

```python
>> > a = FixedArray(["1.10", "2.25", "-0.05"])
>> > a
FixedArray([1.10, 2.25, -0.05], scale=2)
>> > a * 2
FixedArray([2.20, 4.50, -0.10], scale=2)
>> > a.sum()
FixedArray(3.30, scale=2)
>> > bool(FixedArray("0.1") + FixedArray("0.2") == FixedArray("0.3"))
True
```

[← API 参考](index.md)

## 核心机制与设计约束

### 定点数表示与不可变性

每个 `FixedArray` 由两部分组成：

$$v = \text{mantissa} \times 10^{-\text{scale}}$$

- **mantissa**：只读的 `int64` NumPy 数组，形状即 `FixedArray` 的形状；
- **scale**：整个数组共享的非负整数（小数位数）。

`FixedArray` 是 **不可变**的：没有原地运算，`mantissa` 是只读视图，不支持元素赋值，不可哈希
（不能作为 `dict` 键或放入 `set`）。形状变换、索引、`rescale` 到相同 scale 等操作会直接共享尾数而不复制，
这正是不可变性带来的好处。`FixedArray` 只认识“数”，不认识币种与单位——带量纲的运算由上层的
`UnitArray` / `Money` 负责。

### scale 传播规则

| 运算                                      | 结果 scale                                           | 是否舍入                  |
|-------------------------------------------|------------------------------------------------------|---------------------------|
| `a + b`、`a - b`、`a % b`                 | `max(a.scale, b.scale)`                              | 精确                      |
| `+a`、`-a`、`abs(a)`、`a.sum()`           | `a.scale`                                            | 精确                      |
| `a * b`、`a @ b`                          | `a.scale + b.scale`                                  | 精确                      |
| `a / b`                                   | `max(a.scale, b.scale)`                              | `half_up`                 |
| `a // b`                                  | 返回整数（`int` 或 `int64` 数组），不是 `FixedArray` | 向下取整                  |
| `a.mul_round(b, scale, rounding)`         | `scale`，默认 `max(a.scale, b.scale)`                | 按 `rounding`，只舍入一次 |
| `a.matmul(b, scale, rounding)`            | `scale`，默认 `max(a.scale, b.scale)`                | 按 `rounding`，只舍入一次 |
| `a.divide(b, scale, rounding)`            | `scale`，默认 `max(a.scale, b.scale)`                | 按 `rounding`             |
| `a.mean(axis, keepdims, scale, rounding)` | `scale`，默认 `a.scale`                              | 按 `rounding`             |
| `a.rescale(new_scale, rounding)`          | `new_scale`                                          | 缩小时按 `rounding`       |
| `a.round_to_step(step, rounding)`         | `max(a.scale, step.scale)`                           | 按 `rounding`             |

### 操作数白名单与溢出保证

**操作数白名单。** 运算、比较与 `mul_round` / `divide` / `round_to_step` 等方法的另一方只能是：
`FixedArray`、`int`、`float`、`Decimal`、`Fraction`、整数或浮点 `ndarray`，以及只含这些元素的 `list` / `tuple`。
它们先按[构造规则](#fixedarrayvalues-scalenone)转换为 `FixedArray`。字符串、布尔值（含布尔数组）与其他对象一律拒绝：
运算符返回 `NotImplemented`，最终由 Python 抛出 `TypeError`。字符串只能出现在构造函数里。

**溢出保证。** 所有运算都在 `int64` 内完成，不升级为 Python 大整数，也不退化为 float。
结果真实超出 `int64` 时抛出 `OverflowError`。对于中间量容易越界的运算，采用“快慢双路径”：

| 运算             | 快路径（绝大多数输入）                                      | 慢路径（最坏上界超出 `int64` 时自动切换）                            |
|------------------|-------------------------------------------------------------|----------------------------------------------------------------------|
| `+` `-`          | 一次整数加减 + 符号位溢出检测                               | —                                                                    |
| `*`、`mul_round` | `max(abs(a)) · max(abs(b))` 不越界时一次乘法 + 一次带余除法 | 尾数拆成 3 个 10⁹ 进制块做长乘法                                     |
| `@`、`matmul`    | `max(abs(a)) · max(abs(b)) · K` 不越界时直接 `np.matmul`    | 尾数拆成 3 个 21 位二进制块，9 次块矩阵乘法后进位合并                |
| `sum`            | `max(abs(a)) · n` 不越界时直接 `np.sum`                     | 尾数拆成 3 个 10⁹ 进制块分别求和后进位合并                           |
| `divide`         | 带余除法                                                    | 分段长除法；除数接近 `int64` 上限时改用 `10r = 8r + 2r` 的二进制倍增 |

慢路径能正确处理“中间量越界、最终结果在范围内”的情况（例如正负相抵的求和），只有最终结果真实越界才报错。
NumPy 的 `int64` 运算在同样的输入上会静默回绕。

## API 速查表

| 分类                   | 接口 / 运算符                                                                                                                | 功能说明                                                           |
|------------------------|------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| **舍入模式**           | [`RoundingMode`](#roundingmode)                                                                                              | 六种金融与会计舍入模式枚举                                         |
| **构造与属性**         | [`FixedArray(values, scale=None)`](#fixedarrayvalues-scalenone)                                                              | 从数值、字符串、`Decimal`、`Fraction`、数组构造                    |
|                        | [`FixedArray.mantissa`](#fixedarraymantissa)                                                                                 | 只读 `int64` 尾数数组视图                                          |
|                        | [`FixedArray.scale`](#fixedarrayscale)                                                                                       | 全局十进制小数位数                                                 |
|                        | [`FixedArray.shape`](#fixedarrayshape) / [`ndim`](#fixedarrayndim) / [`size`](#fixedarraysize) / [`dtype`](#fixedarraydtype) | 形状、维数、元素总数、尾数数据类型                                 |
|                        | [`FixedArray.T`](#fixedarrayt)                                                                                               | 反转全部轴的转置视图                                               |
| **算术运算**           | [`+a`、`-a`](#a-与--a一元正负号)                                                                                             | 一元正负号                                                         |
|                        | [`abs(a)`](#absa绝对值)                                                                                                      | 逐元素绝对值                                                       |
|                        | [`a + b`、`a - b`](#a--b-与-a---b加减法)                                                                                     | 逐元素加减法，自动对齐 scale                                       |
|                        | [`a * b`](#a--b精确乘法)                                                                                                     | 逐元素精确乘法（scale 相加）                                       |
|                        | [`FixedArray.mul_round`](#fixedarraymul_round)                                                                               | 融合乘法 + 一次舍入                                                |
|                        | [`a / b`](#a--b真除法)                                                                                                       | 逐元素真除法（默认保留 `max(scale)` 位并 `half_up`）               |
|                        | [`FixedArray.divide`](#fixedarraydivide)                                                                                     | 指定目标 scale 与舍入模式的精确除法                                |
|                        | [`a // b`、`a % b`、`divmod(a, b)`](#a--ba--b-与-divmoda-b整除与取余)                                                        | 向下整除、取余与带余除法                                           |
| **矩阵运算**           | [`a @ b`](#a--b精确矩阵乘法)                                                                                                 | 精确矩阵乘法（scale 相加）                                         |
|                        | [`FixedArray.matmul`](#fixedarraymatmul)                                                                                     | 矩阵乘法 + 一次舍入                                                |
| **规约统计**           | [`FixedArray.sum`](#fixedarraysum)                                                                                           | 沿轴精确求和                                                       |
|                        | [`FixedArray.mean`](#fixedarraymean)                                                                                         | 沿轴求算术均值并精确舍入                                           |
| **精度与网格**         | [`FixedArray.rescale`](#fixedarrayrescale)                                                                                   | 调整小数位数                                                       |
|                        | [`FixedArray.round_to_step`](#fixedarrayround_to_step)                                                                       | 对齐到步长网格（最小变动价位、整手）                               |
|                        | [`FixedArray.is_on_step`](#fixedarrayis_on_step)                                                                             | 逐元素判断是否落在步长网格上                                       |
| **比较运算**           | [`a == b`、`a != b`](#a--b-与-a--b相等性比较)                                                                                | 对齐 scale 后逐元素判断相等或不等                                  |
|                        | [`a < b`、`a <= b`、`a > b`、`a >= b`](#a--ba--ba--ba--b大小比较)                                                            | 对齐 scale 后逐元素比较大小                                        |
| **形状与协议**         | [`FixedArray.reshape`](#fixedarrayreshape)                                                                                   | 不复制数据改变数组形状                                             |
|                        | [`FixedArray.transpose`](#fixedarraytranspose)                                                                               | 按指定顺序置换各轴                                                 |
|                        | [`a[key]`](#akey索引与切片)                                                                                                  | 索引、切片、布尔掩码与花式索引                                     |
|                        | [`len(a)`、`bool(a)`、`iter(a)`](#lenaboola-与-itera容器与真值协议)                                                          | 首维长度、单元素真值测试与按首维迭代                               |
| **导出与互操作**       | [`FixedArray.to_float`](#fixedarrayto_float)                                                                                 | 导出为 `float` / `float64` 数组（有损）                            |
|                        | [`FixedArray.to_decimal`](#fixedarrayto_decimal)                                                                             | 导出为 `Decimal` / `object` 数组（无损）                           |
|                        | [`FixedArray.to_pandas`](#fixedarrayto_pandas)                                                                               | 导出为 `pandas.Series` / `DataFrame`                               |
|                        | [`FixedArray.to_polars`](#fixedarrayto_polars)                                                                               | 导出为 `polars.Series`                                             |
|                        | [`str(a)`、`repr(a)`](#stra-与-repra文本表示)                                                                                | 按 scale 补零的精确文本表示                                        |
|                        | [NumPy 互操作机制](#numpy-互操作机制)                                                                                        | 可用与被禁止的 NumPy 函数、常见写法配方                            |
| **内部构造与转换**     | [`FixedArray._set`](#fixedarray_set)                                                                                         | 构造收口：绑定只读 `int64` 尾数视图与 `scale`                      |
|                        | [`FixedArray._from_mantissa`](#fixedarray_from_mantissa)                                                                     | 由已有 `int64` 尾数与 `scale` 零拷贝构造 `FixedArray`              |
|                        | [`FixedArray._is_numeric_operand`](#fixedarray_is_numeric_operand)                                                           | 校验对象是否属于合法的数值操作数白名单                             |
|                        | [`FixedArray._coerce`](#fixedarray_coerce)                                                                                   | 将运算另一侧的操作数安全转换为 `FixedArray`                        |
|                        | [`FixedArray._check_scale`](#fixedarray_check_scale)                                                                         | 校验 `scale` 是否为非负整数                                        |
|                        | [`FixedArray._as_int64`](#fixedarray_as_int64)                                                                               | 将整数标量或整数数组转换为 `int64` 数组并检查越界                  |
|                        | [`FixedArray._align_scales`](#fixedarray_align_scales)                                                                       | 将两个 `FixedArray` 的 `scale` 对齐到较大值                        |
| **内部输入解析**       | [`FixedArray._parse_decimal_str`](#fixedarray_parse_decimal_str)                                                             | 解析十进制字符串为 `(mantissa, scale)`                             |
|                        | [`FixedArray._parse_fraction`](#fixedarray_parse_fraction)                                                                   | 解析有限十进制 `Fraction` 为 `(mantissa, scale)`                   |
|                        | [`FixedArray._parse_float_scalar`](#fixedarray_parse_float_scalar)                                                           | 按最短十进制写法（Dragonbox）解析单个浮点数                        |
|                        | [`FixedArray._parse_object_element`](#fixedarray_parse_object_element)                                                       | 解析单个标量对象为 `(mantissa, scale)`                             |
|                        | [`FixedArray._parse_float_array`](#fixedarray_parse_float_array)                                                             | 向量化往返试探解析浮点数组为 `(尾数数组, scale)`                   |
|                        | [`FixedArray._align_mantissas`](#fixedarray_align_mantissas)                                                                 | 将不同 `scale` 的尾数向量化对齐到目标 `target`                     |
|                        | [`FixedArray._round_float_to_scale`](#fixedarray_round_float_to_scale)                                                       | 将浮点数组的最短十进制写法按 `half_up` 快速舍入到指定 `scale`      |
|                        | [`FixedArray._parse_text_array`](#fixedarray_parse_text_array)                                                               | 去重解析字符串、`Decimal`、`Fraction` 或混合对象数组               |
| **底层数学与舍入内核** | [`FixedArray._check_rounding`](#fixedarray_check_rounding)                                                                   | 校验并规范化舍入模式名称                                           |
|                        | [`FixedArray._check_division`](#fixedarray_check_division)                                                                   | 校验除数非零及 `-2**63 / -1` 溢出边界                              |
|                        | [`FixedArray._max_abs`](#fixedarray_max_abs)                                                                                 | 安全计算 `int64` 数组绝对值的最大值（`-2**63` 不溢出）             |
|                        | [`FixedArray._mul_pow10`](#fixedarray_mul_pow10)                                                                             | 带前置越界检查的 `x · 10^k` 整数放大                               |
|                        | [`FixedArray._decide_bump`](#fixedarray_decide_bump)                                                                         | 六种舍入模式的统一进位裁决函数                                     |
|                        | [`FixedArray._round_quotient`](#fixedarray_round_quotient)                                                                   | 由向下取整的商 `q` 与余数 `r` 按指定模式取整                       |
|                        | [`FixedArray._divmod_round`](#fixedarray_divmod_round)                                                                       | 纯 `int64` 整数带余除法与精确舍入                                  |
|                        | [`FixedArray._doubling_step_10r`](#fixedarray_doubling_step_10r)                                                             | 大除数无溢出二进制倍增单步长除（计算 `(10r) // d` 与 `(10r) % d`） |
|                        | [`FixedArray._chunked_divmod_round`](#fixedarray_chunked_divmod_round)                                                       | 分段十进制长除法与舍入 `round(m · 10^k / d)`                       |
|                        | [`FixedArray._round_div_pow10`](#fixedarray_round_div_pow10)                                                                 | 将带余商 `(q + r/d)` 除以 `10^k` 并精确舍入（支持任意 `k ≥ 0`）    |
|                        | [`FixedArray._chunked_mul_round`](#fixedarray_chunked_mul_round)                                                             | Base 10⁹ 分块卷积长乘法与一次舍入                                  |
|                        | [`FixedArray._chunked_sum`](#fixedarray_chunked_sum)                                                                         | Base 10⁹ 分块沿轴精确求和（消除中间溢出误报）                      |
|                        | [`FixedArray._chunked_matmul`](#fixedarray_chunked_matmul)                                                                   | Base 2²¹ 分块精确矩阵乘法（消除中间溢出误报）                      |
| **内部比较与格式化**   | [`FixedArray._compare`](#fixedarray_compare)                                                                                 | 对齐 `scale` 后执行逐元素比较运算                                  |
|                        | [`FixedArray._format_scalar`](#fixedarray_format_scalar)                                                                     | 将单个整数尾数格式化为固定小数位字符串                             |

## 舍入模式

### RoundingMode

```python
class moneta.RoundingMode(str, Enum)
```

金融与会计常用的六种舍入模式。

**成员**

| 成员        | 字符串值      | 规则                         | 典型用途                                      | 对应 `decimal` 常量 |
|-------------|---------------|------------------------------|-----------------------------------------------|---------------------|
| `HALF_UP`   | `"half_up"`   | 四舍五入，恰好一半时远离零   | 零售报价、大多数默认场景                      | `ROUND_HALF_UP`     |
| `HALF_EVEN` | `"half_even"` | 银行家舍入：恰好一半时取偶数 | 会计（GAAP / IFRS）、利息计提，消除系统性偏差 | `ROUND_HALF_EVEN`   |
| `DOWN`      | `"down"`      | 向零截断                     | 截尾、保守计息                                | `ROUND_DOWN`        |
| `UP`        | `"up"`        | 远离零进位                   | 收费、保证金                                  | `ROUND_UP`          |
| `FLOOR`     | `"floor"`     | 向负无穷                     | 可用余额、数量向下取整                        | `ROUND_FLOOR`       |
| `CEIL`      | `"ceil"`      | 向正无穷                     | 最低应付额                                    | `ROUND_CEILING`     |

各模式舍入到整数的结果：

| 输入 | `half_up` | `half_even` | `down` | `up` | `floor` | `ceil` |
|-----:|----------:|------------:|-------:|-----:|--------:|-------:|
|  2.5 |         3 |           2 |      2 |    3 |       2 |      3 |
|  3.5 |         4 |           4 |      3 |    4 |       3 |      4 |
|  2.4 |         2 |           2 |      2 |    3 |       2 |      3 |
| -2.5 |        -3 |          -2 |     -2 |   -3 |      -3 |     -2 |
| -2.6 |        -3 |          -3 |     -2 |   -3 |      -3 |     -2 |

**机制说明**

`RoundingMode` 继承自 `str`，所以所有 `rounding=` 参数都既可以传枚举成员，也可以传对应的字符串，两者完全等价；
无效的模式名抛出 `ValueError`。接受 `rounding` 的 API：`mul_round`、`divide`、`matmul`、`mean`、`rescale`、`round_to_step`。
所有舍入都在整数商与余数上判定，不经过 float，结果与 `decimal` 模块对应模式完全一致。

**示例**

```python
>> > from moneta import RoundingMode
>> > x = FixedArray(["2.5", "3.5", "2.4", "-2.5", "-2.6"])
>> > for mode in RoundingMode:
    ...
print(f"{mode.value:>9}", x.rescale(0, rounding=mode))
half_up[3, 4, 2, -3, -3]
half_even[2, 4, 2, -2, -3]
down[2, 3, 2, -2, -2]
up[3, 4, 3, -3, -3]
floor[2, 3, 2, -3, -3]
ceil[3, 4, 3, -2, -2]
>> > RoundingMode.HALF_EVEN == "half_even"
True
>> > FixedArray("0.125").rescale(2, RoundingMode.HALF_EVEN)
FixedArray(0.12, scale=2)
>> > FixedArray("0.125").rescale(2, "nearest")
Traceback(most
recent
call
last):
...
ValueError: rounding
只能是('half_up', 'half_even', 'down', 'up', 'floor', 'ceil')
之一，收到
'nearest'
```

**另请参阅**

- [`FixedArray.rescale`](#fixedarrayrescale) – 最直接的舍入入口。

## 构造与核心属性

### FixedArray (values, scale=None)

```python
class moneta.FixedArray(values, scale=None)
```

从数值、字符串、`Decimal`、`Fraction`、序列或 NumPy 数组构造定点数数组。

**参数**

- **values** : *int, float, str, Decimal, Fraction, array_like 或 FixedArray* — 输入数值。标量得到 0 维数组；
  也可以是嵌套序列或 `ndarray`。不支持 `bool`、复数、日期时间等类型。
- **scale** : *int，可选* — 结果的小数位数，必须是非负整数（`int` 或 NumPy 整数，不接受 `bool`、`float`）。
  默认 `None` 时按下表自动推断。

**返回值**

- **out** : *FixedArray* — 形状与 `np.asarray(values)` 相同；`scale` 见下表。

**异常**

- **TypeError** — `values` 的类型不受支持（`bool`、复数、`datetime64` 等）。
- **ValueError** — `scale` 不是非负整数；输入含 NaN 或无穷大；字符串无法解析；`Fraction` 不是有限小数（如 `1/3`）；
  字符串 / `Decimal` / `Fraction` 的小数位数超过显式 `scale`。
- **OverflowError** — 数值（或对齐到共同 scale 后的尾数）超出 `int64` 范围。

**机制说明**

| 输入                      | `scale=None` 时                                | 显式 `scale` 时                                |
|---------------------------|------------------------------------------------|------------------------------------------------|
| `int` / 整数数组          | `0`                                            | 数值不变，尾数乘以 `10**scale`                 |
| `float` 标量              | 最短十进制写法的小数位数（`0.1` → 1）          | 比推断值小时按 `half_up` 舍入                  |
| `float` 数组              | 每个元素取最短往返写法，再取最大小数位数       | 每个元素直接舍入到 `scale`，不必先对齐到全精度 |
| `str` / `Decimal`         | 字面声明的小数位数（`"1.50"` → 2，保留尾随零） | 只能扩大；比字面位数小时报 `ValueError`        |
| `Fraction`                | 能精确表示它的最少小数位数（`3/8` → 3）        | 只能扩大                                       |
| 混合 `object` 数组 / 序列 | 各元素按上述规则解析后取最大值                 | 只能扩大                                       |
| `FixedArray`              | 原 scale，共享尾数（不复制）                   | 等价于 `values.rescale(scale)`，`half_up`      |

- **浮点数按最短写法解析。** 浮点数不按其二进制精确值解析（`0.1` 的精确值有 55 位小数），而是按
  “最短的、能往返还原为同一个浮点数的十进制写法”（与 `repr(float)` 一致，即 Dragonbox 算法的结果）解析。
  数组路径对每个元素逐位试探 `s = 0, 1, 2, …`：在 `|m| ≤ 2^(nmant+1)`（`float64` 为 `2**53`，`float32` 为 `2**24`）
  且 `s ≤ 22` 的精确范围内，必须在 **首次通过** `(m / 10^s).astype(dtype) == x` 往返检验时记录 `(m, s)`
  （大数在更多小数位上也能往返，但那已不是最短写法）；超出精确试探范围的元素去重后逐个按最短写法解析，最后统一对齐。
- **显式 scale 时的快速安全边距舍入。** float 数组指定 `scale` 时，最多只试探到 `scale + 1` 位，不必先把整列对齐到全精度。
  对超过 `scale + 1` 位的元素，先用浮点计算 `z = x · 10^scale`：只要 `x` 为正规数、`|z| < 2**51` 且小数部分离 `0.5` 的距离超过
  安全余量 `|z| · (eps + 2**-50)`，`x` 与其最短十进制写法必然位于半数边界同侧，可直接向量化取整；
  仅贴近 `0.5` 边界或极大的少数元素退回最短写法做整数精确舍入。舍入严格定义在 **最短十进制写法**上：
  `FixedArray(2.675, scale=2)` 得到 `2.68`，而内置 `round(2.675, 2)` 得到 `2.67`。
- **全零数组与极值边界。** 全零输入（如 `0`、`[0, 0]`、`"0"`）指定任意大的 `scale`（即使 `scale > 18`）都不会溢出；
  字符串与 `Decimal` 支持精确解析到 `int64` 下限 `-2**63`（`-9223372036854775808`）。
- **字符串不会被隐式舍入。** 字符串、`Decimal`、`Fraction` 被视为精确输入，小数位数超过 `scale` 时报错，
  避免悄悄丢掉精度；需要舍入时先构造再 [`rescale`](#fixedarrayrescale)。
- **不可变。** 从整数 `ndarray` 构造时，如果结果会与调用方的数组共享内存，会先复制一份。

**示例**

整数、字符串与 `Decimal`：

```python
>> > FixedArray(5)
FixedArray(5, scale=0)
>> > FixedArray(5, scale=2)
FixedArray(5.00, scale=2)
>> > FixedArray(0, scale=25)  # 全零数组支持超过 18 位的任意 scale
FixedArray(0.0000000000000000000000000, scale=25)
>> > from decimal import Decimal
>> > FixedArray("1.50")
FixedArray(1.50, scale=2)
>> > FixedArray(Decimal("1.100"))
FixedArray(1.100, scale=3)
>> > FixedArray("1.5e-3")
FixedArray(0.0015, scale=4)
>> > FixedArray(" 1_000.50 ")
FixedArray(1000.50, scale=2)
```

浮点数按最短写法解析，数组对齐到最大位数：

```python
>> > FixedArray(0.1)
FixedArray(0.1, scale=1)
>> > FixedArray([0.1, 0.25])
FixedArray([0.10, 0.25], scale=2)
>> > FixedArray(np.array([0.1, 2.5], dtype=np.float32))
FixedArray([0.1, 2.5], scale=1)
>> > FixedArray(2.675, scale=2)  # 在 "2.675" 上 half_up
FixedArray(2.68, scale=2)
>> > round(2.675, 2)  # 对比：内置 round 受二进制误差影响
2.67
```

显式 `scale` 时每个元素直接舍入，不会因为全精度对齐或无限小数而溢出：

```python
>> > big_and_fine = np.array([123456789012.5, 0.123456789])
>> > FixedArray(big_and_fine)  # 需对齐到 9 位小数，尾数越界
Traceback(most
recent
call
last):
...
OverflowError: 浮点数需要超过
7
位小数，对齐后超出
int64
范围
>> > FixedArray(big_and_fine, scale=2)
FixedArray([123456789012.50, 0.12], scale=2)
>> > FixedArray(np.array([100000.0, 1 / 3]), scale=2)
FixedArray([100000.00, 0.33], scale=2)
```

`Fraction`、混合类型与另一个 `FixedArray`：

```python
>> > from fractions import Fraction
>> > FixedArray(Fraction(3, 8))
FixedArray(0.375, scale=3)
>> > FixedArray([1, Decimal("2.5"), "0.125"])
FixedArray([1.000, 2.500, 0.125], scale=3)
>> > FixedArray(FixedArray("1.25"), scale=1)  # 等价于 rescale(1)
FixedArray(1.3, scale=1)
```

非法输入：

```python
>> > FixedArray(Fraction(1, 3))
Traceback(most
recent
call
last):
...
ValueError: 1 / 3
不是有限小数
>> > FixedArray("1.234", scale=2)
Traceback(most
recent
call
last):
...
ValueError: 输入的小数位数(3)
超出指定
scale(2)
>> > FixedArray(float("nan"))
Traceback(most
recent
call
last):
...
ValueError: FixedArray
不能包含
NaN
或无穷大
>> > FixedArray(True)
Traceback(most
recent
call
last):
...
TypeError: 不支持从
bool(dtype=bool)
构造
FixedArray
>> > FixedArray(2 ** 63)
Traceback(most
recent
call
last):
...
OverflowError: 整数超出
int64
范围
```

**性能对比**

构造过程需要逐元素解析数值并对齐精度，耗时明显高于后续的纯整数运算；建议在初始化阶段将输入数据一次性构造为 `FixedArray`
，后续计算直接复用该对象，避免在循环中重复调用构造函数。

<!-- bench:construct_float,construct_str -->
*构造：float64 数组 → FixedArray*，对照代码：`np.round(x * 100).astype(np.int64)`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|     1,000 |      53 µs |       1.97 µs |       303 µs |
| 1,000,000 |    32.7 ms |       1.19 ms |       318 ms |

*构造：字符串数组 → FixedArray*，对照代码：`s.astype(np.float64)`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|     1,000 |    1.31 ms |        158 µs |       123 µs |
| 1,000,000 |      1.4 s |        158 ms |       144 ms |

<!-- /bench -->

**另请参阅**

- [`FixedArray.rescale`](#fixedarrayrescale) – 改变已有数组的 scale。
- [`FixedArray.to_decimal`](#fixedarrayto_decimal) – 构造的逆运算。
- [`FixedArray._parse_float_array`](#fixedarray_parse_float_array)、[
  `_round_float_to_scale`](#fixedarray_round_float_to_scale)、[`_parse_text_array`](#fixedarray_parse_text_array) –
  底层输入解析引擎。

### FixedArray.mantissa

```python
FixedArray.mantissa
```

底层整数尾数数组（只读视图）。

**机制说明**

`value = mantissa × 10**(-scale)`。返回的数组 `writeable=False`，修改它会报 `ValueError`；需要可写副本时调用 `.copy()`。
返回的是视图而非拷贝，读取它没有开销。同一个 `FixedArray` 内 scale 相同，所以尾数的大小顺序就是数值的大小顺序，
可以放心交给 `np.argmax`、`np.argsort` 等只读整数函数。

**返回值**

- **mantissa** : *ndarray（int64）* — 0 维 `FixedArray` 返回 0 维数组。

**示例**

```python
>> > a = FixedArray(["1.50", "-2.25"])
>> > a.mantissa
array([150, -225])
>> > FixedArray("19.99").mantissa
array(1999)
>> > a.mantissa[0] = 0
Traceback(most
recent
call
last):
...
ValueError: assignment
destination is read - only
>> > prices = FixedArray(["10.05", "10.20", "9.95"])
>> > prices[int(np.argmax(prices.mantissa))]
FixedArray(10.20, scale=2)
```

**另请参阅**

- [`FixedArray.scale`](#fixedarrayscale)
- [NumPy 互操作机制](#numpy-互操作机制) – 基于尾数的常用写法配方。

### FixedArray.scale

```python
FixedArray.scale
```

小数位数，整个数组共享的非负 Python `int`。

**机制说明**

与 `object` 类型的 `Decimal` 数组（每个元素各自携带指数）不同，`FixedArray` 让整个张量共享唯一一个整数 `scale`，
从而将所有元素的尾数紧凑存储在连续的 `int64` 内存中，实现向量化算术与零精度漂移。

**返回值**

- **scale** : *int*

**示例**

```python
>> > FixedArray("1.50").scale
2
>> > FixedArray(7).scale
0
>> > FixedArray([0.1, 0.125]).scale
3
```

**另请参阅**

- [`FixedArray.mantissa`](#fixedarraymantissa)
- [scale 传播规则](#scale-传播规则)
- [`FixedArray.rescale`](#fixedarrayrescale)

### FixedArray.shape

```python
FixedArray.shape
```

数组形状，直接读取底层 `mantissa.shape`。

**机制说明**

标量输入构造出的 `FixedArray` 是 0 维张量，其 `shape` 为空元组 `()`；`np.shape(a)` 也会自动转发到此属性。

**返回值**

- **shape** : *tuple of int* — 标量构造得到的 `FixedArray` 为 `()`。

**示例**

```python
>> > FixedArray([[1, 2, 3], [4, 5, 6]]).shape
(2, 3)
>> > FixedArray("1.5").shape
()
```

**另请参阅**

- [`FixedArray.ndim`](#fixedarrayndim)、[`FixedArray.size`](#fixedarraysize)
- [`FixedArray.reshape`](#fixedarrayreshape)

### FixedArray.ndim

```python
FixedArray.ndim
```

数组的维度数（轴数），等于 `len(a.shape)`。

**机制说明**

直接读取 `mantissa.ndim`；0 维标量张量的 `ndim` 为 `0`，不支持 `len()`
与迭代（见[容器与真值协议](#lenaboola-与-itera容器与真值协议)）。

**返回值**

- **ndim** : *int*

**示例**

```python
>> > FixedArray("1.5").ndim
0
>> > FixedArray([[1, 2], [3, 4]]).ndim
2
```

**另请参阅**

- [`FixedArray.shape`](#fixedarrayshape)、[`FixedArray.size`](#fixedarraysize)

### FixedArray.size

```python
FixedArray.size
```

数组中元素的总个数，等于 `shape` 各维长度的乘积。

**机制说明**

直接读取 `mantissa.size`。注意 0 维标量张量的 `size` 为 `1`，空数组的 `size` 为 `0`；
`bool(a)` 真值测试仅在 `a.size == 1` 时允许执行。

**返回值**

- **size** : *int*

**示例**

```python
>> > FixedArray([[1, 2, 3], [4, 5, 6]]).size
6
>> > FixedArray("1.5").size
1
```

**另请参阅**

- [`FixedArray.shape`](#fixedarrayshape)、[`FixedArray.ndim`](#fixedarrayndim)
- [`len(a)`、`bool(a)`](#lenaboola-与-itera容器与真值协议)

### FixedArray.dtype

```python
FixedArray.dtype
```

底层尾数的数据类型，恒为 `numpy.int64`。

**机制说明**

它描述的是 **尾数**而不是数值本身：`FixedArray` 表示的是十进制定点数，不是整数。

**返回值**

- **dtype** : *numpy.dtype*

**示例**

```python
>> > FixedArray("1.5").dtype
dtype('int64')
```

**另请参阅**

- [`FixedArray.mantissa`](#fixedarraymantissa)
- [`FixedArray.to_float`](#fixedarrayto_float)、[`FixedArray.to_decimal`](#fixedarrayto_decimal)

### FixedArray.T

```python
FixedArray.T
```

反转全部轴的转置视图，等价于 `a.transpose()`。

**机制说明**

只构造底层 `mantissa.T` 的只读视图，零拷贝且 `scale` 保持不变；0 维与 1 维数组的 `.T` 形状不变。

**返回值**

- **out** : *FixedArray* — `scale` 不变，与原数组共享尾数。

**示例**

```python
>> > FixedArray([["1.1", "2.2", "3.3"], ["4.4", "5.5", "6.6"]]).T
FixedArray([[1.1, 4.4],
            [2.2, 5.5],
            [3.3, 6.6]], scale=1)
```

**另请参阅**

- [`FixedArray.transpose`](#fixedarraytranspose)
- [`FixedArray.reshape`](#fixedarrayreshape)

## 算术运算

本节运算都遵循 NumPy 广播规则，返回新的 `FixedArray`（整除除外），不会修改操作数。
另一方的合法类型见[操作数白名单](#操作数白名单与溢出保证)；反向运算同样可用，
`10 - a`、`np.array([1, 2]) * a`、`Decimal("1.5") + a` 都返回 `FixedArray`。

### +a 与 -a（一元正负号）

```python
+a
-a
```

一元正号返回自身；一元负号逐元素取负。

**返回值**

- **out** : *FixedArray* — `scale` 不变。`+a` 返回的就是 `a` 本身（`FixedArray` 不可变，无需复制）。

**异常**

- **OverflowError** — `-a` 时含有尾数 `-2**63`（其相反数超出 `int64`）。

**机制说明**

`-a` 先检查是否存在 `-2**63`，再对尾数整体取负，开销与 NumPy 取负相当。

**示例**

```python
>> > a = FixedArray(["1.5", "-2"])
>> > -a
FixedArray([-1.5, 2.0], scale=1)
>> > (+a) is a
True
```

**性能对比**

<!-- bench:neg -->
对照代码：`-x`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    3.72 µs |       52.5 ns |       241 ns |
|     1,000 |    3.57 µs |        441 ns |      62.7 µs |
| 1,000,000 |     821 µs |        179 µs |      76.1 ms |

<!-- /bench -->

**另请参阅**

- [`abs(a)`](#absa绝对值)

### abs (a)（绝对值）

```python
abs(a)
```

逐元素绝对值。

**返回值**

- **out** : *FixedArray* — `scale` 不变。

**异常**

- **OverflowError** — 含有尾数 `-2**63`。

**机制说明**

`np.abs(a)` 是 ufunc，会被拒绝（见 [NumPy 互操作机制](#numpy-互操作机制)），请使用内置 `abs()`。

**示例**

```python
>> > abs(FixedArray(["-1.5", "2"]))
FixedArray([1.5, 2.0], scale=1)
>> > np.abs(FixedArray("-1.5"))
Traceback(most
recent
call
last):
...
TypeError: operand
'FixedArray'
does
not support
ufuncs(__array_ufunc__=None)
```

**性能对比**

<!-- bench:abs -->
对照代码：`np.abs(x)`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    3.71 µs |        376 ns |       221 ns |
|     1,000 |    3.57 µs |        463 ns |      46.9 µs |
| 1,000,000 |     732 µs |        279 µs |        59 ms |

<!-- /bench -->

**另请参阅**

- [`+a` 与 `-a`](#a-与--a一元正负号)

### a + b 与 a - b（加减法）

```python
a + b
a - b
```

逐元素加减，结果精确。

**参数**

- **b** : *FixedArray 或数值 array_like* — 另一个操作数，可广播。

**返回值**

- **out** : *FixedArray* — `scale = max(a.scale, b.scale)`，形状为广播后的形状。

**异常**

- **TypeError** — `b` 不是合法的数值操作数。
- **ValueError** — 形状无法广播。
- **OverflowError** — 对齐 scale 时尾数越界，或结果超出 `int64`。

**机制说明**

先把两个操作数对齐到 `max(a.scale, b.scale)`（较小 scale 一方的尾数乘以 10 的幂，越界即报错），再做整数加减。
溢出检测用符号位：两个同号数相加得到异号结果、或异号相减后结果与被减数异号，即为溢出，只多几次位运算。
scale 相同时省去对齐，速度最快。

**示例**

```python
>> > FixedArray("1.10") + FixedArray("0.005")
FixedArray(1.105, scale=3)
>> > 0.1 + 0.2  # 浮点误差
0.30000000000000004
>> > FixedArray(0.1) + 0.2  # float 操作数按最短写法解析
FixedArray(0.3, scale=1)
>> > FixedArray(["100.00", "50.00"]) - FixedArray("0.01")
FixedArray([99.99, 49.99], scale=2)
>> > 10 - FixedArray("0.01")  # 反向运算
FixedArray(9.99, scale=2)
>> > FixedArray([[1], [2]]) + FixedArray([10, 20])  # 广播
FixedArray([[11, 21],
            [12, 22]], scale=0)
>> > np.array([1, 2]) + FixedArray("0.5")  # ndarray 在左侧也返回 FixedArray
FixedArray([1.5, 2.5], scale=1)
>> > FixedArray(1) + "1"
Traceback(most
recent
call
last):
...
TypeError: unsupported
operand
type(s)
for +: 'FixedArray' and 'str'
```

对齐 scale 也可能溢出——`10**17` 在 scale 2 下的尾数是 `10**19`：

```python
>> > FixedArray(10 ** 17) + FixedArray("0.01")
Traceback(most
recent
call
last):
...
OverflowError: 乘以
10 ^ 2
后超出
int64
范围
```

**性能对比**

<!-- bench:add,add_align,sub -->
*add（同 scale）*，对照代码：`x + y`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    4.32 µs |       72.5 ns |       264 ns |
|     1,000 |    5.61 µs |        498 ns |      79.4 µs |
| 1,000,000 |    2.95 ms |        572 µs |      92.7 ms |

*add（scale 2 + scale 4，需对齐）*，对照代码：`x + z`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    8.84 µs |       72.5 ns |       272 ns |
|     1,000 |    11.5 µs |        501 ns |        88 µs |
| 1,000,000 |    4.06 ms |        397 µs |       103 ms |

*sub*，对照代码：`x - y`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    4.25 µs |       71.6 ns |       261 ns |
|     1,000 |    5.64 µs |        495 ns |      78.7 µs |
| 1,000,000 |    3.06 ms |        408 µs |      90.7 ms |

<!-- /bench -->

**另请参阅**

- [`FixedArray.sum`](#fixedarraysum) – 沿轴求和。

### a * b（精确乘法）

```python
a * b
```

逐元素精确乘法，不做任何舍入。

**参数**

- **b** : *FixedArray 或数值 array_like* — 乘数，可广播。

**返回值**

- **out** : *FixedArray* — `scale = a.scale + b.scale`。

**异常**

- **TypeError** — `b` 不是合法的数值操作数（布尔数组也不行，需先 `.astype(np.int64)`）。
- **OverflowError** — 精确乘积超出 `int64`。

**机制说明**

尾数直接相乘、`scale` 相加，等价于 `a.mul_round(b, rounding=None)`。当数组整体的最坏上界 `max(abs(a)) · max(abs(b))` 超出
`int64`
（例如大数与小数分布在不同下标位置）时，自动切换至 Base 10⁹ 分块长乘法，仅当同一位置的真实乘积 `a[i]·b[i]` 超出 `int64`
时才报错。
连乘会让 scale 不断累加（`1.01 ** 3` 的 scale 为 6），尾数也随之增大，
链式计算中应在适当位置用 [`mul_round`](#fixedarraymul_round) 或 [`rescale`](#fixedarrayrescale) 收敛精度。

**示例**

```python
>> > FixedArray("19.99") * 3
FixedArray(59.97, scale=2)
>> > FixedArray("1.25") * FixedArray("0.5")
FixedArray(0.625, scale=3)
>> > FixedArray("1.1") * 1.1
FixedArray(1.21, scale=2)
>> > 1.1 * 1.1
1.2100000000000002
>> > FixedArray("1.01") * FixedArray("1.01") * FixedArray("1.01")  # scale 累加
FixedArray(1.030301, scale=6)
>> > FixedArray(10 ** 10) * FixedArray(10 ** 10)
Traceback(most
recent
call
last):
...
OverflowError: 乘法结果超出
int64
范围
```

**性能对比**

<!-- bench:mul -->
对照代码：`x * r`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    4.71 µs |       72.2 ns |       267 ns |
|     1,000 |    5.73 µs |        498 ns |      79.7 µs |
| 1,000,000 |     995 µs |        395 µs |      89.7 ms |

<!-- /bench -->

**另请参阅**

- [`FixedArray.mul_round`](#fixedarraymul_round) – 乘法后舍入到指定 scale，中间乘积越界也能算。

### FixedArray.mul_round

```python
FixedArray.mul_round(other, scale=None, rounding="half_up")
```

融合乘法与舍入（Fused Multiply-Round）：先求精确乘积，再 **只舍入一次**到目标 `scale`。

**参数**

- **other** : *FixedArray 或数值 array_like* — 乘数，可广播。
- **scale** : *int，可选* — 结果的小数位数。`rounding=None` 时默认为 `self.scale + other.scale`（精确乘积的自然精度），
  否则默认为 `max(self.scale, other.scale)`。
- **rounding** : *RoundingMode、str 或 None，默认 `"half_up"`* — 舍入模式，见 [`RoundingMode`](#roundingmode)。
  传 `None` 表示不舍入：此时 `scale` 不能小于自然精度，大于自然精度时尾数补零。

**返回值**

- **out** : *FixedArray* — 广播后的新数组，`scale` 如上。

**异常**

- **TypeError** — `other` 不是合法的数值操作数。
- **ValueError** — `rounding` 无效；`scale` 不是非负整数；`rounding=None` 时 `scale` 小于自然精度。
- **OverflowError** — 舍入后的结果超出 `int64`。

**机制说明**

- **快路径**：`max(abs(a)) · max(abs(b))` 在 `int64` 内且无需放大时，执行一次 `int64` 乘法加一次带余除法取整。
- **分块慢路径**：中间乘积超出 `int64` 时，把两端尾数的绝对值拆成 3 个 10⁹ 进制块（单块乘积 < 10¹⁸，不会溢出）
  做卷积长乘法，再直接在十进制块上截取、判定余数并舍入。因此只要 **最终舍入结果**在 `int64` 内就能精确算出，
  不会假溢出，也不退化为 Python 大整数。这在“大额本金 × 高精度汇率/费率”场景中很常见。

**示例**

```python
>> > amount = FixedArray("100.00")
>> > rate = FixedArray("0.0325")
>> > amount.mul_round(rate)  # 默认 scale = max(2, 4)
FixedArray(3.2500, scale=4)
>> > amount.mul_round(rate, scale=2)
FixedArray(3.25, scale=2)
>> > FixedArray("2.5").mul_round(FixedArray("0.5"), scale=1)  # 1.25 → half_up
FixedArray(1.3, scale=1)
>> > FixedArray("2.5").mul_round(FixedArray("0.5"), scale=1, rounding="half_even")  # 1.25 → half_even
FixedArray(1.2, scale=1)
>> > FixedArray("1.25").mul_round(2, scale=5, rounding=None)  # 不舍入，扩大 scale
FixedArray(2.50000, scale=5)
```

中间乘积超出 `int64`、但结果在范围内时，`*` 报错而 `mul_round` 给出正确答案：

```python
>> > notional = FixedArray("12345678901.123456")  # 尾数约 1.2e16
>> > fx = FixedArray("1.234567")  # 尾数约 1.2e6，乘积尾数约 1.5e22
>> > notional * fx
Traceback(most
recent
call
last):
...
OverflowError: 乘法结果超出
int64
范围
>> > notional.mul_round(fx, scale=6)  # 精确值 15241567763.923281703552
FixedArray(15241567763.923282, scale=6)
>> > FixedArray("1.25").mul_round(FixedArray("1.5"), scale=1, rounding=None)
Traceback(most
recent
call
last):
...
ValueError: rounding = None
时不支持缩小
scale（需要指定舍入模式）
```

**性能对比**

<!-- bench:mul_round,mul_round_slow -->
*mul_round（scale=2, half_even）*，对照代码：`np.round(x * r, 2)`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    18.4 µs |       2.35 µs |       345 ns |
|     1,000 |    30.7 µs |       2.53 µs |       167 µs |
| 1,000,000 |    14.4 ms |       1.42 ms |       179 ms |

*mul_round 慢路径（尾数积超出 int64）*，对照代码：`np.round(w * r, 6)`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    18.7 µs |       2.33 µs |       348 ns |
|     1,000 |     143 µs |       2.55 µs |       167 µs |
| 1,000,000 |    70.4 ms |       1.28 ms |       181 ms |

<!-- /bench -->

**另请参阅**

- [`a * b`](#a--b精确乘法) — 不舍入的精确乘法（`scale` 相加）。
- [`FixedArray.matmul`](#fixedarraymatmul) — 矩阵乘法融合舍入。
- [`FixedArray.rescale`](#fixedarrayrescale) — 单独调整已有数组的 `scale`。
- [`FixedArray._chunked_mul_round`](#fixedarray_chunked_mul_round)、[`_divmod_round`](#fixedarray_divmod_round) —
  底层分块乘法与带余除法舍入内核。

### a / b（真除法）

```python
a / b
```

逐元素除法，商保留 `max(a.scale, b.scale)` 位小数，按 `half_up` 舍入。

**参数**

- **b** : *FixedArray 或数值 array_like* — 除数，可广播，不能含 0。

**返回值**

- **out** : *FixedArray* — `scale = max(a.scale, b.scale)`。

**异常**

- **TypeError** — `b` 不是合法的数值操作数。
- **ZeroDivisionError** — 除数含 0。
- **OverflowError** — 商超出 `int64`。

**机制说明**

等价于 `a.divide(b)`，需要其他精度或舍入模式时请直接调用 [`divide`](#fixedarraydivide)。

> **注意**：两个整数相除得到的也是整数精度（scale 0）。`FixedArray(1) / 3` 的结果是 `0`，不是 `0.333…`。
> 需要小数时请让被除数带上 scale（如 `FixedArray("1.00")`），或使用 `divide(..., scale=n)`。

**示例**

```python
>> > FixedArray("10.00") / 3
FixedArray(3.33, scale=2)
>> > FixedArray(["10.00", "20.00", "30.00"]) / FixedArray([3, 6, 7])
FixedArray([3.33, 3.33, 4.29], scale=2)
>> > 1 / FixedArray("4.00")
FixedArray(0.25, scale=2)
>> > FixedArray(1) / 3  # 整数 / 整数 → scale 0
FixedArray(0, scale=0)
>> > FixedArray(1) / 0
Traceback(most
recent
call
last):
...
ZeroDivisionError: 除数包含
0
```

**性能对比**

对照代码是不舍入的 `x / y`；与同样舍入的 `np.round(x / y, n)` 对比见 [`divide`](#fixedarraydivide)。

<!-- bench:truediv -->
对照代码：`x / y`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    28.7 µs |         72 ns |       379 ns |
|     1,000 |    46.9 µs |        515 ns |       223 µs |
| 1,000,000 |    21.5 ms |        585 µs |       235 ms |

<!-- /bench -->

**另请参阅**

- [`FixedArray.divide`](#fixedarraydivide)

### FixedArray.divide

```python
FixedArray.divide(other, scale=None, rounding="half_up")
```

定点数除法：商保留 `scale` 位小数，按 `rounding` 精确舍入。

**参数**

- **other** : *FixedArray 或数值 array_like* — 除数，可广播，不能含 0。
- **scale** : *int，可选* — 商的小数位数，默认 `max(self.scale, other.scale)`。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式，六种均支持，见 [`RoundingMode`](#roundingmode)。

**返回值**

- **out** : *FixedArray* — 广播后的新数组，`scale` 如上。

**异常**

- **TypeError** — `other` 不是合法的数值操作数。
- **ValueError** — `rounding` 无效或 `scale` 不是非负整数。
- **ZeroDivisionError** — 除数含 0。
- **OverflowError** — 商超出 `int64`。

**机制说明**

商的尾数等于 `round(a.mantissa × 10^k / b.mantissa)`，其中 `k = scale + b.scale - a.scale`，按 `k` 的符号分三种情况：

- **k > 0（需要更多小数位）**：分段长除法。先做带余除法，再把 `10^k` 分段乘进余数，每段位数保证 `|余数|·10^c`
  不越界；当除数绝对值超过 `int64` 上限的 1/10 时，普通 `r·10` 会溢出，自动切换至无溢出二进制倍增内核
  （将余数与除数转为补码 `uint64`，利用 `10r = 8r + 2r` 且 `2r < 2·abs(d) ≤ 2**64` 逐位推进）。每段与终局进位均做精确的商越界检查，因此只有
  **商本身**超出 `int64` 才会报错。
- **k = 0**：一次带余除法加舍入判定（此时 `-2**63 / -1 = 2**63` 超出 `int64`，抛出 `OverflowError`）。
- **k < 0（需要更少小数位）**：先做带余除法得到整数商 `q` 与余数 `r`，再将 `(q + r/d)` 整体除以 `10^|k|` 并按 `rounding`
  一次性舍入（支持任意大的 `|k|`，包括 `|k| > 18`），不会因放大除数而假溢出。对于被除数尾数为 `-2**63` 且除数尾数为 `-1`
  的边界情形，由于继续除以 `10^|k|`（`|k| ≥ 1`）后结果已在 `int64` 范围内，内部通过翻转 `floor` / `ceil` 方向在负数域完成精确舍入后再取反，避免
  `np.divmod(-2**63, -1)` 硬件溢出。

**示例**

```python
>> > FixedArray("10").divide(3, scale=4)
FixedArray(3.3333, scale=4)
>> > FixedArray(2).divide(3, scale=2, rounding="down")
FixedArray(0.66, scale=2)
>> > FixedArray(2).divide(3, scale=2, rounding="up")
FixedArray(0.67, scale=2)
>> > FixedArray(-1).divide(8, scale=2)  # -0.125，half_up 远离零
FixedArray(-0.13, scale=2)
>> > FixedArray(-1).divide(8, scale=2, rounding="half_even")
FixedArray(-0.12, scale=2)
```

按 USD/CNY 汇率 7.1234 把人民币金额折算为美元（银行家舍入到分）：

```python
>> > cny = FixedArray(["100.00", "250.00"])
>> > cny.divide(FixedArray("7.1234"), scale=2, rounding="half_even")
FixedArray([14.04, 35.10], scale=2)
```

被除数或除数尾数接近 `int64` 上限时也能精确计算：

```python
>> > FixedArray("92233720368.54775807").divide(FixedArray("0.00000001"), scale=0)
FixedArray(9223372036854775807, scale=0)
>> > FixedArray(1).divide(FixedArray(2 ** 62), scale=20)  # 除数 > int64 上限 / 10
FixedArray(0.00000000000000000022, scale=20)
```

**性能对比**

<!-- bench:divide -->
对照代码：`np.round(x / y, 6)`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    28.9 µs |       2.32 µs |       377 ns |
|     1,000 |    46.7 µs |       2.58 µs |       222 µs |
| 1,000,000 |    21.3 ms |       1.22 ms |       234 ms |

<!-- /bench -->

**另请参阅**

- [`a / b`](#a--b真除法)
- [`FixedArray.mean`](#fixedarraymean)
- [`FixedArray._chunked_divmod_round`](#fixedarray_chunked_divmod_round)、[
  `_doubling_step_10r`](#fixedarray_doubling_step_10r)、[`_round_div_pow10`](#fixedarray_round_div_pow10) —
  底层的分段长除、二进制倍增与精度缩小舍入函数。

### a // b、a % b 与 divmod (a, b)（整除与取余）

```python
a // b
a % b
divmod(a, b)
```

向下取整的商、与除数同号的余数，以及两者的组合。

**参数**

- **b** : *FixedArray 或数值 array_like* — 除数，可广播，不能含 0。

**返回值**

- **q** : *int 或 ndarray（int64）* — `a // b` 的结果；两个都是 0 维时为 Python `int`。
- **r** : *FixedArray* — `a % b` 的结果，`scale = max(a.scale, b.scale)`。
- `divmod(a, b)` 返回 `(q, r)`，只做一次除法。

**异常**

- **TypeError** — `b` 不是合法的数值操作数。
- **ZeroDivisionError** — 除数含 0。
- **OverflowError** — `-2**63` 除以 `-1`（仅 `//` 与 `divmod`）。

**机制说明**

两个操作数先对齐到相同 scale，再对尾数做整数带余除法，语义与 Python / NumPy 的 `//`、`%` 一致：
商向负无穷取整，余数与除数同号，满足 `a == (a // b) * b + a % b`。
商是“包含几个 b”的 **无量纲计数**，所以返回整数而不是 `FixedArray`；余数与 `a`、`b` 同量纲，所以仍是 `FixedArray`。
需要“对齐到 b 的整数倍”时请用 [`round_to_step`](#fixedarrayround_to_step)。

**示例**

```python
>> > FixedArray("7.5") // 2
3
>> > FixedArray("-7.5") // 2  # 向负无穷
-4
>> > FixedArray(["7.5", "8.5"]) // FixedArray("2.5")
array([3, 3])
>> > FixedArray("7.5") % 2
FixedArray(1.5, scale=1)
>> > FixedArray("-7.5") % 2  # 与除数同号
FixedArray(0.5, scale=1)
>> > FixedArray("7.5") % -2
FixedArray(-0.5, scale=1)
>> > divmod(FixedArray("1000.00"), FixedArray("3.30"))  # 1000 元能买几份 3.30 元的商品、找零多少
(303, FixedArray(0.10, scale=2))
>> > a, b = FixedArray("-7.5"), FixedArray(2)
>> > bool((a // b) * b + a % b == a)
True
```

**性能对比**

NumPy 的 float64 `floor_divide` / `remainder` 本身较慢，这里 `FixedArray` 在大数组上反而更快。

<!-- bench:floordiv,mod -->
*floordiv（`//`）*，对照代码：`x // y`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    6.13 µs |       90.4 ns |       268 ns |
|     1,000 |    7.36 µs |       10.5 µs |      80.6 µs |
| 1,000,000 |    2.69 ms |       23.5 ms |      89.3 ms |

*mod（`%`）*，对照代码：`x % y`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    3.87 µs |       86.9 ns |       268 ns |
|     1,000 |    5.29 µs |       9.82 µs |      80.8 µs |
| 1,000,000 |    2.37 ms |       22.4 ms |      89.9 ms |

<!-- /bench -->

**另请参阅**

- [`FixedArray.round_to_step`](#fixedarrayround_to_step)

## 矩阵运算

### a @ b（精确矩阵乘法）

```python
a @ b
```

精确矩阵乘法，不做舍入。

**参数**

- **b** : *FixedArray 或数值 array_like* — 右操作数。`ndarray` 或列表在左侧时（`ndarray @ a`）同样返回 `FixedArray`。

**返回值**

- **out** : *FixedArray* — `scale = a.scale + b.scale`。

**异常**

- **TypeError** — `b` 不是合法的数值操作数。
- **ValueError** — 形状不匹配或存在 0 维操作数（由 NumPy 报出）。
- **OverflowError** — 结果真实超出 `int64`。

**机制说明**

等价于 `a.matmul(b, rounding=None)`。形状规则与 `numpy.matmul` 完全相同：1 维 × 1 维为内积，2 维为矩阵乘法，
更高维按批处理广播。快慢双路径见 [`FixedArray.matmul`](#fixedarraymatmul)。

**示例**

两个组合（行）持有三种资产（列）的数量，乘以资产价格得到组合市值：

```python
>> > qty = FixedArray([[100, 200, 0],
                       ...[0, 50, 300]])
>> > price = FixedArray(["10.25", "3.10", "99.99"])
>> > qty @ price
FixedArray([1645.00, 30152.00], scale=2)
>> > FixedArray([["1.5", "2"], ["0", "1"]]) @ FixedArray([[2, 0], [1, 1]])
FixedArray([[5.0, 2.0],
            [1.0, 1.0]], scale=1)
>> > batch = FixedArray(np.ones((4, 2, 3), dtype=np.int64))
>> > (batch @ FixedArray(np.ones((3, 5), dtype=np.int64))).shape
(4, 2, 5)
```

结果越界时报错，而 NumPy 的 `int64` 矩阵乘法静默回绕；逐项乘积很大但正负相抵时，能给出正确结果：

```python
>> > x = 3_037_000_499  # x**2 ≈ 9.22e18，刚好在 int64 内
>> > FixedArray([[x, x]]) @ FixedArray([[x], [x]])
Traceback(most
recent
call
last):
...
OverflowError: 矩阵乘法结果超出
int64
范围
>> > np.array([[x, x]]) @ np.array([[x], [x]])  # 错误结果，没有任何警告
array([[-11857053614]])
>> > FixedArray([[x, x]]) @ FixedArray([[x], [-x]])
FixedArray([[0]], scale=0)
```

**另请参阅**

- [`FixedArray.matmul`](#fixedarraymatmul) – 矩阵乘法 + 一次舍入，含性能对比。
- [`a * b`](#a--b精确乘法) – 逐元素乘法。

### FixedArray.matmul

```python
FixedArray.matmul(other, scale=None, rounding="half_up")
```

矩阵乘法与可选的一次舍入（Fused Matmul-Round）。

**参数**

- **other** : *FixedArray 或数值 array_like* — 右操作数，形状规则同 `numpy.matmul`。
- **scale** : *int，可选* — 结果的小数位数。`rounding=None` 时默认为 `self.scale + other.scale`，
  否则默认为 `max(self.scale, other.scale)`。
- **rounding** : *RoundingMode、str 或 None，默认 `"half_up"`* — 舍入模式。传 `None` 表示不舍入，此时 `scale` 不能小于自然精度。

**返回值**

- **out** : *FixedArray* — `scale` 如上。

**异常**

- **TypeError** — `other` 不是合法的数值操作数。
- **ValueError** — `rounding` 无效；`rounding=None` 时 `scale` 小于自然精度；形状不匹配。
- **OverflowError** — 精确内积或舍入后的结果超出 `int64`。

**机制说明**

先计算精确内积（scale 为 `self.scale + other.scale`），再一次性舍入到目标 `scale`。

- **快路径**：`max(abs(a)) · max(abs(b)) · K`（`K` 为内积长度）不超过 `int64` 时直接调用 `np.matmul`。
  NumPy 的整数矩阵乘法不走 BLAS，比 float64 矩阵乘法慢得多，这是大矩阵上与 float 差距的主要来源。
- **分块慢路径**：否则把每个尾数按二进制拆成 3 个 21 位块，做 9 次块矩阵乘法后逐级进位合并，全程 `int64`；
  块乘积上界为 2⁴²，设计上的内积长度上限约为 2²⁰。
- 与 `mul_round` 不同，`matmul` 要求 **精确内积**本身能放进 `int64`（之后才舍入）。

**示例**

```python
>> > w = FixedArray(["0.333", "0.333", "0.334"])  # 权重，scale 3
>> > v = FixedArray(["100.01", "200.02", "300.03"])  # 价格，scale 2
>> > w @ v  # 精确，scale 5
FixedArray(200.12001, scale=5)
>> > w.matmul(v)  # 默认 scale = max(3, 2)
FixedArray(200.120, scale=3)
>> > w.matmul(v, scale=2)
FixedArray(200.12, scale=2)
>> > w.matmul(v, scale=6, rounding=None)  # 不舍入，扩大 scale
FixedArray(200.120010, scale=6)
```

**性能对比**

<!-- bench:matmul_fast,matmul_slow -->
*matmul（`@`，快路径）*，对照代码：`a @ b`

|     n×n | FixedArray | NumPy float64 | Decimal 循环 |
|--------:|-----------:|--------------:|-------------:|
|   10×10 |    5.93 µs |        762 ns |            — |
| 100×100 |     500 µs |        590 µs |            — |
| 500×500 |    46.5 ms |       7.49 ms |            — |

*matmul（`@`，分块慢路径）*，对照代码：`a @ b`

|     n×n | FixedArray | NumPy float64 | Decimal 循环 |
|--------:|-----------:|--------------:|-------------:|
| 100×100 |    4.23 ms |        158 µs |            — |
| 500×500 |     426 ms |        6.8 ms |            — |

<!-- /bench -->

**另请参阅**

- [`a @ b`](#a--b精确矩阵乘法)
- [`FixedArray.mul_round`](#fixedarraymul_round) – 逐元素版本。
- [`FixedArray._chunked_matmul`](#fixedarray_chunked_matmul) – 底层 Base 2²¹ 分块精确矩阵乘法内核。

## 规约统计

`np.sum(a)` 与 `np.mean(a)` 会自动转发到下面两个方法；其他 NumPy
规约操作（如最值、累计和）的定点数写法见 [NumPy 互操作机制](#numpy-互操作机制)。

### FixedArray.sum

```python
FixedArray.sum(axis=None, keepdims=False, out=None, dtype=None)
```

沿指定轴精确求和。

**参数**

- **axis** : *None、int 或 int 元组，可选* — 求和的轴，默认对全部元素求和。
- **keepdims** : *bool，默认 False* — 为 `True` 时保留被求和的轴（长度为 1）。
- **out, dtype** : *None* — 仅为兼容 `np.sum` 的调用约定，必须为 `None`；需要 float 请先 [
  `to_float()`](#fixedarrayto_float)。

**返回值**

- **out** : *FixedArray* — `scale` 与原数组相同；对全部元素求和时为 0 维；空数组的和为 0。

**异常**

- **TypeError** — 传入了非 `None` 的 `out` 或 `dtype`。
- **OverflowError** — 求和结果真实超出 `int64`。

**机制说明**

- **快路径**：`max(abs(a)) × 每个结果累加的元素数` 不超过 `int64` 时，直接调用 NumPy 的 `int64` 求和。
- **分块慢路径**：否则把每个尾数拆成 3 个 10⁹ 进制块分别求和，再逐级进位合并，
  能正确处理“部分和越界、最终结果在范围内”的情况，全程不使用 Python 大整数。

**示例**

```python
>> > m = FixedArray([["1.10", "2.20"],
                     ...["3.30", "4.40"]])
>> > m.sum()
FixedArray(11.00, scale=2)
>> > m.sum(axis=0)
FixedArray([4.40, 6.60], scale=2)
>> > m.sum(axis=1, keepdims=True)
FixedArray([[3.30],
            [7.70]], scale=2)
>> > np.sum(m, axis=0)  # 转发到 m.sum(axis=0)
FixedArray([4.40, 6.60], scale=2)
>> > float(np.array([0.1, 0.1, 0.1]).sum())
0.30000000000000004
>> > FixedArray([0.1, 0.1, 0.1]).sum()
FixedArray(0.3, scale=1)
```

溢出一定报错，而不是像 NumPy `int64` 那样回绕：

```python
>> > FixedArray([2 ** 62, 2 ** 62]).sum()
Traceback(most
recent
call
last):
...
OverflowError: 求和结果超出
int64
范围
>> > np.array([2 ** 62, 2 ** 62]).sum()
-9223372036854775808
>> > FixedArray([2 ** 62, 2 ** 62, -2 ** 62]).sum()  # 部分和越界、结果在范围内
FixedArray(4611686018427387904, scale=0)
```

**性能对比**

<!-- bench:sum,sum_slow -->
*sum*，对照代码：`np.sum(x)`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|     1,000 |    4.36 µs |        833 ns |      30.1 µs |
| 1,000,000 |     304 µs |        189 µs |      31.7 ms |

*sum 慢路径（尾数接近 int64 上限）*，对照代码：`np.sum(big)`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|     1,000 |    21.6 µs |        831 ns |      30.8 µs |
| 1,000,000 |    9.15 ms |        189 µs |        33 ms |

<!-- /bench -->

**另请参阅**

- [`FixedArray.mean`](#fixedarraymean)
- [`FixedArray._chunked_sum`](#fixedarray_chunked_sum) – 底层 Base 10⁹ 分块精确求和内核。

### FixedArray.mean

```python
FixedArray.mean(axis=None, keepdims=False, scale=None, rounding="half_up", out=None, dtype=None)
```

沿指定轴求算术平均值，按 `rounding` 精确舍入到 `scale` 位小数。

**参数**

- **axis** : *None、int 或 int 元组，可选* — 求均值的轴，默认对全部元素。
- **keepdims** : *bool，默认 False* — 为 `True` 时保留被规约的轴。
- **scale** : *int，可选* — 结果的小数位数，默认与原数组相同。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式。
- **out, dtype** : *None* — 仅为兼容 `np.mean` 的调用约定，必须为 `None`。

**返回值**

- **out** : *FixedArray* — `scale` 为参数 `scale`（默认与原数组相同）。

**异常**

- **TypeError** — 传入了非 `None` 的 `out` 或 `dtype`；`axis` 类型不正确。
- **ValueError** — `rounding` 无效；`scale` 不是非负整数；`axis` 元组中有重复的轴。
- **AxisError** — `axis` 越界（NumPy 的 `AxisError`）。
- **ZeroDivisionError** — 参与平均的元素个数为 0。
- **OverflowError** — 提升 scale 后的均值超出 `int64`。

**机制说明**

先对每个元素做带余除法 `m = q·n + r`（`n` 为元素个数），分别累加商 `q_sum` 与余数 `r_sum`，再把 `r_sum // n` 合并回 `q_sum`
（利用 `int64` 模运算性质，即使正负元素混合时局部 `q_sum` 回绕，与 `r_sum // n` 相加后仍能精确还原在 `int64`
范围内的真实商总和，且全程不触发溢出警告）。
结果 `scale` 小于原 `scale` 时，将整数商与余数一起除以 `10^k` 并一次性舍入（支持 `k > 18` 的任意位数）；大于原 `scale`
时，把余数分段放大继续长除。

> **注意**：默认 `scale` 与原数组相同。整数数组（scale 0）的均值会被舍入为整数：
> `FixedArray([1, 2, 3, 4]).mean()` 得到 `3`（2.5 按 half_up）。需要小数请传 `scale=`。

**示例**

```python
>> > b = FixedArray(["1.00", "2.00", "4.00"])
>> > b.mean()  # 7/3 = 2.333…，保留 2 位
FixedArray(2.33, scale=2)
>> > b.mean(scale=4)
FixedArray(2.3333, scale=4)
>> > b.mean(scale=1, rounding="down")
FixedArray(2.3, scale=1)
>> > c = FixedArray([1, 2, 3, 4])
>> > c.mean()  # 2.5 → half_up
FixedArray(3, scale=0)
>> > c.mean(rounding="half_even")
FixedArray(2, scale=0)
>> > c.mean(scale=1)
FixedArray(2.5, scale=1)
>> > d = FixedArray([[1, 2], [3, 4]])
>> > d.mean(axis=1, scale=1)
FixedArray([1.5, 3.5], scale=1)
>> > np.mean(d, axis=1)  # 注意：scale 仍为 0
FixedArray([2, 4], scale=0)
>> > FixedArray(np.array([], dtype=np.int64)).mean()
Traceback(most
recent
call
last):
...
ZeroDivisionError: 空数组没有均值
```

**性能对比**

<!-- bench:mean -->
对照代码：`np.round(np.mean(x), 4)`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|     1,000 |    22.6 µs |       4.67 µs |      30.9 µs |
| 1,000,000 |    5.79 ms |        202 µs |      31.9 ms |

<!-- /bench -->

**另请参阅**

- [`FixedArray.sum`](#fixedarraysum)
- [`FixedArray.divide`](#fixedarraydivide)
- [`FixedArray._round_quotient`](#fixedarray_round_quotient)、[`_round_div_pow10`](#fixedarray_round_div_pow10) –
  底层商余舍入函数。

## 精度调整与网格对齐

### FixedArray.rescale

```python
FixedArray.rescale(new_scale, rounding="half_up")
```

把小数位数调整为 `new_scale`。

**参数**

- **new_scale** : *int* — 新的小数位数，必须是非负整数。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 缩小 scale 时的舍入模式。扩大或不变时不起作用，但仍会前置校验其合法性。

**返回值**

- **out** : *FixedArray* — `scale = new_scale`，形状不变。

**异常**

- **ValueError** — `new_scale` 不是非负整数；`rounding` 无效。
- **OverflowError** — 扩大 scale 后非零尾数超出 `int64`。

**机制说明**

- **扩大（`new_scale > scale`）**：全零数组直接返回对应 `new_scale` 的零数组（支持任意大位数，如 `scale=30`）；含非零元素时先用最大绝对值校验
  `max(abs(m)) · 10^k` 是否越界，再将尾数乘以 `10^k`。
- **缩小（`new_scale < scale`）**：设缩小位数 `k = scale - new_scale`。`k ≤ 18` 时用带余除法判定余数与半数关系；`k = 19` 时因
  `|m| < 2**63 < 10**19`，直接与半数 `5×10**18` 比较；`k ≥ 20` 时真实绝对值严格小于 `0.5`，仅根据符号和是否非零决定进位，全程无需
  Python 大整数。
- **相等（`new_scale == scale`）**：校验 `rounding` 合法后直接返回 `self`（零拷贝）。

`np.round` 在 float 上舍入，结果受二进制表示影响而不稳定（`1.005` 被舍成 `1.0`，`2.675` 却被舍成 `2.68`），
且只支持“四舍六入五成双”。`rescale` 在十进制尾数上舍入，结果确定，并支持六种模式。

**示例**

```python
>> > a = FixedArray("1.25")
>> > a.rescale(4)
FixedArray(1.2500, scale=4)
>> > a.rescale(1)
FixedArray(1.3, scale=1)
>> > a.rescale(1, rounding="half_even")
FixedArray(1.2, scale=1)
>> > a.rescale(2) is a
True
>> > FixedArray(["1.005", "2.675", "-0.125"]).rescale(2)
FixedArray([1.01, 2.68, -0.13], scale=2)
>> > np.round([1.005, 2.675, -0.125], 2)
array([1., 2.68, -0.12])
>> > FixedArray("0.5000000000000000000").rescale(0)  # 一次缩小 19 位
FixedArray(1, scale=0)
>> > FixedArray(10 ** 17).rescale(2)
Traceback(most
recent
call
last):
...
OverflowError: 乘以
10 ^ 2
后超出
int64
范围
```

**性能对比**

<!-- bench:rescale -->
对照代码：`np.round(r, 2)`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    10.7 µs |       2.32 µs |       300 ns |
|     1,000 |    27.6 µs |       2.09 µs |       131 µs |
| 1,000,000 |    12.1 ms |       1.09 ms |       139 ms |

<!-- /bench -->

**另请参阅**

- [`FixedArray(values, scale=…)`](#fixedarrayvalues-scalenone) – 构造时指定 scale。
- [`FixedArray.round_to_step`](#fixedarrayround_to_step) – 舍入到任意步长，而不只是 10 的幂。
- [`FixedArray._mul_pow10`](#fixedarray_mul_pow10)、[`_round_div_pow10`](#fixedarray_round_div_pow10) – 底层放大与缩小舍入实现。

### FixedArray.round_to_step

```python
FixedArray.round_to_step(step, rounding="half_up")
```

把每个元素舍入到最近的 `step` 整数倍上。

**参数**

- **step** : *FixedArray、int、float、Decimal 或数值 array_like* — 正的步长，可以是标量，也可以是与 `self`
  可广播的数组（每个元素使用不同步长）。不接受字符串，请用 `FixedArray("0.05")` 包一层。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 对商 `a / step` 取整的方式，六种均支持；
  例如数量向下取整到整手用 `"floor"`。

**返回值**

- **out** : *FixedArray* — `scale = max(self.scale, step.scale)`，形状为广播后的形状。

**异常**

- **TypeError** — `step` 为字符串或不受支持的类型。
- **ValueError** — `step` 含有非正数；`rounding` 无效。
- **OverflowError** — 结果超出 `int64`。

**机制说明**

两者对齐 scale 后，对尾数计算 `round(a / step)`（整数带余除法 + 舍入判定），再精确乘回 `step`。
典型用途：最小变动价位（tick size）、整手交易单位（lot size）、现金面额（如 0.05 元）。

**示例**

```python
>> > px = FixedArray(["10.02", "10.03", "10.07"])
>> > px.round_to_step(FixedArray("0.05"))
FixedArray([10.00, 10.05, 10.05], scale=2)
>> > px.round_to_step(0.05)  # float 步长按最短写法解析为 0.05
FixedArray([10.00, 10.05, 10.05], scale=2)
>> > FixedArray([150, 249, 251]).round_to_step(100, rounding="floor")  # 向下取整到 100 股一手
FixedArray([100, 200, 200], scale=0)
>> > FixedArray(["10.02", "100.3"]).round_to_step(FixedArray(["0.05", "0.5"]))  # 每个元素不同步长
FixedArray([10.00, 100.50], scale=2)
>> > FixedArray(["0.125", "0.375"]).round_to_step(FixedArray("0.25"), rounding="half_even")
FixedArray([0.000, 0.500], scale=3)
>> > px.round_to_step("0.05")
Traceback(most
recent
call
last):
...
TypeError: FixedArray
只能与数值类型运算、比较或对齐，收到
str
>> > px.round_to_step(0)
Traceback(most
recent
call
last):
...
ValueError: 步长
step
必须为正数
```

**性能对比**

<!-- bench:round_to_step -->
对照代码：`np.round(x / 0.05) * 0.05`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    23.8 µs |       1.11 µs |       447 ns |
|     1,000 |    34.4 µs |       2.23 µs |       280 µs |
| 1,000,000 |    14.5 ms |        856 µs |       293 ms |

<!-- /bench -->

**另请参阅**

- [`FixedArray.is_on_step`](#fixedarrayis_on_step)
- [`FixedArray.rescale`](#fixedarrayrescale)

### FixedArray.is_on_step

```python
FixedArray.is_on_step(step)
```

逐元素判断是否恰好落在 `step` 的整数倍上。

**参数**

- **step** : *FixedArray、int、float、Decimal 或数值 array_like* — 正的步长，可广播，不接受字符串。

**返回值**

- **out** : *bool 或 ndarray of bool* — `self` 与 `step` 都是 0 维时返回 Python `bool`，否则返回布尔数组。

**异常**

- **TypeError** — `step` 为字符串或不受支持的类型。
- **ValueError** — `step` 含有非正数。

**机制说明**

对齐 scale 后判断尾数余数是否为 0，是精确的整除判定。用 float 取余做同样的判断会因二进制误差而失败。

**示例**

```python
>> > FixedArray(["10.05", "10.07"]).is_on_step(FixedArray("0.05"))
array([True, False])
>> > FixedArray("10.05").is_on_step(0.05)
True
>> > FixedArray("-10.05").is_on_step(FixedArray("0.05"))
True
>> > 10.05 % 0.05 == 0  # float 取余的误差
False
```

**另请参阅**

- [`FixedArray.round_to_step`](#fixedarrayround_to_step)

## 比较运算

### a == b 与 a != b（相等性比较）

```python
a == b
a != b
```

逐元素判断数值是否相等。

**参数**

- **b** : *FixedArray 或数值 array_like* — 另一个操作数，可广播。

**返回值**

- **out** : *numpy.bool_ 或 ndarray of bool* — 两个都是 0 维时返回 NumPy 布尔标量，否则返回布尔数组。

**异常**

- **ValueError** — 形状无法广播。
- **OverflowError** — 对齐到共同 scale 时尾数越界（例如 `FixedArray(10**17) == FixedArray("0.01")`）。

**机制说明**

两个操作数对齐到相同 scale 后比较整数尾数，结果精确：`FixedArray("1.10") == FixedArray("1.1")` 为真，
不需要 `np.isclose` 式的容差。

- `b` 不是数值类型（`str`、`bool`、`None` 等）时，`==` 返回 `False`、`!=` 返回 `True`，与 Python 默认语义一致，不报错。
- `b` 是数值类型但无法表示为 `FixedArray`（NaN、无穷大、超出 `int64` 的整数）时，同样视为不相等。
- 数组的比较结果是布尔数组，不能直接用于 `if`，请使用 `.all()` / `.any()`。

**示例**

```python
>> > bool(FixedArray("1.10") == FixedArray("1.1"))
True
>> > FixedArray(["1.5", "2.0"]) == FixedArray([1.5, 2])
array([True, True])
>> > np.array([1, 2]) == FixedArray(["1.0", "2.5"])  # ndarray 在左侧
array([True, False])
>> > FixedArray(1) == "1"
False
>> > FixedArray(1) != "1"
True
>> > FixedArray(1) == True  # bool 不是合法的数值操作数
False
>> > FixedArray(1) == float("nan")
False
>> > pair = FixedArray([1, 2])
>> > if pair == FixedArray([1, 2]):
    ...
pass
Traceback(most
recent
call
last):
...
ValueError: The
truth
value
of
an
array
with more than one element is ambiguous.Use a.any() or a.all()
>> > bool((pair == FixedArray([1, 2])).all())
True
```

**性能对比**

<!-- bench:eq -->
对照代码：`x == z`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |    5.07 µs |       59.4 ns |       246 ns |
|     1,000 |    6.05 µs |        424 ns |      63.9 µs |
| 1,000,000 |    1.66 ms |        276 µs |      63.8 ms |

<!-- /bench -->

**另请参阅**

- [`a < b` 等大小比较](#a--ba--ba--ba--b大小比较)

### a < b、a <= b、a > b、a >= b（大小比较）

```python
a < b
a <= b
a > b
a >= b
```

逐元素比较大小。

**参数**

- **b** : *FixedArray 或数值 array_like* — 另一个操作数，可广播。

**返回值**

- **out** : *numpy.bool_ 或 ndarray of bool*

**异常**

- **TypeError** — `b` 不是合法的数值操作数（如字符串）。
- **ValueError** — 形状无法广播；`b` 含 NaN 等无法表示的数值。
- **OverflowError** — 对齐到共同 scale 时尾数越界。

**机制说明**

与相等性比较相同，对齐 scale 后比较尾数；scale 相同时没有对齐开销，速度与 NumPy 原生比较相当。
比较结果常用于布尔掩码索引。

**示例**

```python
>> > FixedArray(["1", "2", "3"]) > 1.5
array([False, True, True])
>> > FixedArray(["0.1", "0.2"]) <= Decimal("0.1")
array([True, False])
>> > 1.5 < FixedArray(["1", "2"])  # 反向比较
array([False, True])
>> > p = FixedArray(["9.99", "10.01", "12.50"])
>> > p[p > 10]
FixedArray([10.01, 12.50], scale=2)
>> > FixedArray(1) < "1"
Traceback(most
recent
call
last):
...
TypeError: '<'
not supported
between
instances
of
'FixedArray' and 'str'
```

**性能对比**

<!-- bench:lt -->
对照代码：`x < y`

|         N | FixedArray | NumPy float64 | Decimal 循环 |
|----------:|-----------:|--------------:|-------------:|
|  0-d 标量 |     616 ns |       59.1 ns |       245 ns |
|     1,000 |     756 ns |        426 ns |      64.2 µs |
| 1,000,000 |     262 µs |        258 µs |      64.4 ms |

<!-- /bench -->

**另请参阅**

- [`a[key]`](#akey索引与切片) – 布尔掩码索引。

## 形状变换、索引与容器协议

形状操作只改变尾数数组的视图，不复制数据，`scale` 保持不变。

### FixedArray.reshape

```python
FixedArray.reshape(*shape)
```

返回同样数据、新形状的 `FixedArray`。

**参数**

- **shape** : *int 或 int 元组* — 新形状，可以写成多个整数 `reshape(2, 3)` 或一个元组 `reshape((2, 3))`；某一维可以为
  `-1`。

**返回值**

- **out** : *FixedArray* — `scale` 不变；尾数是原数组的只读视图。

**异常**

- **ValueError** — 元素总数与新形状不一致。

**机制说明**

`np.reshape(a, shape)` 不可用：NumPy 会以关键字参数 `order=` 调用 `a.reshape`，失败后回退到隐式数组转换而报错。
请使用方法形式 `a.reshape(shape)`。

**示例**

```python
>> > six = FixedArray(["1.1", "2.2", "3.3", "4.4", "5.5", "6.6"])
>> > six.reshape(2, 3)
FixedArray([[1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6]], scale=1)
>> > six.reshape((3, -1)).shape
(3, 2)
>> > six.reshape(4, 2)
Traceback(most
recent
call
last):
...
ValueError: cannot
reshape
array
of
size
6
into
shape(4, 2)
```

**另请参阅**

- [`FixedArray.transpose`](#fixedarraytranspose)

### FixedArray.transpose

```python
FixedArray.transpose(*axes)
```

按给定顺序置换各轴；不传参数时反转全部轴（与 `.T` 相同）。

**参数**

- **axes** : *int 或 int 元组，可选* — 轴的新顺序，可以写成多个整数或一个元组。

**返回值**

- **out** : *FixedArray* — `scale` 不变，与原数组共享尾数。

**异常**

- **ValueError** — `axes` 与维数不匹配。

**机制说明**

`np.transpose(a)` 会转发到 `a.transpose()`，可以直接使用。

**示例**

```python
>> > cube = FixedArray(np.arange(6).reshape(1, 2, 3))
>> > cube.transpose(2, 0, 1).shape
(3, 1, 2)
>> > np.transpose(FixedArray([[1, 2], [3, 4]]))
FixedArray([[1, 3],
            [2, 4]], scale=0)
```

**另请参阅**

- [`FixedArray.T`](#fixedarrayt)

### a[key]（索引与切片）

```python
a[key]
```

支持 NumPy 的全部读取索引方式：整数、切片、`...`、`None`（`np.newaxis`）、整数数组（花式索引）与布尔掩码。

**参数**

- **key** : *int、slice、tuple、ndarray 或比较得到的布尔数组*

**返回值**

- **out** : *FixedArray* — 始终返回 `FixedArray`；取单个元素时得到 0 维 `FixedArray`（而不是 Python 数值）。`scale` 不变。

**异常**

- **IndexError** — 下标越界。
- **TypeError** — 尝试赋值（`a[0] = …`）：`FixedArray` 不可变。

**机制说明**

索引直接作用在尾数数组上，基本索引返回视图、花式索引与布尔索引返回拷贝，与 NumPy 相同。
`FixedArray` 不支持元素赋值；需要“修改”某些元素时，用运算构造新数组。
布尔数组不是合法的数值操作数，用作乘法掩码时需先 `.astype(np.int64)`。

**示例**

```python
>> > g = FixedArray([["1.1", "2.2", "3.3"],
                     ...["4.4", "5.5", "6.6"]])
>> > g[0]
FixedArray([1.1, 2.2, 3.3], scale=1)
>> > g[0, 1]
FixedArray(2.2, scale=1)
>> > g[:, 0]
FixedArray([1.1, 4.4], scale=1)
>> > g[..., -1]
FixedArray([3.3, 6.6], scale=1)
>> > six[::2]
FixedArray([1.1, 3.3, 5.5], scale=1)
>> > six[[0, 2]]
FixedArray([1.1, 3.3], scale=1)
>> > six[six > 3]
FixedArray([3.3, 4.4, 5.5, 6.6], scale=1)
>> > six[0] = 1
Traceback(most
recent
call
last):
...
TypeError: 'FixedArray'
object
does
not support
item
assignment
>> > y = FixedArray(["1.5", "-2.0", "3.0"])
>> > y * (y > 0).astype(np.int64)  # 用掩码把负数清零
FixedArray([1.5, 0.0, 3.0], scale=1)
>> > y * (y > 0)  # 布尔数组不能直接参与运算
Traceback(most
recent
call
last):
...
TypeError: operand
'FixedArray'
does
not support
ufuncs(__array_ufunc__=None)
```

**另请参阅**

- [`a < b` 等大小比较](#a--ba--ba--ba--b大小比较)

### len (a)、bool (a) 与 iter (a)（容器与真值协议）

```python
len(a)
bool(a)
iter(a)
```

第一维长度、真值测试与按第一维迭代，语义与 `ndarray` 一致。

**返回值**

- `len(a)` : *int* — 第一维长度。
- `bool(a)` : *bool* — 恰好 1 个元素时，按该元素是否非零判断。
- `iter(a)` : *iterator of FixedArray* — 依次产出去掉第一维的子数组（1 维时为 0 维 `FixedArray`）。

**异常**

- **TypeError** — 对 0 维 `FixedArray` 调用 `len()` 或迭代。
- **ValueError** — 对元素个数不是 1 的数组（含空数组）做真值测试。

**机制说明**

`bool(a)` 与 NumPy 的规则相同：只有单元素数组能转为布尔值，其余情况报错，防止 `if a:` 写法掩盖“是否全部满足”的意图。
多元素数组请使用 `bool((a > 0).all())` 之类的显式归约。迭代时每个元素都是 `FixedArray`，scale 保持不变。

**示例**

```python
>> > len(g)
2
>> > [row for row in g]
[FixedArray([1.1, 2.2, 3.3], scale=1), FixedArray([4.4, 5.5, 6.6], scale=1)]
>> > list(FixedArray(["1.5", "2.5"]))
[FixedArray(1.5, scale=1), FixedArray(2.5, scale=1)]
>> > bool(FixedArray("0.01")), bool(FixedArray(0)), bool(FixedArray([0]))
(True, False, False)
>> > bool(FixedArray([1, 2]))
Traceback(most
recent
call
last):
...
ValueError: The
truth
value
of
an
array
with more than one element is ambiguous.Use a.any() or a.all()
>> > len(FixedArray("1.5"))
Traceback(most
recent
call
last):
...
TypeError: len()
of
unsized
object
>> > list(FixedArray("1.5"))
Traceback(most
recent
call
last):
...
TypeError: iteration
over
a
0 - d
FixedArray
```

**另请参阅**

- [`FixedArray.shape`](#fixedarrayshape)、[`FixedArray.size`](#fixedarraysize)
- [`a[key]`](#akey索引与切片)
- [`a == b`](#a--b-与-a--b相等性比较)

## 类型导出、格式化与 NumPy 互操作

`FixedArray` 不会隐式转换为 float（`float(a)`、`np.asarray(a)` 都会报错），所有导出都必须显式调用：
需要精确值用 `to_decimal`，喂给统计/绘图库用 `to_float`，需要整数尾数直接读 `mantissa`。

### FixedArray.to_float

```python
FixedArray.to_float()
```

转换为浮点数（有损）。

**返回值**

- **out** : *float 或 ndarray（float64）* — 0 维时返回 Python `float`，否则返回形状相同的 `float64` 数组。

**机制说明**

每个元素得到的是 **最接近真实值的 float64**（正确舍入）：

- 常规情况（`scale ≤ 22` 且 `|mantissa| ≤ 2⁵³`）：`mantissa / 10.0**scale`，两个操作数都能精确表示为 float64，
  IEEE 除法保证结果正确舍入，全程向量化。
- 否则（尾数超过 2⁵³ 或 `10**scale` 无法精确表示为 float64）：逐元素用 Python 整数做 `int(m) / 10**scale`，
  Python 的整数真除法同样正确舍入，但速度较慢。

float64 只有约 15–16 位十进制有效数字，导出 float 应该是计算链的 **最后一步**，不要把结果再转回 `FixedArray` 继续计算。
`float(a)` 不能直接作用于 `FixedArray`。

**示例**

```python
>> > FixedArray("1.25").to_float()
1.25
>> > FixedArray(["1.50", "-2.25"]).to_float()
array([1.5, -2.25])
>> > FixedArray(2 ** 53 + 1).to_float()  # 超出 float64 精度，低位丢失
9007199254740992.0
>> > FixedArray("1e-30").to_float()  # scale 30：10**30 不能精确表示为 float64，走整数路径
1e-30
>> > float(FixedArray("1.5"))
Traceback(most
recent
call
last):
...
TypeError: float()
argument
must
be
a
string or a
real
number, not 'FixedArray'
```

**另请参阅**

- [`FixedArray.to_decimal`](#fixedarrayto_decimal) – 无损导出。

### FixedArray.to_decimal

```python
FixedArray.to_decimal()
```

无损转换为 Python 标准库 `decimal.Decimal`。

**返回值**

- **out** : *Decimal 或 ndarray（object）* — 0 维时返回 `Decimal`，否则返回元素为 `Decimal` 的 `object` 数组。
  结果保留 `scale` 位小数（含尾随零），与 `FixedArray` 的数值完全相等。

**机制说明**

数组路径逐元素创建 `Decimal` 对象，速度远低于 `to_float`，适合导出、对账、序列化，不适合放在热循环里。
`to_decimal` 与构造函数互为逆运算。

**示例**

```python
>> > FixedArray("1.50").to_decimal()
Decimal('1.50')
>> > FixedArray([["1.5", "2"], ["3", "4"]]).to_decimal()
array([[Decimal('1.5'), Decimal('2.0')],
       [Decimal('3.0'), Decimal('4.0')]], dtype=object)
>> > orig = FixedArray(["0.10", "123.45"])
>> > bool((FixedArray(orig.to_decimal()) == orig).all())
True
```

**另请参阅**

- [`FixedArray(values, scale=None)`](#fixedarrayvalues-scalenone)

### FixedArray.to_pandas

```python
FixedArray.to_pandas(index=None, name=None, as_float=True)
```

导出为 `pandas.Series`（0 维、1 维）或 `pandas.DataFrame`（2 维）。需要安装 pandas。

**参数**

- **index** : *Index 或 array_like，可选* — 行索引。
- **name** : *str，可选* — `Series` 的名称；导出为 `DataFrame` 时忽略。
- **as_float** : *bool，默认 True* — `True` 导出 `float64` 列（便于统计与绘图，有损）；
  `False` 导出元素为 `Decimal` 的 `object` 列（无损）。

**返回值**

- **out** : *pandas.Series 或 pandas.DataFrame* — 0 维返回长度为 1 的 `Series`；1 维返回 `Series`；2 维返回 `DataFrame`。

**异常**

- **ImportError** — 未安装 pandas。
- **ValueError** — 3 维及以上（由 pandas 报出）。

**机制说明**

`as_float=True` 时数据来自 [`to_float()`](#fixedarrayto_float)，`False` 时来自 [`to_decimal()`](#fixedarrayto_decimal)。
`pd.Series(a)` 会触发隐式数组转换而报错，必须使用本方法。

**示例**

```python
>> > px = FixedArray(["1.50", "2.25"])
>> > px.to_pandas(index=["a", "b"], name="px")
a
1.50
b
2.25
Name: px, dtype: float64
>> > px.to_pandas(as_float=False)
0
1.50
1
2.25
dtype: object
>> > FixedArray([["1.5", "2"], ["3", "4"]]).to_pandas()
0
1
0
1.5
2.0
1
3.0
4.0
```

**另请参阅**

- [`FixedArray.to_polars`](#fixedarrayto_polars)
- [`FixedArray.to_float`](#fixedarrayto_float)、[`FixedArray.to_decimal`](#fixedarrayto_decimal)

### FixedArray.to_polars

```python
FixedArray.to_polars(name="value", as_float=True)
```

导出为 `polars.Series`，仅支持 0 维与 1 维。需要安装 polars。

**参数**

- **name** : *str，默认 `"value"`* — `Series` 的名称。
- **as_float** : *bool，默认 True* — `True` 导出 `Float64` 列；`False` 导出 `Decimal(38, scale)` 列（无损）。

**返回值**

- **out** : *polars.Series* — 0 维返回长度为 1 的 `Series`。

**异常**

- **ImportError** — 未安装 polars。
- **ValueError** — 2 维及以上；`as_float=False` 且 `scale > 38`（polars `Decimal` 最多 38 位小数，请先 `rescale` 或改用
  float）。

**机制说明**

`as_float=False` 时先通过 [`to_decimal()`](#fixedarrayto_decimal) 得到精确的 `Decimal`，再以
`pl.Decimal(precision=38, scale=self.scale)` 构造列，数值与 `FixedArray` 完全一致。

**示例**

> 以下示例依赖 polars，未纳入自动校验。

```python
>> > s = FixedArray(["1.50", "2.25"]).to_polars(name="px")  # doctest: +SKIP
>> > s.to_list()  # doctest: +SKIP
[1.5, 2.25]
>> > FixedArray([[1, 2], [3, 4]]).to_polars()  # doctest: +SKIP
Traceback(most
recent
call
last):
...
ValueError: to_polars
仅支持
1
维或标量
FixedArray，当前维度为
2
```

**另请参阅**

- [`FixedArray.to_pandas`](#fixedarrayto_pandas)
- [`FixedArray.to_float`](#fixedarrayto_float)、[`FixedArray.to_decimal`](#fixedarrayto_decimal)

### str (a) 与 repr (a)（文本表示）

```python
str(a)
repr(a)
```

按 `scale` 输出固定小数位的十进制文本；`repr` 额外标注 `scale`。

**返回值**

- **out** : *str*

**机制说明**

每个元素由整数尾数直接拆位格式化（整数部分 + 补零后的 `scale` 位小数），不经过 float，
因此输出的是 **精确值**，并保留尾随零。大数组的省略规则与 NumPy 的 `array2string` 相同。
`FixedArray` 没有实现格式说明符：`f"{a}"` 可用，`f"{a:.2f}"` 会报 `TypeError`；
需要定制位数时先 [`rescale`](#fixedarrayrescale)，或使用上层的 `MonetaryFormatter`。

**示例**

```python
>> > neg = FixedArray("-0.05")
>> > str(neg)
'-0.05'
>> > neg
FixedArray(-0.05, scale=2)
>> > print(FixedArray(["1.5", "-2"]))
[1.5, -2.0]
>> > print(FixedArray(np.arange(2000)))
[0, 1, 2, ..., 1997, 1998, 1999]
>> > f"总额：{FixedArray('1.50')}"
'总额：1.50'
>> > f"{FixedArray('1.50'):.1f}"
Traceback(most
recent
call
last):
...
TypeError: unsupported
format
string
passed
to
FixedArray.__format__
```

**另请参阅**

- [`FixedArray.rescale`](#fixedarrayrescale)
- [`FixedArray.to_decimal`](#fixedarrayto_decimal)

### NumPy 互操作机制

`FixedArray` 的接口刻意模仿 `ndarray`，但它 **不是** `ndarray` 的子类，也不会被 NumPy 当作普通数组处理。

**机制说明**

两道“闸门”阻止定点数被悄悄转成 `float64` 或 `object` 数组：

| 机制                         | 效果                                                                                              |
|------------------------------|---------------------------------------------------------------------------------------------------|
| `__array_ufunc__ = None`     | 所有 ufunc（`np.add`、`np.sqrt`、`np.abs`、`np.maximum`……）作用于 `FixedArray` 时报 `TypeError`。 |
| `__array__` 抛出 `TypeError` | `np.asarray(a)`、`np.concatenate([a, b])`、`pd.Series(a)` 等需要把对象转成数组的操作全部报错。    |

`__array_ufunc__ = None` 还让 `ndarray` 位于运算符 **左侧**时，NumPy 主动放弃处理，由 Python 调用 `FixedArray`
的反向运算，所以 `np.array([1, 2]) * a` 的结果仍是 `FixedArray`。

以下 NumPy 函数会转发到 `FixedArray` 自身的方法或属性，可以直接使用：

| NumPy 调用                                  | 等价于                                      | 备注                         |
|---------------------------------------------|---------------------------------------------|------------------------------|
| `np.sum(a, axis=…, keepdims=…)`             | [`a.sum(...)`](#fixedarraysum)              | `out`、`dtype` 必须为 `None` |
| `np.mean(a, axis=…, keepdims=…)`            | [`a.mean(...)`](#fixedarraymean)            | 结果 scale 默认与 `a` 相同   |
| `np.transpose(a, axes)`                     | [`a.transpose(axes)`](#fixedarraytranspose) |                              |
| `np.shape(a)` / `np.ndim(a)` / `np.size(a)` | `a.shape` / `a.ndim` / `a.size`             |                              |

其余 NumPy 函数（`np.max`、`np.round`、`np.reshape`、`np.concatenate`、`np.where`、`np.cumsum`……）会报 `TypeError`，
对应的定点数写法如下：

| NumPy 写法                   | FixedArray 写法                    | 说明                         |
|------------------------------|------------------------------------|------------------------------|
| `np.array(x)`                | `FixedArray(x)`                    |                              |
| `a.astype(float)`            | `a.to_float()`                     | 有损                         |
| `np.round(a, 2)`             | `a.rescale(2)`                     | 支持六种舍入模式             |
| `np.round(a / 0.05) * 0.05`  | `a.round_to_step(0.05)`            | 精确                         |
| `np.isclose(a, b)`           | `a == b`                           | 定点数比较精确，不需要容差   |
| `np.abs(a)`                  | `abs(a)`                           |                              |
| `np.reshape(a, s)`           | `a.reshape(s)`                     |                              |
| `np.max(a)` / `np.argmax(a)` | `a[np.argmax(a.mantissa)]`         | 同一数组内尾数顺序即数值顺序 |
| `np.sort(a)`                 | `a[np.argsort(a.mantissa)]`        |                              |
| `np.concatenate([a, b])`     | 对齐 scale 后拼接尾数，见下方配方  |                              |
| `np.where(c, a, b)`          | 对齐 scale 后选择尾数，见下方配方  |                              |
| `np.cumsum(a)`               | 对尾数 `cumsum` 后重建，见下方配方 | **无溢出检查**               |
| `a[i] = v`                   | 不支持                             | 用运算或配方构造新数组       |

**示例**

被拒绝的隐式转换，以及 `ndarray` 在左侧的运算：

```python
>> > h = FixedArray(["1.50", "2.25"])
>> > np.sqrt(h)
Traceback(most
recent
call
last):
...
TypeError: operand
'FixedArray'
does
not support
ufuncs(__array_ufunc__=None)
>> > np.asarray(h)
Traceback(most
recent
call
last):
...
TypeError: FixedArray
不能隐式转成数组：要
float
请用
to_float()，要整数尾数请用.mantissa
>> > np.array([1, 2]) * h
FixedArray([1.50, 4.50], scale=2)
>> > np.shape(m), np.ndim(m), np.size(m)
((2, 2), 2, 4)
```

配方：由尾数重建 `FixedArray`。公开构造器接收的是十进制数值而非底层尾数，若需从运算后的 `int64` 尾数重建 `FixedArray`
，可借助精确乘法实现（乘以 `10^(-scale)` 不会发生舍入）：

```python
>> >

def from_mantissa(mant, scale):


    ...
"""由 int64 尾数与 scale 构造 FixedArray（精确）。"""
...
return FixedArray(mant) * FixedArray(f"1e-{scale}")
>> > from_mantissa(np.array([150, 225]), 2)
FixedArray([1.50, 2.25], scale=2)
>> > p = FixedArray(["10.05", "9.80", "10.20"])
>> > p[np.argsort(p.mantissa)]  # 排序
FixedArray([9.80, 10.05, 10.20], scale=2)
>> > from_mantissa(m.mantissa.max(axis=0), m.scale)  # 按轴取最大值
FixedArray([3.30, 4.40], scale=2)
>> > u, v = FixedArray(["1.5", "2.5"]), FixedArray(["0.25"])
>> > s = max(u.scale, v.scale)
>> > from_mantissa(np.concatenate([u.rescale(s).mantissa, v.rescale(s).mantissa]), s)  # 拼接
FixedArray([1.50, 2.50, 0.25], scale=2)
>> > from_mantissa(np.where([True, False], u.rescale(s).mantissa, v.rescale(s).mantissa), s)  # 条件选择
FixedArray([1.50, 0.25], scale=2)
>> > cash = FixedArray(["100.00", "-30.50", "12.25"])
>> > from_mantissa(np.cumsum(cash.mantissa), cash.scale)  # 累计和：int64 溢出会静默回绕，仅在确定不越界时使用
FixedArray([100.00, 69.50, 81.75], scale=2)
```

**另请参阅**

- [`FixedArray.mantissa`](#fixedarraymantissa)
- [`FixedArray._from_mantissa`](#fixedarray_from_mantissa)
- [`FixedArray.to_float`](#fixedarrayto_float)、[`to_pandas`](#fixedarrayto_pandas)、[`to_polars`](#fixedarrayto_polars)

## 内部构造收口与类型转换（私有 API）

以下私有方法负责 `FixedArray` 的底层实例初始化、操作数白名单校验、类型强制转换与 `scale` 对齐。

### FixedArray._set

```python
FixedArray._set(mantissa, scale)
```

所有构造路径的唯一收口方法，为实例绑定只读 `int64` 尾数视图与 `scale`。

**参数**

- **mantissa** : *int 或整数 array_like* — 底层整数尾数。
- **scale** : *int* — 已通过校验的非负小数位数。

**返回值**

- **None** — 原地设置 `self._mantissa` 与 `self._scale`。

**异常**

- **TypeError** — `mantissa` 不是整数类型（如 `float`、`bool`、`object`）。
- **OverflowError** — `mantissa` 超出 `int64` 范围。

**机制说明**

先通过 [`_as_int64`](#fixedarray_as_int64) 将输入统一转换为 `int64` NumPy 数组（0 维标量也会包装为 0 维 `ndarray`），
随后调用 `.view()` 创建浅视图并将 `view.flags.writeable = False` 设为只读。这样既锁住了 `FixedArray` 自身的尾数不可变性，
又不会修改调用方原本传入的外部数组的可写标志。

**示例**

```python
>> > obj = FixedArray.__new__(FixedArray)
>> > obj._set(np.array([150, 225], dtype=np.int64), 2)
>> > obj
FixedArray([1.50, 2.25], scale=2)
>> > obj.mantissa.flags.writeable
False
```

**另请参阅**

- [`FixedArray._from_mantissa`](#fixedarray_from_mantissa)
- [`FixedArray._as_int64`](#fixedarray_as_int64)

### FixedArray._from_mantissa

```python
FixedArray._from_mantissa(mantissa, scale)
```

直接使用已有的整数尾数矩阵与 `scale` 构造 `FixedArray`（内部零拷贝快速路径）。

**参数**

- **mantissa** : *int 或整数 array_like* — 底层整数尾数；若已是 `int64` 数组则直接共享内存，不发生拷贝。
- **scale** : *int* — 非负整数小数位数。

**返回值**

- **out** : *FixedArray* — 新构造的 `FixedArray` 实例。

**异常**

- **TypeError** — `mantissa` 不是整数类型。
- **OverflowError** — `mantissa` 超出 `int64` 范围。

**机制说明**

通过 `cls.__new__(cls)` 跳过公开构造器 [`__init__`](#fixedarrayvalues-scalenone) 的类型分派与数值解析，直接调用 [
`_set`](#fixedarray_set)
绑定只读视图。内部所有算术、规约、切片操作在计算出结果尾数后均通过此方法返回新实例。

**示例**

```python
>> > FixedArray._from_mantissa(np.array([1234, -500], dtype=np.int64), scale=2)
FixedArray([12.34, -5.00], scale=2)
>> > FixedArray._from_mantissa(42, scale=0)
FixedArray(42, scale=0)
```

**另请参阅**

- [`FixedArray._set`](#fixedarray_set)
- [`FixedArray(values, scale=None)`](#fixedarrayvalues-scalenone)

### FixedArray._is_numeric_operand

```python
FixedArray._is_numeric_operand(val)
```

判断对象是否属于允许与 `FixedArray` 运算、比较或对齐的数值操作数白名单。

**参数**

- **val** : *Any* — 待检查的任意对象。

**返回值**

- **ok** : *bool* — 合法数值操作数返回 `True`，否则返回 `False`。

**机制说明**

白名单包含：`FixedArray`、`int`、`np.integer`、`float`、`np.floating`、`Decimal`、`Fraction`、整数/浮点 `ndarray`，
以及递归仅包含上述元素的 `list`、`tuple` 或 `object` 类型的 `ndarray`。
特别地，由于 Python 中 `bool` 是 `int` 的子类，该方法首先显式排除 `(bool, np.bool_)`；字符串及上层带量纲对象（如 `Money`、
`UnitArray`）也一律返回 `False`。

**示例**

```python
>> > FixedArray._is_numeric_operand(10), FixedArray._is_numeric_operand([1.5, Decimal("2.0")])
(True, True)
>> > FixedArray._is_numeric_operand(True), FixedArray._is_numeric_operand("1.5")
(False, False)
```

**另请参阅**

- [`FixedArray._coerce`](#fixedarray_coerce)

### FixedArray._coerce

```python
FixedArray._coerce(other, strict=True)
```

将二元运算或比较另一侧的操作数转换为 `FixedArray`。

**参数**

- **other** : *Any* — 待转换的操作数。
- **strict** : *bool，默认 True* — 当 `other` 不在数值白名单内时，`True` 直接抛出 `TypeError`，`False` 返回 `None`（供运算符重载返回
  `NotImplemented`）。

**返回值**

- **out** : *FixedArray 或 None* — 若 `other` 已是 `FixedArray` 则原样返回；若在白名单内则调用 `cls(other)` 构造并返回；
  `strict=False` 且不在白名单内时返回 `None`。

**异常**

- **TypeError** — `strict=True` 且 `other` 不在数值白名单内。
- **ValueError** / **OverflowError** — `other` 虽在白名单内但数值非法（如 `NaN`、无限循环小数 `Fraction(1, 3)` 或超出
  `int64`）。

**机制说明**

先调用 [`_is_numeric_operand`](#fixedarray_is_numeric_operand) 拦截字符串与布尔值（尽管字符串可以传给构造函数，但绝不允许隐式参与算术或比较运算），
通过后再调用 `cls(other)` 完成精确解析。

**示例**

```python
>> > FixedArray._coerce(1.25)
FixedArray(1.25, scale=2)
>> > FixedArray._coerce("1.25", strict=False) is None
True
>> > FixedArray._coerce("1.25", strict=True)
Traceback(most
recent
call
last):
...
TypeError: FixedArray
只能与数值类型运算、比较或对齐，收到
str
```

**另请参阅**

- [`FixedArray._is_numeric_operand`](#fixedarray_is_numeric_operand)
- [`FixedArray(values, scale=None)`](#fixedarrayvalues-scalenone)

### FixedArray._check_scale

```python
FixedArray._check_scale(scale)
```

校验 `scale` 是否为合法的非负整数，并返回标准 Python `int`。

**参数**

- **scale** : *Any* — 待校验的小数位数。

**返回值**

- **out** : *int* — 非负 Python `int`。

**异常**

- **ValueError** — `scale` 为负数、浮点数、布尔值或其他非整数类型。

**机制说明**

显式拒绝 `bool` 与 `np.bool_`，仅接受 `int` 或 `np.integer` 且满足 `scale >= 0`。

**示例**

```python
>> > FixedArray._check_scale(4)
4
>> > FixedArray._check_scale(-1)
Traceback(most
recent
call
last):
...
ValueError: scale
必须是非负整数，收到 - 1
>> > FixedArray._check_scale(True)
Traceback(most
recent
call
last):
...
ValueError: scale
必须是非负整数，收到
True
```

**另请参阅**

- [`FixedArray.scale`](#fixedarrayscale)
- [`FixedArray.rescale`](#fixedarrayrescale)

### FixedArray._as_int64

```python
FixedArray._as_int64(x)
```

将整数标量或整数数组安全转换为 `int64` NumPy 数组，对已是 `int64` 的数组不发生拷贝。

**参数**

- **x** : *int 或整数 array_like* — 输入整数或整数数组。

**返回值**

- **out** : *ndarray（int64）* — 转换后的 0 维或多维 `int64` 数组。

**异常**

- **TypeError** — 输入非整数（含 `bool`、`float`、`object`）。
- **OverflowError** — Python 大整数超出 `[-2**63, 2**63 - 1]`，或 `uint64` 数组最大值超过 `2**63 - 1`。

**机制说明**

对单个 Python `int` 先在 Python 层检查 `_MIN <= x <= _MAX` 再转 `int64`；对数组先调用 `np.asarray(x)` 检查
`dtype.kind in "iu"`，
若为 `uint64` 则额外检查上界 `_MAX`，最后通过 `astype(np.int64, copy=False)` 零拷贝返回。

**示例**

```python
>> > FixedArray._as_int64(100)
array(100)
>> > FixedArray._as_int64(2 ** 63)
Traceback(most
recent
call
last):
...
OverflowError: 9223372036854775808
超出
int64
范围
```

**另请参阅**

- [`FixedArray._set`](#fixedarray_set)

### FixedArray._align_scales

```python
FixedArray._align_scales(a, b)
```

将两个 `FixedArray` 的 `scale` 自动对齐到 `max(a.scale, b.scale)`。

**参数**

- **a, b** : *FixedArray* — 待对齐的两个定点数张量。

**返回值**

- **aligned** : *tuple of (FixedArray, FixedArray)* — `scale` 相同的两个 `FixedArray`。

**异常**

- **OverflowError** — 较小 `scale` 的一方在放大尾数时超出 `int64` 范围。

**机制说明**

若 `a.scale == b.scale` 则直接返回 `(a, b)`（零开销）；否则计算 `target = max(a.scale, b.scale)` 并调用 [
`rescale(target)`](#fixedarrayrescale) 将较小一方的尾数乘以 $10^{\text{target} - s}$。

**示例**

```python
>> > a_al, b_al = FixedArray._align_scales(FixedArray("1.5"), FixedArray("0.25"))
>> > a_al, b_al
(FixedArray(1.50, scale=2), FixedArray(0.25, scale=2))
```

**另请参阅**

- [`FixedArray.rescale`](#fixedarrayrescale)
- [`FixedArray._align_mantissas`](#fixedarray_align_mantissas)

## 内部输入解析引擎（私有 API）

以下私有方法负责将字符串、`Decimal`、`Fraction`、浮点数标量与数组精确解析为 `(mantissa, scale)`。

### FixedArray._parse_decimal_str

```python
FixedArray._parse_decimal_str(s)
```

解析十进制字符串（支持正负号、科学计数法、下划线与首尾空白），返回 `(mantissa, scale)`。

**参数**

- **s** : *str* — 十进制数值文本。

**返回值**

- **mantissa, scale** : *tuple of (int, int)* — 整数尾数与非负小数位数。

**异常**

- **ValueError** — 字符串无法解析为合法十进制数，或为 `NaN` / `Inf` 非有限值。
- **OverflowError** — 解析出的尾数超出 `int64` 范围 `[-2**63, 2**63 - 1]`。

**机制说明**

借助 `Decimal(s.strip()).as_tuple()` 提取符号、十进制数字元组与指数 `exp`：

- 当 `exp < 0` 时，字面声明的小数位数 `-exp` 原样保留（例如 `"0.00"` 返回 `(0, 2)`）；
- 当 `exp >= 0` 时，若尾数为 `0` 则返回 `(0, 0)`，否则先检查 `exp <= 19` 及 `_MIN <= m * 10**exp <= _MAX`（支持精确解析
  `-2**63`），再返回 `(m * 10**exp, 0)`。

**示例**

```python
>> > FixedArray._parse_decimal_str("19.99")
(1999, 2)
>> > FixedArray._parse_decimal_str("1.5e-3")
(15, 4)
>> > FixedArray._parse_decimal_str("-9223372036854775808")
(-9223372036854775808, 0)
```

**另请参阅**

- [`FixedArray._parse_text_array`](#fixedarray_parse_text_array)
- [`FixedArray._parse_float_scalar`](#fixedarray_parse_float_scalar)

### FixedArray._parse_fraction

```python
FixedArray._parse_fraction(f)
```

将有理数 `fractions.Fraction` 解析为精确的有限十进制定点数 `(mantissa, scale)`。

**参数**

- **f** : *Fraction* — 标准库有理数对象（已自动约分且分母为正）。

**返回值**

- **mantissa, scale** : *tuple of (int, int)* — 精确对应的整数尾数与最少小数位数。

**异常**

- **ValueError** — 分母含有除 `2` 和 `5` 以外的素因子（即无限循环小数，如 `Fraction(1, 3)`）。

**机制说明**

不断从分母 `f.denominator` 中消去因子 `2`（计数 `twos`）和因子 `5`（计数 `fives`）。若最终剩余分母不为 `1`，说明不能表示为有限位十进制小数，抛出
`ValueError`；否则最少小数位数为 `scale = max(twos, fives)`，尾数为
`f.numerator * 2**(scale - twos) * 5**(scale - fives)`。

**示例**

```python
>> > from fractions import Fraction
>> > FixedArray._parse_fraction(Fraction(3, 8))
(375, 3)
>> > FixedArray._parse_fraction(Fraction(1, 3))
Traceback(most
recent
call
last):
...
ValueError: 1 / 3
不是有限小数
```

**另请参阅**

- [`FixedArray._parse_object_element`](#fixedarray_parse_object_element)

### FixedArray._parse_float_scalar

```python
FixedArray._parse_float_scalar(val)
```

按最短往返十进制写法（Dragonbox 算法）解析单个浮点数，返回 `(mantissa, scale)`。

**参数**

- **val** : *float 或 np.floating* — 单个浮点数标量。

**返回值**

- **mantissa, scale** : *tuple of (int, int)* — 最短十进制表示对应的尾数与小数位数。

**异常**

- **ValueError** — 输入为 `NaN` 或无穷大。
- **OverflowError** — 数值超出 `int64` 范围。

**机制说明**

调用 `np.format_float_positional(val, unique=True, trim="-")` 生成能往返还原为同一个浮点数的最短十进制字符串（`trim="-"`
去掉整数末尾的 `".0"`，使 `1.0` 解析为 `scale=0`），再交给 [`_parse_decimal_str`](#fixedarray_parse_decimal_str) 解析。

**示例**

```python
>> > FixedArray._parse_float_scalar(0.1)
(1, 1)
>> > FixedArray._parse_float_scalar(1.0)
(1, 0)
```

**另请参阅**

- [`FixedArray._parse_float_array`](#fixedarray_parse_float_array)
- [`FixedArray._parse_decimal_str`](#fixedarray_parse_decimal_str)

### FixedArray._parse_object_element

```python
FixedArray._parse_object_element(val)
```

将单个任意受支持的标量（`int`、`float`、`Fraction`、`str`、`Decimal`）分派解析为 `(mantissa, scale)`。

**参数**

- **val** : *int, float, Fraction, Decimal 或 str* — 单个标量元素。

**返回值**

- **mantissa, scale** : *tuple of (int, int)*

**异常**

- **TypeError** — `val` 为布尔类型。
- **ValueError** / **OverflowError** — 元素内容非法或越界。

**机制说明**

按类型依次分派：布尔值报 `TypeError`；整数直接返回 `(int(val), 0)`；浮点数转 [
`_parse_float_scalar`](#fixedarray_parse_float_scalar)；`Fraction` 转 [`_parse_fraction`](#fixedarray_parse_fraction)；其余（
`str`、`Decimal`）转 `str(val)` 后交给 [`_parse_decimal_str`](#fixedarray_parse_decimal_str)。

**示例**

```python
>> > FixedArray._parse_object_element(Decimal("1.250"))
(1250, 3)
>> > FixedArray._parse_object_element(42)
(42, 0)
```

**另请参阅**

- [`FixedArray._parse_text_array`](#fixedarray_parse_text_array)

### FixedArray._parse_float_array

```python
FixedArray._parse_float_array(arr, target_scale=None)
```

将多维浮点数组解析为 `(尾数数组, scale)`，结果与逐个元素取最短十进制写法完全一致。

**参数**

- **arr** : *ndarray of floating* — 浮点数组（`float64`、`float32` 等）。
- **target_scale** : *int 或 None，可选* — 显式指定的目标小数位数；`None` 表示自动推断并对齐到全数组最大位数。

**返回值**

- **mantissa, scale** : *tuple of (ndarray, int)* — 形状与 `arr` 相同的 `int64` 尾数数组及最终 `scale`。

**异常**

- **OverflowError** — 对齐后或放大后的尾数超出 `int64` 范围。

**机制说明**

- **逐位往返检验**：对 `s = 0, 1, …, max_probe`（`target_scale=None` 时为 `22`，否则为 `min(22, target_scale + 1)`）向量化计算
  `m = np.rint(x · 10^s)`。当 `|m| ≤ 2^(nmant+1)`（`float64` 为 `2**53`，`float32` 为 `2**24`）且
  `(m / 10^s).astype(arr.dtype) == x` 时，说明 `x` 的最短写法有 `s` 位小数，在 **首次通过**时记录 `(m, s)`。当剩余未通过元素不足
  1/10 或出现超出 `2^(nmant+1)` 的大数时，收缩下标仅对剩余元素继续试探。
- **自动推断（`target_scale=None`）**：对试探未覆盖的元素按 `np.unique` 去重后调用 [
  `_parse_float_scalar`](#fixedarray_parse_float_scalar)，取全数组最大 `scale`，再由 [
  `_align_mantissas`](#fixedarray_align_mantissas) 统一对齐。
- **显式精度（`target_scale` 非空）**：位数 `≤ target_scale` 的元素由 [`_align_mantissas`](#fixedarray_align_mantissas)
  对齐；位数恰好为 `target_scale + 1` 的元素由 [`_round_div_pow10`](#fixedarray_round_div_pow10) 除以 10
  舍入；其余元素直接交由 [`_round_float_to_scale`](#fixedarray_round_float_to_scale) 快速舍入到 `target_scale`，无需先对齐到全精度。

**示例**

```python
>> > FixedArray._parse_float_array(np.array([0.1, 0.25]))
(array([10, 25]), 2)
>> > FixedArray._parse_float_array(np.array([100000.0, 1 / 3]), target_scale=2)
(array([10000000, 33]), 2)
```

**另请参阅**

- [`FixedArray._align_mantissas`](#fixedarray_align_mantissas)
- [`FixedArray._round_float_to_scale`](#fixedarray_round_float_to_scale)
- [`FixedArray._parse_float_scalar`](#fixedarray_parse_float_scalar)

### FixedArray._align_mantissas

```python
FixedArray._align_mantissas(mantissas, scales, target)
```

将各自 `scale` 不同的整数尾数数组向量化对齐到统一的目标 `target`（要求 `scales ≤ target`）。

**参数**

- **mantissas** : *ndarray（int64）* — 一维或多维初始尾数数组。
- **scales** : *ndarray（int64）* — 与 `mantissas` 同形状的各元素当前小数位数（满足 `scales ≤ target`）。
- **target** : *int* — 目标小数位数。

**返回值**

- **out** : *ndarray（int64）* — 对齐到 `target` 位小数后的尾数数组。

**异常**

- **OverflowError** — 非零元素放大 `10**(target - scale)` 后超出 `int64` 范围。

**机制说明**

先计算位移差 `shift = target - scales`，若存在非零尾数且 `shift > 18` 则直接报 `OverflowError`；随后计算因子数组
`factor = 10**min(shift, 18)`，整列校验 `np.abs(mantissas) <= _MAX // factor` 通过后执行向量化乘法 `mantissas * factor`。

**示例**

```python
>> > m_in = np.array([1, 25], dtype=np.int64)
>> > s_in = np.array([1, 2], dtype=np.int64)
>> > FixedArray._align_mantissas(m_in, s_in, target=3)
array([100, 250])
```

**另请参阅**

- [`FixedArray._parse_float_array`](#fixedarray_parse_float_array)
- [`FixedArray._align_scales`](#fixedarray_align_scales)

### FixedArray._round_float_to_scale

```python
FixedArray._round_float_to_scale(values, scale)
```

将浮点数组每个元素的最短十进制写法按 `half_up` 舍入到 `scale` 位小数，返回 `int64` 尾数数组。

**参数**

- **values** : *ndarray of floating* — 一维浮点数组。
- **scale** : *int* — 目标小数位数。

**返回值**

- **out** : *ndarray（int64）* — 舍入后的 `int64` 尾数数组。

**异常**

- **OverflowError** — 浮点数乘以 `10**scale` 后超出 `int64` 范围。

**机制说明**

先用 `float64` 计算 `z = values · 10.0**scale`，拆出整数部分 `floor = np.floor(z)` 与小数部分 `frac = z - floor`。
由于浮点数 `x` 与其最短十进制写法之差不超过半个 `ulp`（放大后不超过 `|z| · eps / 2`），且 `z` 自身的浮点乘法误差不超过
`|z| · 2**-53`，
因此只要满足安全条件：`x` 为正规数、`|z| < 2**51` 且 `|frac - 0.5| > |z| · (eps + 2**-50)`（两倍安全余量），`x`
与其最短十进制写法就必然落在半数边界 `k + 0.5` 的同一侧，直接用 `floor + (frac > 0.5)` 即可得到精确结果。
对极少数不满足安全条件（贴近 `0.5` 边界、`|z| ≥ 2**51` 或次正规数）的元素，去重后逐个调用 [
`_parse_float_scalar`](#fixedarray_parse_float_scalar) 与 [`_round_div_pow10`](#fixedarray_round_div_pow10) 做纯整数精确舍入。

**示例**

```python
>> > FixedArray._round_float_to_scale(np.array([2.675, 0.123456]), scale=2)
array([268, 12])
```

**另请参阅**

- [`FixedArray._parse_float_array`](#fixedarray_parse_float_array)
- [`FixedArray._round_div_pow10`](#fixedarray_round_div_pow10)

### FixedArray._parse_text_array

```python
FixedArray._parse_text_array(arr)
```

解析字符串、`Decimal`、`Fraction` 或混合对象数组（含 0 维），返回 `(对齐后的尾数数组, scale)`。

**参数**

- **arr** : *ndarray（dtype 为 U 或 O）* — 文本或对象数组。

**返回值**

- **mantissa, scale** : *tuple of (ndarray, int)* — 对齐后的 `int64` 尾数数组与最大 `scale`。

**异常**

- **TypeError** / **ValueError** / **OverflowError** — 元素类型不支持、文本无法解析或对齐后超出 `int64`。

**机制说明**

- 0 维标量直接调用 [`_parse_object_element`](#fixedarray_parse_object_element)，省去去重开销；
- 字符串数组（`kind == "U"`）先通过 `np.unique` 排序去重，仅对不同字符串调用 [
  `_parse_decimal_str`](#fixedarray_parse_decimal_str)；
- 对象数组（`kind == "O"`）按 `(type(val).__name__, str(val))` 字典去重（不能按数值相等去重，否则 `Decimal("1.1")` 与
  `Decimal("1.100")` 会互相覆盖各自声明的小数位数）；
- 最后对齐到全数组最大 `scale`，并通过索引映射还原原数组形状。

**示例**

```python
>> > FixedArray._parse_text_array(np.array(["1.5", "2.25", "1.5"]))
(array([150, 225, 150]), 2)
```

**另请参阅**

- [`FixedArray._parse_decimal_str`](#fixedarray_parse_decimal_str)
- [`FixedArray._parse_object_element`](#fixedarray_parse_object_element)

## 底层数学与防溢出舍入引擎（私有 API）

以下私有方法构成了 `FixedArray` 的纯整数算术、统一舍入裁决、大除数二进制倍增与分块防假溢出内核。

### FixedArray._check_rounding

```python
FixedArray._check_rounding(mode)
```

校验并规范化舍入模式参数，返回对应的模式字符串。

**参数**

- **mode** : *RoundingMode 或 str* — 待校验的舍入模式。

**返回值**

- **out** : *str* — `"half_up"`、`"half_even"`、`"down"`、`"up"`、`"floor"`、`"ceil"` 之一。

**异常**

- **ValueError** — `mode` 不属于上述六种合法模式。

**机制说明**

若传入 [`RoundingMode`](#roundingmode) 枚举实例则直接取 `mode.value`；否则在 `_ROUNDINGS` 元组中校验成员关系。

**示例**

```python
>> > FixedArray._check_rounding(RoundingMode.HALF_EVEN)
'half_even'
>> > FixedArray._check_rounding("floor")
'floor'
```

**另请参阅**

- [`RoundingMode`](#roundingmode)
- [`FixedArray._decide_bump`](#fixedarray_decide_bump)

### FixedArray._check_division

```python
FixedArray._check_division(m, d)
```

整数除法前置安全检查：除数不能含 `0`，且禁止 `-2**63` 除以 `-1`。

**参数**

- **m, d** : *ndarray（int64）* — 被除数与除数尾数数组。

**返回值**

- **None**

**异常**

- **ZeroDivisionError** — `d` 中存在 `0`。
- **OverflowError** — 存在对应位置同时满足 `m == -2**63` 且 `d == -1`（其商 `+2**63` 超出有符号 `int64` 上限，且会在底层 C
  的 `divmod` 中触发硬件异常）。

**机制说明**

使用向量化布尔判断 `np.any(d == 0)` 与 `np.any((d == -1) & (m == _MIN))`，在调用 `np.divmod` 前拦截非法整数除法。

**示例**

```python
>> > FixedArray._check_division(np.array([-2 ** 63], dtype=np.int64), np.array([-1], dtype=np.int64))
Traceback(most
recent
call
last):
...
OverflowError: -2 ** 63
除以 - 1
超出
int64
范围
```

**另请参阅**

- [`FixedArray._divmod_round`](#fixedarray_divmod_round)
- [`FixedArray.divide`](#fixedarraydivide)

### FixedArray._max_abs

```python
FixedArray._max_abs(x)
```

安全计算 `int64` 数组元素绝对值的最大值，返回 Python `int`。

**参数**

- **x** : *ndarray（int64）* — 整数数组。

**返回值**

- **top** : *int* — 数组元素的最大绝对值；空数组返回 `0`。

**机制说明**

不在 NumPy `int64` 内部直接调用 `np.abs(x).max()`（因为 `-2**63` 在 `int64` 下取绝对值会溢出回绕为 `-2**63`），
而是先将 `x.min()` 与 `x.max()` 转为任意精度 Python `int`，再计算 `max(-int(x.min()), int(x.max()))`，彻底消除 `-2**63`
极值隐患。

**示例**

```python
>> > FixedArray._max_abs(np.array([-2 ** 63, 100], dtype=np.int64)) == 2 ** 63
True
>> > FixedArray._max_abs(np.array([], dtype=np.int64))
0
```

**另请参阅**

- [`FixedArray._mul_pow10`](#fixedarray_mul_pow10)

### FixedArray._mul_pow10

```python
FixedArray._mul_pow10(x, k)
```

计算 `x · 10^k`，预先通过上下界判定是否超出 `int64`。

**参数**

- **x** : *ndarray（int64）* — 输入整数尾数数组。
- **k** : *int* — 非负十进制放大位数。

**返回值**

- **out** : *ndarray（int64）* — 放大 `10**k` 倍后的数组。

**异常**

- **OverflowError** — 存在非零元素在乘以 `10**k` 后超出 `int64` 上限。

**机制说明**

- `k == 0` 时直接返回 `x`；
- 调用 [`_max_abs(x)`](#fixedarray_max_abs) 获取最大绝对值 `top`：若 `top == 0`（全零数组或空数组），无论 `k` 多大（即使
  `k > 18`）都直接返回 `np.zeros_like(x)`；
- 若 `top > 0`，先检查 `k > 18` 或 `top > _MAX // 10**k`（由此同时拒绝不能存正号对称值的 `-2**63` 放大），安全时再计算
  `x * 10**k`。

**示例**

```python
>> > FixedArray._mul_pow10(np.array([12, -5], dtype=np.int64), 2)
array([1200, -500])
>> > FixedArray._mul_pow10(np.array([0, 0], dtype=np.int64), 30)
array([0, 0])
```

**另请参阅**

- [`FixedArray._max_abs`](#fixedarray_max_abs)
- [`FixedArray.rescale`](#fixedarrayrescale)

### FixedArray._decide_bump

```python
FixedArray._decide_bump(mode, *, is_negative, odd_q, has_fraction, over_half, exact_half, floor_base=False)
```

六种舍入模式的统一进位裁决函数：根据余数状态决定商基准值是否需要加 `1`。

**参数**

- **mode** : *str* — 已通过校验的舍入模式名称。
- **is_negative** : *bool 或 ndarray of bool* — 真实数值是否为负。
- **odd_q** : *bool 或 ndarray of bool* — 商基准值 `q` 是否为奇数（供 `half_even` 判定）。
- **has_fraction** : *bool 或 ndarray of bool* — 是否存在非零余数（小数部分）。
- **over_half** : *bool 或 ndarray of bool* — 小数部分是否严格大于 `0.5`。
- **exact_half** : *bool 或 ndarray of bool* — 小数部分是否恰好等于 `0.5`。
- **floor_base** : *bool，默认 False* — 商基准值的坐标系：
    - `True`（向下取整基准）：基准商 `q = floor(m / d)` 为带符号整数，余数 `r/d ∈ [0, 1)`，返回布尔值或布尔数组（`q + bump`
      即为最终商）；
    - `False`（向零截断基准）：基准商 `|q| = trunc(|m| / |d|)` 为无符号绝对值幅度，返回 `uint64(0)` 或 `uint64(1)`（供 [
      `_chunked_mul_round`](#fixedarray_chunked_mul_round) 对绝对值幅度进位）。

**返回值**

- **bump** : *bool / ndarray of bool（当 `floor_base=True`）或 uint64 / ndarray of uint64（当 `floor_base=False`）*

**异常**

- **ValueError** — `mode` 非法。

**机制说明**

将所有除法、乘法舍入、均值、精度缩放的进位逻辑收口到同一处。尤其注意在 `floor_base=True`（向下取整坐标系）下，负数 `-2.5` 被拆分为
`q = -3` 加正小数 `+0.5`：此时 `half_up`（远离零，目标为 `-3`）在 `exact_half` 时 **不**加 `1`（仅 `over_half` 时加 `1` 向
`0` 靠近），而正数 `+2.5`（`q = 2` 加 `+0.5`）在 `over_half | exact_half` 时加 `1` 变成 `3`。

**示例**

```python
>> >  # 2.5 = 2 + 0.5 (floor_base=True)：half_up 进位 (+1 -> 3)，half_even 不进位 (2 为偶数 -> 2)
>> > bool(FixedArray._decide_bump("half_up", is_negative=False, odd_q=False, has_fraction=True, over_half=False,
                                  exact_half=True, floor_base=True))
True
>> > bool(FixedArray._decide_bump("half_even", is_negative=False, odd_q=False, has_fraction=True, over_half=False,
                                  exact_half=True, floor_base=True))
False
```

**另请参阅**

- [`FixedArray._round_quotient`](#fixedarray_round_quotient)
- [`FixedArray._round_div_pow10`](#fixedarray_round_div_pow10)
- [`FixedArray._chunked_mul_round`](#fixedarray_chunked_mul_round)

### FixedArray._round_quotient

```python
FixedArray._round_quotient(q, r, d, mode)
```

由向下取整的商 `q` 与余数 `r`（满足 `m = q·d + r`，`r` 与 `d` 同号且 `|r| < |d|`）按 `mode` 精确取整。

**参数**

- **q, r, d** : *ndarray（int64）* — `np.divmod(m, d)` 得到的整数商、余数与除数。
- **mode** : *str* — 舍入模式名称。

**返回值**

- **out** : *ndarray（int64）* — 舍入后的整数商 `q + bump`。

**机制说明**

真实商为 `q + r/d`，其中 `r/d ∈ [0, 1)`。比较 `r/d` 与 `1/2` 的大小时，不计算 `2 * r`（当 `|r|` 接近 `int64` 上限时会溢出），
而是比较 `r` 与 `d - r`：由于 `r` 与 `d` 同号，`|d - r| ≤ |d|` 恒在 `int64` 范围内，绝对不会溢出。随后调用 [
`_decide_bump(..., floor_base=True)`](#fixedarray_decide_bump) 完成进位。

**示例**

```python
>> > q, r = np.divmod(np.array([7, -7]), np.array([2, 2]))  # 3.5 与 -3.5 (= -4 + 0.5)
>> > FixedArray._round_quotient(q, r, np.array([2, 2]), "half_up")
array([4, -4])
>> > FixedArray._round_quotient(q, r, np.array([2, 2]), "half_even")
array([4, -4])
```

**另请参阅**

- [`FixedArray._decide_bump`](#fixedarray_decide_bump)
- [`FixedArray._divmod_round`](#fixedarray_divmod_round)

### FixedArray._divmod_round

```python
FixedArray._divmod_round(m, d, mode="half_up")
```

纯 `int64` 整数带余除法与舍入：计算 `round(m / d)`。

**参数**

- **m, d** : *int 或整数 array_like* — 被除数与除数，必须在 `int64` 范围内，支持广播；`d` 不能含 `0`。
- **mode** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式（六种均支持）。

**返回值**

- **out** : *int 或 ndarray（int64）* — 当 `m` 与 `d` 均为整数标量时返回 Python `int`，否则返回 `int64` 数组。

**异常**

- **ValueError** — `mode` 无效。
- **ZeroDivisionError** — `d` 含 `0`。
- **OverflowError** — 输入超出 `int64` 范围，或 `-2**63` 除以 `-1`。

**机制说明**

先校验 `mode` 并通过 [`_as_int64`](#fixedarray_as_int64) 与 [`_check_division`](#fixedarray_check_division) 校验输入，再执行
`q, r = np.divmod(m, d)` 并交由 [`_round_quotient`](#fixedarray_round_quotient) 取整。

**示例**

```python
>> > FixedArray._divmod_round(5, 2, mode="half_up")
3
>> > FixedArray._divmod_round(5, 2, mode="half_even")
2
>> > FixedArray._divmod_round(np.array([10, -10]), 3, mode="down")
array([3, -3])
```

**另请参阅**

- [`FixedArray._round_quotient`](#fixedarray_round_quotient)
- [`FixedArray.round_to_step`](#fixedarrayround_to_step)

### FixedArray._doubling_step_10r

```python
FixedArray._doubling_step_10r(r, d)
```

大除数无溢出单步倍增：在 `|d| > _MAX // 10` 时精确计算 `(r * 10) // d` 与 `(r * 10) % d`。

**参数**

- **r, d** : *ndarray（int64）* — 当前余数与除数（满足 `r` 与 `d` 同号且 `0 ≤ |r| < |d|`）。

**返回值**

- **digits, new_r** : *tuple of (ndarray, ndarray)* — 下一位十进制商数字 `digits ∈ [0, 9]` 与新余数 `new_r`（均为 `int64`
  数组）。

**机制说明**

当除数绝对值 `|d| > (2**63 - 1) // 10` 时，直接计算 `r * 10` 会超出 `int64`。
本方法先通过补码位运算将 `r` 与 `d` 的绝对值無溢出地映射到无符号 `uint64`（满足 `0 ≤ r_u < d_u ≤ 2**63`），
由于 `2 * r_u < 2 * d_u ≤ 2**64` 在 `uint64` 内绝不溢出，定義单步倍增函数 `_step2(cur_r)` 计算 `(2 * cur_r) // d_u` 与
`(2 * cur_r) % d_u`：
连续调用三次分别得到 `2r`、`4r`、`8r` 的商与余数，再利用 `10r = 8r + 2r` 合并商与余数，最后按 `d` 的符号还原有符号余数
`new_r`。

**示例**

```python
>> > d = np.array([2 ** 62], dtype=np.int64)
>> > r = np.array([2 ** 61], dtype=np.int64)  # r / d = 0.5，乘以 10 后商为 5、余数为 0
>> > FixedArray._doubling_step_10r(r, d)
(array([5]), array([0]))
```

**另请参阅**

- [`FixedArray._chunked_divmod_round`](#fixedarray_chunked_divmod_round)
- [`FixedArray.divide`](#fixedarraydivide)

### FixedArray._chunked_divmod_round

```python
FixedArray._chunked_divmod_round(m, d, k, mode)
```

分段十进制长除法与精确舍入：计算 `round(m · 10^k / d)`，全程纯 `int64` / `uint64` 运算。

**参数**

- **m, d** : *ndarray（int64）* — 被除数与除数尾数数组。
- **k** : *int* — 正整数放大幂次（`k > 0`）。
- **mode** : *str* — 舍入模式名称。

**返回值**

- **out** : *ndarray（int64）* — 舍入后的商尾数数组。

**异常**

- **ZeroDivisionError** — `d` 含 `0`。
- **OverflowError** — 真实商或终局舍入进位后超出 `int64` 范围。

**机制说明**

先做初始带余除法 `q, r = np.divmod(m, d)`，再循环将 `10^k` 分段乘进余数 `r`：

- **常规除数（`max|d| ≤ _MAX // 10`）**：自动选取最大步长 `chunk ∈ [1, 18]` 使得 `max|d| · 10^chunk ≤ _MAX`，每次推进
  `c = min(chunk, k)` 位，直接计算 `digits, r = np.divmod(r * 10**c, d)`；
- **超大除数（`max|d| > _MAX // 10`）**：每次推进 `c = 1` 位，调用 [
  `_doubling_step_10r(r, d)`](#fixedarray_doubling_step_10r) 避免 `r * 10` 溢出；
- **精确越界检查**：每步合并 `q = q * 10^c + digits` 前，严格检验 `q * 10^c + digits` 是否落在 `[_MIN, _MAX]` 内；全部 `k`
  位推进完毕后，由终局余数 `r` 调用 [`_decide_bump`](#fixedarray_decide_bump) 进位，并额外检查 `(q == _MAX) & bump` 进位溢出。

**示例**

```python
>> >  # 100 / 3 保留 4 位小数 (k=4)：333333... -> 3333
>> > FixedArray._chunked_divmod_round(np.array([1], dtype=np.int64), np.array([3], dtype=np.int64), k=4, mode="half_up")
array([3333])
```

**另请参阅**

- [`FixedArray._doubling_step_10r`](#fixedarray_doubling_step_10r)
- [`FixedArray._round_div_pow10`](#fixedarray_round_div_pow10)
- [`FixedArray.divide`](#fixedarraydivide)

### FixedArray._round_div_pow10

```python
FixedArray._round_div_pow10(q, r, d, k, mode)
```

将带余商 `q + r/d`（其中 `q` 为整数商，`r/d ∈ [0, 1)` 为额外小数且 `r` 与 `d` 同号）整体除以 `10^k` 并按 `mode` 精确舍入。

**参数**

- **q, r, d** : *int 或 ndarray（int64）* — 整数部分 `q`、余数分子 `r` 与余数分母 `d`（若无额外分数部分，可传 `r=0, d=1`）。
- **k** : *int* — 非负整数缩小位数（`k ≥ 0`）。
- **mode** : *str* — 舍入模式名称。

**返回值**

- **out** : *ndarray（int64）* — 计算得到的 `round((q + r/d) / 10^k)` 尾数结果。

**机制说明**

该函数解决了“缩小精度或除法 `shift < 0` 时，若把除数 `d` 乘以 `10^k` 会导致 `d · 10^k` 假溢出（尤其 `k > 18` 时 `10^k`
本身已超出 `int64`）”的核心难题，按 `k` 分为四段处理：

1. **`k == 0`**：直接转发给 [`_round_quotient(q, r, d, mode)`](#fixedarray_round_quotient)。
2. **`1 ≤ k ≤ 18`**：令 `p = 10**k`，计算 `q_k, r_k = np.divmod(q, p)`，此时被舍去的总小数部分为 `(r_k + r/d) / p`。令
   `delta = 2 * r_k - p`：
    - 当 `delta > 0`，或 `delta == 0` 且 `r != 0`，或 `delta == -1` 且 `r/d > 0.5` 时，严格大于半数（`over_half = True`）；
    - 当 `delta == 0` 且 `r == 0`，或 `delta == -1` 且 `r/d == 0.5` 时，恰好等于半数（`exact_half = True`）。
3. **`k == 19`**：由于 `|q| ≤ 2**63 < 10**19`，向下取整商 `q_k` 对非负数恒为 `0`、对负数恒为 `-1`；直接将 `q` 与半数常量
   `half_19 = 5_000_000_000_000_000_000`（`5 × 10^18`）比较并结合 `r != 0` 判定 `over_half` 与 `exact_half`。
4. **`k ≥ 20`**：由于 `|q + r/d| ≤ 2**63 < 5 × 10^19 ≤ 10^k / 2`，真实值的绝对值严格小于 `0.5`，因此 `exact_half` 恒为
   `False`，在向下取整坐标系下 `over_half` 简化为 `q < 0`。

最后统一交由 [`_decide_bump(..., floor_base=True)`](#fixedarray_decide_bump) 对 `q_k` 进位。该方法同时被 [
`__init__`](#fixedarrayvalues-scalenone)、[`divide`](#fixedarraydivide)、[`mean`](#fixedarraymean) 和 [
`rescale`](#fixedarrayrescale) 复用。

**示例**

```python
>> >  # 纯整数缩小精度：(125 + 0/1) / 10^1 = 12.5 -> half_up 得 13，half_even 得 12
>> > int(FixedArray._round_div_pow10(125, 0, 1, k=1, mode="half_up"))
13
>> > int(FixedArray._round_div_pow10(125, 0, 1, k=1, mode="half_even"))
12
>> >  # 带余数缩小精度：(12 + 1/2) / 10^1 = 1.25 -> 1
>> > int(FixedArray._round_div_pow10(12, 1, 2, k=1, mode="half_up"))
1
>> >  # k=19 超大跨度缩小：5*10^18 / 10^19 = 0.5 -> half_up 得 1，half_even 得 0
>> > int(FixedArray._round_div_pow10(5_000_000_000_000_000_000, 0, 1, k=19, mode="half_up"))
1
>> > int(FixedArray._round_div_pow10(5_000_000_000_000_000_000, 0, 1, k=19, mode="half_even"))
0
```

**另请参阅**

- [`FixedArray._decide_bump`](#fixedarray_decide_bump)
- [`FixedArray._round_quotient`](#fixedarray_round_quotient)
- [`FixedArray.divide`](#fixedarraydivide)、[`FixedArray.mean`](#fixedarraymean)、[
  `FixedArray.rescale`](#fixedarrayrescale)

### FixedArray._chunked_mul_round

```python
FixedArray._chunked_mul_round(a, b, shift, mode="half_up")
```

十进制 Base $10^9$ 分块卷积长乘法与一次舍入：计算 `round((a * b) / 10^shift, mode)`。

**参数**

- **a, b** : *int 或 ndarray（int64）* — 两个乘数尾数数组，支持广播。
- **shift** : *int* — 十进制右移位数（`natural_scale - target_scale`）；可正（缩小 scale 并舍入）、可为 `0`（精确乘积）、可负（精确乘积再乘以
  `10**(-shift)`）。
- **mode** : *str，默认 `"half_up"`* — 舍入模式名称。

**返回值**

- **out** : *ndarray（int64）* — 运算并舍入后的 `int64` 尾数数组。

**异常**

- **OverflowError** — 最终舍入或放大后的结果超出 `int64` 范围 `[-2**63, 2**63 - 1]`。

**机制说明**

1. **无符号绝对值提取与三块拆分**：利用补码转换提取 `a` 与 `b` 的 `uint64` 绝对值（正确处理 `-2**63`），拆分为 Base $10^9$ 的
   3 个数字块 `a0, a1, a2` 与 `b0, b1, b2`（单块 `< 10^9`，两块乘积 `< 10^18` 仅占 `uint64` 的约 5.4%，卷积累加绝不溢出）。
2. **卷积与进位传递**：计算 5 个卷积项 `c0..c4` 并逐级传递进位，得到 6 个 Base $10^9$ 十进制数字块
   `digits = [d0, d1, d2, d3, d4, d5]`（完整表示 128 位精确乘积）。
3. **截取、舍入与补码符号还原**：
    - `shift == 0`：检查高位块全为 0 且低三块组合不越界；
    - `shift > 0`：按 `s = shift // 9` 与 `r = shift % 9` 在十进制块上直接由高位向低位除以 $10^r$，提取商块 `q`
      、紧邻的第一个被舍弃十进制位 `first_dropped` 以及更低位是否非零 `has_lower`，交由 [
      `_decide_bump(..., floor_base=False)`](#fixedarray_decide_bump) 对绝对值幅度进位；
    - `shift < 0`：精确乘积不越界时再乘以 $10^{-\text{shift}}$；
    - 最终通过补码 `(~val) + uint64(1)` 还原负数，使结果恰好等于 `-2**63` 时也不会触发任何 NumPy 溢出警告。

**示例**

```python
>> >  # 10^12 * 10^10 = 10^22（中间乘积远超 int64），除以 10^12 后为 10^10
>> > FixedArray._chunked_mul_round(10 ** 12, 10 ** 10, shift=12, mode="half_up")
array(10000000000)
```

**另请参阅**

- [`FixedArray.mul_round`](#fixedarraymul_round)
- [`FixedArray._decide_bump`](#fixedarray_decide_bump)

### FixedArray._chunked_sum

```python
FixedArray._chunked_sum(arr, axis=None, keepdims=False)
```

十进制 Base $10^9$ 分块沿轴精确求和，消除正负相抵时的中间溢出假阳性误报。

**参数**

- **arr** : *ndarray（int64）* — 输入尾数数组。
- **axis** : *None、int 或 tuple of int，可选* — 求和轴。
- **keepdims** : *bool，默认 False* — 是否保留被规约的维度。

**返回值**

- **out** : *ndarray（int64）* — 精确求和结果。

**异常**

- **OverflowError** — 最终求和结果真实超出 `int64` 范围。

**机制说明**

将每个 `int64` 元素通过两次 `np.divmod(..., 10**9)` 拆分为 3 个 Base $10^9$ 的部分：`r0, r1 ∈ [0, 10^9)` 以及最高块
`r2 ∈ [-10, 9]`。
分别对 `r0`、`r1`、`r2` 沿轴调用 `sum`（在元素个数 $N \le 9 \times 10^9$ 内单块累加绝不超出 `int64`），再将低位和的进位逐级向上折叠得到
`(s2_total, d1, d0)`。
精确比对 `(s2_total, d1, d0)` 与 `2**63 - 1`（`9_223372036_854775807`）及 `-2**63`（`-10 * 10^18 + 776627963_145224192`
）的边界，未越界时重组为 `int64` 返回。

**示例**

```python
>> > big_arr = np.array([2 ** 62, 2 ** 62, -2 ** 62], dtype=np.int64)
>> > FixedArray._chunked_sum(big_arr)
array(4611686018427387904)
```

**另请参阅**

- [`FixedArray.sum`](#fixedarraysum)

### FixedArray._chunked_matmul

```python
FixedArray._chunked_matmul(a, b)
```

二进制 Base $2^{21}$ 分块精确矩阵乘法，消除内积正负相抵时的中间溢出假阳性误报。

**参数**

- **a, b** : *ndarray（int64）* — 两个待相乘的整数尾数矩阵/张量。

**返回值**

- **out** : *ndarray（int64）* — 精确矩阵乘积尾数数组。

**异常**

- **OverflowError** — 最终内积结果真实超出 `int64` 范围。

**机制说明**

利用位运算将每个 `int64` 元素按 `21` 位拆分为 3 个二进制块：`a0, a1 ∈ [0, 2^21)` 与带符号高块 `a2 ∈ [-2^21, 2^21)`（`b`
同理）。
由于单块乘积上界为 $2^{42}$，在内积维度 $K \le 2^{20}$（约 100 万）内，9 次分块 `np.matmul(ai, bj)` 的累加值绝对不会超出
`int64`。
随后将 9 个分块乘积按位权聚合并逐级进位折叠，当且仅当 63 位以外的高位 `high` 不属于 `{0, -1}`（即符号扩展位不一致）时判定为真实溢出并抛出
`OverflowError`。

**示例**

```python
>> > x = 3_037_000_499
>> > a_mat = np.array([[x, x]], dtype=np.int64)
>> > b_mat = np.array([[x], [-x]], dtype=np.int64)
>> > FixedArray._chunked_matmul(a_mat, b_mat)
array([[0]])
```

**另请参阅**

- [`FixedArray.matmul`](#fixedarraymatmul)
- [`a @ b`](#a--b精确矩阵乘法)

## 内部比较与文本格式化辅助（私有 API）

### FixedArray._compare

```python
FixedArray._compare(other, op)
```

对齐 `scale` 后对底层尾数执行逐元素比较运算 `op`。

**参数**

- **other** : *Any* — 比较的右操作数。
- **op** : *Callable* — NumPy 比较函数（如 `np.equal`、`np.less`、`np.less_equal`、`np.greater`、`np.greater_equal`）。

**返回值**

- **out** : *numpy.bool_、ndarray of bool 或 NotImplemented* — `other` 不在数值白名单内时返回 `NotImplemented`；否则返回比较结果。

**异常**

- **ValueError** — 形状无法广播，或 `other` 含 `NaN`。
- **OverflowError** — 对齐 `scale` 时超出 `int64`。

**机制说明**

先调用 [`_coerce(other, strict=False)`](#fixedarray_coerce)，若返回 `None` 则返回 `NotImplemented`；否则调用 [
`_align_scales`](#fixedarray_align_scales) 对齐精度后返回 `op(a._mantissa, b._mantissa)`。六个比较运算符（`==`、`!=`、`<`、
`<=`、`>`、`>=`）均共用此入口。

**示例**

```python
>> > bool(FixedArray("1.5")._compare(FixedArray("1.50"), np.equal))
True
>> > FixedArray("1.5")._compare("1.5", np.equal) is NotImplemented
True
```

**另请参阅**

- [`a == b`、`a != b`](#a--b-与-a--b相等性比较)
- [`a < b` 等大小比较](#a--ba--ba--ba--b大小比较)

### FixedArray._format_scalar

```python
FixedArray._format_scalar(m, scale)
```

将单个整数尾数 `m` 按 `scale` 格式化为固定小数位的十进制字符串。

**参数**

- **m** : *int* — 单个元素的整数尾数。
- **scale** : *int* — 非负小数位数。

**返回值**

- **out** : *str* — 格式化后的十进制文本（保留尾随零）。

**机制说明**

- `scale == 0` 时直接返回 `str(m)`；
- `scale > 0` 时提取符号位，将 `str(abs(m)).zfill(scale + 1)` 补齐前导零，再切分为整数部分 `digits[:-scale]` 与小数部分
  `digits[-scale:]` 拼接输出，全程純字符串与整数操作，不经过浮点数。

**示例**

```python
>> > FixedArray._format_scalar(-5, scale=2)
'-0.05'
>> > FixedArray._format_scalar(123450, scale=3)
'123.450'
```

**另请参阅**

- [`str(a)` 与 `repr(a)`](#stra-与-repra文本表示)
