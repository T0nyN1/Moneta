# FixedArray — 定点数张量引擎

`moneta.FixedArray` 是基于 `int64` 尾数的定点数 N 维数组，`moneta.RoundingMode` 是配套的舍入模式枚举。
它们是 Moneta 全部精确运算的数值内核：加减乘、求和、矩阵乘法全程走整数运算，
不会产生 IEEE 754 浮点累积误差；舍入只在你明确要求的地方、按你指定的规则发生一次；
结果超出 `int64` 时一定报错，不会静默回绕。

```python
>>> a = FixedArray(["1.10", "2.25", "-0.05"])
>>> a
FixedArray([1.10, 2.25, -0.05], scale=2)
>>> a * 2
FixedArray([2.20, 4.50, -0.10], scale=2)
>>> a.sum()
FixedArray(3.30, scale=2)
>>> bool(FixedArray("0.1") + FixedArray("0.2") == FixedArray("0.3"))
True
```

[← API 参考](index.md)

## 核心机制与设计约束

### 定点数表示、三槽位结构与不可变性

每个 `FixedArray` 实例仅包含三个槽位 `__slots__ = ("_mantissa", "_scale", "_mask")`：

$$v = \text{mantissa} \times 10^{-\text{scale}}$$

- **`mantissa`**：只读的 `int64` NumPy 数组，形状即 `FixedArray` 的形状；
- **`scale`**：整个数组共享的非负整数（十进制小数位数）；
- **`mask`**：可选的只读 `bool` NumPy 数组（`True` 表示缺失值 `<NA>`）；当不存在任何缺失值时恒为 `None`（零额外内存与计算开销），当存在缺失值时，
  **缺失位置上的底层 `mantissa` 强制恒为 `0`**（由 [`_set`](#fixedarray_set) 统一收口保证）。

`FixedArray` 是 **不可变**的：没有原地运算，`mantissa` 与 `mask` 均为只读视图，不支持元素赋值，不可哈希（`__hash__ = None`
）。形状变换、索引、`rescale` 到相同 scale 等操作直接共享尾数与掩码视图而不复制。`FixedArray`
只认识“数”，不认识币种与单位——带量纲的运算由上层的 [`UnitArray`](unit_array.md) / [`Money`](money.md) / [
`Price`](price.md) / [`Quantity`](quantity.md) 负责。

### 缺失值掩码体系（Null / NA Mask）

- **自动识别与显式掩码**：构造器自动将 `None`、`float("nan")` / `np.nan`、`Decimal("NaN")`、文本 `"nan"` / `"NaN"` 以及
  `pandas.NA` 识别为缺失值 `<NA>`，同时支持通过 `mask=` 参数显式传入布尔掩码（与输入自带的缺失值取并集）。
- **算术与矩阵掩码传播**：
  - 逐元素运算（`+`、`-`、`*`、`mul_round`、`/`、`divide`、`//`、`%`、`divmod`、`rescale`、`round_to_step`）通过 [
    `_combine_masks`](#fixedarray_combine_masks-与-fixedarray_matmul_mask) 按位或（`|`）传播掩码；缺失位置的被除数自动置
    `0`、除数临时置 `1`，绝不会因缺失位置触发 `ZeroDivisionError` 或 `OverflowError`。
  - 矩阵乘法（`@`、`matmul`）通过 [`_matmul_mask`](#fixedarray_combine_masks-与-fixedarray_matmul_mask) 传播掩码：只要参与内积点乘的任一元素为
    `<NA>`，对应的输出单元格即为 `<NA>`。
- **规约统计（`skipna`）**：[`sum`](#fixedarraysum) 与 [`mean`](#fixedarraymean) 均提供 `skipna: bool = True` 参数。
  `skipna=True`（默认）时忽略 `<NA>`（`sum` 将 `<NA>` 视作 `0`；`mean` 除以有效非缺失元素个数，若整个切片全为 `<NA>` 则返回
  `<NA>`）；`skipna=False` 时含 `<NA>` 的切片结果即为 `<NA>`。
- **比较与相等性语义**：在 `<NA>` 位置上，`!=` 恒为 `True`，而 `==`、`<`、`<=`、`>`、`>=` 恒为 `False`；若需在单元测试或数据分析中将同位置的
  `<NA>` 视为相等，可使用 `np.array_equal(a, b, equal_nan=True)` 或 `np.array_equiv(a, b, equal_nan=True)`。

### scale 传播规则

| 运算                                                  | 结果 scale                                                                        | 是否舍入                  |
|-------------------------------------------------------|-----------------------------------------------------------------------------------|---------------------------|
| `a + b`、`a - b`、`a % b`                             | `max(a.scale, b.scale)`                                                           | 精确                      |
| `+a`、`-a`、`abs(a)`、`a.sum(axis, keepdims, skipna)` | `a.scale`                                                                         | 精确                      |
| `a * b`、`a @ b`                                      | `a.scale + b.scale`                                                               | 精确                      |
| `a / b`                                               | `max(a.scale, b.scale)`                                                           | `half_up`                 |
| `a // b`                                              | `0`（返回 `scale=0` 的 `FixedArray`，传播 `mask`）                                | 向下取整                  |
| `a.mul_round(b, scale, rounding)`                     | `scale`，默认 `max(a.scale, b.scale)`（`rounding=None` 时为 `a.scale + b.scale`） | 按 `rounding`，只舍入一次 |
| `a.matmul(b, scale, rounding)`                        | `scale`，默认 `max(a.scale, b.scale)`（`rounding=None` 时为 `a.scale + b.scale`） | 按 `rounding`，只舍入一次 |
| `a.divide(b, scale, rounding)`                        | `scale`，默认 `max(a.scale, b.scale)`                                             | 按 `rounding`             |
| `a.mean(axis, keepdims, scale, rounding, skipna)`     | `scale`，默认 `a.scale`                                                           | 按 `rounding`             |
| `a.rescale(new_scale, rounding)`                      | `new_scale`                                                                       | 缩小时按 `rounding`       |
| `a.round_to_step(step, rounding)`                     | `max(a.scale, step.scale)`                                                        | 按 `rounding`             |

### 操作数白名单与溢出保证

**操作数白名单。** 运算、比较与 `mul_round` / `divide` / `round_to_step` / `fillna` 等方法的另一方只能是：`FixedArray`、
`int`、`float`、`Decimal`、`Fraction`、`None`、整数或浮点 `ndarray`，以及只含这些元素的 `list` / `tuple` / `object` 数组（见 [
`_is_numeric_operand`](#fixedarray_is_numeric_operand)）。字符串、布尔值（含布尔数组）与带单位对象一律拒绝：运算符返回
`NotImplemented`，方法直接抛出 `TypeError`。

**快慢双路径与底层引擎分层。** 所有运算均在原生 `int64` / `uint64` 内完成，不升级为 Python `object` 大整数，也不退化为
`float`。舍入规则与进位裁决统一由 [`moneta._rounding`](_rounding.md) 提供，而多精度无溢出慢路径统一由 [
`moneta._limbs`](_limbs.md)（二进制 Base $2^{21}$ 动态分块内核）提供：

| 运算             | 快路径（绝大多数输入）                                                                               | 慢路径（最坏上界超出 `int64` 时自动切换）                                                                                                    |
|------------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `+` `-`          | 一次整数加减 + 符号位异或溢出检测                                                                    | —                                                                                                                                            |
| `*`、`mul_round` | `max                                                                                                 | a                                                                                                                                            | · max|b| <= 2^63 - 1` 且 `0 <= shift <= 18` 时单指令乘法 + 带余除法 | [`limbs_mul`](_limbs.md#limbs_mul)（Base $2^{21}$ 的 $3\times 3$ 块卷积）+ [`_limbs_to_mantissa`](#fixedarray_limbs_to_mantissa) |
| `@`、`matmul`    | `max                                                                                                 | a                                                                                                                                            | · max|b| · K <= 2^63 - 1` 时直接调用 `np.matmul` | [`limbs_matmul`](_limbs.md#limbs_matmul)（按 `STEP_K = 2^19` 分段 Base $2^{21}$ 块内积）+ [`_limbs_to_mantissa`](#fixedarray_limbs_to_mantissa) |
| `sum`            | `max                                                                                                 | a                                                                                                                                            | · count <= 2^63 - 1` 时直接调用 `np.sum` | [`limbs_sum`](_limbs.md#limbs_sum)（拆为 3 个 21 位块沿轴求和后进位规整收口） |
| `divide`         | [`round_quotient`](_rounding.md#round_quotient) 或 [`_round_div_pow10`](#fixedarray_round_div_pow10) | [`_chunked_divmod_round`](#fixedarray_chunked_divmod_round) 分段长除法；大除数自动切换 [`_doubling_step_10r`](#fixedarray_doubling_step_10r) |

## API 速查表

| 分类                   | 接口 / 运算符                                                                                                                                                                                                                                                                                                                                  | 功能说明                                                                 |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **舍入模式**           | [`RoundingMode`](#roundingmode)                                                                                                                                                                                                                                                                                                                | 六种金融与会计舍入模式枚举                                               |
| **构造与属性**         | [`FixedArray(values, scale=None, mask=None)`](#fixedarrayvalues-scalenone-masknone)                                                                                                                                                                                                                                                            | 从数值、字符串、`Decimal`、`Fraction`、数组构造                          |
|                        | [`FixedArray.mantissa`](#fixedarraymantissa)                                                                                                                                                                                                                                                                                                   | 只读 `int64` 尾数数组视图                                                |
|                        | [`FixedArray.scale`](#fixedarrayscale)                                                                                                                                                                                                                                                                                                         | 全局十进制小数位数                                                       |
|                        | [`FixedArray.shape`](#fixedarrayshape) / [`ndim`](#fixedarrayndim) / [`size`](#fixedarraysize) / [`dtype`](#fixedarraydtype)                                                                                                                                                                                                                   | 形状、维数、元素总数、尾数数据类型                                       |
|                        | [`FixedArray.T`](#fixedarrayt)                                                                                                                                                                                                                                                                                                                 | 反转全部轴的转置视图                                                     |
| **缺失值（NA）处理**   | [`FixedArray.mask`](#fixedarraymask)                                                                                                                                                                                                                                                                                                           | 只读缺失值布尔掩码视图（无缺失值时为 `None`）                            |
|                        | [`FixedArray.isna`](#fixedarrayisnafixedarrayisnullfixedarraynotna-与-fixedarraynotnull) / [`isnull`](#fixedarrayisnafixedarrayisnullfixedarraynotna-与-fixedarraynotnull) / [`notna`](#fixedarrayisnafixedarrayisnullfixedarraynotna-与-fixedarraynotnull) / [`notnull`](#fixedarrayisnafixedarrayisnullfixedarraynotna-与-fixedarraynotnull) | 返回标记缺失位置或有效数值位置的布尔矩阵                                 |
|                        | [`FixedArray.fillna`](#fixedarrayfillna)                                                                                                                                                                                                                                                                                                       | 用指定数值填充缺失位置                                                   |
|                        | [`FixedArray.dropna`](#fixedarraydropna)                                                                                                                                                                                                                                                                                                       | 丢弃 1 维向量中的缺失值                                                  |
| **算术运算**           | [`+a`、`-a`](#a-与--a一元正负号)                                                                                                                                                                                                                                                                                                               | 一元正负号                                                               |
|                        | [`abs(a)`](#absa绝对值)                                                                                                                                                                                                                                                                                                                        | 逐元素绝对值                                                             |
|                        | [`a + b`、`a - b`](#a--b-与-a---b加减法)                                                                                                                                                                                                                                                                                                       | 逐元素加减法，自动对齐 scale                                             |
|                        | [`a * b`](#a--b精确乘法)                                                                                                                                                                                                                                                                                                                       | 逐元素精确乘法（scale 相加）                                             |
|                        | [`FixedArray.mul_round`](#fixedarraymul_round)                                                                                                                                                                                                                                                                                                 | 融合乘法 + 一次舍入                                                      |
|                        | [`a / b`](#a--b真除法)                                                                                                                                                                                                                                                                                                                         | 逐元素真除法（默认保留 `max(scale)` 位并 `half_up`）                     |
|                        | [`FixedArray.divide`](#fixedarraydivide)                                                                                                                                                                                                                                                                                                       | 指定目标 scale 与舍入模式的精确除法                                      |
|                        | [`a // b`、`a % b`、`divmod(a, b)`](#a--ba--b-与-divmoda-b整除与取余)                                                                                                                                                                                                                                                                          | 向下整除（返回 `scale=0` 的 `FixedArray`）、取余与带余除法               |
| **矩阵运算**           | [`a @ b`](#a--b精确矩阵乘法)                                                                                                                                                                                                                                                                                                                   | 精确矩阵乘法（scale 相加）                                               |
|                        | [`FixedArray.matmul`](#fixedarraymatmul)                                                                                                                                                                                                                                                                                                       | 矩阵乘法 + 一次舍入                                                      |
| **规约统计**           | [`FixedArray.sum`](#fixedarraysum)                                                                                                                                                                                                                                                                                                             | 沿轴精确求和（支持 `skipna=True`）                                       |
|                        | [`FixedArray.mean`](#fixedarraymean)                                                                                                                                                                                                                                                                                                           | 沿轴求算术均值并精确舍入（支持 `skipna=True`）                           |
| **精度与网格**         | [`FixedArray.rescale`](#fixedarrayrescale)                                                                                                                                                                                                                                                                                                     | 调整小数位数                                                             |
|                        | [`FixedArray.round_to_step`](#fixedarrayround_to_step)                                                                                                                                                                                                                                                                                         | 对齐到步长网格（最小变动价位、整手）                                     |
|                        | [`FixedArray.is_on_step`](#fixedarrayis_on_step)                                                                                                                                                                                                                                                                                               | 逐元素判断是否落在步长网格上                                             |
| **比较运算**           | [`a == b`、`a != b`](#a--b-与-a--b相等性比较)                                                                                                                                                                                                                                                                                                  | 无溢出跨 scale 逐元素判断相等或不等                                      |
|                        | [`a < b`、`a <= b`、`a > b`、`a >= b`](#a--ba--ba--ba--b大小比较)                                                                                                                                                                                                                                                                              | 无溢出跨 scale 逐元素比较大小                                            |
| **形状与协议**         | [`FixedArray.reshape`](#fixedarrayreshape)                                                                                                                                                                                                                                                                                                     | 不复制数据改变数组形状                                                   |
|                        | [`FixedArray.transpose`](#fixedarraytranspose)                                                                                                                                                                                                                                                                                                 | 按指定顺序置换各轴                                                       |
|                        | [`FixedArray.swapaxes`](#fixedarrayswapaxes-与-fixedarraysqueeze) / [`FixedArray.squeeze`](#fixedarrayswapaxes-与-fixedarraysqueeze)                                                                                                                                                                                                           | 交换两个轴 / 移除长度为 1 的轴                                           |
|                        | [`a[key]`](#akey索引与切片)                                                                                                                                                                                                                                                                                                                    | 索引、切片、布尔掩码与花式索引                                           |
|                        | [`len(a)`、`bool(a)`、`iter(a)`](#lenaboola-与-itera容器与真值协议)                                                                                                                                                                                                                                                                            | 首维长度、单元素真值测试与按首维迭代                                     |
| **导出与互操作**       | [`FixedArray.to_float`](#fixedarrayto_float)                                                                                                                                                                                                                                                                                                   | 导出为 `float` / `float64` 数组（缺失值转 `nan`）                        |
|                        | [`FixedArray.to_decimal`](#fixedarrayto_decimal)                                                                                                                                                                                                                                                                                               | 导出为 `Decimal` / `object` 数组（缺失值转 `None`）                      |
|                        | [`FixedArray.to_pandas`](#fixedarrayto_pandas)                                                                                                                                                                                                                                                                                                 | 导出为 `pandas.Series` / `DataFrame`                                     |
|                        | [`FixedArray.to_polars`](#fixedarrayto_polars)                                                                                                                                                                                                                                                                                                 | 导出为 `polars.Series`                                                   |
|                        | [`str(a)`、`repr(a)`](#stra-与-repra文本表示)                                                                                                                                                                                                                                                                                                  | 按 scale 补零的精确文本表示（缺失值显示 `<NA>`）                         |
|                        | [NumPy 互操作机制](#numpy-互操作机制)                                                                                                                                                                                                                                                                                                          | 可用与被禁止的 NumPy 函数（含 `np.array_equal`）、常见写法配方           |
| **内部构造与转换**     | [`FixedArray._set`](#fixedarray_set)                                                                                                                                                                                                                                                                                                           | 构造收口：绑定只读 `int64` 尾数视图、`scale` 与只读 `mask`               |
|                        | [`FixedArray._from_mantissa`](#fixedarray_from_mantissa)                                                                                                                                                                                                                                                                                       | 由已有 `int64` 尾数、`scale` 与可选 `mask` 零拷贝构造                    |
|                        | [`FixedArray._combine_masks`](#fixedarray_combine_masks-与-fixedarray_matmul_mask) / [`_matmul_mask`](#fixedarray_combine_masks-与-fixedarray_matmul_mask)                                                                                                                                                                                     | 逐元素按位或合并掩码 / 矩阵内积掩码传播                                  |
|                        | [`FixedArray._is_numeric_operand`](#fixedarray_is_numeric_operand)                                                                                                                                                                                                                                                                             | 校验对象是否属于合法的数值操作数白名单                                   |
|                        | [`FixedArray._coerce`](#fixedarray_coerce)                                                                                                                                                                                                                                                                                                     | 将运算另一侧的操作数安全转换为 `FixedArray`                              |
|                        | [`FixedArray._lose_precision`](#fixedarray_lose_precision)                                                                                                                                                                                                                                                                                     | 判断缩减到 `target_scale` 是否会丢弃非零有效小数位                       |
|                        | [`FixedArray._check_scale`](#fixedarray_check_scale)                                                                                                                                                                                                                                                                                           | 校验 `scale` 是否为非负整数                                              |
|                        | [`FixedArray._as_int64`](#fixedarray_as_int64)                                                                                                                                                                                                                                                                                                 | 将整数标量或整数数组转换为 `int64` 数组并检查越界                        |
|                        | [`FixedArray._align_scales`](#fixedarray_align_scales)                                                                                                                                                                                                                                                                                         | 将两个 `FixedArray` 的 `scale` 对齐到较大值                              |
| **内部输入解析**       | [`FixedArray._parse_decimal_str`](#fixedarray_parse_decimal_str)                                                                                                                                                                                                                                                                               | 解析十进制字符串为 `(mantissa, scale)`                                   |
|                        | [`FixedArray._parse_fraction`](#fixedarray_parse_fraction)                                                                                                                                                                                                                                                                                     | 解析有限十进制 `Fraction` 为 `(mantissa, scale)`                         |
|                        | [`FixedArray._parse_float_scalar`](#fixedarray_parse_float_scalar)                                                                                                                                                                                                                                                                             | 按最短十进制写法（Dragonbox）解析单个浮点数                              |
|                        | [`FixedArray._is_missing`](#fixedarray_is_missing-与-fixedarray_contains_bool) / [`_contains_bool`](#fixedarray_is_missing-与-fixedarray_contains_bool)                                                                                                                                                                                        | 判定单个元素是否为缺失值 / 检查嵌套序列中是否混有 `bool`                 |
|                        | [`FixedArray._expand_nested`](#fixedarray_expand_nested)                                                                                                                                                                                                                                                                                       | 将嵌套序列中的 `FixedArray` 替换为等值 `Decimal`（保留小数位数）         |
|                        | [`FixedArray._parse_object_element`](#fixedarray_parse_object_element)                                                                                                                                                                                                                                                                         | 解析单个标量对象为 `(mantissa, scale)`                                   |
|                        | [`FixedArray._parse_float_array`](#fixedarray_parse_float_array)                                                                                                                                                                                                                                                                               | 向量化往返试探解析浮点数组为 `(尾数数组, scale)`                         |
|                        | [`FixedArray._align_mantissas`](#fixedarray_align_mantissas)                                                                                                                                                                                                                                                                                   | 将不同 `scale` 的尾数向量化对齐到目标 `target`                           |
|                        | [`FixedArray._round_float_to_scale`](#fixedarray_round_float_to_scale)                                                                                                                                                                                                                                                                         | 将浮点数组的最短十进制写法按 `half_up` 快速舍入到指定 `scale`            |
|                        | [`FixedArray._parse_text_array`](#fixedarray_parse_text_array)                                                                                                                                                                                                                                                                                 | 去重解析字符串、`Decimal`、`Fraction` 或混合对象数组并返回掩码           |
| **底层数学与舍入内核** | [`FixedArray._check_division`](#fixedarray_check_division)                                                                                                                                                                                                                                                                                     | 校验非缺失位置的除数非零及 `-2**63 / -1` 溢出边界                        |
|                        | [`FixedArray._max_abs`](#fixedarray_max_abs)                                                                                                                                                                                                                                                                                                   | 安全计算 `int64` 数组绝对值的最大值（`-2**63` 不溢出）                   |
|                        | [`FixedArray._mul_pow10`](#fixedarray_mul_pow10) / [`_mul_pow10_float`](#fixedarray_mul_pow10)                                                                                                                                                                                                                                                 | 带前置越界检查的 `x · 10^k` 整数放大与防溢出两步浮点放大                 |
|                        | [`FixedArray._divmod_round`](#fixedarray_divmod_round)                                                                                                                                                                                                                                                                                         | 纯 `int64` 整数带余除法与精确舍入                                        |
|                        | [`FixedArray._doubling_step_10r`](#fixedarray_doubling_step_10r)                                                                                                                                                                                                                                                                               | 大除数无溢出二进制倍增单步长除（计算 `(10r) // d` 与 `(10r) % d`）       |
|                        | [`FixedArray._chunked_divmod_round`](#fixedarray_chunked_divmod_round)                                                                                                                                                                                                                                                                         | 分段十进制长除法与舍入 `round(m · 10^k / d)`                             |
|                        | [`FixedArray._round_div_pow10`](#fixedarray_round_div_pow10)                                                                                                                                                                                                                                                                                   | 将带余商 `(q + r/d)` 除以 `10^k` 并精确舍入（支持任意 `k ≥ 0`）          |
|                        | [`FixedArray._limbs_to_mantissa`](#fixedarray_limbs_to_mantissa)                                                                                                                                                                                                                                                                               | 将 [`_limbs`](_limbs.md) 多精度分块结果按目标 `shift` 舍入收口为 `int64` |
|                        | [`moneta._rounding`](_rounding.md) / [`moneta._limbs`](_limbs.md)                                                                                                                                                                                                                                                                              | 独立拆分的统一舍入裁决引擎与 Base $2^{21}$ 多精度分块整数算术模块        |
| **内部比较与格式化**   | [`FixedArray._compare_fixed_array`](#fixedarray_compare_fixed_arrayfixedarray_compare-与-fixedarray_eq_unrepresentable) / [`_compare`](#fixedarray_compare_fixed_arrayfixedarray_compare-与-fixedarray_eq_unrepresentable) / [`_eq_unrepresentable`](#fixedarray_compare_fixed_arrayfixedarray_compare-与-fixedarray_eq_unrepresentable)       | 无溢出整除跨 scale 比较网关与不可表示值逐元素判等                        |
|                        | [`FixedArray._array_function`](#fixedarray_array_function-与-fixedarray_numpy_equal) / [`_numpy_equal`](#fixedarray_array_function-与-fixedarray_numpy_equal)                                                                                                                                                                                  | `__array_function__` 白名单分派与 `np.array_equal` / `array_equiv` 实现  |
|                        | [`FixedArray._format_scalar`](#fixedarray_format_scalar)                                                                                                                                                                                                                                                                                       | 将单个整数尾数格式化为固定小数位字符串                                   |

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
>>> from moneta import RoundingMode
>>> x = FixedArray(["2.5", "3.5", "2.4", "-2.5", "-2.6"])
>>> for mode in RoundingMode:
...     print(f"{mode.value:>9}", x.rescale(0, rounding=mode))
  half_up [3, 4, 2, -3, -3]
half_even [2, 4, 2, -2, -3]
     down [2, 3, 2, -2, -2]
       up [3, 4, 3, -3, -3]
    floor [2, 3, 2, -3, -3]
     ceil [3, 4, 3, -2, -2]
>>> RoundingMode.HALF_EVEN == "half_even"
True
>>> FixedArray("0.125").rescale(2, RoundingMode.HALF_EVEN)
FixedArray(0.12, scale=2)
>>> FixedArray("0.125").rescale(2, "nearest")
Traceback (most recent call last):
    ...
ValueError: rounding 只能是 ('half_up', 'half_even', 'down', 'up', 'floor', 'ceil') 之一，收到 'nearest'
```

**另请参阅**

- [`FixedArray.rescale`](#fixedarrayrescale) – 最直接的舍入入口。

## 构造与核心属性

### `FixedArray(values, scale=None, mask=None)`

```python
class moneta.FixedArray(values, scale=None, mask=None)
```

从数值、字符串、`Decimal`、`Fraction`、序列或 NumPy 数组构造定点数数组。

**参数**

- **values** : *int, float, str, Decimal, Fraction, array_like 或 FixedArray* — 输入数值。标量得到 0 维数组；
  也可以是嵌套序列或 `ndarray`。不支持 `bool`、复数、日期时间等类型。
- **scale** : *int，可选* — 结果的小数位数，必须是非负整数（`int` 或 NumPy 整数，不接受 `bool`、`float`）。
  默认 `None` 时按下表自动推断。
- **mask** : *bool 或 bool array_like，可选* — 缺失值布尔掩码（`True` 表示对应位置为缺失值 `<NA>`），可广播到 `values` 的形状；与
  `values` 中自带的缺失值（`None`、`NaN`、`Decimal("NaN")`、文本 `"nan"`、`pandas.NA`）取并集。

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
>>> FixedArray(5)
FixedArray(5, scale=0)
>>> FixedArray(5, scale=2)
FixedArray(5.00, scale=2)
>>> FixedArray(0, scale=25)  # 全零数组支持超过 18 位的任意 scale
FixedArray(0.0000000000000000000000000, scale=25)
>>> from decimal import Decimal
>>> FixedArray("1.50")
FixedArray(1.50, scale=2)
>>> FixedArray(Decimal("1.100"))
FixedArray(1.100, scale=3)
>>> FixedArray("1.5e-3")
FixedArray(0.0015, scale=4)
>>> FixedArray(" 1_000.50 ")
FixedArray(1000.50, scale=2)
```

浮点数按最短写法解析，数组对齐到最大位数：

```python
>>> FixedArray(0.1)
FixedArray(0.1, scale=1)
>>> FixedArray([0.1, 0.25])
FixedArray([0.10, 0.25], scale=2)
>>> FixedArray(np.array([0.1, 2.5], dtype=np.float32))
FixedArray([0.1, 2.5], scale=1)
>>> FixedArray(2.675, scale=2)  # 在 "2.675" 上 half_up
FixedArray(2.68, scale=2)
>>> round(2.675, 2)  # 对比：内置 round 受二进制误差影响
2.67
```

显式 `scale` 时每个元素直接舍入，不会因为全精度对齐或无限小数而溢出：

```python
>>> big_and_fine = np.array([123456789012.5, 0.123456789])
>>> FixedArray(big_and_fine)  # 需对齐到 9 位小数，尾数越界
Traceback (most recent call last):
    ...
OverflowError: 浮点数需要超过 7 位小数，对齐后超出 int64 范围
>>> FixedArray(big_and_fine, scale=2)
FixedArray([123456789012.50, 0.12], scale=2)
>>> FixedArray(np.array([100000.0, 1 / 3]), scale=2)
FixedArray([100000.00, 0.33], scale=2)
```

`Fraction`、混合类型与另一个 `FixedArray`：

```python
>>> from fractions import Fraction
>>> FixedArray(Fraction(3, 8))
FixedArray(0.375, scale=3)
>>> FixedArray([1, Decimal("2.5"), "0.125"])
FixedArray([1.000, 2.500, 0.125], scale=3)
>>> FixedArray([FixedArray("1.50"), 2])      # 序列中含 FixedArray 元素，自动保留各自 scale
FixedArray([1.50, 2.00], scale=2)
>>> FixedArray([FixedArray([1, 2]), FixedArray(["3.5", "4"])])  # 相同形状的 FixedArray 沿首维堆叠
FixedArray([[1.0, 2.0],
            [3.5, 4.0]], scale=1)
>>> FixedArray(FixedArray("1.25"), scale=1)  # 等价于 rescale(1)
FixedArray(1.3, scale=1)
```

非法输入：

缺失值（`None`、`NaN`、`Decimal("NaN")`、`"nan"`）与显式 `mask`：

```python
>> > FixedArray(None), FixedArray(float("nan"))
(FixedArray( < NA >, scale=0), FixedArray( < NA >, scale = 0))
>> > FixedArray(["1.50", None, "nan", "2.25"])
FixedArray([1.50, < NA >, < NA >, 2.25], scale = 2)
>> > FixedArray([10, 20, 30], scale=1, mask=[False, True, False])
FixedArray([10.0, < NA >, 30.0], scale = 1)
```

非法输入：

```python
>>> FixedArray(Fraction(1, 3))
Traceback (most recent call last):
    ...
ValueError: 1/3 不是有限小数
>>> FixedArray("1.234", scale=2)
Traceback (most recent call last):
    ...
ValueError: 输入的小数位数 (3) 超出指定 scale (2)
>> > FixedArray(float("inf"))
Traceback (most recent call last):
    ...
ValueError: FixedArray
不能包含无穷大(inf)
>>> FixedArray(True)
Traceback (most recent call last):
    ...
TypeError: 不支持从 bool (dtype=bool) 构造 FixedArray
>>> FixedArray(2 ** 63)
Traceback (most recent call last):
    ...
OverflowError: 整数超出 int64 范围
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
>>> a = FixedArray(["1.50", "-2.25"])
>>> a.mantissa
array([ 150, -225])
>>> FixedArray("19.99").mantissa
array(1999)
>>> a.mantissa[0] = 0
Traceback (most recent call last):
    ...
ValueError: assignment destination is read-only
>>> prices = FixedArray(["10.05", "10.20", "9.95"])
>>> prices[int(np.argmax(prices.mantissa))]
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
>>> FixedArray("1.50").scale
2
>>> FixedArray(7).scale
0
>>> FixedArray([0.1, 0.125]).scale
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
>>> FixedArray([[1, 2, 3], [4, 5, 6]]).shape
(2, 3)
>>> FixedArray("1.5").shape
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
>>> FixedArray("1.5").ndim
0
>>> FixedArray([[1, 2], [3, 4]]).ndim
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
>>> FixedArray([[1, 2, 3], [4, 5, 6]]).size
6
>>> FixedArray("1.5").size
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
>>> FixedArray("1.5").dtype
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
>>> FixedArray([["1.1", "2.2", "3.3"], ["4.4", "5.5", "6.6"]]).T
FixedArray([[1.1, 4.4],
            [2.2, 5.5],
            [3.3, 6.6]], scale=1)
```

**另请参阅**

- [`FixedArray.transpose`](#fixedarraytranspose)
- [`FixedArray.reshape`](#fixedarrayreshape)

## 缺失值（NA）查询、填充与过滤

### FixedArray.mask

```python
FixedArray.mask
```

返回底层缺失值掩码矩阵（只读 `bool` 视图，`True` 表示缺失值 `<NA>`）；若当前数组中不存在任何缺失值则返回 `None`。

**返回值**

- **mask** : *numpy.ndarray（bool）或 None*

**示例**

```python
>> > FixedArray(["1.5", "2.0"]).mask is None
True
>> > m = FixedArray(["1.5", None, "2.0"]).mask
>> > m, m.flags.writeable
(array([False, True, False]), False)
```

**另请参阅**

- [`FixedArray.isna`](#fixedarrayisnafixedarrayisnullfixedarraynotna-与-fixedarraynotnull)

### `FixedArray.isna`、`FixedArray.isnull`、`FixedArray.notna` 与 `FixedArray.notnull`

```python
FixedArray.isna()
FixedArray.isnull()
FixedArray.notna()
FixedArray.notnull()
```

返回与当前 `FixedArray` 同形状的布尔 `ndarray`，标记哪些位置是缺失值（`isna` / `isnull`）或有效数值（`notna` / `notnull`）。

**返回值**

- **out** : *numpy.ndarray（bool）* — 始终返回可独立修改的新布尔数组（即使 `self.mask is None` 也会返回全 `False` 或全
  `True` 的布尔数组）。

**示例**

```python
>> > a = FixedArray(["1.50", None, "2.25"])
>> > a.isna(), a.notna()
(array([False, True, False]), array([True, False, True]))
>> > bool((a.isnull() == a.isna()).all() and (a.notnull() == a.notna()).all())
True
```

**另请参阅**

- [`FixedArray.mask`](#fixedarraymask)
- [`FixedArray.fillna`](#fixedarrayfillna)、[`FixedArray.dropna`](#fixedarraydropna)

### FixedArray.fillna

```python
FixedArray.fillna(value)
```

用给定的无量纲数值或数组填充所有缺失位置（`<NA>`），自动对齐 `scale` 并返回无缺失值的新 `FixedArray`。

**参数**

- **value** : *int, float, Decimal, Fraction, FixedArray 或可广播的数值 array_like* — 填充值。在 `self` 需要被填充的位置上，
  `value` 本身不能是缺失值。

**返回值**

- **out** : *FixedArray* — `scale = max(self.scale, value.scale)` 且 `out.mask is None` 的新实例（若原数组无缺失值则直接返回
  `self`）。

**异常**

- **TypeError** — `value` 不在数值白名单内（如传入字符串）。
- **ValueError** — `value` 在需要填充的位置上含有缺失值（`<NA>` / `None` / `NaN`），或形状无法广播。

**示例**

```python
>> > a = FixedArray(["1.5", None, "2.25"])
>> > a.fillna(0)
FixedArray([1.50, 0.00, 2.25], scale=2)
>> > a.fillna(FixedArray(["0", "9.999", "0"]))
FixedArray([1.500, 9.999, 2.250], scale=3)
>> > a.fillna(None)
Traceback(most
recent
call
last):
...
ValueError: fillna
的填充值不能是缺失值
```

**另请参阅**

- [`FixedArray.dropna`](#fixedarraydropna)
- [`FixedArray.isna`](#fixedarrayisnafixedarrayisnullfixedarraynotna-与-fixedarraynotnull)

### FixedArray.dropna

```python
FixedArray.dropna()
```

移除 1 维 `FixedArray` 中的所有缺失值（`<NA>`），返回仅包含有效元素的新 1 维 `FixedArray`。

**返回值**

- **out** : *FixedArray* — `scale` 不变、`mask` 为 `None` 的 1 维 `FixedArray`。

**异常**

- **ValueError** — `self.ndim != 1`（报 `"dropna() 仅支持 1 维 FixedArray"`）。

**示例**

```python
>> > FixedArray(["1.50", None, "2.25", None]).dropna()
FixedArray([1.50, 2.25], scale=2)
>> > FixedArray([[1, None], [2, 3]]).dropna()
Traceback(most
recent
call
last):
...
ValueError: dropna()
仅支持
1
维
FixedArray
```

**另请参阅**

- [`FixedArray.fillna`](#fixedarrayfillna)

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
>>> a = FixedArray(["1.5", "-2"])
>>> -a
FixedArray([-1.5, 2.0], scale=1)
>>> (+a) is a
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

### `abs(a)`（绝对值）

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
>>> abs(FixedArray(["-1.5", "2"]))
FixedArray([1.5, 2.0], scale=1)
>>> np.abs(FixedArray("-1.5"))
Traceback (most recent call last):
    ...
TypeError: operand 'FixedArray' does not support ufuncs (__array_ufunc__=None)
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
>>> FixedArray("1.10") + FixedArray("0.005")
FixedArray(1.105, scale=3)
>>> 0.1 + 0.2  # 浮点误差
0.30000000000000004
>>> FixedArray(0.1) + 0.2  # float 操作数按最短写法解析
FixedArray(0.3, scale=1)
>>> FixedArray(["100.00", "50.00"]) - FixedArray("0.01")
FixedArray([99.99, 49.99], scale=2)
>>> 10 - FixedArray("0.01")  # 反向运算
FixedArray(9.99, scale=2)
>>> FixedArray([[1], [2]]) + FixedArray([10, 20])  # 广播
FixedArray([[11, 21],
            [12, 22]], scale=0)
>>> np.array([1, 2]) + FixedArray("0.5")  # ndarray 在左侧也返回 FixedArray
FixedArray([1.5, 2.5], scale=1)
>>> FixedArray(1) + "1"
Traceback (most recent call last):
    ...
TypeError: unsupported operand type(s) for +: 'FixedArray' and 'str'
```

对齐 scale 也可能溢出——`10**17` 在 scale 2 下的尾数是 `10**19`：

```python
>>> FixedArray(10 ** 17) + FixedArray("0.01")
Traceback (most recent call last):
    ...
OverflowError: 乘以 10^2 后超出 int64 范围
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
>>> FixedArray("19.99") * 3
FixedArray(59.97, scale=2)
>>> FixedArray("1.25") * FixedArray("0.5")
FixedArray(0.625, scale=3)
>>> FixedArray("1.1") * 1.1
FixedArray(1.21, scale=2)
>>> 1.1 * 1.1
1.2100000000000002
>>> FixedArray("1.01") * FixedArray("1.01") * FixedArray("1.01")  # scale 累加
FixedArray(1.030301, scale=6)
>>> FixedArray(10 ** 10) * FixedArray(10 ** 10)
Traceback (most recent call last):
    ...
OverflowError: 结果超出
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
>>> amount = FixedArray("100.00")
>>> rate = FixedArray("0.0325")
>>> amount.mul_round(rate)  # 默认 scale = max(2, 4)
FixedArray(3.2500, scale=4)
>>> amount.mul_round(rate, scale=2)
FixedArray(3.25, scale=2)
>>> FixedArray("2.5").mul_round(FixedArray("0.5"), scale=1)  # 1.25 → half_up
FixedArray(1.3, scale=1)
>>> FixedArray("2.5").mul_round(FixedArray("0.5"), scale=1, rounding="half_even")  # 1.25 → half_even
FixedArray(1.2, scale=1)
>>> FixedArray("1.25").mul_round(2, scale=5, rounding=None)  # 不舍入，扩大 scale
FixedArray(2.50000, scale=5)
```

中间乘积超出 `int64`、但结果在范围内时，`*` 报错而 `mul_round` 给出正确答案：

```python
>>> notional = FixedArray("12345678901.123456")  # 尾数约 1.2e16
>>> fx = FixedArray("1.234567")  # 尾数约 1.2e6，乘积尾数约 1.5e22
>>> notional * fx
Traceback (most recent call last):
    ...
OverflowError: 结果超出
int64
范围
>>> notional.mul_round(fx, scale=6)  # 精确值 15241567763.923281703552
FixedArray(15241567763.923282, scale=6)
>>> FixedArray("1.25").mul_round(FixedArray("1.5"), scale=1, rounding=None)
Traceback (most recent call last):
    ...
ValueError: rounding=None 时不支持缩小 scale（需要指定舍入模式）
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
- [`limbs_mul`](_limbs.md#limbs_mul)、[`_divmod_round`](#fixedarray_divmod_round) —
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
- **ZeroDivisionError** — 非缺失位置上除数含 0（被 `mask` 标记为 `<NA>` 的位置跳过检查并传播 `<NA>`）。
- **OverflowError** — 商超出 `int64`。

**机制说明**

等价于 `a.divide(b)`，需要其他精度或舍入模式时请直接调用 [`divide`](#fixedarraydivide)。

> **注意**：两个整数相除得到的也是整数精度（scale 0）。`FixedArray(1) / 3` 的结果是 `0`，不是 `0.333…`。
> 需要小数时请让被除数带上 scale（如 `FixedArray("1.00")`），或使用 `divide(..., scale=n)`。

**示例**

```python
>>> FixedArray("10.00") / 3
FixedArray(3.33, scale=2)
>>> FixedArray(["10.00", "20.00", "30.00"]) / FixedArray([3, 6, 7])
FixedArray([3.33, 3.33, 4.29], scale=2)
>>> 1 / FixedArray("4.00")
FixedArray(0.25, scale=2)
>>> FixedArray(1) / 3  # 整数 / 整数 → scale 0
FixedArray(0, scale=0)
>>> FixedArray(1) / 0
Traceback (most recent call last):
    ...
ZeroDivisionError: 除数包含 0
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
- **ZeroDivisionError** — 非缺失位置上除数含 0（被 `mask` 标记为 `<NA>` 的位置跳过检查并传播 `<NA>`）。
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
>>> FixedArray("10").divide(3, scale=4)
FixedArray(3.3333, scale=4)
>>> FixedArray(2).divide(3, scale=2, rounding="down")
FixedArray(0.66, scale=2)
>>> FixedArray(2).divide(3, scale=2, rounding="up")
FixedArray(0.67, scale=2)
>>> FixedArray(-1).divide(8, scale=2)  # -0.125，half_up 远离零
FixedArray(-0.13, scale=2)
>>> FixedArray(-1).divide(8, scale=2, rounding="half_even")
FixedArray(-0.12, scale=2)
```

按 USD/CNY 汇率 7.1234 把人民币金额折算为美元（银行家舍入到分）：

```python
>>> cny = FixedArray(["100.00", "250.00"])
>>> cny.divide(FixedArray("7.1234"), scale=2, rounding="half_even")
FixedArray([14.04, 35.10], scale=2)
```

被除数或除数尾数接近 `int64` 上限时也能精确计算：

```python
>>> FixedArray("92233720368.54775807").divide(FixedArray("0.00000001"), scale=0)
FixedArray(9223372036854775807, scale=0)
>>> FixedArray(1).divide(FixedArray(2 ** 62), scale=20)  # 除数 > int64 上限 / 10
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

### `a // b`、`a % b` 与 `divmod(a, b)`（整除与取余）

```python
a // b
a % b
divmod(a, b)
```

向下取整的商、与除数同号的余数，以及两者的组合。

**参数**

- **b** : *FixedArray 或数值 array_like* — 除数，可广播，不能含 0。

**返回值**

- **q** : *FixedArray* — `a // b` 的结果，`scale = 0`（无量纲整数商，支持缺失值 `mask` 传播）。
- **r** : *FixedArray* — `a % b` 的结果，`scale = max(a.scale, b.scale)`（支持缺失值 `mask` 传播）。
- `divmod(a, b)` 返回 `(q, r)`。

**异常**

- **TypeError** — `b` 不是合法的数值操作数。
- **ZeroDivisionError** — 非缺失位置上除数含 0。
- **OverflowError** — 非缺失位置上 `-2**63` 除以 `-1`（仅 `//` 与 `divmod`）。

**机制说明**

两个操作数先对齐到相同 scale，再对尾数做整数带余除法，语义与 Python / NumPy 的 `//`、`%` 一致：
商向负无穷取整，余数与除数同号，满足 `a == (a // b) * b + a % b`。
商返回 `scale = 0` 的 `FixedArray`，因此当任一操作数含有 `<NA>` 时，商和余数的对应位置都能一致传播 `<NA>` 掩码。
需要“对齐到 b 的整数倍”时请用 [`round_to_step`](#fixedarrayround_to_step)。

**示例**

```python
>>> FixedArray("7.5") // 2
FixedArray(3, scale=0)
>>> FixedArray("-7.5") // 2  # 向负无穷
FixedArray(-4, scale=0)
>>> FixedArray(["7.5", "8.5"]) // FixedArray("2.5")
FixedArray([3, 3], scale=0)
>>> FixedArray("7.5") % 2
FixedArray(1.5, scale=1)
>>> FixedArray("-7.5") % 2  # 与除数同号
FixedArray(0.5, scale=1)
>>> FixedArray("7.5") % -2
FixedArray(-0.5, scale=1)
>>> divmod(FixedArray("1000.00"), FixedArray("3.30"))  # 1000 元能买几份 3.30 元的商品、找零多少
(FixedArray(303, scale=0), FixedArray(0.10, scale=2))
>>> a, b = FixedArray("-7.5"), FixedArray(2)
>>> bool((a // b) * b + a % b == a)
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
>>> qty = FixedArray([[100, 200, 0],
...                   [0, 50, 300]])
>>> price = FixedArray(["10.25", "3.10", "99.99"])
>>> qty @ price
FixedArray([1645.00, 30152.00], scale=2)
>>> FixedArray([["1.5", "2"], ["0", "1"]]) @ FixedArray([[2, 0], [1, 1]])
FixedArray([[5.0, 2.0],
            [1.0, 1.0]], scale=1)
>>> batch = FixedArray(np.ones((4, 2, 3), dtype=np.int64))
>>> (batch @ FixedArray(np.ones((3, 5), dtype=np.int64))).shape
(4, 2, 5)
```

结果越界时报错，而 NumPy 的 `int64` 矩阵乘法静默回绕；逐项乘积很大但正负相抵时，能给出正确结果：

```python
>>> x = 3_037_000_499  # x**2 ≈ 9.22e18，刚好在 int64 内
>>> FixedArray([[x, x]]) @ FixedArray([[x], [x]])
Traceback (most recent call last):
    ...
OverflowError: 结果超出
int64
范围
>>> np.array([[x, x]]) @ np.array([[x], [x]])  # 错误结果，没有任何警告
array([[-11857053614]])
>>> FixedArray([[x, x]]) @ FixedArray([[x], [-x]])
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
>>> w = FixedArray(["0.333", "0.333", "0.334"])  # 权重，scale 3
>>> v = FixedArray(["100.01", "200.02", "300.03"])  # 价格，scale 2
>>> w @ v  # 精确，scale 5
FixedArray(200.12001, scale=5)
>>> w.matmul(v)  # 默认 scale = max(3, 2)
FixedArray(200.120, scale=3)
>>> w.matmul(v, scale=2)
FixedArray(200.12, scale=2)
>>> w.matmul(v, scale=6, rounding=None)  # 不舍入，扩大 scale
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
- [`limbs_matmul`](_limbs.md#limbs_matmul)、[`FixedArray._limbs_to_mantissa`](#fixedarray_limbs_to_mantissa) – 底层
  Base $2^{21}$ 分块精确矩阵乘法与收口内核。

## 规约统计

`np.sum(a)` 与 `np.mean(a)` 会自动转发到下面两个方法；其他 NumPy
规约操作（如最值、累计和）的定点数写法见 [NumPy 互操作机制](#numpy-互操作机制)。

### FixedArray.sum

```python
FixedArray.sum(axis=None, keepdims=False, skipna=True, out=None, dtype=None)
```

沿指定轴精确求和。

**参数**

- **axis** : *None、int 或 int 元组，可选* — 求和的轴，默认对全部元素求和。
- **keepdims** : *bool，默认 False* — 为 `True` 时保留被求和的轴（长度为 1）。
- **skipna** : *bool，默认 True* — 是否忽略缺失值 `<NA>`。为 `True` 时缺失位置按 `0` 处理；为 `False` 时若求和切片中含有
  `<NA>` 则结果对应位置为 `<NA>`。
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
>>> m = FixedArray([["1.10", "2.20"],
...                   ["3.30", "4.40"]])
>>> m.sum()
FixedArray(11.00, scale=2)
>>> m.sum(axis=0)
FixedArray([4.40, 6.60], scale=2)
>>> m.sum(axis=1, keepdims=True)
FixedArray([[3.30],
            [7.70]], scale=2)
>>> np.sum(m, axis=0)  # 转发到 m.sum(axis=0)
FixedArray([4.40, 6.60], scale=2)
>>> float(np.array([0.1, 0.1, 0.1]).sum())
0.30000000000000004
>>> FixedArray([0.1, 0.1, 0.1]).sum()
FixedArray(0.3, scale=1)
```

溢出一定报错，而不是像 NumPy `int64` 那样回绕：

```python
>>> FixedArray([2 ** 62, 2 ** 62]).sum()
Traceback (most recent call last):
    ...
OverflowError: 结果超出
int64
范围
>> > int(np.array([2 ** 62, 2 ** 62]).sum())
-9223372036854775808
>>> FixedArray([2 ** 62, 2 ** 62, -2 ** 62]).sum()  # 部分和越界、结果在范围内
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
- [`limbs_sum`](_limbs.md#limbs_sum) – 底层 Base $2^{21}$ 分块精确求和内核。

### FixedArray.mean

```python
FixedArray.mean(axis=None, keepdims=False, scale=None, rounding="half_up", skipna=True, out=None, dtype=None)
```

沿指定轴求算术平均值，按 `rounding` 精确舍入到 `scale` 位小数。

**参数**

- **axis** : *None、int 或 int 元组，可选* — 求均值的轴，默认对全部元素。
- **keepdims** : *bool，默认 False* — 为 `True` 时保留被规约的轴。
- **scale** : *int，可选* — 结果的小数位数，默认与原数组相同。
- **rounding** : *RoundingMode 或 str，默认 `"half_up"`* — 舍入模式。
- **skipna** : *bool，默认 True* — 是否忽略缺失值 `<NA>`。为 `True` 时仅对有效非缺失元素求均值（若某切片全部为 `<NA>`
  则该位置返回 `<NA>`）；为 `False` 时若切片含 `<NA>` 则结果为 `<NA>`。
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
>>> b = FixedArray(["1.00", "2.00", "4.00"])
>>> b.mean()  # 7/3 = 2.333…，保留 2 位
FixedArray(2.33, scale=2)
>>> b.mean(scale=4)
FixedArray(2.3333, scale=4)
>>> b.mean(scale=1, rounding="down")
FixedArray(2.3, scale=1)
>>> c = FixedArray([1, 2, 3, 4])
>>> c.mean()  # 2.5 → half_up
FixedArray(3, scale=0)
>>> c.mean(rounding="half_even")
FixedArray(2, scale=0)
>>> c.mean(scale=1)
FixedArray(2.5, scale=1)
>>> d = FixedArray([[1, 2], [3, 4]])
>>> d.mean(axis=1, scale=1)
FixedArray([1.5, 3.5], scale=1)
>>> np.mean(d, axis=1)  # 注意：scale 仍为 0
FixedArray([2, 4], scale=0)
>>> FixedArray(np.array([], dtype=np.int64)).mean()
Traceback (most recent call last):
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
- [`round_quotient`](_rounding.md#round_quotient)、[`_round_div_pow10`](#fixedarray_round_div_pow10) –
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
>>> a = FixedArray("1.25")
>>> a.rescale(4)
FixedArray(1.2500, scale=4)
>>> a.rescale(1)
FixedArray(1.3, scale=1)
>>> a.rescale(1, rounding="half_even")
FixedArray(1.2, scale=1)
>>> a.rescale(2) is a
True
>>> FixedArray(["1.005", "2.675", "-0.125"]).rescale(2)
FixedArray([1.01, 2.68, -0.13], scale=2)
>>> np.round([1.005, 2.675, -0.125], 2)
array([ 1.  ,  2.68, -0.12])
>>> FixedArray("0.5000000000000000000").rescale(0)  # 一次缩小 19 位
FixedArray(1, scale=0)
>>> FixedArray(10 ** 17).rescale(2)
Traceback (most recent call last):
    ...
OverflowError: 乘以 10^2 后超出 int64 范围
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

- [`FixedArray(values, scale=…)`](#fixedarrayvalues-scalenone-masknone) – 构造时指定 scale。
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
>>> px = FixedArray(["10.02", "10.03", "10.07"])
>>> px.round_to_step(FixedArray("0.05"))
FixedArray([10.00, 10.05, 10.05], scale=2)
>>> px.round_to_step(0.05)  # float 步长按最短写法解析为 0.05
FixedArray([10.00, 10.05, 10.05], scale=2)
>>> FixedArray([150, 249, 251]).round_to_step(100, rounding="floor")  # 向下取整到 100 股一手
FixedArray([100, 200, 200], scale=0)
>>> FixedArray(["10.02", "100.3"]).round_to_step(FixedArray(["0.05", "0.5"]))  # 每个元素不同步长
FixedArray([10.00, 100.50], scale=2)
>>> FixedArray(["0.125", "0.375"]).round_to_step(FixedArray("0.25"), rounding="half_even")
FixedArray([0.000, 0.500], scale=3)
>>> px.round_to_step("0.05")
Traceback (most recent call last):
    ...
TypeError: FixedArray 只能与数值类型运算、比较或对齐，收到 str
>>> px.round_to_step(0)
Traceback (most recent call last):
    ...
ValueError: 步长 step 必须为正数
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
>>> FixedArray(["10.05", "10.07"]).is_on_step(FixedArray("0.05"))
array([ True, False])
>>> FixedArray("10.05").is_on_step(0.05)
True
>>> FixedArray("-10.05").is_on_step(FixedArray("0.05"))
True
>>> 10.05 % 0.05 == 0  # float 取余的误差
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
>>> bool(FixedArray("1.10") == FixedArray("1.1"))
True
>>> FixedArray(["1.5", "2.0"]) == FixedArray([1.5, 2])
array([ True,  True])
>>> np.array([1, 2]) == FixedArray(["1.0", "2.5"])  # ndarray 在左侧
array([ True, False])
>>> FixedArray(1) == "1"
False
>>> FixedArray(1) != "1"
True
>>> FixedArray(1) == True  # bool 不是合法的数值操作数
False
>> > bool(FixedArray(1) == float("nan"))
False
>>> pair = FixedArray([1, 2])
>>> if pair == FixedArray([1, 2]):
...     pass
Traceback (most recent call last):
    ...
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
>>> bool((pair == FixedArray([1, 2])).all())
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
>>> FixedArray(["1", "2", "3"]) > 1.5
array([False, True, True])
>>> FixedArray(["0.1", "0.2"]) <= Decimal("0.1")
array([ True, False])
>>> 1.5 < FixedArray(["1", "2"])  # 反向比较
array([False, True])
>>> p = FixedArray(["9.99", "10.01", "12.50"])
>>> p[p > 10]
FixedArray([10.01, 12.50], scale=2)
>>> FixedArray(1) < "1"
Traceback (most recent call last):
    ...
TypeError: '<' not supported between instances of 'FixedArray' and 'str'
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
>>> six = FixedArray(["1.1", "2.2", "3.3", "4.4", "5.5", "6.6"])
>>> six.reshape(2, 3)
FixedArray([[1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6]], scale=1)
>>> six.reshape((3, -1)).shape
(3, 2)
>>> six.reshape(4, 2)
Traceback (most recent call last):
    ...
ValueError: cannot reshape array of size 6 into shape (4,2)
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
>>> cube = FixedArray(np.arange(6).reshape(1, 2, 3))
>>> cube.transpose(2, 0, 1).shape
(3, 1, 2)
>>> np.transpose(FixedArray([[1, 2], [3, 4]]))
FixedArray([[1, 3],
            [2, 4]], scale=0)
```

**另请参阅**

- [`FixedArray.T`](#fixedarrayt)

### `FixedArray.swapaxes` 与 `FixedArray.squeeze`

```python
FixedArray.swapaxes(axis1, axis2)
FixedArray.squeeze(axis=None)
```

交换张量的两个指定轴（`swapaxes`），或移除长度为 `1` 的单维轴（`squeeze`），同时保持 `scale` 并同步变换 `mask` 视图。

**参数**

- **axis1, axis2** : *int* — `swapaxes` 要互换的两个轴编号（支持负数下标）。
- **axis** : *None, int 或 tuple of int，可选* — `squeeze` 要移除的单维轴；默认 `None` 时移除所有长度为 `1` 的轴（若选定轴的长度不为
  `1` 则由 NumPy 抛出 `ValueError`）。

**返回值**

- **out** : *FixedArray* — 共享底层尾数与掩码视图的新 `FixedArray`。

**示例**

```python
>> > a = FixedArray(np.arange(6).reshape(1, 2, 3), scale=1)
>> > a.swapaxes(0, 2).shape
(3, 2, 1)
>> > a.squeeze().shape, a.squeeze(axis=0).shape
((2, 3), (2, 3))
```

**另请参阅**

- [`FixedArray.transpose`](#fixedarraytranspose)、[`FixedArray.reshape`](#fixedarrayreshape)

### `a[key]`（索引与切片）

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
>>> g = FixedArray([["1.1", "2.2", "3.3"],
...                   ["4.4", "5.5", "6.6"]])
>>> g[0]
FixedArray([1.1, 2.2, 3.3], scale=1)
>>> g[0, 1]
FixedArray(2.2, scale=1)
>>> g[:, 0]
FixedArray([1.1, 4.4], scale=1)
>>> g[..., -1]
FixedArray([3.3, 6.6], scale=1)
>>> six[::2]
FixedArray([1.1, 3.3, 5.5], scale=1)
>>> six[[0, 2]]
FixedArray([1.1, 3.3], scale=1)
>>> six[six > 3]
FixedArray([3.3, 4.4, 5.5, 6.6], scale=1)
>>> six[0] = 1
Traceback (most recent call last):
    ...
TypeError: 'FixedArray' object does not support item assignment
>>> y = FixedArray(["1.5", "-2.0", "3.0"])
>>> y * (y > 0).astype(np.int64)  # 用掩码把负数清零
FixedArray([1.5, 0.0, 3.0], scale=1)
>>> y * (y > 0)  # 布尔数组不能直接参与运算
Traceback (most recent call last):
    ...
TypeError: operand 'FixedArray' does not support ufuncs (__array_ufunc__=None)
```

**另请参阅**

- [`a < b` 等大小比较](#a--ba--ba--ba--b大小比较)

### `len(a)`、`bool(a)` 与 `iter(a)`（容器与真值协议）

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

- **TypeError** — 对 0 维 `FixedArray` 调用 `len()` 或迭代；或对单个缺失值（`<NA>`）做真值测试（报
  `"缺失值（NA）的真值不明确"`）。
- **ValueError** — 对元素个数不是 1 的数组（含空数组）做真值测试。

**机制说明**

`bool(a)` 与 NumPy 的规则相同：只有单元素数组能转为布尔值，其余情况报错，防止 `if a:` 写法掩盖“是否全部满足”的意图。
多元素数组请使用 `bool((a > 0).all())` 之类的显式归约。迭代时每个元素都是 `FixedArray`，scale 保持不变。

**示例**

```python
>>> len(g)
2
>>> [row for row in g]
[FixedArray([1.1, 2.2, 3.3], scale=1), FixedArray([4.4, 5.5, 6.6], scale=1)]
>>> list(FixedArray(["1.5", "2.5"]))
[FixedArray(1.5, scale=1), FixedArray(2.5, scale=1)]
>>> bool(FixedArray("0.01")), bool(FixedArray(0)), bool(FixedArray([0]))
(True, False, False)
>> > bool(FixedArray(None))
Traceback(most
recent
call
last):
...
TypeError: 缺失值（NA）的真值不明确
>>> bool(FixedArray([1, 2]))
Traceback (most recent call last):
    ...
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
>>> len(FixedArray("1.5"))
Traceback (most recent call last):
    ...
TypeError: len() of unsized object
>>> list(FixedArray("1.5"))
Traceback (most recent call last):
    ...
TypeError: iteration over a 0-d FixedArray
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
>>> FixedArray("1.25").to_float()
1.25
>>> FixedArray(["1.50", "-2.25"]).to_float()
array([ 1.5 , -2.25])
>>> FixedArray(2 ** 53 + 1).to_float()  # 超出 float64 精度，低位丢失
9007199254740992.0
>>> FixedArray("1e-30").to_float()  # scale 30：10**30 不能精确表示为 float64，走整数路径
1e-30
>>> float(FixedArray("1.5"))
Traceback (most recent call last):
    ...
TypeError: float() argument must be a string or a real number, not 'FixedArray'
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
>>> FixedArray("1.50").to_decimal()
Decimal('1.50')
>>> FixedArray([["1.5", "2"], ["3", "4"]]).to_decimal()
array([[Decimal('1.5'), Decimal('2.0')],
       [Decimal('3.0'), Decimal('4.0')]], dtype=object)
>>> orig = FixedArray(["0.10", "123.45"])
>>> bool((FixedArray(orig.to_decimal()) == orig).all())
True
```

**另请参阅**

- [`FixedArray(values, scale=None, mask=None)`](#fixedarrayvalues-scalenone-masknone)

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
>>> px = FixedArray(["1.50", "2.25"])
>>> px.to_pandas(index=["a", "b"], name="px")
a
1.50
b
2.25
Name: px, dtype: float64
>>> px.to_pandas(as_float=False)
0
1.50
1
2.25
dtype: object
>>> FixedArray([["1.5", "2"], ["3", "4"]]).to_pandas()
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

- `as_float=True` 时先调用 [`to_float()`](#fixedarrayto_float)，再传入 `nan_to_null=True` 构造 `pl.Series`，将缺失值（
  `<NA>`）统一转换为 Polars 原生 `null`（而非浮点 `NaN`）；
- `as_float=False` 时先通过 [`to_decimal()`](#fixedarrayto_decimal) 得到精确的 `Decimal`（缺失位置为 `None`），再以
  `pl.Decimal(precision=38, scale=self.scale)` 构造列，缺失位置同样导出为 `null`。

**示例**

> 以下示例依赖 polars，未纳入自动校验。

```python
>>> s = FixedArray(["1.50", "2.25"]).to_polars(name="px")  # doctest: +SKIP
>>> s.to_list()  # doctest: +SKIP
[1.5, 2.25]
>>> FixedArray([[1, 2], [3, 4]]).to_polars()  # doctest: +SKIP
Traceback (most recent call last):
    ...
ValueError: to_polars 仅支持 1 维或标量 FixedArray，当前维度为 2
```

**另请参阅**

- [`FixedArray.to_pandas`](#fixedarrayto_pandas)
- [`FixedArray.to_float`](#fixedarrayto_float)、[`FixedArray.to_decimal`](#fixedarrayto_decimal)

### `str(a)` 与 `repr(a)`（文本表示）

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
>>> neg = FixedArray("-0.05")
>>> str(neg)
'-0.05'
>>> neg
FixedArray(-0.05, scale=2)
>>> print(FixedArray(["1.5", "-2"]))
[1.5, -2.0]
>>> print(FixedArray(np.arange(2000)))
[0, 1, 2, ..., 1997, 1998, 1999]
>>> f"总额：{FixedArray('1.50')}"
'总额：1.50'
>>> f"{FixedArray('1.50'):.1f}"
Traceback (most recent call last):
    ...
TypeError: unsupported format string passed to FixedArray.__format__
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
>>> h = FixedArray(["1.50", "2.25"])
>>> np.sqrt(h)
Traceback (most recent call last):
    ...
TypeError: operand 'FixedArray' does not support ufuncs (__array_ufunc__=None)
>>> np.asarray(h)
Traceback (most recent call last):
    ...
TypeError: FixedArray 不能隐式转成数组：要 float 请用 to_float()，要整数尾数请用 .mantissa
>>> np.array([1, 2]) * h
FixedArray([1.50, 4.50], scale=2)
>>> np.shape(m), np.ndim(m), np.size(m)
((2, 2), 2, 4)
```

配方：由尾数重建 `FixedArray`。公开构造器接收的是十进制数值而非底层尾数，若需从运算后的 `int64` 尾数重建 `FixedArray`
，可借助精确乘法实现（乘以 `10^(-scale)` 不会发生舍入）：

```python
>>> def from_mantissa(mant, scale):
...     """由 int64 尾数与 scale 构造 FixedArray（精确）。"""
...     return FixedArray(mant) * FixedArray(f"1e-{scale}")
>>> from_mantissa(np.array([150, 225]), 2)
FixedArray([1.50, 2.25], scale=2)
>>> p = FixedArray(["10.05", "9.80", "10.20"])
>>> p[np.argsort(p.mantissa)]  # 排序
FixedArray([9.80, 10.05, 10.20], scale=2)
>>> from_mantissa(m.mantissa.max(axis=0), m.scale)  # 按轴取最大值
FixedArray([3.30, 4.40], scale=2)
>>> u, v = FixedArray(["1.5", "2.5"]), FixedArray(["0.25"])
>>> s = max(u.scale, v.scale)
>>> from_mantissa(np.concatenate([u.rescale(s).mantissa, v.rescale(s).mantissa]), s)  # 拼接
FixedArray([1.50, 2.50, 0.25], scale=2)
>>> from_mantissa(np.where([True, False], u.rescale(s).mantissa, v.rescale(s).mantissa), s)  # 条件选择
FixedArray([1.50, 0.25], scale=2)
>>> cash = FixedArray(["100.00", "-30.50", "12.25"])
>>> from_mantissa(np.cumsum(cash.mantissa), cash.scale)  # 累计和：int64 溢出会静默回绕，仅在确定不越界时使用
FixedArray([100.00, 69.50, 81.75], scale=2)
```

**另请参阅**

- [`FixedArray.mantissa`](#fixedarraymantissa)
- [`FixedArray._from_mantissa`](#fixedarray_from_mantissa)
- [`FixedArray.to_float`](#fixedarrayto_float)、[`to_pandas`](#fixedarrayto_pandas)、[`to_polars`](#fixedarrayto_polars)

## 内部构造收口与类型转换（私有 API）

以下私有方法负责 `FixedArray` 的底层实例初始化、缺失值掩码合成、操作数白名单校验、类型强制转换与 `scale` 对齐。

### FixedArray._set

```python
FixedArray._set(mantissa, scale, mask=None)
```

所有构造路径的唯一收口方法，为实例绑定只读 `int64` 尾数视图、`scale` 以及可选的只读缺失值布尔掩码视图 `_mask`。

**参数**

- **mantissa** : *int 或整数 array_like* — 底层整数尾数。
- **scale** : *int* — 已通过校验的非负小数位数。
- **mask** : *bool 或 bool array_like，可选* — 缺失值掩码（`True` 表示缺失）。

**返回值**

- **None** — 原地设置 `self._mantissa`、`self._scale` 与 `self._mask`。

**机制说明**

1. 先通过 [`_as_int64`](#fixedarray_as_int64) 将尾数转换为 `int64` 数组，创建浅视图 `.view()` 并设
   `view.flags.writeable = False`；
2. 若传入了 `mask` 且其中存在至少一个 `True`：将 `mask` 广播到 `view.shape`，保存为只读布尔视图 `self._mask`
   ，并检查缺失位置上的尾数是否非零——若非零，则复制一份尾数将缺失位置 **强制重置为 `0`** 后再锁定为只读；
3. 若 `mask is None` 或全为 `False`，则将 `self._mask` 设为 `None`。

**示例**

```python
>>> obj = FixedArray.__new__(FixedArray)
>> > obj._set(np.array([150, 999], dtype=np.int64), 2, mask=[False, True])
>> > obj, obj.mantissa.tolist(), obj.mask.tolist()
(FixedArray([1.50, < NA >], scale=2), [150, 0], [False, True])
>> > obj.mantissa.flags.writeable, obj.mask.flags.writeable
(False, False)
```

**另请参阅**

- [`FixedArray._from_mantissa`](#fixedarray_from_mantissa)
- [`FixedArray.mask`](#fixedarraymask)

### FixedArray._from_mantissa

```python
FixedArray._from_mantissa(mantissa, scale, mask=None)
```

直接使用已有的整数尾数矩阵、`scale` 与可选 `mask` 构造 `FixedArray`（内部零拷贝快速路径）。

**参数**

- **mantissa** : *int 或整数 array_like* — 底层整数尾数；若已是 `int64` 数组且无非零被掩码位置，则直接共享内存。
- **scale** : *int* — 非负整数小数位数。
- **mask** : *bool 或 bool array_like，可选* — 缺失值掩码。

**返回值**

- **out** : *FixedArray*

**示例**

```python
>>> FixedArray._from_mantissa(np.array([1234, -500], dtype=np.int64), scale=2)
FixedArray([12.34, -5.00], scale=2)
>> > FixedArray._from_mantissa(np.array([1234, 0], dtype=np.int64), scale=2, mask=[False, True])
FixedArray([12.34, < NA >], scale = 2)
```

**另请参阅**

- [`FixedArray._set`](#fixedarray_set)

### `FixedArray._combine_masks` 与 `FixedArray._matmul_mask`

```python
FixedArray._combine_masks(m1, m2)
FixedArray._matmul_mask(m1, m2, s1, s2)
```

二元逐元素运算与矩阵乘法的缺失值掩码传播辅助函数。

**参数**

- **m1, m2** : *ndarray (bool) 或 None* — 左右操作数的 `_mask`。
- **s1, s2** : *tuple of int* — 矩阵乘法左右操作数的形状 `(a.shape, b.shape)`。

**返回值**

- **out** : *ndarray (bool) 或 None* — 若合并结果中不含任何 `True` 则返回 `None`，否则返回布尔掩码数组。

**机制说明**

- `_combine_masks`：逐元素取按位或 `m1 | m2`（支持广播），只要任一方缺失则结果缺失；
- `_matmul_mask`：利用布尔矩阵乘法 `np.matmul(u1, ones_2) | np.matmul(ones_1, u2)`（在 `bool` 代数下即 OR-AND
  半环，无整数累加回绕风险），只要参与内积点乘的任一项为 `True`，对应输出单元格即为 `True`。

**示例**

```python
>> > FixedArray._combine_masks(None, None) is None
True
>> > FixedArray._combine_masks(np.array([True, False]), np.array([False, False])).tolist()
[True, False]
>> > FixedArray._matmul_mask(np.array([[True, False]]), None, (1, 2), (2, 1)).tolist()
[[True]]
```

**另请参阅**

- [`FixedArray.mask`](#fixedarraymask)

### FixedArray._is_numeric_operand

```python
FixedArray._is_numeric_operand(val)
```

判断对象是否属于允许与 `FixedArray` 运算、比较或对齐的无量纲数值操作数白名单。

**参数**

- **val** : *Any* — 待检查的任意对象。

**返回值**

- **ok** : *bool*

**机制说明**

白名单包含：`FixedArray`、`int`、`np.integer`、`float`、`np.floating`、`Decimal`、`Fraction`、`None`（表示缺失值 `<NA>`）、整数/浮点
`ndarray`，以及递归仅包含上述元素的 `list`、`tuple` 或 `object` 类型的 `ndarray`。显式排除 `(bool, np.bool_)`、字符串以及上层带量纲对象（
`UnitArray` 子类）。

**示例**

```python
>> > FixedArray._is_numeric_operand(10), FixedArray._is_numeric_operand([1.5, None, FixedArray("2.0")])
(True, True)
>>> FixedArray._is_numeric_operand(True), FixedArray._is_numeric_operand("1.5")
(False, False)
```

**另请参阅**

- [`FixedArray._coerce`](#fixedarray_coerce)

### FixedArray._coerce

```python
FixedArray._coerce(other, strict=True)
```

将运算或比较另一侧的操作数转换为 `FixedArray`。

**参数**

- **other** : *Any* — 待转换的操作数。
- **strict** : *bool，默认 True* — 当 `other` 不在数值白名单内时，`True` 抛出 `TypeError`，`False` 返回 `None`（供运算符返回
  `NotImplemented`）。

**返回值**

- **out** : *FixedArray 或 None*

**示例**

```python
>>> FixedArray._coerce(1.25)
FixedArray(1.25, scale=2)
>> > FixedArray._coerce(None)
FixedArray( < NA >, scale = 0)
>>> FixedArray._coerce("1.25", strict=False) is None
True
```

**另请参阅**

- [`FixedArray._is_numeric_operand`](#fixedarray_is_numeric_operand)

### FixedArray._lose_precision

```python
FixedArray._lose_precision(target_scale)
```

判断将当前 `FixedArray` 的小数位数缩减到 `target_scale` 是否会丢失非零有效小数位（末尾多余的 `0` 以及被 `mask`
标记为缺失的位置不计为精度丢失）。

**参数**

- **target_scale** : *int* — 目标小数位数。

**返回值**

- **out** : *bool* — 若存在有效（非缺失）元素在 `target_scale` 之后仍有非零十进制数字，返回 `True`；否则返回 `False`。

**机制说明**

设 `k = self._scale - target_scale`：若 `k <= 0` 直接返回 `False`；若 `k > 18`（由于 `|mantissa| < 10^19 <= 10^k`
），只要存在非零有效尾数即返回 `True`；否则检查 `(self._mantissa % (10 ** k) != 0) & valid`。供上层 [`Money`](money.md)、[
`Price`](price.md)、[`Quantity`](quantity.md) 判断是否需要发出 [
`PrecisionLossWarning`](unit_array.md#precisionlosswarning)。

**示例**

```python
>> > FixedArray("1.500")._lose_precision(2)
False
>> > FixedArray("1.505")._lose_precision(2)
True
```

### FixedArray._check_scale

```python
FixedArray._check_scale(scale)
```

校验 `scale` 是否为合法的非负整数，并返回标准 Python `int`。

**示例**

```python
>>> FixedArray._check_scale(4)
4
>>> FixedArray._check_scale(-1)
Traceback (most recent call last):
    ...
ValueError: scale 必须是非负整数，收到 -1
```

### FixedArray._as_int64

```python
FixedArray._as_int64(x)
```

将整数标量或整数数组安全转换为 `int64` NumPy 数组，对已是 `int64` 的数组不发生拷贝。

**示例**

```python
>> > FixedArray._as_int64([1, 2, 3])
array([1, 2, 3])
>>> FixedArray._as_int64(2 ** 63)
Traceback (most recent call last):
    ...
OverflowError: 9223372036854775808 超出 int64 范围
```

### FixedArray._align_scales

```python
FixedArray._align_scales(a, b)
```

将两个 `FixedArray` 的 `scale` 自动对齐到 `max(a.scale, b.scale)`。

**示例**

```python
>> > FixedArray._align_scales(FixedArray("1.5"), FixedArray("2.25"))
(FixedArray(1.50, scale=2), FixedArray(2.25, scale=2))
```

## 内部输入解析引擎（私有 API）

### FixedArray._parse_decimal_str

```python
FixedArray._parse_decimal_str(s)
```

解析单个十进制字符串（支持正负号、下划线分隔符与科学计数法），返回 `(mantissa, scale)`。

**示例**

```python
>> > FixedArray._parse_decimal_str("-12.340")
(-12340, 3)
>> > FixedArray._parse_decimal_str("1.5e-2")
(15, 3)
```

### FixedArray._parse_fraction

```python
FixedArray._parse_fraction(f)
```

将标准库 ` fractions.Fraction` 解析为精确有限十进制表示 `(mantissa, scale)`（分母含 `2` 和 `5` 以外的质因子时抛
`ValueError`）。

**示例**

```python
>>> from fractions import Fraction
>>> FixedArray._parse_fraction(Fraction(3, 8))
(375, 3)
```

### FixedArray._parse_float_scalar

```python
FixedArray._parse_float_scalar(val)
```

按 IEEE 754 最短往返十进制写法（Dragonbox 算法）解析单个浮点数，返回 `(mantissa, scale)`。

**示例**

```python
>>> FixedArray._parse_float_scalar(0.1)
(1, 1)
>>> FixedArray._parse_float_scalar(1.0)
(1, 0)
```

### `FixedArray._is_missing` 与 `FixedArray._contains_bool`

```python
FixedArray._is_missing(val)
FixedArray._contains_bool(values)
```

- **`_is_missing(val)`**：判断单个标量元素是否表示缺失值（涵盖 `None`、浮点 `NaN`、`Decimal("NaN")`、字符串 `"nan"` /
  `"NaN"` / `"-NaN"` 以及 `pandas.NA`）；
- **`_contains_bool(values)`**：递归检查嵌套 `list` / `tuple` 中是否混有 `bool` 标量或 `bool` 数组（防止
  `np.asarray([1, True])` 将布尔值静默转为整数 `1`）。

**示例**

```python
>> > FixedArray._is_missing(None), FixedArray._is_missing(float("nan")), FixedArray._is_missing("NaN")
(True, True, True)
>> > FixedArray._is_missing("1.5"), FixedArray._contains_bool([1, [2, True]])
(False, True)
```

### FixedArray._expand_nested

```python
FixedArray._expand_nested(values)
```

递归遍历嵌套的 `list` 或 `tuple`，将其中的 `FixedArray` 元素转换为等值的 `Decimal`（0 维标量）或 `Decimal`
对象数组（多维数组），保留原有的十进制小数位数与缺失值。

**示例**

```python
>>> FixedArray._expand_nested([FixedArray("1.50"), (2, FixedArray(["0.1", "0.25"]))])
[Decimal('1.50'), [2, array([Decimal('0.10'), Decimal('0.25')], dtype=object)]]
```

### FixedArray._parse_object_element

```python
FixedArray._parse_object_element(val)
```

将单个受支持的非缺失标量（`int`、`float`、`Fraction`、`str`、`Decimal` 或 0 维 `FixedArray`）分派解析为 `(mantissa, scale)`。

**示例**

```python
>>> FixedArray._parse_object_element(Decimal("1.250"))
(1250, 3)
>>> FixedArray._parse_object_element(FixedArray("1.50"))
(150, 2)
```

### FixedArray._parse_float_array

```python
FixedArray._parse_float_array(arr, target_scale=None)
```

向量化解析有限浮点数数组，返回 `(尾数数组, scale)`，结果与逐元素取最短十进制写法完全一致。

**示例**

```python
>>> FixedArray._parse_float_array(np.array([0.1, 0.25]))
(array([10, 25]), 2)
>> > FixedArray._parse_float_array(np.array([0.126, 2.5]), target_scale=2)
(array([13, 250]), 2)
```

### FixedArray._align_mantissas

```python
FixedArray._align_mantissas(mantissas, scales, target)
```

将各自具有不同小数位数 `scales`（满足 `scales <= target`）的整数尾数向量化对齐到统一的目标精度 `target`。

**示例**

```python
>> > FixedArray._align_mantissas(np.array([15, 225]), np.array([1, 2]), target=3)
array([1500, 2250])
```

### FixedArray._round_float_to_scale

```python
FixedArray._round_float_to_scale(values, scale)
```

将浮点数组元素的最短十进制写法按 `half_up` 快速舍入到指定 `scale`，返回 `int64` 尾数数组。

**示例**

```python
>> > FixedArray._round_float_to_scale(np.array([2.675, 0.3333333]), scale=2)
array([268, 33])
```

### FixedArray._parse_text_array

```python
FixedArray._parse_text_array(arr)
```

解析字符串、`Decimal`、`Fraction` 或混合对象数组（含 0 维），识别缺失值并返回三元组 `(对齐后的尾数数组, scale, null_mask)`。

**返回值**

- **mantissa, scale, null_mask** : *tuple of (ndarray, int, Optional[ndarray])* — 对齐后的 `int64` 尾数数组（缺失位置已置
  `0`）、最大 `scale` 以及缺失值布尔掩码（无缺失值时为 `None`）。

**示例**

```python
>>> FixedArray._parse_text_array(np.array(["1.5", "2.25", "1.5"]))
(array([150, 225, 150]), 2, None)
>> > m, s, mask = FixedArray._parse_text_array(np.array(["1.5", "nan"]))
>> > m.tolist(), s, mask.tolist()
([15, 0], 1, [False, True])
```

## 底层数学与防溢出舍入引擎（私有 API）

`FixedArray` 的通用舍入规则与多精度分块算术已解耦拆分至两个独立底层模块：

- **[`moneta._rounding`](_rounding.md)**：提供 [`RoundingMode`](_rounding.md#roundingmode)、[
  `check_rounding`](_rounding.md#check_rounding)、[`decide_bump`](_rounding.md#decide_bump) 与 [
  `round_quotient`](_rounding.md#round_quotient)；
- **[`moneta._limbs`](_limbs.md)**：提供二进制 Base $2^{21}$ 多精度分块算术 [`split_int64`](_limbs.md#split_int64)、[
  `normalize_limbs`](_limbs.md#normalize_limbs)、[`limbs_mul`](_limbs.md#limbs_mul)、[
  `limbs_matmul`](_limbs.md#limbs_matmul)、[`limbs_sum`](_limbs.md#limbs_sum)、[
  `limbs_round_div_pow10`](_limbs.md#limbs_round_div_pow10) 与 [`limbs_to_int64`](_limbs.md#limbs_to_int64)。

以下为保留在 `FixedArray` 类上的定点缩放与除法私有方法：

### FixedArray._check_division

```python
FixedArray._check_division(m, d, mask=None)
```

整数除法前置安全检查：在非缺失位置上（`~mask`），除数 `d` 不能含 `0`，且禁止 `-2**63` 除以 `-1`。

**示例**

```python
>>> FixedArray._check_division(np.array([-2 ** 63], dtype=np.int64), np.array([-1], dtype=np.int64))
Traceback (most recent call last):
    ...
OverflowError: -2**63 除以 -1 超出 int64 范围
>> >  # 被 mask 标记为缺失的位置会被安全跳过
>> > FixedArray._check_division(np.array([1], dtype=np.int64), np.array([0], dtype=np.int64), mask=np.array([True]))
```

### FixedArray._max_abs

```python
FixedArray._max_abs(x)
```

安全计算 `int64` 数组元素绝对值的最大值，返回 Python `int`（`-2**63` 不溢出，空数组返回 `0`）。

**示例**

```python
>>> FixedArray._max_abs(np.array([-2 ** 63, 100], dtype=np.int64)) == 2 ** 63
True
>>> FixedArray._max_abs(np.array([], dtype=np.int64))
0
```

### `FixedArray._mul_pow10`

```python
FixedArray._mul_pow10(x, k)
FixedArray._mul_pow10_float(x, scale)
```

- **`_mul_pow10(x, k)`**：计算整数尾数 `x · 10^k`，预先通过 [`_max_abs`](#fixedarray_max_abs) 判断是否超出 `int64`
  （全零数组支持任意大 `k`）；
- **`_mul_pow10_float(x, scale)`**：计算浮点数 `x · 10.0^scale`，当 `scale > 300` 时拆成
  `x * 10.0**300 * 10.0**(scale - 300)` 两步相乘，避免 `10.0 ** scale` 本身在 IEEE 754 `float64` 中溢出为 `inf`
  （从而支持极小次正规浮点数如 `5e-324` 放大到 `scale=330`）。

**示例**

```python
>>> FixedArray._mul_pow10(np.array([12, -5], dtype=np.int64), 2)
array([1200, -500])
>> > round(FixedArray._mul_pow10_float(1e-310, 312), 6)
100.0
```

### FixedArray._divmod_round

```python
FixedArray._divmod_round(m, d, mode="half_up")
```

纯 `int64` 整数带余除法与舍入：计算 `round(m / d)`，内部调用 [`check_rounding`](_rounding.md#check_rounding)、[
`_check_division`](#fixedarray_check_division) 与 [`round_quotient`](_rounding.md#round_quotient)。

**示例**

```python
>> > FixedArray._divmod_round(5, 2, mode="half_up"), FixedArray._divmod_round(5, 2, mode="half_even")
(3, 2)
>>> FixedArray._divmod_round(np.array([10, -10]), 3, mode="down")
array([ 3, -3])
```

### FixedArray._doubling_step_10r

```python
FixedArray._doubling_step_10r(r, d)
```

大除数无溢出单步倍增：在 `|d| > _MAX // 10` 时利用 `uint64` 下 $10r = 8r + 2r$ 精确计算 `(r * 10) // d` 与
`(r * 10) % d`。

**示例**

```python
>>> d = np.array([2 ** 62], dtype=np.int64)
>> > r = np.array([2 ** 61], dtype=np.int64)
>>> FixedArray._doubling_step_10r(r, d)
(array([5]), array([0]))
```

### FixedArray._chunked_divmod_round

```python
FixedArray._chunked_divmod_round(m, d, k, mode)
```

分段十进制长除法与精确舍入：计算 `round(m · 10^k / d)`，全程纯 `int64` / `uint64` 运算，大除数自动切换至 [
`_doubling_step_10r`](#fixedarray_doubling_step_10r)，终局调用 [`round_quotient`](_rounding.md#round_quotient)。

**示例**

```python
>>> FixedArray._chunked_divmod_round(np.array([1], dtype=np.int64), np.array([3], dtype=np.int64), k=4, mode="half_up")
array([3333])
```

### FixedArray._round_div_pow10

```python
FixedArray._round_div_pow10(q, r, d, k, mode)
```

将带余商 `q + r/d`（其中 `q` 为整数商，`r/d ∈ [0, 1)` 为额外小数且 `r` 与 `d` 同号）整体除以 `10^k` 并调用 [
`decide_bump(..., floor_base=True)`](_rounding.md#decide_bump) 按 `mode` 精确舍入（支持任意 $k \ge 0$）。

**示例**

```python
>>> int(FixedArray._round_div_pow10(125, 0, 1, k=1, mode="half_up"))
13
>>> int(FixedArray._round_div_pow10(125, 0, 1, k=1, mode="half_even"))
12
>>> int(FixedArray._round_div_pow10(5_000_000_000_000_000_000, 0, 1, k=19, mode="half_up"))
1
```

### FixedArray._limbs_to_mantissa

```python
FixedArray._limbs_to_mantissa(limbs, shift, mode)
```

将 [`_limbs`](_limbs.md) 模块计算出的精确多精度分块结果 `limbs` 按十进制右移位数 `shift = natural_scale - target_scale`
转换为目标 `scale` 的 `int64` 尾数数组。

**参数**

- **limbs** : *list of np.ndarray* — 由 [`limbs_mul`](_limbs.md#limbs_mul) 或 [`limbs_matmul`](_limbs.md#limbs_matmul)
  产出的规整分块列表。
- **shift** : *int* — 需要缩小的十进制小数位数（`shift > 0` 表示除以 $10^{ ext{shift}}$ 并舍入；`shift <= 0`
  表示精确收口后乘以 $10^{- ext{shift}}$）。
- **mode** : *str* — 舍入模式名称。

**返回值**

- **out** : *np.ndarray（int64）*

**异常**

- **OverflowError** — 舍入或放大后的最终真实结果超出 `int64` 范围。

**机制说明**

- 当 `shift > 0` 时：调用 `limbs_to_int64(limbs_round_div_pow10(limbs, shift, mode))`
  直接在多精度分块上除以 $10^{ ext{shift}}$ 并只舍入一次，然后再收口检验是否落在 `int64` 范围内（因此即使未缩放前的精确积或精确内积超出
  `int64`，只要除以 $10^{ ext{shift}}$ 舍入后能放进 `int64`，就绝不会溢出！）；
- 当 `shift <= 0` 时：先调用 [`limbs_to_int64(limbs)`](_limbs.md#limbs_to_int64) 收口为 `int64`，再通过 [
  `_mul_pow10`](#fixedarray_mul_pow10) 放大 $10^{- ext{shift}}$ 倍。

**示例**

```python
>> > from moneta._limbs import limbs_mul
>> > limbs = limbs_mul(np.array([3 * 10 ** 18]), np.array([2 * 10 ** 18]))
>> > FixedArray._limbs_to_mantissa(limbs, shift=18, mode="half_up")
array([6000000000000000000])
```

**另请参阅**

- [`limbs_round_div_pow10`](_limbs.md#limbs_round_div_pow10)、[`limbs_to_int64`](_limbs.md#limbs_to_int64)
- [`FixedArray.mul_round`](#fixedarraymul_round)、[`FixedArray.matmul`](#fixedarraymatmul)

## 内部比较与 NumPy 协议辅助（私有 API）

### `FixedArray._compare_fixed_array`、`FixedArray._compare` 与 `FixedArray._eq_unrepresentable`

```python
FixedArray._compare_fixed_array(other, op)
FixedArray._compare(other, op)
FixedArray._eq_unrepresentable(other)
```

无溢出跨 `scale` 精确比较网关：

- **`_compare_fixed_array(other, op)`**：当两个 `FixedArray` 的 `scale`
  不同（设 $k = ext{other.scale} - ext{self.scale} > 0$）时， **不将低 `scale` 尾数乘以 $10^k$**（否则当 $k > 18$
  或尾数较大时会触发假溢出），而是将高 `scale` 尾数 `m_hi` 向下整除 $10^k$ 得到 `(q, r)`，直接比较 `m_lo` 与 `q` 及余数
  `r != 0`，从而在 **任意 `scale` 差距下都绝对不会溢出**！同时对任一侧为 `<NA>` 的位置，将 `!=` 置为 `True`、其余比较置为
  `False`。
- **`_compare(other, op)`**：将非 `FixedArray` 的右操作数通过 `_coerce(other, strict=False)` 转换；若右操作数含 `inf` 或超出
  `int64` 的大数，在 `==` / `!=` 时交由 `_eq_unrepresentable` 逐元素安全判定为不相等。

**示例**

```python
>> >  # 哪怕 scale 相差 30 位（10^30 远超 int64），跨 scale 比较也绝不溢出
>> > bool(FixedArray(1, scale=0) > FixedArray(0, scale=30))
True
>> > bool(FixedArray(1, scale=0) == FixedArray("1.00"))
True
```

### `FixedArray._array_function` 与 `FixedArray._numpy_equal`

```python
FixedArray._array_function(func, args, kwargs, name)
FixedArray._numpy_equal(a1, a2, broadcast, equal_nan=False)
```

实现 `__array_function__` 协议白名单分派（供 `FixedArray` 与 [`UnitArray`](unit_array.md) 共用）：

- 允许 `np.sum`、`np.mean`、`np.transpose`、`np.shape`、`np.ndim`、`np.size` 转发至对象自身方法；
- 将 `np.array_equal`（要求形状相同，`broadcast=False`）与 `np.array_equiv`（允许广播，`broadcast=True`）转发给 `_numpy_equal`
  ，支持 `equal_nan=True` 将对应位置均为 `<NA>` 的元素视为相等；
- 其余未列入白名单的 NumPy 函数一律抛出 `TypeError`。

**示例**

```python
>> > a = FixedArray(["1.50", None])
>> > np.array_equal(a, FixedArray(["1.5", None]), equal_nan=False)
False
>> > np.array_equal(a, FixedArray(["1.5", None]), equal_nan=True)
True
```

### FixedArray._format_scalar

```python
FixedArray._format_scalar(m, scale)
```

将单个整数尾数 `m` 按指定 `scale` 格式化为十进制字符串（补齐前导零与小数点）。

**示例**

```python
>> > FixedArray._format_scalar(150, 2), FixedArray._format_scalar(-5, 2), FixedArray._format_scalar(42, 0)
('1.50', '-0.05', '42')
```
