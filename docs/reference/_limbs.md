# _limbs — Base $2^{21}$ 二进制分块多精度整数算术内核

`moneta._limbs` 是支撑 [`FixedArray`](fixed_array.md) 慢速防溢出通道（大数乘法、长向量矩阵内积、正负抵消大数求和、缩放长除法）的纯函数式多精度整数内核。
它将 `int64` 整数张量拆分为 **$2^{21}$ 进制（Base $2^{21}$，每块 21 比特）** 的动态分块列表（`limbs`），全程在原生 `int64` /
`uint64` 数组上用单周期位移（`>> 21`）与位掩码（`& MASK`）完成进位规整与位模式溢出检测，无需回退到慢速的 Python `object`
任意精度大整数，也绝不发生硬件溢出回绕。

```python
>> > import numpy as np
>> > from moneta._limbs import (

...
split_int64, normalize_limbs, limbs_mul, limbs_matmul,
...
limbs_sum, limbs_round_div_pow10, limbs_to_int64,
... )
>> > big = np.array([10 ** 15, -10 ** 15], dtype=np.int64)
>> > prod_limbs = limbs_mul(big, np.array([10 ** 6, 10 ** 6], dtype=np.int64))  # 中间积 10^21 > 2^63 - 1
>> > div_limbs = limbs_round_div_pow10(prod_limbs, k=6, mode="half_up")  # 在分块上除以 10^6 并舍入
>> > limbs_to_int64(div_limbs).tolist()  # 落回 int64 范围，精确无溢出
[1000000000000000, -1000000000000000]
>> >  # 正负相抵的大数求和：中间部分和超出 int64，但最终和为 -1
>> > MAX, MIN = (1 << 63) - 1, -(1 << 63)
>> > int(limbs_sum(np.array([MAX, MAX, MIN, MIN + 1], dtype=np.int64)))
-1
```

[← API 参考](index.md)

## 核心机制与设计约束

### 1. Base $2^{21}$ 分块表示法与位运算进位

任意多精度有符号整数张量 $V$ 由低位在前（Little-Endian）的 `limbs` 列表表示：

$$V = \sum_{i=0}^{L-1} \text{limbs}[i] \cdot 2^{21 i}$$

经 [`normalize_limbs`](#normalize_limbs) 规整后的规范化 `limbs` 满足以下三条不变式：

1. **至少包含 4 个分块（$L \ge 4$）**：前 3 个低位分块 `limbs[0]`, `limbs[1]`, `limbs[2]` 刚好完整覆盖 $3 \times 21 = 63$
   个数值比特；`limbs[3:]` 为符号扩展或超 63 位的高位块。
2. **低位块无符号、最高块带符号**：除最后一个分块 `limbs[-1]` 外，所有低位块严格位于非负区间 $[0, 2^{21} - 1]$
   （即 $[0, \text{MASK}]$）；最高块 `limbs[-1]` 为带符号 `int64`，且绝对值严格小于 $2^{42}$（一旦达到 $2^{42}$
   则自动向上动态追加新分块）。
3. **单周期位运算进位**：由于基数是 2 的幂（$2^{21}$），进位传递只需算术右移 `carry = temp >> 21`，取余只需按位与
   `r = temp & MASK`（Python/C 补码的算术右移天然向负无穷取整，使得 `r = temp & MASK` 对正负数均严格落在 $[0, 2^{21}-1]$
   ），彻底取代了昂贵的整数除法指令 `np.divmod`。

### 2. $3 \times 3$ 块卷积与内积无溢出上界证明

为什么选 21 比特而不是 32 比特或十进制 $10^9$？

- [`split_int64`](#split_int64) 将任意有符号 `int64`（范围 $[-2^{63}, 2^{63}-1]$）无损拆为 3 块：
  `pa[0], pa[1]` $\in [0, 2^{21})$，最高块 `pa[2]` $\in [-2^{21}, 2^{21})$（当输入落在区间 $[-2^{63}, -2^{63} + 2^{42})$ 时
  `pa[2] == -2**21`）。
- 两个 `int64` 做 $3 \times 3$ 块卷积时，任意单项乘积满足 $|pa_i \cdot pb_j| \le 2^{21} \cdot 2^{21} = 2^{42}$。
- 在第 $k = 2$ 个位权（$2^{42}$ 项：`pa[0]*pb[2] + pa[1]*pb[1] + pa[2]*pb[0]`）上最多叠加 3
  项，单次元素乘法的未规整块和满足 $|t_2| \le 3 \times 2^{42} \approx 1.32 \times 10^{13} \ll 2^{63} - 1$。
- 在矩阵乘法 [`limbs_matmul`](#limbs_matmul) 中，沿内积维以步长 [
  `STEP_K = 2**19 = 524,288`](#shiftbasemaskstep_k_max-与-_min) 分段累加，单段内同一位权累加的绝对值上界为：
  $$3 \times \text{STEP\_K} \times 2^{42} = 3 \times 2^{19} \times 2^{42} = 1.5 \times 2^{62} < 2^{63} - 1$$
  即使算上进位传递（$< 2^{43}$），也严格处于 `int64` 安全区间内，任意长度的内积都绝不发生 `int64` 溢出。

### 3. 高位块位模式溢出检测（零乘法收口）

当需要将规整后的 `limbs` 收口回 `int64` 时，[`limbs_to_int64`](#limbs_to_int64) **不做任何乘法**，而是直接检查 `limbs[3:]`
的二进制补码位模式：

- 低 63 位组合值 `low63 = (limbs[2] << 42) | (limbs[1] << 21) | limbs[0]` 的取值范围为 $[0, 2^{63}-1]$；
- **非负 `int64`（$[0, 2^{63}-1]$）**：当且仅当所有高位块 `limbs[3:]` 全为 `0`；
- **负 `int64`（$[-2^{63}, -1]$）**：在二进制补码中，第 63 位及更高位必须全为 `1`，即当且仅当 `limbs[3:-1]` 全等于 `MASK`（21
  个 `1`）且最高块 `limbs[-1] == -1`；
- 只要任一元素不满足上述两种位模式之一，说明其真实数值超出了 $[-2^{63}, 2^{63}-1]$，立即抛出 `OverflowError`。

## API 速查表

| 分类                 | 接口 / 常量                                                                           | 功能说明                                                              |
|----------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **模块常量**         | [`SHIFT`、`BASE`、`MASK`、`STEP_K`、`_MAX`、`_MIN`](#shiftbasemaskstep_k_max-与-_min) | Base $2^{21}$ 位宽、基数、位掩码、矩阵内积分段步长与 `int64` 极值常量 |
| **拆分、规整与收口** | [`split_int64`](#split_int64)                                                         | 将 `int64` 数组拆分为 3 个 21 位分块（低两块无符号，最高块带符号）    |
|                      | [`normalize_limbs`](#normalize_limbs)                                                 | 自低向高单周期位运算进位规整，支持大数动态扩容（输出至少 4 块）       |
|                      | [`limbs_to_int64`](#limbs_to_int64)                                                   | 基于高位块补码位模式检测溢出并将 `limbs` 还原为 `int64` 数组          |
| **多精度张量算术**   | [`limbs_mul`](#limbs_mul)                                                             | 逐元素精确大数乘法（$3 \times 3$ 块卷积 + 进位规整）                  |
|                      | [`limbs_matmul`](#limbs_matmul)                                                       | 分段防溢出精确矩阵内积（沿内积维按 `STEP_K = 2**19` 分段累加）        |
|                      | [`limbs_sum`](#limbs_sum)                                                             | 沿指定轴分块精确求和并收口为 `int64`（消除中间和溢出假阳性）          |
|                      | [`limbs_round_div_pow10`](#limbs_round_div_pow10)                                     | 大数分块除以 $10^k$ 的竖式长除法与粘滞位（Sticky Bit）一次舍入        |

## 模块常量

### `SHIFT`、`BASE`、`MASK`、`STEP_K`、`_MAX` 与 `_MIN`

```python
moneta._limbs.SHIFT = 21
moneta._limbs.BASE = np.int64(2097152)       # 1 << 21
moneta._limbs.MASK = np.int64(2097151)       # (1 << 21) - 1
moneta._limbs.STEP_K = 524288                # 1 << 19
moneta._limbs._MAX = 9223372036854775807     # (1 << 63) - 1
moneta._limbs._MIN = -9223372036854775808    # -(1 << 63)
```

| 常量            | 值                                  | 含义                                                       |
|-----------------|-------------------------------------|------------------------------------------------------------|
| `SHIFT`         | `21`                                | 每个分块的二进制位宽（21 bits）                            |
| `BASE`          | `np.int64(2**21)`（`2,097,152`）    | 分块基数，用于长除法余数进位 `rem * BASE + limb`           |
| `MASK`          | `np.int64(2**21 - 1)`（`0x1FFFFF`） | 低 21 位掩码，用于提取无符号分块 `temp & MASK`             |
| `STEP_K`        | `2**19`（`524,288`）                | [`limbs_matmul`](#limbs_matmul) 沿内积维单次累加的最大项数 |
| `_MAX` / `_MIN` | `2**63 - 1` / `-2**63`              | 有符号 64 位整数的上下界                                   |

**示例**

```python
>>> from moneta._limbs import SHIFT, BASE, MASK, STEP_K
>>> (SHIFT, int(BASE), int(MASK), STEP_K)
(21, 2097152, 2097151, 524288)
```

## 拆分、规整与收口

### `split_int64`

```python
moneta._limbs.split_int64(a)
```

将输入的 `int64` 标量或数组无损分解为 3 个 21 位分块构成的列表 `[low, mid, high]`（低位在前）。

**参数**

- **a** : *np.ndarray 或 int64 array_like* — 输入的整数数组。

**返回值**

- **limbs** : *list of np.ndarray（长度为 3）* —
    - `limbs[0] = a & MASK` $\in [0, 2^{21})$（第 0~20 位）；
    - `limbs[1] = (a >> 21) & MASK` $\in [0, 2^{21})$（第 21~41 位）；
    - `limbs[2] = a >> 42` $\in [-2^{21}, 2^{21})$（第 42~63 位，带符号）。

**机制说明**

利用二次算术右移 `q1 = a_arr >> 21` 与按位与 `& MASK`，仅需 4
条向量化位运算指令即可完成拆分，且对任意 $x \in [-2^{63}, 2^{63}-1]$
均严格满足 $x = \text{limbs}[0] + \text{limbs}[1] \cdot 2^{21} + \text{limbs}[2] \cdot 2^{42}$。

**示例**

```python
>>> import numpy as np
>>> from moneta._limbs import split_int64, SHIFT
>>> parts = split_int64(np.array([0, -1, -(1 << 63)], dtype=np.int64))
>>> [p.tolist() for p in parts]
[[0, 2097151, 0], [0, 2097151, 0], [0, -1, -2097152]]
>>> sum(int(parts[i][2]) << (SHIFT * i) for i in range(3)) == -(1 << 63)
True
```

**另请参阅**

- [`normalize_limbs`](#normalize_limbs)
- [`limbs_to_int64`](#limbs_to_int64)

### `normalize_limbs`

```python
moneta._limbs.normalize_limbs(limbs)
```

进位规整器：自低位向高位逐块传递进位，将未规整的分块序列转换为至少包含 4 块、低位块位于 $[0, 2^{21})$
、最高块绝对值 $< 2^{42}$ 的规范化 `limbs` 列表。

**参数**

- **limbs** : *Sequence[np.ndarray]* — 未完全规整的分块列表（各块可正可负，绝对值 $\le 1.5 \times 2^{62}$）。

**返回值**

- **out** : *list of np.ndarray（长度 $\ge 4$）* — 规范化后的分块列表（对空数组同样安全）。

**机制说明**

1. **逐块推波进位**：遍历 `limbs[:-1]`，对每块计算 `temp = limb + carry`，以 `carry = temp >> 21` 抽出向高一位的带符号进位，以
   `r = temp & MASK` 留下 $[0, 2^{21})$ 内的非负余数；
2. **补齐至少 4 块**：若已产出的低位块不足 3 个（即总块数不足 4），继续对 `top` 执行 `& MASK` 与 `>> 21` 拆分，确保前 3 块完整覆盖
   63 位无符号有效位；
3. **大数动态扩容**：当最高块 `top` 中存在绝对值 $\ge 2^{42}$ 的元素时，自动循环拆分 `top & MASK` 与 `top >> 21`
   追加新分块，直到最高块绝对值严格小于 $2^{42}$。

**示例**

```python
>>> import numpy as np
>>> from moneta._limbs import normalize_limbs, limbs_to_int64
>>> norm = normalize_limbs([np.asarray(-1, dtype=np.int64)])
>>> len(norm), [int(x) for x in norm]
(4, [2097151, 2097151, 2097151, -1])
>>> int(limbs_to_int64(norm))
-1
```

**另请参阅**

- [`split_int64`](#split_int64)
- [`limbs_to_int64`](#limbs_to_int64)

### `limbs_to_int64`

```python
moneta._limbs.limbs_to_int64(limbs)
```

检验 `limbs` 表示的大数是否落在有符号 `int64` 范围内，并将其无溢出地合成为 `int64` NumPy 数组。

**参数**

- **limbs** : *Sequence[np.ndarray]* — 分块列表（若长度小于 4 会先自动调用 [`normalize_limbs`](#normalize_limbs)）。

**返回值**

- **out** : *np.ndarray（int64）*

**异常**

- **OverflowError** — 存在任一元素的真实值超出 $[-2^{63}, 2^{63}-1]$（报 `"结果超出 int64 范围"`）。

**机制说明**

1. 检查高位块 `high = limbs[3:]`：
    - 非负合法位模式：`is_zero = all(h == 0 for h in high)`；
    - 负数合法位模式（补码符号扩展）：`is_neg = all(h == MASK for h in high[:-1]) & (high[-1] == -1)`；
    - 若並非所有元素都满足 `is_zero | is_neg`，立即抛出 `OverflowError`。
2. 合成低 63 位：`low63 = (limbs[2] << 42) | (limbs[1] << 21) | limbs[0]`；
3. 对负数位置通过 `(low63.astype(np.uint64) | (1 << 63)).view(np.int64)` 置上最高符号位，全程无乘法、无溢出。

**示例**

```python
>>> import numpy as np
>>> from moneta._limbs import split_int64, limbs_mul, limbs_to_int64
>>> a = np.array([(1 << 63) - 1, -(1 << 63)], dtype=np.int64)
>>> limbs_to_int64(split_int64(a)).tolist() == a.tolist()
True
>>> limbs_to_int64(limbs_mul(np.array([10**10]), np.array([10**10])))
Traceback (most recent call last):
    ...
OverflowError: 结果超出 int64 范围
```

**另请参阅**

- [`normalize_limbs`](#normalize_limbs)
- [`FixedArray._limbs_to_mantissa`](fixed_array.md#fixedarray_limbs_to_mantissa)

## 多精度张量算术

### `limbs_mul`

```python
moneta._limbs.limbs_mul(a, b)
```

计算两个广播兼容的 `int64` 数组 `a * b` 的精确 126 位大数乘积，返回规整后的 `limbs` 列表。

**参数**

- **a, b** : *np.ndarray（int64）* — 可广播的整数数组。

**返回值**

- **limbs** : *list of np.ndarray* — 精确乘积的规范化 `limbs` 列表。

**机制说明**

分别调用 [`split_int64`](#split_int64) 将 `a` 和 `b` 拆为 3 个 21 位块，执行 $3 \times 3$ 块卷积得到 5 个位权项 `t[0..4]`
（其中 `t[i + j] += pa[i] * pb[j]`），最后调用 [`normalize_limbs(t)`](#normalize_limbs) 规整进位。即使输入为
`-2**63 * -2**63 = +2**126`，也能在 `int64` 块数组内精确表示。

**示例**

```python
>>> import numpy as np
>>> from moneta._limbs import limbs_mul, limbs_round_div_pow10, limbs_to_int64
>>> # (3 * 10^18) * (2 * 10^18) / 10^18 = 6 * 10^18
>>> prod = limbs_mul(np.array([3 * 10**18]), np.array([2 * 10**18]))
>>> int(limbs_to_int64(limbs_round_div_pow10(prod, 18, "half_up"))[0]) == 6 * 10**18
True
```

**性能对比**

|         N | `limbs_mul(a, b)`（126 位精确大数） | `a * b`（原生 `int64`，越界静默回绕） |
|----------:|------------------------------------:|--------------------------------------:|
|  0-d 标量 |                             5.43 µs |                                287 ns |
|     1,000 |                             31.1 µs |                                660 ns |
| 1,000,000 |                             21.9 ms |                                712 µs |

**另请参阅**

- [`limbs_round_div_pow10`](#limbs_round_div_pow10)
- [`FixedArray.mul_round`](fixed_array.md#fixedarraymul_round)

### `limbs_matmul`

```python
moneta._limbs.limbs_matmul(a, b)
```

计算两个满足矩阵乘法维度约束的 `int64` 数组 `a @ b` 的精确大数内积，返回规整后的 `limbs` 列表。支持 1D 向量点积、2D
矩阵乘法以及高维批量矩阵乘法（Batch Matmul）。

**参数**

- **a, b** : *np.ndarray（int64）* — 维度 $\ge 1$ 且内积维匹配的整数张量。

**返回值**

- **limbs** : *list of np.ndarray* — 精确矩阵乘积的规范化 `limbs` 列表。

**异常**

- **ValueError** — `a` 或 `b` 为 0 维标量，或矩阵内积维度不匹配（由 `np.matmul` 抛出）。

**机制说明**

沿内积轴 `terms = a.shape[-1]` 以 [`STEP_K = 2**19 = 524,288`](#shiftbasemaskstep_k_max-与-_min) 为步长分片：

1. 对每个分片拆分为 3 个 21 位块，执行 9 次 `np.matmul(pa[i], pb[j])` 累加到 `t[0..4]`
   （单段内任意元素绝对值 $\le 1.5 \times 2^{62} < 2^{63}-1$，绝不溢出）；
2. 调用 [`normalize_limbs(t)`](#normalize_limbs) 规整当前分片结果，并与历史累加器 `acc` 逐块相加后再次 `normalize_limbs`。

**示例**

```python
>>> import numpy as np
>>> from moneta._limbs import limbs_matmul, limbs_to_int64
>>> MAX, MIN = (1 << 63) - 1, -(1 << 63)
>>> # 第一项 MAX * MAX 远超 int64，但与第二项 MAX * (-MAX) 精确抵消后等于 42
>>> a = np.array([MAX, MAX, 6], dtype=np.int64)
>>> b = np.array([MAX, -MAX, 7], dtype=np.int64)
>>> int(limbs_to_int64(limbs_matmul(a, b)))
42
```

**另请参阅**

- [`limbs_mul`](#limbs_mul)
- [`FixedArray.matmul`](fixed_array.md#fixedarraymatmul)

### `limbs_sum`

```python
moneta._limbs.limbs_sum(arr, axis=None, keepdims=False)
```

沿指定轴对 `int64` 数组进行分块精确求和，并在最终结果不越界时收口返回 `int64` 数组。

**参数**

- **arr** : *np.ndarray（int64）* — 输入整数数组。
- **axis** : *None, int 或 tuple of int，可选* — 求和轴；默认 `None` 对全部元素求和。
- **keepdims** : *bool，默认 `False`* — 是否保留被规约的轴（长度为 1）。

**返回值**

- **out** : *np.ndarray（int64）*

**异常**

- **OverflowError** — 沿轴求和的 **最终真实总和**超出 `int64` 范围。

**机制说明**

先调用 `pa = split_int64(arr)` 将每个元素拆为 3 个绝对值 $\le 2^{21}$ 的分块，分别沿指定轴执行原生
`limb.sum(axis=axis, keepdims=keepdims)`（由于每块绝对值 $\le 2^{21}$
，只要单次累加元素个数不超过 $2^{63} / 2^{21} = 2^{42} \approx 4.39 \times 10^{12}$（约 4.39 万亿）个，分块求和就绝不会在
`int64` 中溢出），最后通过 `limbs_to_int64(normalize_limbs(s))` 规整进位并收口。这样即使序列前半部分之和远超 `2**63 - 1`
、只要后半部分正负相抵使最终总和落回 `int64`，就能精确算出结果。

**示例**

```python
>> > import numpy as np
>> > from moneta._limbs import limbs_sum
>> > MAX, MIN = (1 << 63) - 1, -(1 << 63)
>> > int(limbs_sum(np.array([MAX, MAX, MIN, MIN], dtype=np.int64)))
-2
>> > limbs_sum(np.array([[MAX, 1], [MIN, 2]], dtype=np.int64), axis=0).tolist()
[-1, 3]
```

**性能对比**

|         N | `limbs_sum(a)`（防中间溢出慢路径） | `a.sum()`（原生 `int64` 快路径） |
|----------:|-----------------------------------:|---------------------------------:|
|     1,000 |                            18.2 µs |                           753 ns |
| 1,000,000 |                            14.5 ms |                           320 µs |

**另请参阅**

- [`FixedArray.sum`](fixed_array.md#fixedarraysum)

### `limbs_round_div_pow10`

```python
moneta._limbs.limbs_round_div_pow10(limbs, k, mode="half_up")
```

对规整的 `limbs` 大数执行除以 $10^k$ 的竖式长除法，并按 `mode` 一次性精确舍入，返回舍入后的规范化 `limbs` 列表。

**参数**

- **limbs** : *Sequence[np.ndarray]* — 输入大数的分块列表。
- **k** : *int* — 除以 10 的幂次（$k \ge 0$；当 $k = 0$ 时仅调用 `normalize_limbs` 规整）。
- **mode** : *str，默认 `"half_up"`* — 6 种合法舍入模式之一。

**返回值**

- **out** : *list of np.ndarray* — 除以 $10^k$ 并完成舍入后的规范化 `limbs` 列表。

**异常**

- **ValueError** — `mode` 不是合法的舍入模式名。

**机制说明**

1. **分轮除以 $10^c$（$c = \min (k, 12)$）**：每轮取除数 $p = 10^c \le 10^{12}$。从最高块 `cur_limbs[-1]` 向最低块依次执行带余除法：
   $$\text{q}, \text{rem} = \text{divmod} (\text{rem} \cdot 2^{21} + \text{limb}, \; 10^c)$$
   因为 $0 \le \text{rem} < 10^{12}$ 且 $0 \le \text{limb} < 2^{21}$
   ，被除数 $\text{rem} \cdot 2^{21} + \text{limb} < 10^{12} \times 2^{21} \approx 2.10 \times 10^{18} < 2^{63} - 1$
   ，全程在原生 `int64` 内绝不溢出！
2. **跨轮粘滞位（Sticky Bit）追踪**：当 $k > 12$ 需要多轮长除时，中间轮次被舍去的非零余数会通过布尔掩码
   `sticky = sticky | (rem != 0)` 记录下来。
3. **终局一次舍入**：最后一轮长除结束后，余数 `rem` $\in [0, p)$，且商 `cur_limbs` 是向下取整（Floor）的精确整数商：
    - 存在非零小数部分：`has_fraction = (rem != 0) | sticky`；
    - 严格大于半数 `half = p // 2`：`over_half = (rem > half) | ((rem == half) & sticky)`；
    - 恰好等于半数：`exact_half = (rem == half) & ~sticky`；
    - 商为负数：`is_negative = cur_limbs[-1] < 0`；
    - 商为奇数：`odd_q = (cur_limbs[0] & 1) == 1`（因为 $2^{21}$ 是偶数，整个大数的奇偶性完全由最低块 `cur_limbs[0]`
      的最低比特决定！）。
      将上述条件送入 [`decide_bump(..., floor_base=True)`](_rounding.md#decide_bump)，把进位加到 `cur_limbs[0]`
      后再调用 [`normalize_limbs`](#normalize_limbs) 即可。

**示例**

```python
>> > import numpy as np
>> > from moneta._limbs import split_int64, limbs_round_div_pow10, limbs_to_int64
>> > vs = np.array([25, -25, 35, -35], dtype=np.int64)
>> > limbs_to_int64(limbs_round_div_pow10(split_int64(vs), k=1, mode="half_even")).tolist()
[2, -2, 4, -4]
>> > limbs_to_int64(limbs_round_div_pow10(split_int64(vs), k=1, mode="half_up")).tolist()
[3, -3, 4, -4]
```

**另请参阅**

- [`decide_bump`](_rounding.md#decide_bump)
- [`limbs_to_int64`](#limbs_to_int64)
- [`FixedArray._limbs_to_mantissa`](fixed_array.md#fixedarray_limbs_to_mantissa)
