# _rounding — 金融与会计舍入规则与进位裁决引擎

`moneta._rounding` 是全库的统一舍入规则枚举与整数进位裁决内核，为 [`FixedArray`](fixed_array.md)、[`_limbs`](_limbs.md)、[
`Money`](money.md)、[`Price`](price.md) 与 [`Quantity`](quantity.md) 提供一致的 6 大舍入模式校验及无溢出进位判定。
其中公开枚举 [`RoundingMode`](#roundingmode) 同时由顶层 `moneta` 与 `moneta.fixed_array` 重导出。

```python
>> > from moneta._rounding import RoundingMode, check_rounding, decide_bump, round_quotient
>> > check_rounding(RoundingMode.HALF_EVEN), check_rounding("half_up")
('half_even', 'half_up')
>> > q, r = divmod(-7, 2)  # -7 = (-4) * 2 + 1
>> > round_quotient(q, r, 2, "half_up"), round_quotient(q, r, 2, "half_even")
(-4, -4)
>> > q2, r2 = divmod(-5, 2)  # -5 = (-3) * 2 + 1
>> > round_quotient(q2, r2, 2, "half_up"), round_quotient(q2, r2, 2, "half_even")
(-3, -2)
```

[← API 参考](index.md)

## 核心机制与设计约束

### 六种金融与会计舍入模式

| 枚举成员                 | 字符串值      | 规则说明                                                         | 正半数（`+2.5`） | 负半数（`-2.5`） | 正奇半数（`+3.5`） | 负奇半数（`-3.5`） |
|--------------------------|---------------|------------------------------------------------------------------|-----------------:|-----------------:|-------------------:|-------------------:|
| `RoundingMode.HALF_UP`   | `"half_up"`   | 四舍五入（恰好一半时**远离零**，零售金融与票据常用）             |              `3` |             `-3` |                `4` |               `-4` |
| `RoundingMode.HALF_EVEN` | `"half_even"` | 银行家舍入（四舍六入五成双，GAAP / IFRS 推荐，消除长期统计偏差） |              `2` |             `-2` |                `4` |               `-4` |
| `RoundingMode.DOWN`      | `"down"`      | 向零截断（直接舍去非零小数部分）                                 |              `2` |             `-2` |                `3` |               `-3` |
| `RoundingMode.UP`        | `"up"`        | 远离零进位（只要存在非零小数即向绝对值更大方向进 1）             |              `3` |             `-3` |                `4` |               `-4` |
| `RoundingMode.FLOOR`     | `"floor"`     | 向负无穷截断（$\lfloor x \rfloor$）                              |              `2` |             `-3` |                `3` |               `-4` |
| `RoundingMode.CEIL`      | `"ceil"`      | 向正无穷进位（$\lceil x \rceil$）                                |              `3` |             `-2` |                `4` |               `-3` |

### 双基准坐标系裁决机制（`floor_base`）

在底层整数除法与长除法中，不同算法产出的“未舍入商”处于两种不同的基准坐标系。[`decide_bump`](#decide_bump) 通过
`floor_base` 参数同时支持这两种坐标系，避免调用方在负数区间做二次坐标转换：

| 维度                 | 向下取整基准（`floor_base=True`）                                                                                                                                                                                                         | 向零截断基准（`floor_base=False`，默认）                    |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| **商基准定义**       | 带符号向下取整商 $q = \lfloor x \rfloor$，满足 $x = q + f$，$f \in [0, 1)$                                                                                                                                                                | 无符号绝对值截断商 $t = \lfloor                             |x| \rfloor \ge 0$，满足 $|x| = t + f$，$f \in [0, 1)$ |
| **进位作用方式**     | `rounded = q + bump`（向 $+\infty$ 方向加 `0` 或 `1`）                                                                                                                                                                                    | `mag = t + bump`（向远离零方向加 `0` 或 `1`，再恢复原符号） |
| **返回值类型**       | `bool` 或 `ndarray(bool)`                                                                                                                                                                                                                 | `np.uint64(0)` 或 `np.uint64(1)`（及 `uint64` 数组）        |
| **负数时的余量方向** | 例：$x = -2.3$ 时 $q = -3$，$f = 0.7 > 0.5$（注意 `over_half` 是相对于 $q = -3$ 而言的 $f > 0.5$，或由调用方按对称余数传入）                                                                                                              | 例：$x = -2.3$ 时 $t = 2$，$f = 0.3 < 0.5$                  |
| **典型调用方**       | [`round_quotient`](#round_quotient)、[`limbs_round_div_pow10`](_limbs.md#limbs_round_div_pow10)、[`FixedArray._round_div_pow10`](fixed_array.md#fixedarray_round_div_pow10)、[`Money._fused_round_step`](money.md#money_fused_round_step) | 无符号幅值除法内核                                          |

在 `floor_base=True` 坐标系下（设 $x = q + f$，$f \in [0, 1)$，其中 $q < 0 \iff x < 0$），6 种模式对 `q` 是否加 `1` 的真值逻辑为：

- **`"floor"`**：恒为 `False`（$q$ 本身就是 $\lfloor x \rfloor$）；
- **`"ceil"`**：`has_fraction`（只要 $f > 0$ 即为 $q + 1$）；
- **`"down"`（向零）**：`has_fraction & is_negative`（负数且有小数时，$\lfloor x \rfloor$ 比向零截断小 1，需加 1）；
- **`"up"`（远离零）**：`has_fraction & np.logical_not(is_negative)`（正数且有小数时加 1，负数保持 $\lfloor x \rfloor$）；
- **`"half_up"`**：正数时 `over_half | exact_half`，负数时 `over_half`（因负数恰好一半如 $-2.5 = -3 + 0.5$
  远离零就是 $-3 = q$，无需加 1）；
- **`"half_even"`**：`over_half | (exact_half & odd_q)`（当 $q$ 为奇数且恰好一半时，$q + 1$ 必为偶数，无论正负均成立）。

## API 速查表

| 分类               | 接口 / 常量                         | 功能说明                                                   |
|--------------------|-------------------------------------|------------------------------------------------------------|
| **枚举与常量**     | [`RoundingMode`](#roundingmode)     | 6 种金融与会计舍入规则枚举（继承自 `str` 与 `Enum`）       |
|                    | [`_ROUNDINGS`](#_roundings)         | 全部合法舍入模式字符串构成的六元组常量                     |
| **校验与裁决函数** | [`check_rounding`](#check_rounding) | 校验并将 `RoundingMode` 或字符串规范化为小写模式名 `str`   |
|                    | [`decide_bump`](#decide_bump)       | 支持向下取整基准与向零截断基准的统一进位裁决函数           |
|                    | [`round_quotient`](#round_quotient) | 由向下取整商 `q`、余数 `r` 与除数 `d` 无溢出计算舍入后的商 |

## 枚举与常量

### `RoundingMode`

```python
class moneta._rounding.RoundingMode(str, Enum)
```

金融与会计舍入规则枚举。由于同时继承自 `str` 与 `Enum`，每个枚举成员本身就是对应的字符串实例（
`isinstance(RoundingMode.HALF_UP, str)` 为 `True`），在所有接受 `rounding` 参数的 API 中既可传枚举成员也可传同名小写字符串。

**成员列表**

- **`RoundingMode.HALF_UP = "half_up"`** — 四舍五入（逢五远离零）。
- **`RoundingMode.HALF_EVEN = "half_even"`** — 银行家舍入（四舍六入五成双）。
- **`RoundingMode.DOWN = "down"`** — 向零截断。
- **`RoundingMode.UP = "up"`** — 远离零进位。
- **`RoundingMode.FLOOR = "floor"`** — 向负无穷截断。
- **`RoundingMode.CEIL = "ceil"`** — 向正无穷进位。

**示例**

```python
>>> from moneta import RoundingMode
>>> RoundingMode.HALF_EVEN == "half_even" and isinstance(RoundingMode.HALF_EVEN, str)
True
>>> [m.value for m in RoundingMode]
['half_up', 'half_even', 'down', 'up', 'floor', 'ceil']
```

**另请参阅**

- [`check_rounding`](#check_rounding)
- [`_ROUNDINGS`](#_roundings)

### `_ROUNDINGS`

```python
moneta._rounding._ROUNDINGS = ('half_up', 'half_even', 'down', 'up', 'floor', 'ceil')
```

由 `RoundingMode` 全部成员值按定义顺序组成的不可变元组常量，供 [`check_rounding`](#check_rounding) 与 [
`decide_bump`](#decide_bump) 做 $O (1)$ 成员检查与错误信息拼装。

**示例**

```python
>>> from moneta._rounding import _ROUNDINGS
>>> _ROUNDINGS
('half_up', 'half_even', 'down', 'up', 'floor', 'ceil')
```

## 校验与裁决函数

### `check_rounding`

```python
moneta._rounding.check_rounding(mode)
```

校验舍入模式参数是否合法，并统一返回标准小写字符串。

**参数**

- **mode** : *RoundingMode 或 str* — 待校验的舍入模式。

**返回值**

- **out** : *str* — `"half_up"`、`"half_even"`、`"down"`、`"up"`、`"floor"`、`"ceil"` 之一（类型为纯 Python `str`）。

**异常**

- **ValueError** — `mode` 不在 `_ROUNDINGS` 中（如拼写错误、大写字符串 `"HALF_UP"`、`None` 或其他类型）。

**机制说明**

若 `isinstance(mode, RoundingMode)` 直接返回 `mode.value`；否则检查 `mode in _ROUNDINGS`，不匹配则抛出 `ValueError`。

**示例**

```python
>>> from moneta._rounding import RoundingMode, check_rounding
>>> check_rounding(RoundingMode.FLOOR)
'floor'
>>> check_rounding("ceil")
'ceil'
>>> check_rounding("nearest")
Traceback (most recent call last):
    ...
ValueError: rounding 只能是 ('half_up', 'half_even', 'down', 'up', 'floor', 'ceil') 之一，收到 'nearest'
```

**另请参阅**

- [`RoundingMode`](#roundingmode)
- [`decide_bump`](#decide_bump)

### `decide_bump`

```python
moneta._rounding.decide_bump(
    mode,
    *,
    is_negative,
    odd_q,
    has_fraction,
    over_half,
    exact_half,
    floor_base=False,
)
```

根据商的奇偶性、正负号以及小数余量与 $1/2$ 的大小关系，为 6 大舍入模式统一裁决是否需要对商基准值加 `1`。支持标量与任意维度的
NumPy 布尔数组向量化计算。

**参数**

- **mode** : *str* — 已规范化的舍入模式名称（`check_rounding` 的返回值）。
- **is_negative** : *bool 或 array_like of bool* — 数值是否为负（在 `floor_base=True` 下即 `q < 0`）。
- **odd_q** : *bool 或 array_like of bool* — 基准整数商是否为奇数（`q % 2 != 0` 或 `(q & 1) == 1`）。
- **has_fraction** : *bool 或 array_like of bool* — 是否存在非零小数余量。
- **over_half** : *bool 或 array_like of bool* — 小数余量是否严格大于 $0.5$。
- **exact_half** : *bool 或 array_like of bool* — 小数余量是否恰好等于 $0.5$。
- **floor_base** : *bool，默认 `False`* — 商基准坐标系选择：
    - `True`（向下取整基准）：基准商 $q = \lfloor x \rfloor$，返回 `bool` 或 `ndarray(bool)`，满足 `rounded = q + bump`；
    - `False`（向零截断基准）：基准商 $t = \lfloor |x| \rfloor \ge 0$，返回 `np.uint64(0)` 或 `np.uint64(1)`（或 `uint64`
      数组），满足 `rounded_abs = t + bump`。

**返回值**

- **bump** : *bool、ndarray (bool)、np.uint64 或 ndarray (uint64)*

**异常**

- **ValueError** — `mode` 不属于 6 种合法舍入模式。

**示例**

向下取整基准（`floor_base=True`）与向零截断基准（`floor_base=False`）对比：

```python
>>> from moneta._rounding import decide_bump
>>> kw = dict(is_negative=False, odd_q=False, has_fraction=True, over_half=False, exact_half=True, floor_base=True)
>>> bool(decide_bump("half_up", **kw)), bool(decide_bump("half_even", **kw))
(True, False)
>>> kw_neg = dict(is_negative=True, odd_q=False, has_fraction=True, over_half=True, exact_half=False, floor_base=False)
>>> int(decide_bump("down", **kw_neg)), int(decide_bump("floor", **kw_neg))
(0, 1)
```

**另请参阅**

- [`round_quotient`](#round_quotient)
- [`limbs_round_div_pow10`](_limbs.md#limbs_round_div_pow10)
- [`FixedArray._round_div_pow10`](fixed_array.md#fixedarray_round_div_pow10)

### `round_quotient`

```python
moneta._rounding.round_quotient(q, r, d, mode)
```

由向下取整带余除法得到的商 `q`、余数 `r` 与除数 `d`（满足 $m = q \cdot d + r$，$r$ 与 $d$ 同号且 $|r| < |d|$）按 `mode`
计算舍入后的整数商。

**参数**

- **q** : *int, np.integer 或 np.ndarray* — 向下取整的商（`q, r = divmod(m, d)` 或 `np.divmod(m, d)` 的第一项）。
- **r** : *int, np.integer 或 np.ndarray* — 余数（与 `d` 同号，$|r| < |d|$）。
- **d** : *int, np.integer 或 np.ndarray* — 除数（可正可负，不能为 0）。
- **mode** : *str* — 舍入模式名称。

**返回值**

- **out** : *int, np.integer 或 np.ndarray* — 舍入后的商 `q + bump`。当 `q` 为 Python `int` 且 `bump` 为 0 维标量时返回
  Python `int`。

**机制说明**

真实商为 $q + r/d$，其中小数部分 $r/d \in [0, 1)$：

1. 若 `mode == "floor"`，无需任何计算直接返回 `q`（短路优化）；
2. 若 `mode` 为 `"half_up"` 或 `"half_even"`，需要比较 $r/d$ 与 $1/2$ 的大小。若直接计算 `2 * r` 与 `d`
   比较，当 $|r| > 2^{62}$ 时 `2 * r` 会在 `int64` 中溢出回绕！因此函数改为比较 **`r` 与 `d - r`**：由于 $r$ 与 $d$
   同号且 $|r| < |d|$，$d - r$ 与 $d$ 必同号且 $|d - r| \le |d|$， **在任意 `int64` 除数下都绝对不会溢出**：
    - 当 $d > 0$ 时，$r/d > 1/2 \iff r > d - r$；
    - 当 $d < 0$ 时（此时 $r \le 0, d - r < 0$），$r/d > 1/2 \iff r < d - r$；
    - 无论 $d$ 正负，$r/d = 1/2 \iff r == d - r$。
3. 将计算出的布尔条件送入 [`decide_bump(..., floor_base=True)`](#decide_bump) 得到 `bump`，返回 `q + bump`。

**示例**

```python
>> > import numpy as np
>> > from moneta._rounding import round_quotient
>> > q, r = divmod(7, 2)
>> > [round_quotient(q, r, 2, m) for m in ("half_up", "half_even", "down", "up", "floor", "ceil")]
[4, 4, 3, 4, 3, 4]
>> > q_arr, r_arr = np.divmod(np.array([5, -5, 7, -7]), 2)
>> > round_quotient(q_arr, r_arr, 2, "half_even").tolist()
[2, -2, 4, -4]
```

**性能对比**

|         N | `round_quotient(q, r, d, "half_up")` | `np.round(a / b)` (`float64`，大整数有损) |
|----------:|-------------------------------------:|------------------------------------------:|
|  0-d 标量 |                              6.02 µs |                                   1.68 µs |
|     1,000 |                              11.3 µs |                                   2.05 µs |
| 1,000,000 |                               5.9 ms |                                   1.05 ms |

**另请参阅**

- [`decide_bump`](#decide_bump)
- [`FixedArray._divmod_round`](fixed_array.md#fixedarray_divmod_round)
- [`FixedArray._chunked_divmod_round`](fixed_array.md#fixedarray_chunked_divmod_round)
