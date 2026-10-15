# Moneta 文档

**版本**：0.1.0 · **依赖**：Python ≥ 3.9，NumPy ≥ 1.22

Moneta 是一个基于 NumPy 的 **定点数**多币种金融计算库。所有数值在底层都以 `int64` 尾数加十进制小数位数（`scale`）表示，
加减乘、求和、矩阵乘法全程走整数运算，不会出现 IEEE 754 浮点的累积误差；舍入只在你明确要求的地方、按你指定的规则发生一次。

```python
>> > 0.1 + 0.2 == 0.3
False
>> > bool(FixedArray("0.1") + FixedArray("0.2") == FixedArray("0.3"))
True
```

- [API 参考](reference/index.md) — 公开模块、类与函数的完整说明、机制解析、示例与性能对比

## 架构总览

Moneta 自下而上分为四层，上层只依赖下层：

```text
┌────────────────────────────────────────────────────────────────────┐
│ 应用层   Portfolio · FXEngine · LiveFXWatcher · CashFlow · tvm      │
├────────────────────────────────────────────────────────────────────┤
│ 业务对象  Money · Price · Quantity · Instrument · PositionLot        │
│           ExchangeRate · MonetaryFormatter                         │
├────────────────────────────────────────────────────────────────────┤
│ 量纲层   UnitArray（FixedArray + 单位）· Currency · CurrencyRegistry │
├────────────────────────────────────────────────────────────────────┤
│ 数值内核  FixedArray · RoundingMode   （纯 int64 定点数张量）        │
└────────────────────────────────────────────────────────────────────┘
```

- **数值内核**：[`FixedArray`](reference/fixed_array.md) 是全部精确运算的基础，它只认识“数”，不认识币种或单位。
- **量纲层**：在 `FixedArray` 之上附加单位/币种，阻止不同量纲之间的非法运算。
- **业务对象**：金额、价格、数量、标的、持仓批次等金融语义对象，本身也可以是数组（张量）。
- **应用层**：组合估值、汇率引擎、现金流排程与货币时间价值计算。

## 安装

从源码安装：

```bash
git clone <repo-url> moneta
cd moneta
pip install -e .
```

| 可选依赖 | 安装命令                     | 用途                     |
|----------|------------------------------|--------------------------|
| pandas   | `pip install -e ".[pandas]"` | `FixedArray.to_pandas()` |
| polars   | `pip install polars`         | `FixedArray.to_polars()` |
| 测试     | `pip install -e ".[test]"`   | 运行测试套件             |

```python
>> > import moneta
>> > moneta.__version__
'0.1.0'
```

## 五分钟上手

**精确的定点数。** `FixedArray` 用 `int64` 尾数加小数位数 `scale` 表示数值，`v = mantissa × 10^(-scale)`。
浮点数按最短十进制写法解析，`0.1` 就是 `0.1`：

```python
>> > price = FixedArray("19.99")
>> > price.mantissa, price.scale
(array(1999), 2)
>> > FixedArray(0.1) * 3
FixedArray(0.3, scale=1)
>> > 0.1 * 3
0.30000000000000004
```

**像 NumPy 一样做张量运算。**

```python
>> > prices = FixedArray(["19.99", "5.25", "102.40"])
>> > qty = FixedArray([3, 10, 1])
>> > amount = prices * qty  # 乘法精确，scale = 2 + 0
>> > amount
FixedArray([59.97, 52.50, 102.40], scale=2)
>> > amount.sum()
FixedArray(214.87, scale=2)
```

**舍入只发生在你要求的地方。**

```python
>> > rate = FixedArray("0.0325")  # 年利率 3.25%
>> > amount.mul_round(rate, scale=2, rounding="half_even")
FixedArray([1.95, 1.71, 3.33], scale=2)
>> > FixedArray("100.00") / 3  # 除法默认保留 max(scale) 位并 half_up
FixedArray(33.33, scale=2)
```

**显式导出。**

```python
>> > amount.to_float()
array([59.97, 52.5, 102.4])
>> > amount.to_decimal()
array([Decimal('59.97'), Decimal('52.50'), Decimal('102.40')], dtype=object)
```

更多细节请参阅 [`FixedArray` 的 scale 传播规则](reference/fixed_array.md#scale-传播规则)、
[舍入模式](reference/fixed_array.md#roundingmode) 与 [NumPy 互操作机制](reference/fixed_array.md#numpy-互操作机制)。

## 示例约定

- 文档代码中以 `>>>` 开头的行是输入，紧随其后不带提示符的行是输出。
- 所有示例默认已执行 `import numpy as np` 与 `from moneta import FixedArray`，需要其他名称时会在示例中显式导入。
