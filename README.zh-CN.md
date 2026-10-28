# Moneta

[English](README.md) | 简体中文

<p align="center"><img src="docs/_static/moneta-logo.svg" width="128" alt="Moneta logo"></p>

Moneta 是一个基于 NumPy 的**定点数**多币种金融计算库。所有数值在底层都以 `int64` 尾数加十进制小数位数（`scale`）表示，加减乘、求和、矩阵乘法全程走整数运算，不会出现浮点累积误差；舍入只在你明确要求的地方、按你指定的规则发生一次。

```python
>>> from moneta import FixedArray
>>> 0.1 + 0.2 == 0.3
False
>>> bool(FixedArray("0.1") + FixedArray("0.2") == FixedArray("0.3"))
True
```

## 特性

- **精确定点数张量**：[`FixedArray`](docs/zh/reference/fixed_array.md) 支持广播、切片、求和、矩阵乘法与缺失值（`<NA>`），只在超出 `int64` 时报错，不会悄悄丢精度。
- **显式单次舍入**：六种舍入模式（`half_up`、`half_even`、`down`、`up`、`floor`、`ceil`），乘除法融合舍入，结果与对精确值舍入一次完全一致。
- **量纲安全**：[`Money`](docs/zh/reference/money.md)、[`Price`](docs/zh/reference/price.md)、[`Quantity`](docs/zh/reference/quantity.md) 拦截跨币种、跨单位的非法运算，`Price × Quantity → Money` 自动完成量纲相消；金额始终保持币种精度，支持最大余数法守恒分配。
- **外汇**：[`ExchangeRate`](docs/zh/reference/fx.md) 支持买卖价与正反向单次舍入换算；[`FXEngine`](docs/zh/reference/fx.md) 经锚定货币精确套算交叉汇率，可组合多个汇率源，按日期严格取价。
- **持仓与组合**：[`PositionLot`](docs/zh/reference/lot.md) 同时记录计价货币与基准货币成本，[`Portfolio`](docs/zh/reference/portfolio.md) 支持 FIFO / LIFO / WAC 平仓、多币种估值、盈亏与再平衡试算。
- **现金流与时间价值**：[付息排程](docs/zh/reference/cashflow.md)、[计日惯例](docs/zh/reference/conventions.md)（ACT/360、ACT/365 Fixed、30/360 Bond Basis、ACT/ACT ISDA）与 [`pv` / `fv` / `npv`](docs/zh/reference/tvm.md)。
- **展示格式化**：[`MonetaryFormatter`](docs/zh/reference/formatter.md) 输出千分位、币种符号与代码、会计负数括号。

## 安装

要求 Python ≥ 3.9、NumPy ≥ 1.22。从源码安装：

```bash
git clone <repo-url> moneta
cd moneta
pip install -e .
```

| 可选依赖 | 安装命令                     | 用途                     |
|----------|------------------------------|--------------------------|
| pandas   | `pip install -e ".[pandas]"` | `to_pandas()` 导出       |
| polars   | `pip install polars`         | `to_polars()` 导出       |
| 测试     | `pip install -e ".[test]"`   | 运行测试套件             |

## 快速上手

**定点数张量与显式舍入**

```python
>>> prices = FixedArray(["19.99", "5.25", "102.40"])
>>> amount = prices * FixedArray([3, 10, 1])        # 乘法精确，scale = 2 + 0
>>> amount, amount.sum()
(FixedArray([59.97, 52.50, 102.40], scale=2), FixedArray(214.87, scale=2))
>>> amount.mul_round(FixedArray("0.0325"), scale=2, rounding="half_even")
FixedArray([1.95, 1.71, 3.33], scale=2)
```

**带量纲的金额、价格与数量**

```python
>>> from moneta import Money, Price, Quantity
>>> px = Price(["189.50", "200.00"], "USD", base_unit="share")
>>> total = (px * Quantity([10, 5], unit="share")).sum()   # Price × Quantity → Money
>>> print(total)
2895.00 USD
>>> [str(part) for part in Money("100.00", "USD").allocate([1, 1, 1])]   # 守恒分配
['33.34 USD', '33.33 USD', '33.33 USD']
>>> Money("1.00", "USD") + Money("1.00", "CNY")
Traceback (most recent call last):
    ...
TypeError: 币种不匹配: USD 与 CNY
```

**外汇换算与交叉汇率**

```python
>>> from moneta import ExchangeRate, FXEngine, UserDictFXProvider
>>> print(ExchangeRate("USD", "CNY", "7.2").convert(Money("100.00", "USD")))
720.00 CNY
>>> engine = FXEngine(UserDictFXProvider({"EUR/USD": "1.085", "USD/JPY": "155.5"}))
>>> engine.get_rate("EUR", "JPY")                   # 经 USD 精确套算：1.085 × 155.5
ExchangeRate('EUR', 'JPY', '168.7175')
```

**投资组合**

```python
>>> from moneta import Instrument, Portfolio
>>> aapl = Instrument("AAPL", "USD")
>>> p = Portfolio("CNY")
>>> _ = p.deposit(aapl, 100, "2024-01-02", price="100", fx=ExchangeRate("USD", "CNY", "7.0"))
>>> fx = ExchangeRate("USD", "CNY", "7.2")
>>> print(p.total_cost(), p.total_value({"AAPL": 120}, fx=fx), p.unrealized_pnl({"AAPL": 120}, fx=fx), sep=" | ")
70000.00 CNY | 86400.00 CNY | 16400.00 CNY
```

**付息排程与净现值**

```python
>>> from moneta import generate_cash_flow_schedule, npv
>>> flows = generate_cash_flow_schedule("2026-01-01", "2027-01-01", "semi-annual",
...                                     principal=Money("1000", "USD"), coupon_rate="0.05")
>>> [(str(f.pay_date), str(f.amount), f.label) for f in flows]
[('2026-07-01', '25.00 USD', 'Coupon'), ('2027-01-01', '25.00 USD', 'Coupon'), ('2027-01-01', '1000.00 USD', 'Principal')]
>>> print(npv(0.05, flows, "2026-01-01", convention="ACT/ACT"))
1000.59 USD
```

**格式化输出**

```python
>>> from moneta import MonetaryFormatter
>>> MonetaryFormatter.format(Money("-1234567.89", "USD"), accounting=True)
'($1,234,567.89 USD)'
>>> MonetaryFormatter.format(Money("1234567", "JPY")), MonetaryFormatter.format(Money("1234.5", "CNY"))
('¥1,234,567 JPY', '¥1,234.50 CNY')
```

## 文档

在线阅读：**<https://t0nyn1.github.io/Moneta/>**，也可以直接浏览 Markdown 源文件：

- [文档首页](docs/zh/index.md)：架构总览、安装与五分钟上手
- [API 参考](docs/zh/reference/index.md)：各模块的机制说明、完整参数、示例与性能对比

| 层次     | 模块                                                                                                                                                                                                                       |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 数值内核 | [`fixed_array`](docs/zh/reference/fixed_array.md)、[`_rounding`](docs/zh/reference/_rounding.md)、[`_limbs`](docs/zh/reference/_limbs.md)、[`_utils`](docs/zh/reference/_utils.md)                                                        |
| 量纲层   | [`unit_array`](docs/zh/reference/unit_array.md)、[`_units`](docs/zh/reference/_units.md)、[`currency`](docs/zh/reference/currency.md)                                                                                                |
| 业务对象 | [`money`](docs/zh/reference/money.md)、[`price`](docs/zh/reference/price.md)、[`quantity`](docs/zh/reference/quantity.md)、[`instrument`](docs/zh/reference/instrument.md)、[`lot`](docs/zh/reference/lot.md)、[`fx`](docs/zh/reference/fx.md)、[`formatter`](docs/zh/reference/formatter.md) |
| 应用层   | [`portfolio`](docs/zh/reference/portfolio.md)、[`conventions`](docs/zh/reference/conventions.md)、[`cashflow`](docs/zh/reference/cashflow.md)、[`tvm`](docs/zh/reference/tvm.md)                                                        |

## 开发

```bash
pip install -e ".[test]"
pytest
python docs/_tools/check_docs.py
```

`check_docs.py` 会执行文档中的全部示例，并检查链接与示例格式。

## 许可证

[MIT](LICENSE)
