# Moneta

English | [简体中文](README.zh-CN.md)

High-performance, fixed-point multi-currency and FX pricing tensor engine based on NumPy. Zero float drift, portfolio revaluation, and banking-grade precision.

Every value in Moneta is stored as an `int64` mantissa plus a decimal `scale`. Addition, subtraction, multiplication, sums and matrix products all run on integers, so floating-point errors never accumulate; rounding happens only where you ask for it, exactly once, with the rule you choose.

```python
>>> from moneta import FixedArray
>>> 0.1 + 0.2 == 0.3
False
>>> bool(FixedArray("0.1") + FixedArray("0.2") == FixedArray("0.3"))
True
```

## Features

- **Exact fixed-point tensors**: [`FixedArray`](docs/reference/fixed_array.md) supports broadcasting, slicing, sums, matrix products and missing values (`<NA>`). It raises only when a result exceeds `int64`, and never silently loses precision.
- **Explicit single rounding**: six rounding modes (`half_up`, `half_even`, `down`, `up`, `floor`, `ceil`) and fused multiply/divide-and-round, giving exactly the result of rounding the exact value once.
- **Dimensional safety**: [`Money`](docs/reference/money.md), [`Price`](docs/reference/price.md) and [`Quantity`](docs/reference/quantity.md) reject cross-currency and cross-unit operations, and `Price × Quantity → Money` cancels units automatically. Money always stays at its currency's precision and supports conservative allocation by the largest-remainder method.
- **FX**: [`ExchangeRate`](docs/reference/fx.md) supports bid/ask quotes and single-rounding conversion in both directions; [`FXEngine`](docs/reference/fx.md) derives exact cross rates through an anchor currency, chains multiple rate sources and matches dates strictly.
- **Lots and portfolios**: [`PositionLot`](docs/reference/lot.md) records cost in both the quote currency and the base currency; [`Portfolio`](docs/reference/portfolio.md) supports FIFO / LIFO / WAC closing, multi-currency valuation, P&L and rebalancing.
- **Cash flows and time value of money**: [coupon schedules](docs/reference/cashflow.md), [day count conventions](docs/reference/conventions.md) (ACT/360, ACT/365 Fixed, 30/360 Bond Basis, ACT/ACT ISDA) and [`pv` / `fv` / `npv`](docs/reference/tvm.md).
- **Display formatting**: [`MonetaryFormatter`](docs/reference/formatter.md) renders thousands separators, currency symbols and codes, and accounting-style negative parentheses.

## Installation

Requires Python ≥ 3.9 and NumPy ≥ 1.22. Install from source:

```bash
git clone <repo-url> moneta
cd moneta
pip install -e .
```

| Optional | Command                      | Purpose               |
|----------|------------------------------|-----------------------|
| pandas   | `pip install -e ".[pandas]"` | `to_pandas()` export  |
| polars   | `pip install polars`         | `to_polars()` export  |
| tests    | `pip install -e ".[test]"`   | run the test suite    |

## Quick start

**Fixed-point tensors and explicit rounding**

```python
>>> prices = FixedArray(["19.99", "5.25", "102.40"])
>>> amount = prices * FixedArray([3, 10, 1])        # exact product, scale = 2 + 0
>>> amount, amount.sum()
(FixedArray([59.97, 52.50, 102.40], scale=2), FixedArray(214.87, scale=2))
>>> amount.mul_round(FixedArray("0.0325"), scale=2, rounding="half_even")
FixedArray([1.95, 1.71, 3.33], scale=2)
```

**Money, prices and quantities with units**

```python
>>> from moneta import Money, Price, Quantity
>>> px = Price(["189.50", "200.00"], "USD", base_unit="share")
>>> total = (px * Quantity([10, 5], unit="share")).sum()   # Price × Quantity → Money
>>> print(total)
2895.00 USD
>>> [str(part) for part in Money("100.00", "USD").allocate([1, 1, 1])]   # conservative allocation
['33.34 USD', '33.33 USD', '33.33 USD']
>>> Money("1.00", "USD") + Money("1.00", "CNY")   # error messages are currently in Chinese
Traceback (most recent call last):
    ...
TypeError: 币种不匹配: USD 与 CNY
```

**FX conversion and cross rates**

```python
>>> from moneta import ExchangeRate, FXEngine, UserDictFXProvider
>>> print(ExchangeRate("USD", "CNY", "7.2").convert(Money("100.00", "USD")))
720.00 CNY
>>> engine = FXEngine(UserDictFXProvider({"EUR/USD": "1.085", "USD/JPY": "155.5"}))
>>> engine.get_rate("EUR", "JPY")                   # exact cross via USD: 1.085 × 155.5
ExchangeRate('EUR', 'JPY', '168.7175')
```

**Portfolios**

```python
>>> from moneta import Instrument, Portfolio
>>> aapl = Instrument("AAPL", "USD")
>>> p = Portfolio("CNY")
>>> _ = p.deposit(aapl, 100, "2024-01-02", price="100", fx=ExchangeRate("USD", "CNY", "7.0"))
>>> fx = ExchangeRate("USD", "CNY", "7.2")
>>> print(p.total_cost(), p.total_value({"AAPL": 120}, fx=fx), p.unrealized_pnl({"AAPL": 120}, fx=fx), sep=" | ")
70000.00 CNY | 86400.00 CNY | 16400.00 CNY
```

**Coupon schedules and net present value**

```python
>>> from moneta import generate_cash_flow_schedule, npv
>>> flows = generate_cash_flow_schedule("2026-01-01", "2027-01-01", "semi-annual",
...                                     principal=Money("1000", "USD"), coupon_rate="0.05")
>>> [(str(f.pay_date), str(f.amount), f.label) for f in flows]
[('2026-07-01', '25.00 USD', 'Coupon'), ('2027-01-01', '25.00 USD', 'Coupon'), ('2027-01-01', '1000.00 USD', 'Principal')]
>>> print(npv(0.05, flows, "2026-01-01", convention="ACT/ACT"))
1000.59 USD
```

**Formatting**

```python
>>> from moneta import MonetaryFormatter
>>> MonetaryFormatter.format(Money("-1234567.89", "USD"), accounting=True)
'($1,234,567.89 USD)'
>>> MonetaryFormatter.format(Money("1234567", "JPY")), MonetaryFormatter.format(Money("1234.5", "CNY"))
('¥1,234,567 JPY', '¥1,234.50 CNY')
```

## Documentation

The documentation is currently written in Chinese; an English version is planned.

- [Home](docs/index.md): architecture overview, installation and a five-minute tour
- [API reference](docs/reference/index.md): design notes, full parameters, examples and benchmarks for every module

| Layer            | Modules                                                                                                                                                                                                                    |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Numeric core     | [`fixed_array`](docs/reference/fixed_array.md), [`_rounding`](docs/reference/_rounding.md), [`_limbs`](docs/reference/_limbs.md), [`_utils`](docs/reference/_utils.md)                                                        |
| Units            | [`unit_array`](docs/reference/unit_array.md), [`_units`](docs/reference/_units.md), [`currency`](docs/reference/currency.md)                                                                                                |
| Business objects | [`money`](docs/reference/money.md), [`price`](docs/reference/price.md), [`quantity`](docs/reference/quantity.md), [`instrument`](docs/reference/instrument.md), [`lot`](docs/reference/lot.md), [`fx`](docs/reference/fx.md), [`formatter`](docs/reference/formatter.md) |
| Applications     | [`portfolio`](docs/reference/portfolio.md), [`conventions`](docs/reference/conventions.md), [`cashflow`](docs/reference/cashflow.md), [`tvm`](docs/reference/tvm.md)                                                        |

## Development

```bash
pip install -e ".[test]"
pytest
python docs/_tools/check_docs.py
```

`check_docs.py` runs every example in the documentation and checks links and example formatting.

## License

[MIT](LICENSE)
