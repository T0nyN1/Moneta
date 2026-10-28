# formatter — 金额千分位、币种符号与会计括号格式化

`moneta.formatter` 模块提供 [`MonetaryFormatter`](#monetaryformatter) 类，用于将 [`Money`](money.md) 标量（[`format`](#monetaryformatterformat)）或多维张量（[`format_array`](#monetaryformatterformat_array)）格式化为带千分位分隔符、前置币种符号、后置币种代码以及可选会计负数括号的展示字符串。[`Money.format`](money.md#moneyformat) 即对该类方法的便捷封装。

```python
>>> from moneta import Currency, MonetaryFormatter, Money
>>> MonetaryFormatter.format(Money("1234567.89", "USD"))
'$1,234,567.89 USD'
>>> MonetaryFormatter.format(Money("-1234.50", "USD"), accounting=True)
'($1,234.50 USD)'
>>> MonetaryFormatter.format(Money("100.00", "CNY")), MonetaryFormatter.format(Money("100", "JPY"))
('¥100.00 CNY', '¥100 JPY')
>>> MonetaryFormatter.format(Money("-1234.50", "CHF"))  # 符号与代码相同时不重复显示
'-1,234.50 CHF'
>>> MonetaryFormatter.format_array(Money([["1000", None], ["-25.5", "-0.00"]], "USD"), accounting=True)
array([['$1,000.00 USD', '<NA>'],
       ['($25.50 USD)', '$0.00 USD']], dtype=object)
```

[← API 参考](index.md)

## 核心机制与设计约束

### 1. 默认同时显示币种符号与代码的消歧设计

在多币种金融报表中，仅靠货币符号极易造成严重歧义：

- `"¥"` 同时被人民币（`CNY`，2 位小数）与日元（`JPY`，0 位小数）使用；
- `"$"` 同时被美元（`USD`）、加元（`CAD`）、澳元（`AUD`）、新加坡元（`SGD`）等多种货币使用。

因此，`MonetaryFormatter` 默认同时开启 `show_symbol=True`（前置符号）与 `show_code=True`（后置空格加 ISO 代码），将人民币与日元清晰区分显示为 `"¥100.00 CNY"` 与 `"¥100 JPY"`。

在此基础上，格式化器还遵循两条精细的符号排版规则：

1. **符号与代码相同时自动去重（`symbol == currency.code`）**：当某币种的 `symbol` 与其 `code` 完全相同时（如瑞士法郎 `CHF` 的 `symbol="CHF"`、`code="CHF"`），若 `show_symbol=True` 且 `show_code=True`，前置符号会被自动省略，只保留后置代码（显示为 `"1,234.50 CHF"` 而非冗余的 `"CHF 1,234.50 CHF"`）；若显式设 `show_code=False`，则正常显示前置符号（`"-CHF 1,234.50"`）。
2. **字母结尾符号自动加空格（`symbol[-1:].isalpha()`）**：当实际显示的前置符号以字母结尾时（如 `"CHF"`、`"Fr"`、科威特第纳尔 `"KD"`），符号与数字之间自动插入一个空格（如 `"Fr 1.00 XFR"`、`"-KD 1,234.005 KWD"`、关闭后置代码时的 `"-CHF 1,234.50"`）；而以非字母字符结尾的符号（如 `"$"`、`"¥"`、`"€"`、`"£"`、`"HK$"`）则直接紧贴数字（如 `"$1.00 USD"`、`"HK$1.00"`）。

| 币种与金额 | 默认（`show_symbol=True, show_code=True`） | 仅符号（`show_code=False`） | 仅代码（`show_symbol=False`） | 纯数字（两者皆 `False`） |
|---|---|---|---|---|
| `Money("1234.5", "USD")` | `"$1,234.50 USD"` | `"$1,234.50"` | `"1,234.50 USD"` | `"1,234.50"` |
| `Money("100", "CNY")` / `Money("100", "JPY")` | `"¥100.00 CNY"` / `"¥100 JPY"` | `"¥100.00"` / `"¥100"` | `"100.00 CNY"` / `"100 JPY"` | `"100.00"` / `"100"` |
| `Money("-1234.5", "CHF")`（`symbol == code`） | `"-1,234.50 CHF"` | `"-CHF 1,234.50"` | `"-1,234.50 CHF"` | `"-1,234.50"` |
| `Money("1", Currency("XFR", 2, symbol="Fr"))` | `"Fr 1.00 XFR"` | `"Fr 1.00"` | `"1.00 XFR"` | `"1.00"` |

### 2. 显式分隔符契约与校验规则（无隐式地区推断）

`MonetaryFormatter` 不依赖操作系统或地区名（`locale`）做隐式格式猜测，而是由调用方通过 `thousands_sep`（默认 `","`）与 `decimal_sep`（默认 `"."`）显式指定千分位与小数点分隔符，并在 [`_check`](#_na_textmonetaryformatter_check-与-monetaryformatter_format_one) 中执行严格校验：

- `thousands_sep` 与 `decimal_sep` 都必须是 `str`（传入 `None` 或非字符串抛 `TypeError`）；
- `thousands_sep` 允许为空字符串 `""`（表示不对整数部分做千分位分组）；
- `decimal_sep` **不能为空字符串 `""`**（否则抛 `ValueError("decimal_sep 不能为空")`）；
- `thousands_sep` 与 `decimal_sep` **不能相同**（否则抛 `ValueError(f"thousands_sep 与 decimal_sep 不能相同: {decimal_sep!r}")`）。

### 3. 负数会计括号、负零消号与 `<NA>` 缺失值规则

- **负数展示**：当底层整数尾数 `mantissa < 0` 时，`accounting=False`（默认）在最外层加负号 `"-{body}"`（如 `"-$1,234.50 USD"`）；`accounting=True` 用圆括号包裹 `"({body})"`（如 `"($1,234.50 USD)"`）。
- **负零不带负号或括号**：由于 `Money("-0.00", "USD")` 的底层定点尾数 `mantissa == 0`（满足 `mantissa >= 0`），无论 `accounting` 为 `False` 还是 `True`，零金额始终格式化为无符号的 `"$0.00 USD"`，绝不出现 `"-$0.00 USD"` 或 `"($0.00 USD)"`。
- **缺失值恒为 `"<NA>"`**：当金额元素为缺失值（`isna() == True`）时，无论符号、代码或会计括号参数如何设置，一律返回字符串 `"<NA>"`。
- **标量与数组严格分流**：[`format`](#monetaryformatterformat) 仅接受 0 维标量 `Money`（`ndim == 0`，传入 1 维或多维数组抛 `ValueError`）；格式化数组须调用 [`format_array`](#monetaryformatterformat_array)。

## API 速查表

| 分类 | 接口 / 常量 | 功能说明 |
|---|---|---|
| **格式化器类** | [`MonetaryFormatter`](#monetaryformatter) | 金额展示格式化器类 |
| | [`MonetaryFormatter.format`](#monetaryformatterformat) | 格式化 0 维标量 `Money`，返回 `str`（被 [`Money.format`](money.md#moneyformat) 直接调用） |
| | [`MonetaryFormatter.format_array`](#monetaryformatterformat_array) | 逐元素格式化任意维度的 `Money` 张量，返回同形状的 `dtype=object` 字符串数组 |
| **内部辅助（私有 API）** | [`_NA_TEXT`、`_check` 与 `_format_one`](#_na_textmonetaryformatter_check-与-monetaryformatter_format_one) | 缺失值占位常量 `"<NA>"`、参数与分隔符安检及单元素格式化核心逻辑 |

## 金额格式化接口

### `MonetaryFormatter`

```python
class moneta.MonetaryFormatter
```

无状态的金额展示格式化工具类（同时由顶层 `moneta` 与 `moneta.formatter` 导出），所有公开功能通过类方法 [`format`](#monetaryformatterformat) 与 [`format_array`](#monetaryformatterformat_array) 提供。

### `MonetaryFormatter.format`

```python
@classmethod
MonetaryFormatter.format(
    money,
    accounting=False,
    show_symbol=True,
    show_code=True,
    thousands_sep=",",
    decimal_sep=".",
)
```

将单个 0 维标量 [`Money`](money.md) 格式化为字符串。

**参数**

- **money** : *Money* — 待格式化的标量金额（必须满足 `money.ndim == 0`；格式化数组请使用 [`format_array`](#monetaryformatterformat_array)）。
- **accounting** : *bool，默认 `False`* — 为 `True` 时负数采用会计括号表示（如 `"($1,234.50 USD)"`）；为 `False` 时负数前置负号 `"-"`（如 `"-$1,234.50 USD"`）。
- **show_symbol** : *bool，默认 `True`* — 是否在数字前显示币种符号（当 `show_code=True` 且 `symbol == code` 时自动省略前置符号以避免重复；字母结尾的符号与数字之间自动加一个空格）。
- **show_code** : *bool，默认 `True`* — 是否在数字后显示空格与三位币种代码（如 `" USD"`）。
- **thousands_sep** : *str，默认 `","`* — 整数部分的千分位分隔符；传入空字符串 `""` 时不分组。
- **decimal_sep** : *str，默认 `"."`* — 小数点分隔符（当币种 `digits == 0` 如 `JPY` 时不显示小数点与小数位）；不能为空字符串，且不能与 `thousands_sep` 相同。

**返回值**

- **out** : *str* — 格式化后的金额字符串（若 `money.isna()` 为 `True`，返回 `"<NA>"`）。

**异常**

- **TypeError** — `money` 不是 [`Money`](money.md) 实例，或 `thousands_sep` / `decimal_sep` 不是 `str`。
- **ValueError** — `money.ndim != 0`（非标量）；`decimal_sep` 为空字符串 `""`；或 `thousands_sep == decimal_sep`。

**示例**

千分位与小数点定制、符号与代码规则、会计括号、负零及缺失值：

```python
>>> from moneta import Currency, MonetaryFormatter, Money
>>> m = Money("-1234567.89", "USD")
>>> MonetaryFormatter.format(m)
'-$1,234,567.89 USD'
>>> MonetaryFormatter.format(m, accounting=True)
'($1,234,567.89 USD)'
>>> MonetaryFormatter.format(m, thousands_sep=".", decimal_sep=",")  # 欧洲数字分隔风格
'-$1.234.567,89 USD'
>>> MonetaryFormatter.format(m, thousands_sep="", show_symbol=False, show_code=False)
'-1234567.89'
>>> MonetaryFormatter.format(Money("-0.00", "USD"), accounting=True)  # 负零不带负号或括号
'$0.00 USD'
>>> MonetaryFormatter.format(Money(["1.00", None], "USD")[1], accounting=True)  # 缺失值显示 <NA>
'<NA>'
```

符号等于代码（`CHF`）与字母结尾符号（`Fr`、`KD`）的自动排版：

```python
>>> chf = Money("-1234.50", "CHF")
>>> MonetaryFormatter.format(chf), MonetaryFormatter.format(chf, show_code=False)
('-1,234.50 CHF', '-CHF 1,234.50')
>>> kwd = Currency("KWD", 3, symbol="KD")
>>> MonetaryFormatter.format(Money("-1234.005", kwd))
'-KD 1,234.005 KWD'
```

非标量输入与非法分隔符拦截：

```python
>>> MonetaryFormatter.format(Money(["1.00", "2.00"], "USD"))
Traceback (most recent call last):
    ...
ValueError: format 只支持标量 Money (ndim == 0)；数组请用 format_array
>>> MonetaryFormatter.format(Money("1.00", "USD"), decimal_sep="")
Traceback (most recent call last):
    ...
ValueError: decimal_sep 不能为空
>>> MonetaryFormatter.format(Money("1.00", "USD"), thousands_sep=".", decimal_sep=".")
Traceback (most recent call last):
    ...
ValueError: thousands_sep 与 decimal_sep 不能相同: '.'
```

**另请参阅**

- [`MonetaryFormatter.format_array`](#monetaryformatterformat_array)
- [`Money.format`](money.md#moneyformat)

### `MonetaryFormatter.format_array`

```python
@classmethod
MonetaryFormatter.format_array(
    money,
    accounting=False,
    show_symbol=True,
    show_code=True,
    thousands_sep=",",
    decimal_sep=".",
)
```

逐元素格式化任意维度的 [`Money`](money.md) 张量（含 0 维标量、1 维向量、多维矩阵及空数组），返回与输入 `money` 形状完全相同的 `dtype=object` NumPy 数组（每个元素为 `str`，结果与对每个位置调用 [`format`](#monetaryformatterformat) 完全一致）。

**参数**

- **money** : *Money* — 任意维度的 `Money` 实例。
- **accounting, show_symbol, show_code, thousands_sep, decimal_sep** — 与 [`MonetaryFormatter.format`](#monetaryformatterformat) 完全相同。

**返回值**

- **out** : *np.ndarray* — `shape == money.shape`、`dtype == object` 的字符串数组。

**异常**

- **TypeError** — `money` 不是 `Money` 实例，或 `thousands_sep` / `decimal_sep` 不是 `str`。
- **ValueError** — `decimal_sep` 为空字符串 `""`，或 `thousands_sep == decimal_sep`。

**示例**

```python
>>> from moneta import MonetaryFormatter, Money
>>> m = Money([["1.00", None], ["-2.50", "1000.00"]], "USD")
>>> out = MonetaryFormatter.format_array(m, accounting=True)
>>> out.shape, out.dtype
((2, 2), dtype('O'))
>>> out.tolist()
[['$1.00 USD', '<NA>'], ['($2.50 USD)', '$1,000.00 USD']]
```

**性能对比**

| 格式化操作 | 耗时 | 说明 |
|---|---:|---|
| `MonetaryFormatter.format(m)`（0-d 标量） | 1.76 µs | 含参数校验、整数小数拆分与千分位组装 |
| `MonetaryFormatter.format_array(m_1000)`（1,000 元素） | 883 µs | 批量生成 1,000 个格式化字符串对象 |

**另请参阅**

- [`MonetaryFormatter.format`](#monetaryformatterformat)

## 内部辅助方法与常量（私有 API）

### `_NA_TEXT`、`MonetaryFormatter._check` 与 `MonetaryFormatter._format_one`

```python
moneta.formatter._NA_TEXT = "<NA>"
MonetaryFormatter._check(money, thousands_sep, decimal_sep)
MonetaryFormatter._format_one(
    money,
    mantissa,
    na,
    accounting,
    show_symbol,
    show_code,
    thousands_sep,
    decimal_sep,
)
```

- **`_NA_TEXT`**：缺失值格式化输出常量 `"<NA>"`。
- **`_check(money, thousands_sep, decimal_sep)`**：校验 `isinstance(money, Money)`、`thousands_sep` 与 `decimal_sep` 均为 `str`、`bool(decimal_sep)` 非空以及 `thousands_sep != decimal_sep`。
- **`_format_one(...)`**：单个元素的底层格式化实现：当 `na=True` 时直接返回 `"<NA>"`；否则通过 `divmod(abs(mantissa), 10 ** digits)` 无精度损失地拆分整数部分与小数部分（即使 `mantissa == -2**63` 也能在 Python `int` 中精确取绝对值），应用千分位、小数位补零、符号去重与字母空格规则，最后根据 `mantissa >= 0` 与 `accounting` 添加负号或括号。

**示例**

```python
>>> from moneta import MonetaryFormatter, Money
>>> m = Money("-1234.50", "CHF")
>>> MonetaryFormatter._format_one(m, int(m.mantissa), False, True, True, True, ",", ".")
'(1,234.50 CHF)'
```
