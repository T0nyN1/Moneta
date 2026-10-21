# Currency — 币种定义与享元注册表

`moneta.currency` 模块定义法定货币与金融资产的量纲、最小原子结算精度（`digits`）以及全局享元单例注册表（`CurrencyRegistry`）。
它是上层 [`Money`](money.md)、[`Price`](price.md)、`ExchangeRate` 与 `Portfolio`
强制保证量纲安全与法定精度对齐的基础：同一币种代码在全局注册表中保证只有唯一事实定义，不允许被篡改或静默覆盖。

```python
>>> from moneta import Currency, CurrencyRegistry, get_currency, USD, CNY, JPY
>>> USD
Currency(code='USD', digits=2, symbol='$', name='US Dollar')
>>> get_currency("usd") is USD
True
>>> JPY.digits
0
>>> usdt = Currency("USDT", 6, "₮", "Tether")
>>> CurrencyRegistry.register(usdt)
>>> get_currency("USDT") is usdt
True
>>> CurrencyRegistry.clear()
```

[← API 参考](index.md)

## 核心机制与设计约束

### 不可变值对象与原子精度

每个 [`Currency`](#currencycode-digits-symbol-name) 实例包含四个只读槽位（
`__slots__ = ("_code", "_digits", "_symbol", "_name")`）：

- **`code`**：货币或资产代码。构造时自动去除首尾空白并转为大写，必须由 **3 到 5 个英文字母**组成（正则 [
  `_CODE_RE = r"^[A-Z]{3,5}$"`](#_code_re-与-_max_digits)，既涵盖 ISO 4217 的三字母法币代码如 `"USD"`、`"CNY"`，也支持
  `"USDT"`、`"USDC"` 等 4–5 字母的数字资产或自定义代码）。
- **`digits`**：最小不可分割结算单位对应的小数位数（如美元/人民币的“分”对应 `2`，日元对应 `0`，巴林第纳尔对应 `3`）。必须是 **
  `0 <= digits <= 18` 的整数**（[`_MAX_DIGITS = 18`](#_code_re-与-_max_digits)：底层 [`FixedArray`](fixed_array.md) 采用有符号
  `int64` 尾数，在 `scale=18` 下仍可精确表示约 $\pm 9.22$），显式拒绝 `bool` 与 `float`。
- **`symbol` 与 `name`**：展示用的货币符号（默认 `""`）与全称（默认等于标准化后的 `code`）。

`Currency` 在构造后完全 **不可变**：重写了 `__setattr__` 与 `__delattr__`，任何修改或删除属性的操作都会直接抛出
`AttributeError`。

### 享元注册表与唯一事实来源

[`CurrencyRegistry`](#currencyregistry) 维护全局唯一的货币字典 `_registry`，遵循**唯一事实来源（Single Source of Truth）**
原则：

1. **幂等注册与冲突拒绝**：调用 [`CurrencyRegistry.register(ccy)`](#currencyregistryregister) 时，若该 `code`
   尚未注册则存入；若已注册且四元组 `(code, digits, symbol, name)` 完全相同则无副作用直接返回；只要有任何一个字段不同即抛出
   `ValueError`。注册表 **不提供强制覆盖开关**，内置的 10 种常用货币永远不可被篡改。
2. **输入归一化与冲突拦截**：调用 [`get_currency(cur)`](#get_currency) 或 [
   `CurrencyRegistry.get(cur)`](#currencyregistryget)（也是 [`Money`](money.md) 与 [`Price`](price.md) 构造时的统一币种解析入口）时：
   - 传入字符串（如 `"usd"`、`" EUR "`）：去首尾空白并转大写后查找注册表，返回对应的单例；未注册则抛出 `KeyError`。
   - 传入 `Currency` 实例：若其 `code` 已在注册表中且四元组定义一致，则自动归并并返回 **注册表中的单例**
     ；若与注册表中的同代码定义有任何字段冲突（例如传入私自构造的 `Currency("USD", 3)`），立即抛出 `ValueError`
     ；若该代码尚未注册，则原样返回该自定义实例（允许临时使用未注册的自定义币种构造 `Money` 或 `Price`）。

### 相等性、拷贝与序列化语义

| 操作                               | 行为与语义                                                                                                                                                                                                                                                                         |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `c1 == c2`、`c1 != c2`             | 仅比较量纲核心字段 `(code, digits)`；展示字段 `symbol` 与 `name` 不影响相等性。因此同代码但不同 `digits` 的未注册币种（如 `Currency("XYZ", 2)` 与 `Currency("XYZ", 3)`）绝不相等，对应的 `Money` 也不能相加减。与非 `Currency` 对象（如字符串 `"USD"`）比较返回 `False` / `True`。 |
| `hash(c)`                          | 基于 `hash(c.code)` 计算，可安全作为 `dict` 键或放入 `set`。                                                                                                                                                                                                                       |
| `copy.copy(c)`、`copy.deepcopy(c)` | 直接返回 `c` 自身，不创建新对象。                                                                                                                                                                                                                                                  |
| `pickle.loads(pickle.dumps(c))`    | 通过 [`_restore_currency`](#_restore_currency) 还原：若 `code` 已注册且定义一致，直接返回注册表中的同一单例（`is` 恒等）；若与注册表定义冲突则抛 `ValueError`；若未注册则还原为独立实例且不会自动写入注册表。                                                                      |

## API 速查表

| 分类                     | 接口 / 常量                                                                                    | 功能说明                                                     |
|--------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| **货币定义与属性**       | [`Currency(code, digits, symbol="", name="")`](#currencycode-digits-symbol-name)               | 构造不可变的货币/资产量纲定义                                |
|                          | [`Currency.code`](#currencycode)                                                               | 标准化后的大写货币代码（3–5 个英文字母）                     |
|                          | [`Currency.digits`](#currencydigits)                                                           | 最小原子结算单位对应的小数位数（`0`–`18`）                   |
|                          | [`Currency.symbol`](#currencysymbol)                                                           | 货币显示符号（如 `"$"`、`"¥"`）                              |
|                          | [`Currency.name`](#currencyname)                                                               | 货币全称（如 `"US Dollar"`）                                 |
| **比较、序列化与展示**   | [`c1 == c2`、`c1 != c2`、`hash(c)`](#c1--c2c1--c2-与-hashc相等与哈希)                          | 基于 `(code, digits)` 的相等判断与哈希支持                   |
|                          | [`copy`、`deepcopy`、`pickle`](#copydeepcopy-与-pickle单例保持与序列化)                        | 浅/深拷贝返回自身，反序列化自动归并至注册表单例              |
|                          | [`str(c)`、`repr(c)`](#strc-与-reprc文本表示)                                                  | 代码文本输出与完整构造表达式展示                             |
| **注册表与查找**         | [`CurrencyRegistry`](#currencyregistry)                                                        | 全局货币享元注册表类                                         |
|                          | [`CurrencyRegistry.register`](#currencyregistryregister)                                       | 将 `Currency` 实例注册到全局注册表（冲突报错）               |
|                          | [`CurrencyRegistry.get`](#currencyregistryget)                                                 | 按代码或实例获取货币单例（校验冲突）                         |
|                          | [`CurrencyRegistry.has`](#currencyregistryhas)                                                 | 判断货币代码或实例是否已在注册表中                           |
|                          | [`CurrencyRegistry.clear`](#currencyregistryclear)                                             | 清空自定义注册项并恢复 10 个内置默认币种                     |
|                          | [`get_currency`](#get_currency)                                                                | 通用类型提升函数：将 `str` 或 `Currency` 解析为 `Currency`   |
| **内置币种常量**         | [`USD` `CNY` `EUR` `JPY` `GBP` `HKD` `AUD` `CAD` `CHF` `SGD`](#usdcnyeurjpygbphkdaudcadchfsgd) | 预注册的 10 种 ISO 4217 常用法定货币单例                     |
| **内部辅助（私有 API）** | [`_CODE_RE` 与 `_MAX_DIGITS`](#_code_re-与-_max_digits)                                        | 货币代码校验正则（`^[A-Z]{3,5}$`）与最大精度上限常量（`18`） |
|                          | [`Currency._definition`](#currency_definition)                                                 | 返回完整定义四元组 `(code, digits, symbol, name)`            |
|                          | [`_restore_currency`](#_restore_currency)                                                      | `pickle` 反序列化还原与注册表单例归并入口                    |
|                          | [`_init_defaults`](#_init_defaults)                                                            | 初始化并注册 10 个内置法定货币常量                           |

## 货币定义与核心属性

### `Currency(code, digits, symbol="", name="")`

```python
class moneta.Currency(code, digits, symbol="", name="")
```

定义法定货币或金融资产的量纲与最小原子结算精度。

**参数**

- **code** : *str* — 货币代码（如 `"CNY"`、`"USD"`、`"JPY"`、`"USDT"`）。首尾空白会被自动去除并统一转为大写，去除空白后必须为 3
  到 5 个 ASCII 英文字母（正则 `^[A-Z]{3,5}$`）。
- **digits** : *int* — 最小原子结算单位对应的小数位数，满足 `0 <= digits <= 18`。支持 Python `int` 与 NumPy 整数（如
  `np.int64`、`np.uint8`），拒绝 `bool` 与 `float`。
- **symbol** : *str，默认 `""`* — 货币符号（如 `"¥"`、`"$"`、`"€"`）。
- **name** : *str，默认 `""`* — 货币全称（如 `"Chinese Yuan"`）。若传入空字符串 `""`，则默认设为标准化后的 `code`。

**返回值**

- **out** : *Currency* — 不可变的 `Currency` 实例。注意直接调用 `Currency(...)` **不会**自动将其写入全局 [
  `CurrencyRegistry`](#currencyregistry)；如需全局通过代码字符串查找，请显式调用 [
  `CurrencyRegistry.register`](#currencyregistryregister)。

**异常**

- **TypeError** — `code`、`symbol` 或 `name` 不是 `str`；`digits` 是 `bool`、`float` 或其他非整数类型。
- **ValueError** — `code` 去除空白后不是 3 到 5 个英文字母；`digits` 超出 `[0, 18]` 范围。
- **AttributeError** — 构造后尝试对任何属性赋值（`setattr`）或删除属性（`delattr`）。

**机制说明**

构造器内部依次执行以下校验与初始化：

1. 检查 `isinstance(code, str)`，执行 `normalized = code.strip().upper()` 并用预编译正则 [
   `_CODE_RE`](#_code_re-与-_max_digits) 校验格式。
2. 先显式拦截 `isinstance(digits, (bool, np.bool_))`（因为 `bool` 是 `int` 子类，且旧版 NumPy 的 `np.bool_` 可通过
   `operator.index`），再通过 `operator.index(digits)` 将任意整数类型无损转为 Python `int`（同时拒绝 `float`），最后校验
   `0 <= digits <= _MAX_DIGITS`。
3. 校验 `symbol` 与 `name` 为 `str`，若 `name` 为空则回退为 `normalized`。
4. 由于 `Currency.__setattr__` 与 `Currency.__delattr__` 均已被重写为直接抛出 `AttributeError`，构造器内部通过
   `object.__setattr__(self, ...)` 完成底层的只读绑定。

**示例**

构造自定义币种与自动标准化：

```python
>> > from moneta import Currency, USD
>> > bhd = Currency("bhd", 3, "BD", "Bahraini Dinar")
>> > bhd
Currency(code='BHD', digits=3, symbol='BD', name='Bahraini Dinar')
>> > usdt = Currency(" usdt ", np.int64(6))  # 自动去空白、转大写，name 默认等于 code
>> > (usdt.code, usdt.digits, usdt.symbol, usdt.name)
('USDT', 6, '', 'USDT')
```

不可变性保护（禁止修改或删除属性）与非法参数拦截：

```python
>> > USD.digits = 4
Traceback(most
recent
call
last):
...
AttributeError: Currency
是不可变对象，禁止设置属性
'digits'
>> > del USD._digits
Traceback(most
recent
call
last):
...
AttributeError: Currency
是不可变对象，禁止删除属性
'_digits'
>> > Currency("US", 2)
Traceback(most
recent
call
last):
...
ValueError: 货币代码必须是
3
到
5
个英文字母，收到
'US'
>> > Currency("USD", 2.0)
Traceback(most
recent
call
last):
...
TypeError: digits
必须是整数，收到
float
>> > Currency("USD", True)
Traceback(most
recent
call
last):
...
TypeError: digits
不能是
bool
>> > Currency("ETH", 19)
Traceback(most
recent
call
last):
...
ValueError: digits
必须在
0
到
18
之间，收到
19
```

**另请参阅**

- [`CurrencyRegistry.register`](#currencyregistryregister) — 将构造好的币种注册为全局单例。
- [`get_currency`](#get_currency) — 根据代码或实例获取币种。

### Currency.code

```python
Currency.code
```

标准化后的大写货币代码字符串。

**机制说明**

读取底层只读槽位 `self._code`，恒为 3 到 5 个大写英文字母组成的 `str`。

**返回值**

- **code** : *str*

**示例**

```python
>> > from moneta import USD, CNY, Currency
>> > USD.code, CNY.code
('USD', 'CNY')
>> > Currency(" usdc ", 6).code
'USDC'
```

**另请参阅**

- [`Currency.digits`](#currencydigits)
- [`str(c)`](#strc-与-reprc文本表示)

### Currency.digits

```python
Currency.digits
```

最小原子结算单位对应的小数位数。

**机制说明**

读取底层只读槽位 `self._digits`，恒为 `[0, 18]` 范围内的 Python `int`。当使用该币种构造上层 [`Money`](money.md) 对象时，
`Money` 底层 [`FixedArray`](fixed_array.md) 的 `scale` 恒绑定为此 `digits`。

**返回值**

- **digits** : *int*

**示例**

```python
>>> from moneta import USD, JPY, Currency
>>> USD.digits, JPY.digits
(2, 0)
>>> Currency("BHD", 3).digits
3
```

**另请参阅**

- [`Currency.code`](#currencycode)
- [`Money.scale`](money.md#继承自-unitarray-的数组属性)
- [`FixedArray.scale`](fixed_array.md#fixedarrayscale)

### Currency.symbol

```python
Currency.symbol
```

货币显示符号。

**机制说明**

读取底层只读槽位 `self._symbol`。不参与相等性比较（[`==`](#c1--c2c1--c2-与-hashc相等与哈希)），但参与注册表冲突检测（[
`_definition`](#currency_definition)）以及上层 `MonetaryFormatter` 的格式化输出。

**返回值**

- **symbol** : *str*

**示例**

```python
>>> from moneta import USD, CNY, EUR, HKD
>>> (USD.symbol, CNY.symbol, EUR.symbol, HKD.symbol)
('$', '¥', '€', 'HK$')
```

**另请参阅**

- [`Currency.name`](#currencyname)
- [`Currency._definition`](#currency_definition)

### Currency.name

```python
Currency.name
```

货币英文全称或描述名称。

**机制说明**

读取底层只读槽位 `self._name`。构造时若未传入非空 `name`，则自动回退为大写 `code`。与 `symbol` 一样，不参与 `==`
量纲比较，但参与注册表冲突检测。

**返回值**

- **name** : *str*

**示例**

```python
>>> from moneta import USD, JPY, Currency
>>> USD.name, JPY.name
('US Dollar', 'Japanese Yen')
>>> Currency("USDT", 6).name
'USDT'
```

**另请参阅**

- [`Currency.symbol`](#currencysymbol)
- [`Currency._definition`](#currency_definition)

## 相等、哈希、序列化与文本协议

### `c1 == c2`、`c1 != c2` 与 `hash(c)`（相等与哈希）

```python
c1 == c2
c1 != c2
hash(c)
```

按货币代码与原子精度 `(code, digits)` 判断两个币种是否相等，并支持作为字典键或集合元素。

**参数**

- **other** : *Any* — 另一个比较对象。

**返回值**

- **out** : *bool 或 int* — `==` / `!=` 返回 `bool`；`hash(c)` 返回 `int`。

**机制说明**

- `__eq__` 首先检查 `isinstance(other, Currency)`，若不是则返回 `NotImplemented`（因此 `USD == "USD"` 为 `False`
  ，不会抛异常，也不会与裸字符串混同）。
- 若两者均为 `Currency`，当且仅当 `self._code == other._code and self._digits == other._digits` 时返回 `True`；展示属性
  `symbol` 与 `name` 的差异不影响量纲相等性。
- `__hash__` 返回 `hash(self._code)`，保证相等的 `Currency` 实例始终具有相同的哈希值。

**示例**

```python
>> > from moneta import Currency, Money, USD, CNY, get_currency
>>> Currency("ABC", 2, "$", "Alpha") == Currency("abc", 2)  # symbol 与 name 不影响相等
True
>>> Currency("ABC", 2) == Currency("ABC", 3)  # digits 不同则不等
False
>> > Money(1, Currency("XYZ", 2)) + Money(1, Currency("XYZ", 3))  # 同代码但不同 digits 的未注册币种禁止混算
Traceback(most
recent
call
last):
...
TypeError: 币种不匹配: XYZ
与
XYZ
>>> USD == "USD"  # 与字符串比较为 False
False
>>> len({USD, get_currency("usd"), Currency("USD", 2, "$", "US Dollar"), CNY})
2
```

**另请参阅**

- [`Currency._definition`](#currency_definition) — 包含 `symbol` 与 `name` 的完整四元组比对。

### copy、deepcopy 与 pickle（单例保持与序列化）

```python
copy.copy(c)
copy.deepcopy(c)
pickle.dumps(c) / pickle.loads(data)
```

拷贝操作直接返回自身；`pickle` 反序列化时自动与全局注册表单例对齐。

**返回值**

- **out** : *Currency*

**异常**

- **ValueError** — `pickle.loads` 还原出的币种代码已在当前 [`CurrencyRegistry`](#currencyregistry) 中注册，但四元组定义
  `(code, digits, symbol, name)` 与注册表不一致。

**机制说明**

- `__copy__` 与 `__deepcopy__` 均直接返回 `self`，保证包含 `Currency` 的复合数据结构在深拷贝后依然指向同一享元实例。
- `__reduce__` 返回 `(_restore_currency, self._definition())`：反序列化时调用 [`_restore_currency`](#_restore_currency)
  ，若该代码已在注册表中且定义一致，则直接返回注册表单例（保证 `pickle.loads(pickle.dumps(USD)) is USD`）；若未注册则还原为独立
  `Currency` 实例且不自动写入注册表。

**示例**

```python
>> > import copy, pickle
>> > from moneta import Currency, CurrencyRegistry, USD
>> > copy.copy(USD) is USD and copy.deepcopy(USD) is USD
True
>> > pickle.loads(pickle.dumps(USD)) is USD
True
>> > unregistered = Currency("XYZ", 4, "X", "Custom Xyz")
>> > restored = pickle.loads(pickle.dumps(unregistered))
>> > restored == unregistered and restored is not unregistered
True
>> > CurrencyRegistry.has("XYZ")  # 反序列化未注册币种不会污染注册表
False
```

**另请参阅**

- [`_restore_currency`](#_restore_currency)
- [`CurrencyRegistry`](#currencyregistry)

### `str(c)` 与 `repr(c)`（文本表示）

```python
str(c)
repr(c)
```

`str(c)` 返回三到五字母的大写货币代码；`repr(c)` 返回带全部字段的可求值构造形式。

**返回值**

- **out** : *str*

**机制说明**

`__str__` 直接返回 `self._code`，便于嵌入日志、表头或格式化字符串；`__repr__` 输出
`Currency(code=..., digits=..., symbol=..., name=...)` 完整信息。

**示例**

```python
>>> from moneta import USD
>>> str(USD)
'USD'
>>> repr(USD)
"Currency(code='USD', digits=2, symbol='$', name='US Dollar')"
>>> f"结算币种：{USD}"
'结算币种：USD'
```

**另请参阅**

- [`Currency.code`](#currencycode)

## 币种注册表与查找（CurrencyRegistry & get_currency）

### CurrencyRegistry

```python
class moneta.CurrencyRegistry
```

全局货币注册表与享元工厂，维护从大写货币代码到唯一 `Currency` 实例的映射字典 `_registry`。

**机制说明**

`CurrencyRegistry` 的所有公开接口均为类方法（`@classmethod`），无需实例化即可直接调用。模块加载时会自动调用 [
`_init_defaults()`](#_init_defaults) 将 10 种内置常用货币注册进 `CurrencyRegistry._registry`。

**另请参阅**

- [`CurrencyRegistry.register`](#currencyregistryregister)
- [`CurrencyRegistry.get`](#currencyregistryget)
- [`CurrencyRegistry.has`](#currencyregistryhas)
- [`CurrencyRegistry.clear`](#currencyregistryclear)
- [`get_currency`](#get_currency)

### CurrencyRegistry.register

```python
CurrencyRegistry.register(currency)
```

将一个 `Currency` 实例注册到全局注册表中。

**参数**

- **currency** : *Currency* — 待注册的货币定义实例。

**返回值**

- **None**

**异常**

- **TypeError** — `currency` 不是 `Currency` 实例（如传入了字符串）。
- **ValueError** — 相同 `code` 已在注册表中存在，且已有定义的四元组 `(code, digits, symbol, name)` 与 `currency` 不一致。

**机制说明**

先检查 `isinstance(currency, Currency)`，随后在 `cls._registry` 中查找 `currency.code`：

- 若不存在，写入 `cls._registry[currency.code] = currency`；
- 若已存在，比较 `existing._definition() != currency._definition()`：完全一致时静默通过（幂等），任何一项不同（包括 `symbol` 或
  `name` 不同）均抛出 `ValueError`，从而确保内置币种与已注册自定义币种绝不被意外覆盖。

**示例**

```python
>> > from moneta import Currency, CurrencyRegistry, get_currency, USD
>> > bhd = Currency("BHD", 3, "BD", "Bahraini Dinar")
>> > CurrencyRegistry.register(bhd)
>> > CurrencyRegistry.register(Currency("bhd", 3, "BD", "Bahraini Dinar"))  # 相同定义重复注册幂等
>> > get_currency("BHD") is bhd
True
>> > CurrencyRegistry.register(Currency("BHD", 2, "BD", "Bahraini Dinar"))  # 精度冲突报错
Traceback(most
recent
call
last):
...
ValueError: 货币代码
BHD
已注册为
Currency(code='BHD', digits=3, symbol='BD', name='Bahraini Dinar')，不能改注册为
Currency(code='BHD', digits=2, symbol='BD', name='Bahraini Dinar')
>> > CurrencyRegistry.register(Currency("USD", 4, "$", "US Dollar"))  # 内置币种不可覆盖
Traceback(most
recent
call
last):
...
ValueError: 货币代码
USD
已注册为
Currency(code='USD', digits=2, symbol='$', name='US Dollar')，不能改注册为
Currency(code='USD', digits=4, symbol='$', name='US Dollar')
>> > CurrencyRegistry.clear()
```

**另请参阅**

- [`CurrencyRegistry.get`](#currencyregistryget)
- [`CurrencyRegistry.has`](#currencyregistryhas)
- [`Currency._definition`](#currency_definition)

### CurrencyRegistry.get

```python
CurrencyRegistry.get(code)
```

根据货币代码字符串或 `Currency` 实例获取货币单例。

**参数**

- **code** : *str 或 Currency* — 货币代码（忽略大小写与首尾空白）或 `Currency` 实例。

**返回值**

- **out** : *Currency* — 注册表中的 `Currency` 单例（或未注册的自定义 `Currency` 实例自身）。

**异常**

- **TypeError** — `code` 既不是 `str` 也不是 `Currency`。
- **KeyError** — 传入字符串代码，但该代码未在注册表中注册。
- **ValueError** — 传入 `Currency` 实例，但其 `code` 在注册表中已有不同定义（`digits`、`symbol` 或 `name` 不一致）。

**机制说明**

- 当 `code` 是 `Currency` 实例时：先在 `_registry` 中查找 `code.code`。若未注册则原样放行返回 `code`；若已注册则核对
  `registered._definition() == code._definition()`，一致时返回注册表单例 `registered`，不一致时抛出 `ValueError`（拦截如
  `Currency("USD", 3)` 之类的伪造实例）。
- 当 `code` 是 `str` 时：执行 `code.strip().upper()` 后在 `_registry` 中查找，未找到则抛出 `KeyError`。

**示例**

```python
>> > from moneta import Currency, CurrencyRegistry, USD
>> > CurrencyRegistry.get(" usd ") is USD
True
>> > CurrencyRegistry.get(Currency("USD", 2, "$", "US Dollar")) is USD  # 等价实例归并为单例
True
>> > custom = Currency("XYZ", 4)
>> > CurrencyRegistry.get(custom) is custom  # 未注册实例原样返回
True
>> > CurrencyRegistry.get("XYZ")  # 未注册字符串代码报错
Traceback(most
recent
call
last):
...
KeyError: "未注册的货币代码: 'XYZ'"
>> > CurrencyRegistry.get(Currency("USD", 3, "$", "US Dollar"))  # 与已注册定义冲突报错
Traceback(most
recent
call
last):
...
ValueError: Currency(code='USD', digits=3, symbol='$', name='US Dollar')
与注册表中的定义
Currency(code='USD', digits=2, symbol='$', name='US Dollar')
不一致
```

**另请参阅**

- [`get_currency`](#get_currency) — 顶层便捷别名函数。
- [`CurrencyRegistry.has`](#currencyregistryhas)

### CurrencyRegistry.has

```python
CurrencyRegistry.has(code)
```

检查货币代码（或与注册表定义一致的 `Currency` 实例）是否已注册。

**参数**

- **code** : *str、Currency 或 Any* — 待检查的代码或对象。

**返回值**

- **out** : *bool* — 已注册且定义一致返回 `True`，否则返回 `False`（传入非 `str` / `Currency` 类型也安全返回 `False`
  ，不抛异常）。

**机制说明**

- 若传入 `Currency` 实例：不仅检查 `code.code` 是否在 `_registry` 中，还会校验
  `registered._definition() == code._definition()`；因此与注册表定义冲突的实例（如 `Currency("USD", 3)`）会返回 `False`。
- 若传入 `str`：去除首尾空白并转大写后检查是否在 `_registry` 中。
- 其他任何类型直接返回 `False`。

**示例**

```python
>> > from moneta import Currency, CurrencyRegistry, USD
>> > CurrencyRegistry.has("cny"), CurrencyRegistry.has(USD)
(True, True)
>> > CurrencyRegistry.has("XYZ"), CurrencyRegistry.has(Currency("USD", 3)), CurrencyRegistry.has(None)
(False, False, False)
```

**另请参阅**

- [`CurrencyRegistry.get`](#currencyregistryget)
- [`CurrencyRegistry.register`](#currencyregistryregister)

### CurrencyRegistry.clear

```python
CurrencyRegistry.clear()
```

清空所有自定义注册币种，并将注册表恢复为初始的 10 个内置常用货币。

**返回值**

- **None**

**机制说明**

先调用 `cls._registry.clear()` 清空字典，再立即调用 [`_init_defaults()`](#_init_defaults) 重新注册 `USD`、`CNY`、`EUR`、
`JPY`、`GBP`、`HKD`、`AUD`、`CAD`、`CHF`、`SGD`。常用于单元测试环境隔离。

**示例**

```python
>> > from moneta import Currency, CurrencyRegistry, USD
>> > CurrencyRegistry.register(Currency("XYZ", 4))
>> > CurrencyRegistry.has("XYZ")
True
>> > CurrencyRegistry.clear()
>> > CurrencyRegistry.has("XYZ"), CurrencyRegistry.get("USD") is USD
(False, True)
```

**另请参阅**

- [`_init_defaults`](#_init_defaults)
- [`CurrencyRegistry.register`](#currencyregistryregister)

### get_currency

```python
moneta.get_currency(cur)
```

通用币种解析与类型提升函数：将货币代码字符串或 `Currency` 实例统一转换为 `Currency` 对象。

**参数**

- **cur** : *str 或 Currency* — 货币代码字符串或 `Currency` 实例。

**返回值**

- **out** : *Currency* — 对应的 `Currency` 单例或自定义实例。

**异常**

- **TypeError** / **KeyError** / **ValueError** — 与 [`CurrencyRegistry.get`](#currencyregistryget) 完全相同。

**机制说明**

直接转发调用 `CurrencyRegistry.get(cur)`。上层 [`Money`](money.md)、[`Price`](price.md)、`ExchangeRate`、`Portfolio`
在接收用户传入的 `currency="USD"` 或 `currency=USD` 参数时，统一通过此函数收口。

**示例**

```python
>> > from moneta import get_currency, Money, CNY, EUR, USD, Currency
>> > get_currency("cny") is CNY and get_currency(EUR) is EUR
True
>> > Money("10.00", "usd").currency is USD
True
>> > Money(1, Currency("USD", 3))  # 拒绝与注册表冲突的伪造 USD 实例
Traceback(most
recent
call
last):
...
ValueError: Currency(code='USD', digits=3, symbol='', name='USD')
与注册表中的定义
Currency(code='USD', digits=2, symbol='$', name='US Dollar')
不一致
```

**另请参阅**

- [`CurrencyRegistry.get`](#currencyregistryget)
- [`Money(values, currency)`](money.md#moneyvalues-currency)
- [`Price(values, quote_currency, ...)`](price.md#pricevalues-quote_currency-base_unitshare-tick_sizenone-scalenone)

## 内置常用币种常量

### USD、CNY、EUR、JPY、GBP、HKD、AUD、CAD、CHF、SGD

模块预定义了 10 个遵循 ISO 4217 法定结算精度的常用货币单例，可直接从顶层包 `moneta` 或 `moneta.currency` 导入：

| 常量名称 | `code`  | `digits` | `symbol` | `name`                | 最小原子单位说明                 |
|----------|---------|---------:|----------|-----------------------|----------------------------------|
| `USD`    | `"USD"` |      `2` | `"$"`    | `"US Dollar"`         | 1 美分（Cent）$= 10^{-2}$ 美元   |
| `CNY`    | `"CNY"` |      `2` | `"¥"`    | `"Chinese Yuan"`      | 1 分 $= 10^{-2}$ 元              |
| `EUR`    | `"EUR"` |      `2` | `"€"`    | `"Euro"`              | 1 欧分（Cent）$= 10^{-2}$ 欧元   |
| `JPY`    | `"JPY"` |      `0` | `"¥"`    | `"Japanese Yen"`      | 1 日元（法定结算无小数位）       |
| `GBP`    | `"GBP"` |      `2` | `"£"`    | `"British Pound"`     | 1 便士（Penny）$= 10^{-2}$ 英镑  |
| `HKD`    | `"HKD"` |      `2` | `"HK$"`  | `"Hong Kong Dollar"`  | 1 仙 $= 10^{-2}$ 港元            |
| `AUD`    | `"AUD"` |      `2` | `"A$"`   | `"Australian Dollar"` | 1 澳分 $= 10^{-2}$ 澳元          |
| `CAD`    | `"CAD"` |      `2` | `"C$"`   | `"Canadian Dollar"`   | 1 加分 $= 10^{-2}$ 加元          |
| `CHF`    | `"CHF"` |      `2` | `"CHF"`  | `"Swiss Franc"`       | 1 生丁（Rappen）$= 10^{-2}$ 瑞郎 |
| `SGD`    | `"SGD"` |      `2` | `"S$"`   | `"Singapore Dollar"`  | 1 新分 $= 10^{-2}$ 新元          |

**示例**

```python
>>> from moneta import USD, CNY, EUR, JPY, GBP, HKD, AUD, CAD, CHF, SGD, get_currency
>>> [(c.code, c.digits, c.symbol) for c in (USD, CNY, EUR, JPY, GBP)]
[('USD', 2, '$'), ('CNY', 2, '¥'), ('EUR', 2, '€'), ('JPY', 0, '¥'), ('GBP', 2, '£')]
>>> all(get_currency(c.code) is c for c in (USD, CNY, EUR, JPY, GBP, HKD, AUD, CAD, CHF, SGD))
True
```

**另请参阅**

- [`Currency`](#currencycode-digits-symbol-name)
- [`CurrencyRegistry`](#currencyregistry)

## 内部辅助方法与函数（私有 API）

### `_CODE_RE` 与 `_MAX_DIGITS`

```python
moneta.currency._CODE_RE = re.compile(r"^[A-Z]{3,5}$")
moneta.currency._MAX_DIGITS = 18
```

- **`_CODE_RE`**：预编译正则表达式，要求标准化（`strip().upper()`）后的货币代码严格由 3 到 5 个大写 ASCII 英文字母组成；
- **`_MAX_DIGITS = 18`**：`Currency.digits` 允许的最大小数位数上限（有符号 `int64` 尾数在 18
  位小数下仍能精确表示约 $\pm 9.22$）。

**示例**

```python
>> > from moneta.currency import _CODE_RE, _MAX_DIGITS
>> > bool(_CODE_RE.match("USDT")), bool(_CODE_RE.match("US")), _MAX_DIGITS
(True, False, 18)
```

### Currency._definition

```python
Currency._definition()
```

返回表示货币完整定义的四元组 `(code, digits, symbol, name)`。

**返回值**

- **out** : *tuple of (str, int, str, str)*

**机制说明**

供 [`CurrencyRegistry.register`](#currencyregistryregister)、[`CurrencyRegistry.get`](#currencyregistryget)、[
`CurrencyRegistry.has`](#currencyregistryhas) 与 [`_restore_currency`](#_restore_currency) 校验两个同代码 `Currency`
实例是否存在任何字段冲突（包括展示字段 `symbol` 与 `name`），同时作为 `Currency.__reduce__` 的序列化参数元组。

**示例**

```python
>> > from moneta import USD
>> > USD._definition()
('USD', 2, '$', 'US Dollar')
```

**另请参阅**

- [`CurrencyRegistry.register`](#currencyregistryregister)
- [`_restore_currency`](#_restore_currency)

### _restore_currency

```python
moneta.currency._restore_currency(code, digits, symbol, name)
```

`pickle` 反序列化还原钩子函数：已注册币种归并回注册表单例，未注册币种还原为独立实例。

**参数**

- **code** : *str* — 货币代码。
- **digits** : *int* — 小数位数。
- **symbol** : *str* — 货币符号。
- **name** : *str* — 货币全称。

**返回值**

- **out** : *Currency*

**异常**

- **ValueError** — `code` 已在当前 `CurrencyRegistry` 中注册，但四元组定义不一致。

**机制说明**

先构造临时实例 `restored = Currency(code, digits, symbol, name)`，再查询 `CurrencyRegistry._registry.get(restored.code)`
：若未注册则直接返回 `restored`；若已注册且 `registered._definition() == restored._definition()` 则返回注册表单例
`registered`；否则抛出 `ValueError`。

**示例**

```python
>> > from moneta import USD
>> > from moneta.currency import _restore_currency
>> > _restore_currency("USD", 2, "$", "US Dollar") is USD
True
>> > _restore_currency("USD", 3, "$", "US Dollar")
Traceback(most
recent
call
last):
...
ValueError: pickle
中的
Currency(code='USD', digits=3, symbol='$', name='US Dollar')
与注册表中的定义
Currency(code='USD', digits=2, symbol='$', name='US Dollar')
不一致
```

**另请参阅**

- [`copy`、`deepcopy` 与 `pickle`](#copydeepcopy-与-pickle单例保持与序列化)
- [`Currency._definition`](#currency_definition)

### _init_defaults

```python
moneta.currency._init_defaults()
```

将模块内置的 10 种法定货币常量依次注册到 [`CurrencyRegistry`](#currencyregistry)。

**返回值**

- **None**

**机制说明**

遍历 `(USD, CNY, EUR, JPY, GBP, HKD, AUD, CAD, CHF, SGD)` 并依次调用 `CurrencyRegistry.register(c)`
。在模块首次导入及调用 [`CurrencyRegistry.clear()`](#currencyregistryclear) 时执行。

**示例**

```python
>> > from moneta import CurrencyRegistry, JPY
>> > from moneta.currency import _init_defaults
>> > _init_defaults()  # 幂等重入，无副作用
>> > CurrencyRegistry.get("JPY") is JPY
True
```

**另请参阅**

- [`CurrencyRegistry.clear`](#currencyregistryclear)
- [内置常用币种常量](#usdcnyeurjpygbphkdaudcadchfsgd)
