# UnitArray — 带量纲定点数张量抽象基类

`moneta.UnitArray` 是所有携带单位或金融量纲标签的定点数张量（如 [`Money`](money.md)、[`Price`](price.md)、[
`Quantity`](quantity.md)）的统一抽象基类（`ABC`）。
它在底层封装一个只读的 [`FixedArray`](fixed_array.md) 实例承载数值，并在外部通过两个抽象钩子（[
`_create_like`](#unitarray_create_like) 与 [`_check_compatible`](#unitarray_check_compatible)）绑定业务量纲：
凡是不改变单位的运算（取正负、绝对值、沿轴求和、索引切片、首维迭代、形状变换）均由基类统一实现并自动保留原单位；
凡是涉及比较的运算均经由基类的量纲兼容性安检；同时与 `FixedArray` 一样禁止 NumPy `ufunc` 与隐式数组转换，杜绝带量纲对象被静默降级为无单位浮点数或
`object` 数组。

```python
>>> from moneta import UnitArray, Money, Price, Quantity
>>> m = Money(["100.50", "-20.25", "50.00"], "USD")
>>> isinstance(m, UnitArray)
True
>>> (-m).currency, abs(m).currency, m.sum().currency
(Currency(code='USD', digits=2, symbol='$', name='US Dollar'),
 Currency(code='USD', digits=2, symbol='$', name='US Dollar'),
 Currency(code='USD', digits=2, symbol='$', name='US Dollar'))
>>> m.sum()
Money(FixedArray(130.25, scale=2), currency='USD')
>>> m == Money(["100.50", "-20.25", "50.00"], "CNY")  # 跨币种不相等且不报错
False
>>> m < Money("200.00", "CNY")  # 跨币种比大小直接拦截
Traceback (most recent call last):
    ...
TypeError: 币种不匹配: USD
与
CNY
```

[← API 参考](index.md)

## 核心机制与设计约束

### 组合委托与双钩子扩展契约

`UnitArray` 采用 **组合优于继承**的轻量级委托架构，自身仅声明单槽位 `__slots__ = ("_data",)` 指向底层的 [
`FixedArray`](fixed_array.md)。任何具体子类（内置的 [`Money`](money.md)、[`Price`](price.md)、[`Quantity`](quantity.md)
或用户自定义量纲类）只需：

1. 在 `__init__` 中初始化 `self._data`（类型为 `FixedArray`）与子类自身的量纲槽位；
2. 实现 [`_create_like(new_data)`](#unitarray_create_like)：将新的 `FixedArray` 包装为携带当前实例相同单位/配置的新子类对象；
3. 实现 [`_check_compatible(other)`](#unitarray_check_compatible)：当 `other` 类型不符或单位不匹配时抛出 `TypeError`（或
   `ValueError`）。

完成这两步后，子类即刻自动获得属性透传、切片索引、张量变形、正负号、绝对值、沿轴精确求和、零值判定、量纲安全比较、NumPy 防护以及向
`float` / `Decimal` / `pandas` / `polars` 的导出能力。

### 保单位运算与跨量纲代数分工

| 运算类别             | 涵盖接口                                                                                    | 负责层级                 | 量纲行为                                                                                  |
|----------------------|---------------------------------------------------------------------------------------------|--------------------------|-------------------------------------------------------------------------------------------|
| **保单位变换与规约** | `+a`、`-a`、`abs(a)`、`a.sum()`、`a[key]`、`iter(a)`、`a.reshape()`、`a.transpose()`、`a.T` | `UnitArray` 基类统一实现 | 委托 `self._data` 计算后调用 `self._create_like(...)`，结果严格保持原子类类型与原单位     |
| **零值与真值判定**   | `a.is_zero()`、`bool(a)`                                                                    | `UnitArray` 基类统一实现 | 直接检查底层尾数是否为 0，不触发跨类型比较，返回布尔值或布尔数组                          |
| **相等与大小比较**   | `a == b`、`a != b`、`a < b`、`a <= b`、`a > b`、`a >= b`                                    | `UnitArray` 基类统一实现 | 先调 `_check_compatible(b)` 做量纲安检，通过后再委托 `self._data` 逐元素比较              |
| **二元加减与乘除**   | `a + b`、`a - b`、`a * b`、`a / b`、`a @ b`                                                 | 各具体子类自行实现       | 按物理与金融量纲法则定义（如 `Price * Quantity -> Money`、`Money / Money -> FixedArray`） |

### 比较运算的三层量纲安检规则

在当前 [`unit_array.py`](https://github.com/T0nyN1/Moneta/blob/main/src/moneta/unit_array.py) 实现中， **有序大小比较（`< <= > >=`）** 与 **相等性比较（
`== !=`）** 对“量纲不匹配”和“无量纲数”采取了不同的安检策略：

- **大小比较（`<`、`<=`、`>`、`>=`）—— 只要量纲不匹配或对方是无量纲数，一律报 `TypeError`**：
    1. **对方是量纲不匹配的 `UnitArray`**（如 `USD` 与 `CNY` 比大小，或 `Money` 与 [`Quantity`](quantity.md) 比大小）：[
       `_ordered`](#unitarray_ordered) 直接调用 `self._check_compatible(other)`，由子类立即抛出携带具体冲突原因的
       `TypeError`（如 `TypeError: 币种不匹配: USD 与 CNY` 或 `TypeError: 操作数必须是 Money，得到 Quantity`）。
    2. **对方是无量纲数或非 `UnitArray` 对象**（如 `int`、`float`、`Decimal`、`FixedArray`、`ndarray`）：`_ordered` 先通过
       `if not isinstance(other, UnitArray): return NotImplemented` 交还给 Python 解释器；由于无量纲数与 `ndarray`（因
       `__array_ufunc__ = None`）同样拒绝与 `UnitArray` 比较，Python 最终直接抛出
       `TypeError: '<' not supported between instances of ...`。
- **相等性比较（`==`、`!=`）—— 量纲不匹配或遇到无量纲数时不抛异常，直接判定为“不相等”（`False` / `True`）**：
    - `__eq__` / `__ne__` 内部通过 [`_compatible_or_none(other)`](#unitarray_compatible_or_none) 调用
      `self._check_compatible(other)`，但用 `try ... except (TypeError, ValueError)` 捕获了量纲不匹配与非同类型异常并转为返回
      `NotImplemented`，最终由 Python 折叠为标量 `False`（`==`）或 `True`（`!=`）。
    - 这样设计符合 Python 数据模型规范：确保 `a == None`、容器成员检测 `a in [...]` 或字典查找不会因列表中混有其他单位/无量纲对象而崩溃；同时也意味着
      **带单位对象与裸数字 `0` 永远不相等（`a == 0` 恒为 `False`）**，判断是否为零必须使用 [
      `a.is_zero()`](#unitarrayis_zero)。

| 比较场景                                                                                        | `a == b` / `a != b`                                                                                          | `a < b` / `a <= b` / `a > b` / `a >= b`                                                    | 设计原因                                                                                                                  |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **同子类且单位兼容**（如 `Money("1", "USD")` 与 `Money("2", "USD")`）                           | 委托 `self._data == other._data`，0 维返回 `np.bool_`，高维返回 `bool` 数组（形状无法广播时抛 `ValueError`） | 委托 `self._data < other._data`，0 维返回 `np.bool_`，高维返回 `bool` 数组                 | 同量纲数值可直接按定点数对齐 `scale` 后逐元素比较                                                                         |
| **同为 `UnitArray` 但单位不兼容**（如 `USD` 与 `CNY`，或 `Money` 与 [`Quantity`](quantity.md)） | 内部返回 `NotImplemented`，Python 最终判定为 `False` / `True`（**不抛异常**）                                | 调用 `self._check_compatible(other)`，**立即抛出子类的 `TypeError`**（含具体单位冲突原因） | 字典查找、`in` 成员检测要求 `==` 永不因类型/单位不同而崩溃；但“100 美元是否小于 200 人民币”属于逻辑错误，必须显式报错     |
| **与无量纲数 / 非 `UnitArray` 比较**（如 `a == 0`、`a < 0`、`a == FixedArray("0")`）            | 内部返回 `NotImplemented`，允许对方尝试反向比较，最终判定为 `False` / `True`                                 | 内部返回 `NotImplemented`，允许对方尝试反向比较，**最终由 Python 抛出 `TypeError`**        | 带单位对象与裸数字（哪怕是 `0`）量纲不同，绝不静默相等或比大小；判断是否为零请始终使用 [`a.is_zero()`](#unitarrayis_zero) |

## API 速查表

| 分类                     | 接口 / 运算符                                                                                                                                                                                                                                                                                                                 | 功能说明                                                             |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| **抽象基类与扩展钩子**   | [`UnitArray`](#unitarray)                                                                                                                                                                                                                                                                                                     | 带单位/量纲定点数张量的抽象基类（`ABC`）                             |
|                          | [`UnitArray._create_like`](#unitarray_create_like)                                                                                                                                                                                                                                                                            | 抽象钩子：用新 `FixedArray` 构造同单位的当前子类实例                 |
|                          | [`UnitArray._check_compatible`](#unitarray_check_compatible)                                                                                                                                                                                                                                                                  | 抽象钩子：校验另一对象是否与当前实例量纲兼容                         |
| **底层数据与数组属性**   | [`UnitArray.data`](#unitarraydata)                                                                                                                                                                                                                                                                                            | 底层 [`FixedArray`](fixed_array.md) 定点数张量实例                   |
|                          | [`UnitArray.mantissa`](#unitarraymantissa)                                                                                                                                                                                                                                                                                    | 底层只读 `int64` 整数尾数数组视图（缺失位置为 0）                    |
|                          | [`UnitArray.scale`](#unitarrayscale)                                                                                                                                                                                                                                                                                          | 全局十进制小数位数                                                   |
|                          | [`UnitArray.shape`](#unitarrayshape) / [`ndim`](#unitarrayndim) / [`size`](#unitarraysize) / [`dtype`](#unitarraydtype)                                                                                                                                                                                                       | 张量形状、维度数、元素总数、尾数数据类型（`int64`）                  |
|                          | [`UnitArray.T`](#unitarrayt)                                                                                                                                                                                                                                                                                                  | 反转全部轴并保持原单位的转置视图                                     |
| **缺失值（NA）处理**     | [`UnitArray.isna`](#unitarrayisnaunitarrayisnullunitarraynotna-与-unitarraynotnull) / [`isnull`](#unitarrayisnaunitarrayisnullunitarraynotna-与-unitarraynotnull) / [`notna`](#unitarrayisnaunitarrayisnullunitarraynotna-与-unitarraynotnull) / [`notnull`](#unitarrayisnaunitarrayisnullunitarraynotna-与-unitarraynotnull) | 返回标记缺失位置或有效数值位置的布尔数组                             |
|                          | [`UnitArray.fillna`](#unitarrayfillna)                                                                                                                                                                                                                                                                                        | 用兼容单位的同类对象或数值填充缺失位置（保持原单位）                 |
|                          | [`UnitArray.dropna`](#unitarraydropna)                                                                                                                                                                                                                                                                                        | 丢弃 1 维向量中的缺失值（保持原单位）                                |
| **保单位算术与规约**     | [`+a`、`-a`](#a-与--a一元正负号)                                                                                                                                                                                                                                                                                              | 一元正号（返回自身）与逐元素取负（保持原单位）                       |
|                          | [`abs(a)`](#absa绝对值)                                                                                                                                                                                                                                                                                                       | 逐元素绝对值（保持原单位）                                           |
|                          | [`UnitArray.sum`](#unitarraysum)                                                                                                                                                                                                                                                                                              | 沿指定轴精确求和（保持原单位，支持 `skipna=True` 与 `np.sum` 转发）  |
|                          | [`UnitArray.is_zero`](#unitarrayis_zero)                                                                                                                                                                                                                                                                                      | 逐元素或标量判断数值是否为 0（缺失值不是 0）                         |
| **比较运算**             | [`a == b`、`a != b`](#a--b-与-a--b相等性比较)                                                                                                                                                                                                                                                                                 | 量纲兼容时逐元素判等；量纲不兼容或为裸数字时返回 `False` / `True`    |
|                          | [`a < b`、`a <= b`、`a > b`、`a >= b`](#a--ba--ba--ba--b大小比较)                                                                                                                                                                                                                                                             | 量纲兼容时逐元素比较大小；量纲不兼容时抛出 `TypeError`               |
| **形状与容器协议**       | [`UnitArray.reshape`](#unitarrayreshape)                                                                                                                                                                                                                                                                                      | 不复制底层尾数改变张量形状（保持原单位）                             |
|                          | [`UnitArray.transpose`](#unitarraytranspose)                                                                                                                                                                                                                                                                                  | 按指定轴顺序置换张量维度（保持原单位）                               |
|                          | [`UnitArray.swapaxes`](#unitarrayswapaxes-与-unitarraysqueeze) / [`UnitArray.squeeze`](#unitarrayswapaxes-与-unitarraysqueeze)                                                                                                                                                                                                | 交换两个轴 / 移除长度为 1 的轴（保持原单位）                         |
|                          | [`a[key]`](#akey索引与切片)                                                                                                                                                                                                                                                                                                   | 索引、切片、布尔掩码与花式索引（保持原单位）                         |
|                          | [`len(a)`、`bool(a)`、`iter(a)`](#lenaboola-与-itera容器与真值协议)                                                                                                                                                                                                                                                           | 首维长度、单元素真值测试与按首维保单位迭代                           |
| **导出与互操作**         | [`UnitArray.to_float`](#unitarrayto_float)                                                                                                                                                                                                                                                                                    | 剥离单位导出为 `float` 标量或 `float64` 数组                         |
|                          | [`UnitArray.to_decimal`](#unitarrayto_decimal)                                                                                                                                                                                                                                                                                | 剥离单位导出为 `Decimal` 标量或 `object` 数组                        |
|                          | [`UnitArray.to_pandas`](#unitarrayto_pandas)                                                                                                                                                                                                                                                                                  | 导出为 `pandas.Series` 或 `DataFrame`                                |
|                          | [`UnitArray.to_polars`](#unitarrayto_polars)                                                                                                                                                                                                                                                                                  | 导出为 `polars.Series`                                               |
|                          | [`UnitArray.__array__` 与 NumPy 防护机制](#unitarrayarray-与-numpy-防护机制)                                                                                                                                                                                                                                                  | 拦截隐式转 `ndarray` 与 `ufunc`，白名单分派 `__array_function__`     |
| **内部辅助（私有 API）** | [`UnitArray._as_factor`](#unitarray_as_factor)                                                                                                                                                                                                                                                                                | 校验无量纲数值白名单并转为 `FixedArray`                              |
|                          | [`UnitArray._as_grid`](#unitarray_as_grid-与-unitarray_fill_value) / [`UnitArray._fill_value`](#unitarray_as_grid-与-unitarray_fill_value)                                                                                                                                                                                    | 校验不含缺失值的网格参数 / 将 `fillna` 数值填充值换算为 `FixedArray` |
|                          | [`UnitArray._compatible_or_none`](#unitarray_compatible_or_none)                                                                                                                                                                                                                                                              | 捕获 `_check_compatible` 异常并转为布尔返回值（供 `==` / `!=` 使用） |
|                          | [`UnitArray._ordered`](#unitarray_ordered)                                                                                                                                                                                                                                                                                    | 校验有序比较的操作数类型与量纲兼容性并分发比较函数                   |

## 抽象基类契约

### UnitArray

```python
class moneta.UnitArray(ABC)
```

带单位或量纲标签的定点数张量抽象基类。

**类属性与槽位**

- **`__slots__ = ("_data",)`** — 限制实例属性，仅在基类层存储底层 [`FixedArray`](fixed_array.md) 引用 `_data`，消除每个实例的
  `__dict__` 内存开销。子类也应声明各自的 `__slots__`（如 `Money` 的 `("_currency",)`）。
- **`__array_ufunc__ = None`** — 显式关闭 NumPy 通用函数（`ufunc`）分派协议。任何对 `UnitArray` 调用 `np.add`、`np.sqrt` 或将
  `ndarray` 放在运算符左侧（如 `np.array([1, 2]) + a`）的操作都会立即抛出 `TypeError`，避免 NumPy 将其拆解为慢速且破坏量纲的
  `object` 数组。

**异常**

- **TypeError** — 直接实例化 `UnitArray()`，或子类未完整实现 [`_create_like`](#unitarray_create_like) 与 [
  `_check_compatible`](#unitarray_check_compatible) 两个抽象方法时实例化。

**机制说明**

`UnitArray` 继承自 `abc.ABC`。在下面的文档示例中，除了使用内置子类 [`Money`](money.md)、[`Price`](price.md)、[
`Quantity`](quantity.md) 外，我们也会使用一个最小自定义子类 `Length` 来直观展示基类的扩展方式：

**示例**

直接实例化抽象基类或未实现全部钩子的子类会被阻止：

```python
>>> from moneta import UnitArray
>>> UnitArray()
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class UnitArray ...
>>> class PartialUnit(UnitArray):  # 只实现了 _create_like
...     def _create_like(self, new_data):
...         return self
>>> PartialUnit()
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class PartialUnit ...
```

只需实现两个抽象钩子即可定义完整的自定义带单位张量类：

```python
>>> class Length(UnitArray):
...     __slots__ = ("_unit",)
...     def __init__(self, values, unit="m"):
...         self._data = values if isinstance(values, FixedArray) else FixedArray(values)
...         self._unit = str(unit)
...     def _create_like(self, new_data):
...         return Length(new_data, self._unit)
...     def _check_compatible(self, other):
...         if not isinstance(other, Length):
...             raise TypeError(f"操作数必须是 Length，得到 {type(other).__name__}")
...         if other._unit != self._unit:
...             raise TypeError(f"单位不匹配: {self._unit} 与 {other._unit}")
...     def __repr__(self):
...         return f"Length({self._data!r}, unit={self._unit!r})"
>>> d = Length(["1.50", "-2.25", "3.00"], "km")
>>> d.sum()
Length(FixedArray(2.25, scale=2), unit='km')
>>> abs(d[1])
Length(FixedArray(2.25, scale=2), unit='km')
```

**另请参阅**

- [`UnitArray._create_like`](#unitarray_create_like)
- [`UnitArray._check_compatible`](#unitarray_check_compatible)
- [`FixedArray`](fixed_array.md)

### UnitArray._create_like

```python
@abstractmethod
UnitArray._create_like(new_data)
```

抽象工厂钩子：使用给定的新底层 [`FixedArray`](fixed_array.md) 与当前实例相同的单位/配置构造同类型子类实例。

**参数**

- **new_data** : *FixedArray* — 经过取负、绝对值、求和、切片或形状变换后的新底层定点数张量。

**返回值**

- **out** : *UnitArray* — 与 `self` 具有相同具体子类类型和相同量纲元数据的实例。

**机制说明**

`UnitArray` 基类本身不知道子类的构造函数签名（例如 `Money` 需要 `currency`，[`Price`](price.md) 需要 `quote_currency`、
`base_unit`、`tick_size`，[`Quantity`](quantity.md) 需要 `unit`）。
因此，基类中所有不改变单位的方法（`__neg__`、`__abs__`、`sum`、`__getitem__`、`__iter__`、`reshape`、`transpose`）在完成
`self._data` 上的计算后，统一调用 `self._create_like(new_data)` 将结果重新包装为子类实例。

内置子类的实现策略：

- **[`Money._create_like(new_data)`](money.md#money_create_like)**：调用 [
  `Money._wrap(new_data, self._currency)`](money.md#money_wrap) 校验 `new_data.scale == self._currency.digits` 并零拷贝构造同币种
  `Money`；
- **[`Price._create_like(new_data)`](price.md#price_create_like)**：返回携带相同 `quote_currency`、`base_unit` 与
  `tick_size` 的新 [`Price`](price.md)；
- **[`Quantity._create_like(new_data)`](quantity.md#quantity_create_like)**：返回携带相同 `unit` 的新 [
  `Quantity`](quantity.md)。

**示例**

```python
>>> from moneta import Money, Price, Quantity
>>> m = Money(["10.00", "20.00"], "USD")
>>> m._create_like(FixedArray(["5.50", "6.25"]))
Money(FixedArray([5.50, 6.25], scale=2), currency='USD')
>>> p = Price("100.00", "USD", base_unit="barrel", tick_size=FixedArray("0.05"))
>>> p2 = p._create_like(FixedArray("101.25"))
>>> (p2.quote_currency.code, p2.base_unit, p2.tick_size)
('USD', 'barrel', FixedArray(0.05, scale=2))
```

**另请参阅**

- [`UnitArray._check_compatible`](#unitarray_check_compatible)
- [`UnitArray.data`](#unitarraydata)

### UnitArray._check_compatible

```python
@abstractmethod
UnitArray._check_compatible(other)
```

抽象安检钩子：检查另一个对象 `other` 是否与当前实例在类型和单位/量纲上兼容。

**参数**

- **other** : *Any* — 待检验的比较或运算对象。

**返回值**

- **None** — 兼容时静默返回 `None`。

**异常**

- **TypeError 或 ValueError** — 当 `other` 不是当前子类的实例，或其携带的币种/单位与 `self` 不兼容时抛出。

**机制说明**

基类的比较方法（[`==` / `!=`](#a--b-与-a--b相等性比较) 通过 [`_compatible_or_none`](#unitarray_compatible_or_none)，[`<` /
`<=` / `>` / `>=`](#a--ba--ba--ba--b大小比较) 通过 [`_ordered`](#unitarray_ordered)）以及子类的同量纲二元运算（如
`Money + Money`、`Quantity - Quantity`）均通过调用 `self._check_compatible(other)` 执行量纲安检：

- **[`Money._check_compatible(other)`](money.md#money_check_compatible)**：要求 `isinstance(other, Money)` 且
  `self._currency == other._currency`；
- **[`Price._check_compatible(other)`](price.md#price_check_compatible)**：要求 `isinstance(other, Price)` 且
  `quote_currency` 相同、`base_unit` 兼容；
- **[`Quantity._check_compatible(other)`](quantity.md#quantity_check_compatible)**：要求 `isinstance(other, Quantity)`
  且单位相同（或其中一方为通配单位 `"Any"`）。

**示例**

```python
>>> from moneta import Money, Quantity
>>> m_usd = Money("10.00", "USD")
>>> m_usd._check_compatible(Money("20.00", "USD"))  # 兼容：返回 None
>>> m_usd._check_compatible(Money("20.00", "CNY"))
Traceback (most recent call last):
    ...
TypeError: 币种不匹配: USD
与
CNY
>>> m_usd._check_compatible(Quantity(10, "share"))
Traceback (most recent call last):
    ...
TypeError: 操作数必须是
Money，得到
Quantity
```

**另请参阅**

- [`UnitArray._compatible_or_none`](#unitarray_compatible_or_none)
- [`UnitArray._ordered`](#unitarray_ordered)

## 底层数据与数组属性

### UnitArray.data

```python
UnitArray.data
```

返回底层封装的 [`FixedArray`](fixed_array.md) 定点数张量实例。

**机制说明**

直接返回只读槽位 `self._data`（零拷贝）。当你需要执行剥离量纲的底层定点数运算（如调用 `FixedArray` 的特种精度控制方法）时，可通过
`.data` 访问。

**返回值**

- **data** : *FixedArray*

**示例**

```python
>>> a = Length([["1.5", "2"], ["3", "4"]], "m")
>>> a.data
FixedArray([[1.5, 2.0],
            [3.0, 4.0]], scale=1)
>>> type(a.data) is FixedArray
True
```

**另请参阅**

- [`UnitArray.mantissa`](#unitarraymantissa)
- [`UnitArray.scale`](#unitarrayscale)
- [`FixedArray`](fixed_array.md)

### UnitArray.mantissa

```python
UnitArray.mantissa
```

底层纯整数尾数数组（只读 `int64` 视图），等价于 `self.data.mantissa`。

**机制说明**

直接委托 `self._data.mantissa`：返回的 `numpy.ndarray` 的 `flags.writeable` 为 `False`，原位修改会抛出 `ValueError`。
真实十进制数值满足 $\text{value} = \text{mantissa} \times 10^{-\text{scale}}$。由于同一 `UnitArray` 内所有元素共享同一个单位与
`scale`，尾数的大小顺序与带单位数值的大小顺序完全一致，可直接传入 `np.argmax`、`np.argsort` 等只读整数函数。

**返回值**

- **mantissa** : *numpy.ndarray（int64）*

**示例**

```python
>>> a = Length(["1.50", "-2.25"], "m")
>>> a.mantissa
array([ 150, -225])
>>> a.mantissa.flags.writeable
False
>>> a.mantissa[0] = 999
Traceback (most recent call last):
    ...
ValueError: assignment destination is read-only
```

**另请参阅**

- [`UnitArray.data`](#unitarraydata)
- [`UnitArray.scale`](#unitarrayscale)
- [`FixedArray.mantissa`](fixed_array.md#fixedarraymantissa)

### UnitArray.scale

```python
UnitArray.scale
```

全局十进制小数位数，等价于 `self.data.scale`。

**机制说明**

直接返回 `self._data.scale`（非负 Python `int`）。在 [`Money`](money.md) 中，`scale` 恒等于所属币种的 [
`currency.digits`](currency.md#currencydigits)；在 [`Price`](price.md) 与 [`Quantity`](quantity.md) 中，`scale`
由输入数值自动推断或显式指定。

**返回值**

- **scale** : *int*

**示例**

```python
>>> from moneta import Money, JPY, USD
>>> Money("19.99", USD).scale, Money(500, JPY).scale
(2, 0)
>>> Length(["0.1", "0.125"], "m").scale
3
```

**另请参阅**

- [`UnitArray.mantissa`](#unitarraymantissa)
- [`FixedArray.scale`](fixed_array.md#fixedarrayscale)

### UnitArray.shape

```python
UnitArray.shape
```

张量形状元组，等价于 `self.data.shape`。

**机制说明**

直接返回 `self._data.shape`。0 维标量张量的 `shape` 为 `()`。

**返回值**

- **shape** : *tuple of int*

**示例**

```python
>>> Length([["1.5", "2"], ["3", "4"]], "m").shape
(2, 2)
>>> Length("1.5", "m").shape
()
```

**另请参阅**

- [`UnitArray.ndim`](#unitarrayndim)、[`UnitArray.size`](#unitarraysize)
- [`UnitArray.reshape`](#unitarrayreshape)

### UnitArray.ndim

```python
UnitArray.ndim
```

张量的维度数（轴数），等于 `len(self.shape)`。

**机制说明**

直接返回 `self._data.ndim`。`ndim == 0` 的标量张量不支持 `len()` 与 `iter()`。

**返回值**

- **ndim** : *int*

**示例**

```python
>>> Length("1.5", "m").ndim
0
>>> Length([["1.5", "2"], ["3", "4"]], "m").ndim
2
```

**另请参阅**

- [`UnitArray.shape`](#unitarrayshape)、[`UnitArray.size`](#unitarraysize)
- [`len(a)`、`bool(a)` 与 `iter(a)`](#lenaboola-与-itera容器与真值协议)

### UnitArray.size

```python
UnitArray.size
```

张量中元素的总个数，等于 `shape` 各维长度的乘积。

**机制说明**

直接返回 `self._data.size`。0 维标量的 `size` 为 `1`，空数组的 `size` 为 `0`；仅当 `size == 1` 时允许调用 `bool(a)`。

**返回值**

- **size** : *int*

**示例**

```python
>>> Length([["1.5", "2", "3"], ["4", "5", "6"]], "m").size
6
>>> Length("1.5", "m").size
1
```

**另请参阅**

- [`UnitArray.shape`](#unitarrayshape)、[`UnitArray.ndim`](#unitarrayndim)

### UnitArray.dtype

```python
UnitArray.dtype
```

底层整数尾数的数据类型，恒为 `numpy.dtype('int64')`。

**机制说明**

直接返回 `self._data.dtype`。

**返回值**

- **dtype** : *numpy.dtype*

**示例**

```python
>>> Length("1.5", "m").dtype
dtype('int64')
```

**另请参阅**

- [`UnitArray.mantissa`](#unitarraymantissa)
- [`FixedArray.dtype`](fixed_array.md#fixedarraydtype)

### UnitArray.T

```python
UnitArray.T
```

反转全部轴的转置视图，等价于 `self.transpose()`，结果保持原单位。

**机制说明**

内部调用 `self.transpose()`，即 `self._create_like(self._data.transpose())`。底层尾数数组零拷贝共享内存，外层包装为携带原单位的新子类实例。

**返回值**

- **out** : *UnitArray* — 与 `self` 同类型、同单位的转置视图。

**示例**

```python
>>> a = Length(["1", "2", "3", "4"], "km").reshape(2, 2)
>>> a.T
Length(FixedArray([[1, 3],
                   [2, 4]], scale=0), unit='km')
```

**另请参阅**

- [`UnitArray.transpose`](#unitarraytranspose)
- [`UnitArray.reshape`](#unitarrayreshape)

## 缺失值（NA）处理

### `UnitArray.isna`、`UnitArray.isnull`、`UnitArray.notna` 与 `UnitArray.notnull`

```python
UnitArray.isna()
UnitArray.isnull()
UnitArray.notna()
UnitArray.notnull()
```

返回标记缺失位置（`isna` / `isnull`）或有效数值位置（`notna` / `notnull`）的布尔 NumPy 数组（0 维标量时返回 0
维布尔数组）。直接委托给底层 [
`FixedArray.isna()`](fixed_array.md#fixedarrayisnafixedarrayisnullfixedarraynotna-与-fixedarraynotnull) 与 `notna()`。

**示例**

```python
>>> m = Money(["10.50", None, "20.00"], "USD")
>>> m.isna().tolist(), m.notna().tolist()
([False, True, False], [True, False, True])
>>> bool((m.isnull() == m.isna()).all() and (m.notnull() == m.notna()).all())
True
```

**另请参阅**

- [`UnitArray.fillna`](#unitarrayfillna)、[`UnitArray.dropna`](#unitarraydropna)

### UnitArray.fillna

```python
UnitArray.fillna(value)
```

用同类对象（量纲须兼容）或数值填充缺失位置，结果保持原单位。

**参数**

- **value** : *UnitArray 或 数值（int, float, Decimal, Fraction, FixedArray, 数值数组）* — 填充值。若为 `UnitArray`
  实例，会先调用 [`_check_compatible(value)`](#unitarray_check_compatible) 校验单位兼容性；若为无量纲数值，则通过钩子 [
  `_fill_value(value)`](#unitarray_as_grid-与-unitarray_fill_value) 按当前子类的构造规则换算为 `FixedArray`（例如 [
  `Money`](money.md) 会将数值换算到币种精度，精确输入超出精度时发出 [
  `PrecisionLossWarning`](fixed_array.md#precisionlosswarning)）。

**返回值**

- **out** : *UnitArray* — 填充缺失位置后的同单位实例。

**异常**

- **TypeError** — `value` 为不兼容单位的 `UnitArray`，或不属于数值白名单（如字符串、布尔值）。
- **ValueError** — `value` 在需要填充的位置上本身也是缺失值。

**示例**

```python
>>> m = Money(["10.50", None, "20.00"], "USD")
>>> m.fillna(0)
Money(FixedArray([10.50, 0.00, 20.00], scale=2), currency='USD')
>>> m.fillna(Money("99.99", "USD"))
Money(FixedArray([10.50, 99.99, 20.00], scale=2), currency='USD')
>>> m.fillna(Money("99.99", "CNY"))
Traceback (most recent call last):
    ...
TypeError: 币种不匹配: USD 与 CNY
```

**另请参阅**

- [`UnitArray.isna`](#unitarrayisnaunitarrayisnullunitarraynotna-与-unitarraynotnull)、[
  `UnitArray.dropna`](#unitarraydropna)
- [`UnitArray._fill_value`](#unitarray_as_grid-与-unitarray_fill_value)

### UnitArray.dropna

```python
UnitArray.dropna()
```

丢弃 1 维向量中的缺失值，结果保持原单位。仅限 1 维张量（`ndim == 1`），其他维度抛出 `ValueError`。

**示例**

```python
>>> Money(["10.50", None, "20.00"], "USD").dropna()
Money(FixedArray([10.50, 20.00], scale=2), currency='USD')
```

**另请参阅**

- [`UnitArray.fillna`](#unitarrayfillna)
- [`FixedArray.dropna`](fixed_array.md#fixedarraydropna)

## 保单位算术与规约运算

### +a 与 -a（一元正负号）

```python
+a
-a
```

一元正号直接返回自身；一元负号逐元素取反并保持原单位。

**返回值**

- **out** : *UnitArray* — 与 `a` 同类型、同单位、`scale` 不变。`+a` 直接返回 `a` 本身（`+a is a`）。

**异常**

- **OverflowError** — `-a` 时底层尾数含有 `int64` 最小值 `-2**63`（其相反数超出 `int64` 表示范围）。

**机制说明**

- `__pos__`：由于 `UnitArray` 不可变，直接返回 `self`；
- `__neg__`：执行 `self._create_like(-self._data)`，先由 [`FixedArray.__neg__`](fixed_array.md#a-与--a一元正负号) 检查
  `-2**63` 溢出并对尾数取负，再用 `_create_like` 包装回原单位子类。

**示例**

```python
>>> a = Length(["-1.5", "2.0"], "km")
>>> (+a) is a
True
>>> -a
Length(FixedArray([1.5, -2.0], scale=1), unit='km')
>>> -Length(FixedArray._from_mantissa(np.array([-(2 ** 63)]), 0), "km")
Traceback (most recent call last):
    ...
OverflowError: 取负导致数值超出 int64 范围
```

**性能对比**

`UnitArray` 相比 `FixedArray` 仅增加一次 `_create_like` 实例包装（约 `0.2 µs` 常数开销），在大数组上耗时与 `FixedArray`
完全一致：

|         N | UnitArray (`-a`) | FixedArray (`-fa`) | NumPy float64 (`-x`) |
|----------:|-----------------:|-------------------:|---------------------:|
|  0-d 标量 |          4.03 µs |            3.83 µs |              33.5 ns |
|     1,000 |          3.98 µs |            3.76 µs |               417 ns |
| 1,000,000 |           901 µs |             904 µs |               720 µs |

**另请参阅**

- [`abs(a)`](#absa绝对值)
- [`FixedArray` 一元正负号](fixed_array.md#a-与--a一元正负号)

### `abs(a)`（绝对值）

```python
abs(a)
```

逐元素取绝对值，结果保持原单位。

**返回值**

- **out** : *UnitArray* — 与 `a` 同类型、同单位、`scale` 不变。

**异常**

- **OverflowError** — 底层尾数含有 `-2**63`。

**机制说明**

调用 `self._create_like(abs(self._data))`。注意 `np.abs(a)` 属于 NumPy `ufunc`，会被 `__array_ufunc__ = None` 拒绝，请始终使用
Python 内置的 `abs(a)`。

**示例**

```python
>>> a = Length(["-1.5", "2.0"], "km")
>>> abs(a)
Length(FixedArray([1.5, 2.0], scale=1), unit='km')
>>> np.abs(a)
Traceback (most recent call last):
    ...
TypeError: operand 'Length' does not support ufuncs (__array_ufunc__=None)
```

**性能对比**

|         N | UnitArray (`abs(a)`) | FixedArray (`abs(fa)`) | NumPy float64 (`np.abs(x)`) |
|----------:|---------------------:|-----------------------:|----------------------------:|
|  0-d 标量 |              3.97 µs |                3.83 µs |                      379 ns |
|     1,000 |              3.88 µs |                3.72 µs |                      448 ns |
| 1,000,000 |               925 µs |                 943 µs |                      761 µs |

**另请参阅**

- [`+a` 与 `-a`](#a-与--a一元正负号)
- [`FixedArray` 绝对值](fixed_array.md#absa绝对值)

### UnitArray.sum

```python
UnitArray.sum(axis=None, keepdims=False, skipna=True, out=None, dtype=None)
```

沿指定轴对张量元素精确求和，结果保持原单位与原 `scale`。

**参数**

- **axis** : *int 或 tuple of int，可选* — 求和所沿的轴。默认 `None` 时对全部元素求和并返回 0 维同单位标量。
- **keepdims** : *bool，默认 `False`* — 为 `True` 时保留被规约的轴（长度为 1）。
- **skipna** : *bool，默认 `True`* — 是否忽略缺失值 `<NA>`。为 `True` 时缺失位置按 `0` 处理；为 `False` 时若求和切片中含有
  `<NA>`，该切片结果为 `<NA>`。
- **out** : *None* — 兼容 `np.sum(a)` 签名预留，必须为 `None`（不支持原地输出）。
- **dtype** : *None* — 兼容 `np.sum(a)` 签名预留，必须为 `None`（禁止指定 `dtype` 改变定点数类型）。

**返回值**

- **out** : *UnitArray* — 与 `self` 同类型、同单位、`scale` 不变的求和结果。

**异常**

- **TypeError** — 传入了非 `None` 的 `out` 或 `dtype` 参数。
- **OverflowError** — 沿轴求和的最终真实结果超出 `int64` 范围。

**机制说明**

内部执行 `self._create_like(self._data.sum(axis=axis, keepdims=keepdims, skipna=skipna, out=out, dtype=dtype))`：

- 底层直接复用 [`FixedArray.sum`](fixed_array.md#fixedarraysum) 的快慢双路径：当最坏上界 `max(|m|) * count <= 2^63 - 1`
  时直接走 `np.sum` 快路径；当最坏上界可能越界时自动切换为 [`limbs_sum`](_limbs.md#limbs_sum)（Base $2^{21}$
  三块精确累加），哪怕中间部分和越界、只要最终正负相抵后在 `int64` 范围内就不会误报溢出。
- 由于实现了 `__array_function__` 并兼容 `out=None, dtype=None`，调用 `np.sum(a, axis=..., keepdims=...)`
  会自动转发到此方法并返回带原单位的 `UnitArray` 子类实例。

**示例**

```python
>>> a = Length([["1.5", "2"], ["3", "4.25"]], "km")
>>> a.sum()
Length(FixedArray(10.75, scale=2), unit='km')
>>> a.sum(axis=0)
Length(FixedArray([4.50, 6.25], scale=2), unit='km')
>>> a.sum(axis=1, keepdims=True)
Length(FixedArray([[3.50],
                   [7.25]], scale=2), unit='km')
>>> np.sum(a)  # np.sum 自动转发并保留单位
Length(FixedArray(10.75, scale=2), unit='km')
>>> m_na = Money(["1.50", None, "2.50"], "USD")
>>> m_na.sum(), m_na.sum(skipna=False)
(Money(FixedArray(4.00, scale=2), currency='USD'), Money(FixedArray(<NA>, scale=2), currency='USD'))
```

中间和超过 `int64` 上限但正负相抵后未越界时仍能精确求和：

```python
>>> big = Length(FixedArray._from_mantissa(np.array([2**63 - 1, 1, -1]), 0), "m")
>>> big.sum().mantissa
array(9223372036854775807)
>>> np.sum(a, dtype=np.float64)
Traceback (most recent call last):
    ...
TypeError: FixedArray.sum 不支持 out、dtype；要 float 请先 to_float()
```

**性能对比**

|         N | UnitArray (`a.sum()`) | FixedArray (`fa.sum()`) | NumPy float64 (`x.sum()`) |
|----------:|----------------------:|------------------------:|--------------------------:|
|     1,000 |               4.77 µs |                 4.55 µs |                    852 ns |
| 1,000,000 |                323 µs |                  326 µs |                    191 µs |

**另请参阅**

- [`FixedArray.sum`](fixed_array.md#fixedarraysum)
- [`limbs_sum`](_limbs.md#limbs_sum)

### UnitArray.is_zero

```python
UnitArray.is_zero()
```

判断标量或张量中的各个元素是否为 0。

**返回值**

- **out** : *numpy.bool_ 或 numpy.ndarray（bool）* — 0 维标量返回 `np.bool_`（`np.True_` 或 `np.False_`），1 维及以上返回同形状的布尔
  `ndarray`。

**机制说明**

直接计算 `self._data.mantissa == 0`：

- **为什么不能用 `a == 0`？** 在 Moneta 的量纲安全体系中，带单位的 `UnitArray` 与无量纲裸数字 `0` 量纲不兼容，`a == 0` 永远返回
  `False`（即使对数组也是返回单个 `False` 标量而非布尔掩码）。因此，判断金额、价格或数量是否为零时， **必须使用
  `a.is_zero()`**。
- 由于无论 `scale` 是多少，定点数为 0 当且仅当其整数尾数 `mantissa == 0`，该操作无需任何 `scale` 转换，直接在底层 `int64`
  尾数上做向量化零比较。

**示例**

```python
>>> z = Length("0.00", "m")
>>> bool(z.is_zero()), isinstance(z.is_zero(), np.bool_)
(True, True)
>>> z == 0  # 注意：带单位对象与裸数字 0 永远不相等！
False
>>> arr = Length(["0", "1.5", "-0.0"], "m")
>>> arr.is_zero()
array([ True, False,  True])
```

**性能对比**

|         N | UnitArray (`a.is_zero()`) | FixedArray (`fa.mantissa == 0`) | NumPy float64 (`x == 0`) |
|----------:|--------------------------:|--------------------------------:|-------------------------:|
|  0-d 标量 |                    448 ns |                          418 ns |                  34.4 ns |
|     1,000 |                    661 ns |                          644 ns |                   773 ns |
| 1,000,000 |                    127 µs |                          106 µs |                   101 µs |

**另请参阅**

- [`a == b` 与 `a != b`](#a--b-与-a--b相等性比较)
- [`bool(a)`](#lenaboola-与-itera容器与真值协议)

## 比较运算与量纲安检

### a == b 与 a != b（相等性比较）

```python
a == b
a != b
```

在量纲兼容的前提下对齐 `scale` 并逐元素比较是否相等；若类型或单位不兼容，则不抛异常而判定为不相等。

**参数**

- **b** : *Any* — 另一个比较对象。

**返回值**

- **out** : *numpy.bool_、numpy.ndarray（bool）或 bool* — 当 `b` 与 `a` 量纲兼容时，0 维返回 `np.bool_`，高维返回广播后的
  `bool` 数组；当 `b` 与 `a` 量纲不兼容或不是同子类实例时，返回 Python `False`（`==`）或 `True`（`!=`）。

**异常**

- **ValueError** — `a` 与 `b` 量纲兼容，但两个张量的形状无法按 NumPy 规则广播（例如长度为 2 的向量与长度为 3 的向量比较）。

**机制说明**

`__eq__` 与 `__ne__` 内部首先调用 [`self._compatible_or_none(other)`](#unitarray_compatible_or_none)：

1. 若返回 `False`（即 `_check_compatible(other)` 抛出了 `TypeError` 或 `ValueError`），`__eq__` / `__ne__` 直接返回
   `NotImplemented`。这带来两个关键特性：
    - **允许对方接管反向比较**：如果 `other` 是实现了自定义 `__eq__` 的对象，Python 会继续尝试调用 `other.__eq__(self)`；
    - **安全用于容器查找**：若对方也返回 `NotImplemented`，Python 最终将 `==` 折叠为 `False`、将 `!=` 折叠为 `True`，因此
      `Money("10", "USD") == Money("10", "CNY")`、`Money("10", "USD") == 10` 或 `x in [0, "abc", Money("10", "USD")]`
      都能安全工作而不会崩溃。
2. 若返回 `True`（量纲兼容），则委托底层的 `self._data == other._data`（或 `!=`），自动对齐 `scale` 后逐元素精确比较（如
   `Length("1.10", "m") == Length("1.1", "m")` 为 `True`）。

**示例**

同单位比较（自动对齐不同 `scale`）与广播：

```python
>>> bool(Length("1.10", "m") == Length("1.1", "m"))
True
>>> Length(["1", "2"], "m") == Length(["1.0", "3.0"], "m")
array([ True, False])
>>> Length(["1", "2"], "m") != Length(["1.0", "3.0"], "m")
array([False,  True])
```

跨单位、跨类型比较与容器成员检测（永不因量纲不兼容报错）：

```python
>>> Length("1", "m") == Length("1", "km")
False
>>> Length("1", "m") != Length("1", "km")
True
>>> Length("1", "m") == 1, Length("1", "m") == FixedArray("1")
(False, False)
>>> Length("2", "m") in [0, "x", Length("2", "km"), Length("2.00", "m")]
True
```

同单位但形状不兼容属于真实维度错误，会抛出 `ValueError`：

```python
>>> Length(["1", "2"], "m") == Length(["1", "2", "3"], "m")
Traceback (most recent call last):
    ...
ValueError: operands could not be broadcast together with shapes (2,) (3,)
```

**性能对比**

|         N | UnitArray (`a == b`) | FixedArray (`fa1 == fa2`) | NumPy float64 (`x == y`) |
|----------:|---------------------:|--------------------------:|-------------------------:|
|  0-d 标量 |               743 ns |                    603 ns |                  31.5 ns |
|     1,000 |               880 ns |                    745 ns |                   426 ns |
| 1,000,000 |               274 µs |                    270 µs |                   272 µs |

**另请参阅**

- [`a < b`、`a <= b`、`a > b`、`a >= b`](#a--ba--ba--ba--b大小比较)
- [`UnitArray.is_zero`](#unitarrayis_zero)
- [`UnitArray._compatible_or_none`](#unitarray_compatible_or_none)

### a < b、a <= b、a > b、a >= b（大小比较）

```python
a < b
a <= b
a > b
a >= b
```

在严格校验量纲兼容后，对齐 `scale` 并逐元素比较大小。

**参数**

- **b** : *UnitArray* — 必须与 `a` 同属一个具体子类且单位/量纲完全兼容，可广播。

**返回值**

- **out** : *numpy.bool_ 或 numpy.ndarray（bool）* — 0 维标量返回 `np.bool_`，高维数组返回广播后的 `bool` 数组。

**异常**

- **TypeError** — `b` 是单位不兼容的 `UnitArray`（由 `self._check_compatible(b)` 抛出明确原因），或 `b` 不是 `UnitArray`
  且不支持反向比较（由 Python 抛出 `TypeError`）。
- **ValueError** — 形状无法广播。
- **OverflowError** — 对齐 `scale` 时尾数超出 `int64` 范围。

**机制说明**

四个有序比较运算符统一转发给内部方法 [`self._ordered(other, op)`](#unitarray_ordered)：

1. 若 `not isinstance(other, UnitArray)`，返回 `NotImplemented`（允许自定义对象响应反向比较，否则由 Python 报
   `TypeError: '<' not supported between instances of ...`）；
2. 若 `other` 是 `UnitArray`，调用 `self._check_compatible(other)`：一旦单位或子类不匹配（例如拿 `USD` 与 `CNY` 比大小，或拿
   `"m"` 与 `"km"` 比大小），立即抛出子类的 `TypeError`；
3. 安检通过后，在底层 `FixedArray` 上执行 `op(self._data, other._data)`。

**示例**

```python
>>> a, b = Length(["1", "2"], "m"), Length(["2", "1"], "m")
>>> (a < b).tolist(), (a >= b).tolist()
([True, False], [False, True])
>>> min(Length("3", "m"), Length("2", "m"))
Length(FixedArray(2, scale=0), unit='m')
```

不同单位或与裸数字比大小时立即拦截：

```python
>>> Length("1", "m") < Length("1", "km")
Traceback (most recent call last):
    ...
TypeError: 单位不匹配: m 与 km
>>> Length("1", "m") < 0
Traceback (most recent call last):
    ...
TypeError: '<' not supported between instances of 'Length' and 'int'
```

**性能对比**

|         N | UnitArray (`a < b`) | FixedArray (`fa1 < fa2`) | NumPy float64 (`x < y`) |
|----------:|--------------------:|-------------------------:|------------------------:|
|  0-d 标量 |              827 ns |                   528 ns |                 31.5 ns |
|     1,000 |              970 ns |                   663 ns |                  418 ns |
| 1,000,000 |              270 µs |                   263 µs |                  265 µs |

**另请参阅**

- [`a == b` 与 `a != b`](#a--b-与-a--b相等性比较)
- [`UnitArray._ordered`](#unitarray_ordered)
- [`FixedArray` 大小比较](fixed_array.md#a--ba--ba--ba--b大小比较)

## 形状变换、索引与容器协议

### UnitArray.reshape

```python
UnitArray.reshape(*shape)
```

在不改变数值与单位的前提下改变张量形状。

**参数**

- **\*shape** : *int 或 tuple of int* — 新形状，既可传多个整数参数 `a.reshape(2, 3)`，也可传单个元组 `a.reshape((2, 3))`
  ；支持将其中一个维度设为 `-1` 自动推断。

**返回值**

- **out** : *UnitArray* — 与 `self` 同类型、同单位的新形状实例（底层尽量返回零拷贝视图）。

**异常**

- **ValueError** — 新形状的元素总数与 `self.size` 不一致。

**机制说明**

执行 `self._create_like(self._data.reshape(*shape))`。

**示例**

```python
>>> a = Length(["1", "2", "3", "4"], "km").reshape(2, 2)
>>> a
Length(FixedArray([[1, 2],
                   [3, 4]], scale=0), unit='km')
>>> a.reshape(-1)
Length(FixedArray([1, 2, 3, 4], scale=0), unit='km')
```

**另请参阅**

- [`UnitArray.transpose`](#unitarraytranspose)、[`UnitArray.T`](#unitarrayt)
- [`UnitArray.shape`](#unitarrayshape)
- [`FixedArray.reshape`](fixed_array.md#fixedarrayreshape)

### UnitArray.transpose

```python
UnitArray.transpose(*axes)
```

按指定顺序置换张量的各个轴，结果保持原单位。

**参数**

- **\*axes** : *int，可选* — 轴的新排列顺序；不传参时反转全部轴（等价于 [`.T`](#unitarrayt)）。

**返回值**

- **out** : *UnitArray* — 与 `self` 同类型、同单位的转置视图。

**机制说明**

执行 `self._create_like(self._data.transpose(*axes))`，零拷贝共享底层尾数。

**示例**

```python
>>> a = Length(["1", "2", "3", "4", "5", "6"], "km").reshape(2, 3)
>>> a.transpose(1, 0)
Length(FixedArray([[1, 4],
                   [2, 5],
                   [3, 6]], scale=0), unit='km')
```

**另请参阅**

- [`UnitArray.T`](#unitarrayt)
- [`UnitArray.reshape`](#unitarrayreshape)
- [`FixedArray.transpose`](fixed_array.md#fixedarraytranspose)

### `UnitArray.swapaxes` 与 `UnitArray.squeeze`

```python
UnitArray.swapaxes(axis1, axis2)
UnitArray.squeeze(axis=None)
```

- **`swapaxes(axis1, axis2)`**：交换两个轴并保持原单位（委托 [
  `FixedArray.swapaxes`](fixed_array.md#fixedarrayswapaxes-与-fixedarraysqueeze)）。
- **`squeeze(axis=None)`**：移除长度为 1 的轴并保持原单位（委托 [
  `FixedArray.squeeze`](fixed_array.md#fixedarrayswapaxes-与-fixedarraysqueeze)）。

**示例**

```python
>>> a = Money(np.arange(6).reshape(1, 2, 3), "USD")
>>> a.swapaxes(0, 2).shape, a.squeeze().shape
((3, 2, 1), (2, 3))
```

### `a[key]`（索引与切片）

```python
a[key]
```

按整数下标、切片、元组索引、布尔掩码或整数花式索引提取子张量，结果始终保持原子类类型与原单位。

**参数**

- **key** : *int, slice, tuple, ndarray（bool 或整数）* — 支持 NumPy 的全部标准索引方式。

**返回值**

- **out** : *UnitArray* — 与 `a` 同类型、同单位的标量（0 维 `UnitArray`）或子张量。注意：即使是取单个元素（如 `a[0]`
  ），返回的也是携带原单位的 0 维 `UnitArray` 子类实例，而不是裸 Python 数字，从而确保单位信息绝不在索引时丢失。

**异常**

- **IndexError** — 下标越界、维度超出 `ndim` 或布尔掩码形状不匹配。

**机制说明**

执行 `self._create_like(self._data[key])`。基础切片与整数索引共享底层 `FixedArray` 的只读尾数视图（零拷贝）；布尔掩码与花式索引返回拷贝。

**示例**

```python
>>> a = Length(["10", "20", "30"], "km")
>>> a[1]  # 单元素索引返回 0 维同单位对象
Length(FixedArray(20, scale=0), unit='km')
>>> a[1:]
Length(FixedArray([20, 30], scale=0), unit='km')
>>> a[a > Length("15", "km")]  # 布尔掩码过滤
Length(FixedArray([20, 30], scale=0), unit='km')
```

**另请参阅**

- [`len(a)`、`bool(a)` 与 `iter(a)`](#lenaboola-与-itera容器与真值协议)
- [`FixedArray` 索引与切片](fixed_array.md#akey索引与切片)

### `len(a)`、`bool(a)` 与 `iter(a)`（容器与真值协议）

```python
len(a)
bool(a)
iter(a)
```

首维长度查询、单元素真值测试与按首维保单位迭代。

**返回值**

- **`len(a)`** : *int* — 首维长度 `shape[0]`。
- **`bool(a)`** : *bool* — 单元素张量（`size == 1`）非零时为 `True`，为零时为 `False`。
- **`iter(a)`** : *Iterator[UnitArray]* — 沿首维逐项产出保持原单位的 `UnitArray` 子类实例。

**异常**

- **TypeError** — 对 0 维标量张量调用 `len(a)` 或 `iter(a)`（报错信息包含具体子类名称，如 `iteration over a 0-d Money`）。
- **ValueError** — 对 `size != 1`（空数组或多元素数组）调用 `bool(a)`（遵循 NumPy 真值防歧义约定：
  `The truth value of an array with more than one element is ambiguous`）。

**机制说明**

- `__len__` 与 `__bool__` 直接委托 `len(self._data)` 与 `bool(self._data)`；
- `__iter__` 先检查 `if self.ndim == 0: raise TypeError(f"iteration over a 0-d {type(self).__name__}")`，随后返回生成器表达式
  `(self._create_like(item) for item in self._data)`，确保遍历出的每一个子元素都携带原单位。

**示例**

```python
>>> mat = Length([["1", "2"], ["3", "4"]], "km")
>>> len(mat)
2
>>> list(mat)
[Length(FixedArray([1, 2], scale=0), unit='km'), Length(FixedArray([3, 4], scale=0), unit='km')]
>>> bool(Length("0", "km")), bool(Length("0.5", "km"))
(False, True)
```

0 维标量禁止 `len()` 与 `iter()`，多元素数组禁止 `bool()`：

```python
>>> list(Length("5", "km"))
Traceback (most recent call last):
    ...
TypeError: iteration over a 0-d Length
>>> bool(Length(["0", "0"], "km"))
Traceback (most recent call last):
    ...
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

**另请参阅**

- [`a[key]`](#akey索引与切片)
- [`UnitArray.is_zero`](#unitarrayis_zero)
- [`FixedArray` 容器与真值协议](fixed_array.md#lenaboola-与-itera容器与真值协议)

## 导出与互操作

### UnitArray.to_float

```python
UnitArray.to_float()
```

剥离单位标签，将定点数数值转换为 Python `float`（0 维时）或 `numpy.ndarray(dtype=float64)`（1 维及以上）。

**返回值**

- **out** : *float 或 numpy.ndarray（float64）*

**机制说明**

直接委托 [`self._data.to_float()`](fixed_array.md#fixedarrayto_float)。注意转换为二进制浮点数是有损操作（例如超过 $2^{53}$
的整数尾数或 `0.1` 的十进制精确性在浮点运算中会退化），仅建议在绘图或对接外部浮点统计库时显式调用。

**示例**

```python
>>> Length("1.5", "m").to_float()
1.5
>>> Length(["0.1", "0.25"], "m").to_float()
array([0.1 , 0.25])
```

**另请参阅**

- [`UnitArray.to_decimal`](#unitarrayto_decimal)
- [`UnitArray.__array__` 与 NumPy 防护机制](#unitarrayarray-与-numpy-防护机制)
- [`FixedArray.to_float`](fixed_array.md#fixedarrayto_float)

### UnitArray.to_decimal

```python
UnitArray.to_decimal()
```

剥离单位标签，无损转换为标准库 `decimal.Decimal`（0 维时）或元素为 `Decimal` 的 `object` 数组（1 维及以上），并严格保留
`scale` 尾随零。

**返回值**

- **out** : *Decimal 或 numpy.ndarray（object）*

**机制说明**

直接委托 [`self._data.to_decimal()`](fixed_array.md#fixedarrayto_decimal)。

**示例**

```python
>>> Length("1.50", "m").to_decimal()
Decimal('1.50')
>>> Length(["0.1", "0.25"], "m").to_decimal()
array([Decimal('0.10'), Decimal('0.25')], dtype=object)
```

**另请参阅**

- [`UnitArray.to_float`](#unitarrayto_float)
- [`FixedArray.to_decimal`](fixed_array.md#fixedarrayto_decimal)

### UnitArray.to_pandas

```python
UnitArray.to_pandas(index=None, name=None, as_float=True)
```

将 1 维或 2 维 `UnitArray` 导出为 `pandas.Series` 或 `pandas.DataFrame`。

**参数**

- **index** : *array-like 或 Index，可选* — 行索引。
- **name** : *str，可选* — `Series` 的名称（仅 1 维生效）。
- **as_float** : *bool，默认 `True`* — 为 `True` 时导出为 `float64` 列；为 `False` 时导出为无损 `Decimal`（`object`）列。

**返回值**

- **out** : *pandas.Series 或 pandas.DataFrame*（子类可在此基础上附加元数据，如 [
  `Money.to_pandas`](money.md#moneyto_pandas) 会自动写入 `res.attrs["currency"] = self.currency.code`）。

**异常**

- **ValueError** — `ndim` 不是 1 或 2。
- **ImportError** — 当前环境未安装 `pandas`。

**机制说明**

基类直接委托 [`self._data.to_pandas(index=index, name=name, as_float=as_float)`](fixed_array.md#fixedarrayto_pandas)。

**示例**

```python
>>> s = Length(["1.5", "2.0"], "m").to_pandas(index=["a", "b"], name="len")
>>> s.to_dict()
{'a': 1.5, 'b': 2.0}
>>> Length(["1.50"], "m").to_pandas(as_float=False).tolist()
[Decimal('1.50')]
```

**另请参阅**

- [`UnitArray.to_polars`](#unitarrayto_polars)
- [`Money.to_pandas`](money.md#moneyto_pandas)
- [`FixedArray.to_pandas`](fixed_array.md#fixedarrayto_pandas)

### UnitArray.to_polars

```python
UnitArray.to_polars(name="value", as_float=True)
```

将 1 维 `UnitArray` 导出为 `polars.Series`。

**参数**

- **name** : *str，默认 `"value"`* — `Series` 的列名。
- **as_float** : *bool，默认 `True`* — 为 `True` 时导出为 `Float64`；为 `False` 时导出为 `Decimal(scale=self.scale)`。

**返回值**

- **out** : *polars.Series*

**异常**

- **ValueError** — `ndim > 1`。
- **ImportError** — 当前环境未安装 `polars`。

**机制说明**

直接委托 [`self._data.to_polars(name=name, as_float=as_float)`](fixed_array.md#fixedarrayto_polars)。

**示例**

```python
>>> s = Length(["1.50", "2.25"], "m").to_polars(name="len")  # doctest: +SKIP
```

**另请参阅**

- [`UnitArray.to_pandas`](#unitarrayto_pandas)
- [`FixedArray.to_polars`](fixed_array.md#fixedarrayto_polars)

### `UnitArray.__array__` 与 NumPy 防护机制

```python
UnitArray.__array__(dtype=None, copy=None)
UnitArray.__array_function__(func, types, args, kwargs)
UnitArray.__array_ufunc__ = None
```

显式拦截 `np.asarray(a)`、`np.array(a)` 等隐式数组转换以及所有 NumPy `ufunc` 调用，同时通过 `__array_function__`
仅放行白名单内的安全 NumPy 函数。

**参数**

- **dtype / copy** : *Any，默认 `None`* — 兼容 NumPy `__array__` 协议签名。

**异常**

- **TypeError** — 任何调用 `__array__`、NumPy `ufunc` 或非白名单 NumPy 函数（如 `np.concatenate`、`np.max`）的操作均无条件抛出
  `TypeError`。

**机制说明**

如果允许 `np.asarray(a)`、`np.add(a, b)` 或未受控的 NumPy 高阶函数退回默认行为，NumPy 会将 `UnitArray` 拆解为
`dtype=object` 数组或静默丢失单位。为此，`UnitArray` 采取三道防护：

1. **`__array_ufunc__ = None`**：阻止一切 `np.add`、`np.multiply`、`np.sqrt` 以及左操作数为 `ndarray`
   的算术运算（除非子类自身专门实现了对应的反向运算）；
2. **`__array__` 抛 `TypeError`**：阻止 `np.asarray(a)` 等隐式转换，并在报错消息中明确提示子类名与替代方案；
3. **`__array_function__` 白名单分派**：调用 [
   `FixedArray._array_function`](fixed_array.md#fixedarray_array_function-与-fixedarray_numpy_equal)，仅允许 `np.sum`、
   `np.mean`、`np.transpose`、`np.shape`、`np.ndim`、`np.size`、`np.array_equal` 与 `np.array_equiv`，其余函数一律抛出
   `TypeError`。

**示例**

```python
>>> a = Length(["1", "2"], "m")
>>> np.asarray(a)
Traceback (most recent call last):
    ...
TypeError: Length 不能隐式转成数组：要 float 请用 to_float()，要整数尾数请用 .mantissa
>>> np.sqrt(a)
Traceback (most recent call last):
    ...
TypeError: operand 'Length' does not support ufuncs (__array_ufunc__=None)
>>> np.add(a, a)
Traceback (most recent call last):
    ...
TypeError: operand 'Length' does not support ufuncs (__array_ufunc__=None)
>>> np.array_equal(a, Length(["1.0", "2.0"], "m"))
True
```

**另请参阅**

- [`UnitArray.to_float`](#unitarrayto_float)
- [`UnitArray.mantissa`](#unitarraymantissa)
- [`FixedArray` 的 NumPy 互操作机制](fixed_array.md#numpy-互操作机制)

## 内部辅助方法（私有 API）

### UnitArray._as_factor

```python
@staticmethod
UnitArray._as_factor(other, what)
```

校验 `other` 是否属于无量纲数值操作数白名单，并将其转换为 [`FixedArray`](fixed_array.md)。

**参数**

- **other** : *Any* — 待校验的无量纲操作数。
- **what** : *str* — 用于拼装错误提示的角色名称（如 `"乘数"`、`"除数"`、`"步长"`、`"权重"`）。

**返回值**

- **out** : *FixedArray*

**异常**

- **TypeError** — `FixedArray._is_numeric_operand(other)` 为 `False`（拒绝字符串、布尔值与带单位的 `UnitArray` 对象）。

**机制说明**

内部直接调用 [`FixedArray._coerce(other, strict=False)`](fixed_array.md#fixedarray_coerce)：若返回 `None`
（不在数值白名单内），则抛出带有业务角色名 `what` 的 `TypeError`；否则返回转换后的 `FixedArray`。供 [`Money`](money.md)、[
`Price`](price.md)、[`Quantity`](quantity.md) 等子类的乘除与步长对齐方法复用。

**示例**

```python
>>> UnitArray._as_factor(1.5, "乘数")
FixedArray(1.5, scale=1)
>>> UnitArray._as_factor("1.5", "乘数")
Traceback (most recent call last):
    ...
TypeError: 乘数只接受数值（int、float、Decimal、Fraction、FixedArray
或数值数组），收到
str
```

**另请参阅**

- [`FixedArray._is_numeric_operand`](fixed_array.md#fixedarray_is_numeric_operand)
- [`FixedArray._coerce`](fixed_array.md#fixedarray_coerce)

### `UnitArray._as_grid` 与 `UnitArray._fill_value`

```python
@classmethod


UnitArray._as_grid(step, what)

UnitArray._fill_value(value)
```

- **`_as_grid(step, what)`**：供步长 `step`、`tick_size` 等网格参数复用的校验方法——先调用 [
  `_as_factor(step, what)`](#unitarray_as_factor) 转为 `FixedArray`，再检查 `grid.mask is not None`，若含缺失值则抛出
  `ValueError(f"{what}不能是缺失值")`。
- **`_fill_value(value)`**：将 [`fillna(value)`](#unitarrayfillna) 传入的无量纲数值填充值换算为 `FixedArray` 的钩子。基类默认直接调用
  `self._as_factor(value, "填充值")`；具有固定精度的子类（如 [`Money._fill_value`](money.md#money_fill_value)
  ）会覆盖此方法，按子类构造规则换算精度。

**示例**

```python
>>> UnitArray._as_grid("0.05", "步长")
Traceback (most recent call last):
    ...
TypeError: 步长只接受数值（int、float、Decimal、Fraction、FixedArray 或数值数组），收到 str
>>> UnitArray._as_grid(FixedArray(None), "步长")
Traceback (most recent call last):
    ...
ValueError: 步长不能是缺失值
```

### UnitArray._compatible_or_none

```python
UnitArray._compatible_or_none(other)
```

以无异常布尔值形式判定 `other` 是否与 `self` 量纲兼容。

**参数**

- **other** : *Any* — 待检查的对象。

**返回值**

- **out** : *bool* — `self._check_compatible(other)` 正常返回时为 `True`；抛出 `TypeError` 或 `ValueError` 时捕获并返回
  `False`。

**机制说明**

专供 [`__eq__` 与 `__ne__`](#a--b-与-a--b相等性比较) 调用：将子类钩子 `_check_compatible` 的异常信号转换为 `False`，使
`__eq__` / `__ne__` 在遇到跨量纲或非 `UnitArray` 对象时能干净地返回 `NotImplemented`。

**示例**

```python
>>> a = Length("1", "m")
>>> a._compatible_or_none(Length("2", "m"))
True
>>> a._compatible_or_none(Length("2", "km")), a._compatible_or_none(1)
(False, False)
```

**另请参阅**

- [`UnitArray._check_compatible`](#unitarray_check_compatible)
- [`a == b` 与 `a != b`](#a--b-与-a--b相等性比较)

### UnitArray._ordered

```python
UnitArray._ordered(other, op)
```

执行有序大小比较（`<`、`<=`、`>`、`>=`）的统一类型过滤、量纲安检与底层分发。

**参数**

- **other** : *Any* — 右操作数。
- **op** : *Callable[[FixedArray, FixedArray], Any]* — 底层 `FixedArray` 上的二元比较函数（如 `lambda a, b: a < b`）。

**返回值**

- **out** : *numpy.bool_、numpy.ndarray（bool）或 NotImplemented*

**异常**

- **TypeError** — 当 `other` 是 `UnitArray` 实例但未通过 `self._check_compatible(other)` 时抛出。

**机制说明**

```python
if not isinstance(other, UnitArray):
    return NotImplemented  # 非 UnitArray：交由 Python 尝试 other 的反向比较或报 TypeError
self._check_compatible(other)  # 单位不兼容的 UnitArray：抛出子类定义的明确 TypeError
return op(self._data, other._data)  # 单位兼容：委托底层 FixedArray 对齐 scale 后比较
```

**示例**

```python
>>> a = Length("1", "m")
>>> bool(a._ordered(Length("2", "m"), lambda x, y: x < y))
True
>>> a._ordered(10, lambda x, y: x < y)
NotImplemented
>>> a._ordered(Length("2", "km"), lambda x, y: x < y)
Traceback (most recent call last):
    ...
TypeError: 单位不匹配: m
与
km
```

**另请参阅**

- [`UnitArray._check_compatible`](#unitarray_check_compatible)
- [`a < b`、`a <= b`、`a > b`、`a >= b`](#a--ba--ba--ba--b大小比较)
