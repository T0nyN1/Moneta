# _units — 物理单位规范与兼容性工具模块

`moneta._units` 定义金融资产数量与标价中使用的非货币物理单位规范（[`check_unit`](#check_unit)）、通配单位常量（[
`WILDCARD_UNIT`](#wildcard_unit)）以及对称的单位兼容性判定函数（[`units_compatible`](#units_compatible)），为 [
`Quantity`](quantity.md)、[`Price`](price.md)、[`Instrument`](instrument.md) 与 `PositionLot`
提供全库一致的单位校验与通配匹配规则。其中 [`WILDCARD_UNIT`](#wildcard_unit) 同时由 [
`moneta.quantity`](quantity.md#wildcard_unit) 重导出。

```python
>>> from moneta._units import WILDCARD_UNIT, check_unit, units_compatible
>>> WILDCARD_UNIT
'Any'
>>> check_unit("  barrel \t\n"), check_unit("BTC")
('barrel', 'BTC')
>>> units_compatible("share", "share"), units_compatible("Any", "barrel"), units_compatible("share", "barrel")
(True, True, False)
```

[← API 参考](index.md)

## 核心机制与设计约束

### 1. 单位字符串规范（去首尾空白、非空、大小写敏感）

与受 ISO 4217 三位大写字母严格约束的货币代码（[`Currency`](currency.md)）不同，标的资产的物理或合约度量单位（如股票股数
`"share"`、原油桶数 `"barrel"`、期货手数 `"contract"`、贵金属金衡盎司 `"oz"`、加密货币 `"BTC"` 或中文 `"手"` / `"克"`
）开放给用户自由命名，仅由 [`check_unit`](#check_unit) 施加三条基础不变量：

1. **类型必须为 `str`**：拒绝 `None`、整数、浮点数、`bytes` 或容器，防止位置参数错位时静默转字符串；
2. **去除首尾空白后非空**：自动执行 `unit.strip()`，若结果为空字符串 `""` 则抛出 `ValueError("单位不能为空")`；
3. **严格区分大小写**：不做大小写折叠（例如 `"BTC"` 与 `"btc"`、`"share"` 与 `"Share"`、`"Any"` 与 `"any"` 均视为不同单位）。

### 2. 通配单位 `"Any"`（`WILDCARD_UNIT`）与跨组件兼容语义

[`WILDCARD_UNIT = "Any"`](#wildcard_unit) 是全库唯一的通配单位字面量。[`units_compatible(u1, u2)`](#units_compatible)
采用对称判定规则：当且仅当 `u1 == u2` 或其中至少一方等于 `"Any"` 时返回 `True`。各上层业务模块在此规则之上共享统一的校验与收敛约定：

| 业务组件                      | 构造时的单位校验                                                                                                                                                             | 二元运算 / 跨组件交互中的兼容与收敛规则                                                                                                                                                                                |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`Quantity`](quantity.md)     | 调用 [`check_unit(unit)`](#check_unit)（允许 `"Any"`，默认 `"share"`）；从已有 `Quantity` 构造时用 [`units_compatible`](#units_compatible) 校验并可将 `"Any"` 窄化为具体单位 | 加减、比率除法、比较通过 [`units_compatible`](#units_compatible) 判兼容；加减结果单位通过 [`_resolve_unit`](quantity.md#quantity_resolve_unit) 收敛为非 `"Any"` 的具体单位                                             |
| [`Price`](price.md)           | 调用 [`check_unit(base_unit)`](#check_unit)（允许 `"Any"`，默认 `"share"`）；从已有 `Price` 构造时用 [`units_compatible`](#units_compatible) 校验                            | 同币种 `Price` 之间的加减、比率除法、比较以及 `Price * Quantity` 均通过 [`units_compatible`](#units_compatible) 判兼容；加减结果 `base_unit` 收敛为具体单位                                                            |
| [`Instrument`](instrument.md) | 调用 [`check_unit(base_unit)`](#check_unit)，但**额外禁止 `base_unit == "Any"`**（具体金融合约必须绑定明确的物理或合约单位）                                                 | [`as_price`](instrument.md#instrumentas_price) 与 [`calc_notional`](instrument.md#instrumentcalc_notional) 通过 [`units_compatible`](#units_compatible) 接受单位为 `self.base_unit` 或 `"Any"` 的 `Price` / `Quantity` |
| `PositionLot`                 | 证券批次要求持仓 `Quantity` 的单位与 `asset.base_unit` 满足 [`units_compatible`](#units_compatible)，并统一归一化存储为 `asset.base_unit`                                    | `split` 与 `match_lots` 扣减数量时同样通过 [`units_compatible`](#units_compatible) 校验并归一化到批次基准单位                                                                                                          |

## API 速查表

| 分类               | 接口 / 常量                             | 功能说明                                                    |
|--------------------|-----------------------------------------|-------------------------------------------------------------|
| **模块常量**       | [`WILDCARD_UNIT`](#wildcard_unit)       | 通配单位常量 `"Any"`，可与任意具体标的单位兼容匹配          |
| **校验与判定函数** | [`check_unit`](#check_unit)             | 校验单位参数为非空 `str` 并返回去除首尾空白后的规范化字符串 |
|                    | [`units_compatible`](#units_compatible) | 对称判断两个单位字符串是否相同或至少一方为通配单位 `"Any"`  |

## 模块常量

### `WILDCARD_UNIT`

```python
moneta._units.WILDCARD_UNIT = 'Any'
```

通配单位字符串常量 `"Any"`（同时由 [`moneta.quantity.WILDCARD_UNIT`](quantity.md#wildcard_unit) 重导出）。携带 `"Any"`
单位的 [`Quantity`](quantity.md) 或 [`Price`](price.md) 可与任意具体单位的同类对象或对应标的对象兼容运算，并在加减法或批次入账后自动收敛为具体单位。注意仅精确大写的
`"Any"` 具有通配语义，小写的 `"any"` 会被视为普通具体单位名称。

**示例**

```python
>>> from moneta import Price, Quantity
>>> from moneta._units import WILDCARD_UNIT
>>> from moneta.quantity import WILDCARD_UNIT as QTY_WILDCARD
>>> WILDCARD_UNIT == "Any" and WILDCARD_UNIT is QTY_WILDCARD
True
>>> (Quantity(3, unit=WILDCARD_UNIT) + Quantity(2, unit="barrel")).unit
'barrel'
>>> Price("80.00", "USD", base_unit=WILDCARD_UNIT) * Quantity(5, unit="barrel")
Money(FixedArray(400.00, scale=2), currency='USD')
```

**另请参阅**

- [`units_compatible`](#units_compatible)
- [`Quantity`](quantity.md#wildcard_unit)

## 校验与判定函数

### `check_unit`

```python
moneta._units.check_unit(unit)
```

校验标的单位参数是否为非空字符串，并返回去除首尾空白字符后的规范化单位字符串。

**参数**

- **unit** : *Any* — 待校验的单位名称（如 `"share"`、`"  barrel "`、`"BTC"`、`"Any"`、`"合约"`）。

**返回值**

- **out** : *str* — 去除首尾空白（`unit.strip()`）后的非空字符串，保留原始大小写。

**异常**

- **TypeError** — `not isinstance(unit, str)`（报 `f"单位必须是 str，收到 {type(unit).__name__}"`）。
- **ValueError** — `unit.strip()` 为空字符串 `""`（报 `"单位不能为空"`）。

**机制说明**

1. 先检查 `isinstance(unit, str)`，不满足立即抛出 `TypeError`；
2. 执行 `unit = unit.strip()` 剥离首尾空格、制表符与换行符；
3. 若剥离后为空字符串，抛出 `ValueError("单位不能为空")`；否则返回规范化后的 `unit`。

**示例**

```python
>>> from moneta._units import check_unit
>>> check_unit("  barrel \t\n"), check_unit("BTC"), check_unit("btc"), check_unit("合约")
('barrel', 'BTC', 'btc', '合约')
>>> check_unit("   ")
Traceback (most recent call last):
    ...
ValueError: 单位不能为空
>>> check_unit(123)
Traceback (most recent call last):
    ...
TypeError: 单位必须是 str，收到 int
```

**另请参阅**

- [`units_compatible`](#units_compatible)
- [`Quantity`](quantity.md#quantityvalues-unitshare-scalenone)
- [`Price`](price.md#pricevalues-quote_currency-base_unitshare-tick_sizenone-scalenone)
- [
  `Instrument`](instrument.md#instrumentsymbol-quote_currency-base_unitshare-multiplier1-tick_sizenone-name-asset_classequity)

### `units_compatible`

```python
moneta._units.units_compatible(u1, u2)
```

判断两个已规范化的单位字符串 `u1` 与 `u2` 是否量纲兼容：当 `u1 == u2` 或至少一方为通配单位 [`WILDCARD_UNIT`（
`"Any"`）](#wildcard_unit) 时返回 `True`，否则返回 `False`。

**参数**

- **u1, u2** : *str* — 待比较的两个单位字符串。

**返回值**

- **compatible** : *bool* — 等价于 `u1 == u2 or u1 == "Any" or u2 == "Any"`。满足自反性与交换律（对称性），但由于 `"Any"`
  的存在 **不具有传递性**（例如 `"barrel"` 与 `"Any"` 兼容、`"Any"` 与 `"share"` 兼容，但 `"barrel"` 与 `"share"` 不兼容）。

**示例**

```python
>>> from moneta._units import units_compatible
>>> units_compatible("share", "share"), units_compatible("barrel", "Any"), units_compatible("Any", "Any")
(True, True, True)
>>> units_compatible("share", "barrel"), units_compatible("share", "Share"), units_compatible("any", "share")
(False, False, False)
```

**另请参阅**

- [`WILDCARD_UNIT`](#wildcard_unit)
- [`check_unit`](#check_unit)
- [`Quantity._check_compatible`](quantity.md#quantity_check_compatible)
- [`Price._check_compatible`](price.md#price_check_compatible)
