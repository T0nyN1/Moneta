# API 参考

[← 文档首页](../index.md)

本参考手册按模块组织 Moneta 的公开 API。所有公开类与函数均可直接从顶层包导入（如
`from moneta import FixedArray, RoundingMode`）。

## 数值内核

| 模块                                   | 公开接口                                                                      | 说明                                                 |
|----------------------------------------|-------------------------------------------------------------------------------|------------------------------------------------------|
| [`moneta.fixed_array`](fixed_array.md) | [`FixedArray`](fixed_array.md)、[`RoundingMode`](fixed_array.md#roundingmode) | 基于 `int64` 尾数的定点数 N 维张量与六种金融舍入模式 |

## 量纲与币种

| 模块                                 | 公开接口                                                                                                                                                                                                                                                             | 说明                                                                  |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`moneta.unit_array`](unit_array.md) | [`UnitArray`](unit_array.md#unitarray)、[`PrecisionLossWarning`](unit_array.md#precisionlosswarning)                                                                                                                                                                 | 带单位的定点数张量抽象基类，统一提供保单位变换与跨量纲比较安检        |
| [`moneta.currency`](currency.md)     | [`Currency`](currency.md#currencycode-digits-symbol-name)、[`CurrencyRegistry`](currency.md#currencyregistry)、[`get_currency`](currency.md#get_currency)、[`CNY` `USD` `EUR` `JPY` `GBP` `HKD` `AUD` `CAD` `CHF` `SGD`](currency.md#usdcnyeurjpygbphkdaudcadchfsgd) | 币种定义（ISO 代码、原子精度 `digits`）、享元注册表与内置常用币种常量 |

## 金额、价格与数量

| 模块                             | 公开接口                                                                                                   | 说明                                                                               |
|----------------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`moneta.money`](money.md)       | [`Money`](money.md#moneyvalues-currency)、[`PrecisionLossWarning`](money.md#precisionlosswarning)          | 绑定单一币种的金额张量（`scale` 恒等于币种精度）、守恒分配与阶梯费率计算           |
| [`moneta.price`](price.md)       | [`Price`](price.md#pricevalues-quote_currency-base_unitshare-tick_sizenone-scalenone)                      | 单位标的报价张量（`[计价货币/基准单位]`），支持 `tick_size` 报价网格与成交金额计算 |
| [`moneta.quantity`](quantity.md) | [`Quantity`](quantity.md#quantityvalues-unitshare-scalenone)、[`WILDCARD_UNIT`](quantity.md#wildcard_unit) | 份额、手数、合约数等数量张量（默认精确运算，支持通配单位 `"Any"` 与显式步长取整）  |

## 标的、持仓与组合

| 模块                | 公开接口                         | 说明                                          |
|---------------------|----------------------------------|-----------------------------------------------|
| `moneta.instrument` | `Instrument`                     | 金融标的参考数据与持仓头寸表示                |
| `moneta.lot`        | `PositionLot`、`LotMatchingRule` | 持仓成本批次管理与 FIFO / LIFO / WAC 结转规则 |
| `moneta.portfolio`  | `Portfolio`                      | 多币种、多资产组合容器、算术合并与净值重估    |

## 外汇与汇率引擎

| 模块        | 公开接口                                                                                                                                                                                                                                                      | 说明                                                                                   |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `moneta.fx` | `ExchangeRate`、`ExchangeRateNotFoundError`、`FXProvider`、`UserDictFXProvider`、`SnapshotFXProvider`、`FrankfurterFXProvider`、`CompositeFXProvider`、`FXEngine`、`LiveFXWatcher`、`convert_currency`、`decompose_fx_pnl`、`covered_interest_parity_forward` | 货币对汇率、多源汇率责任链、张量批量换汇、实时盯市重估、汇兑损益分解与抛补利率平价远期 |

## 现金流、计日惯例与货币时间价值

| 模块                 | 公开接口                                  | 说明                                                           |
|----------------------|-------------------------------------------|----------------------------------------------------------------|
| `moneta.conventions` | `DayCountConvention`、`year_fraction`     | 金融计日惯例（ACT/360、ACT/365F、30/360 等）与年化时间跨度计算 |
| `moneta.cashflow`    | `CashFlow`、`generate_cash_flow_schedule` | 未来现金流定义与定期付息/摊还排程生成                          |
| `moneta.tvm`         | `pv`、`fv`、`npv`、`discount_factors`     | 现值、终值、净现值与折现因子张量计算                           |

## 格式化输出

| 模块               | 公开接口            | 说明                                               |
|--------------------|---------------------|----------------------------------------------------|
| `moneta.formatter` | `MonetaryFormatter` | 千分位分隔、货币符号、会计负数括号等金融金额格式化 |
