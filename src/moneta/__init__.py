"""Moneta：基于 NumPy 的高性能定点数多币种金额与汇率张量引擎。"""

__version__ = "0.1.0"

from .currency import (
    Currency,
    CurrencyRegistry,
    get_currency,
    CNY,
    USD,
    EUR,
    JPY,
    GBP,
    HKD,
    AUD,
    CAD,
    CHF,
    SGD,
)
from .fixed_array import FixedArray, PrecisionLossWarning, RoundingMode
from .unit_array import UnitArray
from .money import Money
from .price import Price
from .quantity import Quantity
from .instrument import Instrument
from .lot import PositionLot, LotMatchingRule, match_lots

from .portfolio import Portfolio
from .conventions import DayCountConvention, year_fraction
from .cashflow import CashFlow, generate_cash_flow_schedule
from .tvm import pv, fv, npv, discount_factors
from .formatter import MonetaryFormatter
from .fx import (
    ExchangeRate,
    ExchangeRateNotFoundError,
    FXProvider,
    UserDictFXProvider,
    SnapshotFXProvider,
    FrankfurterFXProvider,
    CompositeFXProvider,
    FXEngine,
    decompose_fx_pnl,
    covered_interest_parity_forward,
)

__all__ = [
    # Core
    "FixedArray",
    "RoundingMode",
    "UnitArray",
    "PrecisionLossWarning",

    # Currency
    "Currency",
    "CurrencyRegistry",
    "get_currency",
    "CNY",
    "USD",
    "EUR",
    "JPY",
    "GBP",
    "HKD",
    "AUD",
    "CAD",
    "CHF",
    "SGD",
    # Money & Pricing
    "Money",
    "Price",
    "Quantity",
    # Holdings & Portfolio
    "PositionLot",
    "Portfolio",
    "Instrument",
    "LotMatchingRule",
    "match_lots",
    # Cash Flow & Schedule & Conventions
    "CashFlow",
    "DayCountConvention",
    "year_fraction",
    "generate_cash_flow_schedule",
    # Time Value of Money (TVM)
    "pv",
    "fv",
    "npv",
    "discount_factors",
    # Formatter
    "MonetaryFormatter",
    # FX
    "ExchangeRate",
    "ExchangeRateNotFoundError",
    "FXProvider",
    "UserDictFXProvider",
    "SnapshotFXProvider",
    "FrankfurterFXProvider",
    "CompositeFXProvider",
    "FXEngine",
    "decompose_fx_pnl",
    "covered_interest_parity_forward",
]
