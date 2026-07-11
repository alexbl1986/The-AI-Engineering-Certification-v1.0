"""Shared typed records for the trading assistant."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Position:
    """One line of the tactical book snapshot.

    `quantity` and `position_value` are signed (negative for shorts).
    `position_value` is in the position's native `currency`; multiply by
    `fx_rate_to_base` to compare across the book. Option fields are None for
    stocks. Cost basis is deliberately absent (it comes from the ledger).
    """

    symbol: str
    asset_class: str
    currency: str
    fx_rate_to_base: float
    quantity: float
    mark_price: float
    position_value: float
    strike: float | None = None
    expiry: date | None = None
    right: str | None = None
