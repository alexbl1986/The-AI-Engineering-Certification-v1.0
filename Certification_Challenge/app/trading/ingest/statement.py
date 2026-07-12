"""Parser for the IBKR YTD Activity Statement (multi-section CSV).

Extracts the `Trades` section's fills into typed `Trade` records. The section
also carries IBKR's own SubTotal/Total aggregation rows; only `Order` rows are
real fills, so those are the only ones kept (see the double-count guard test).
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

from app.trading.domain import Trade
from app.trading.symbols import ContractKey, statement_symbol_contract_key

_TIMESTAMP_FORMAT = "%Y-%m-%d, %H:%M:%S"


def _num(value: str) -> float:
    """Coerce an IBKR numeric field to float.

    Drops thousands separators, and treats a blank field as 0.0 (the Forex
    conversion row leaves basis / realized P/L / close price blank).
    """
    value = value.replace(",", "").strip()
    return float(value) if value else 0.0


def parse_activity_statement(content: str) -> list[Trade]:
    rows = list(csv.reader(io.StringIO(content.removeprefix("\ufeff"))))
    trade_rows = [r for r in rows if r and r[0] == "Trades"]

    header = next(r for r in trade_rows if r[1] == "Header")
    trades: list[Trade] = []
    for row in trade_rows:
        record = dict(zip(header, row))
        if record["DataDiscriminator"] != "Order":
            continue
        symbol = record["Symbol"]
        trades.append(
            Trade(
                symbol=symbol,
                root_ticker=symbol.split()[0],
                asset_category=record["Asset Category"],
                currency=record["Currency"],
                timestamp=datetime.strptime(record["Date/Time"], _TIMESTAMP_FORMAT),
                quantity=_num(record["Quantity"]),
                price=_num(record["T. Price"]),
                proceeds=_num(record["Proceeds"]),
                commission=_num(record["Comm/Fee"]),
                basis=_num(record["Basis"]),
                realized_pl=_num(record["Realized P/L"]),
                mtm_pl=_num(record["MTM P/L"]),
                code=record["Code"],
            )
        )
    return trades


def parse_account_nav(content: str) -> float | None:
    """Total NAV (cash-inclusive) from the `Net Asset Value` section, or None.

    Returns the `Total` row's `Current Total` -- a coherent point-in-time figure
    that includes cash, which is the denominator for the % -of-NAV exposure
    rules. None when the section is absent, so the exposure tool can refuse
    (MissingData) rather than divide by a fabricated zero. The trailing
    "Time Weighted Rate of Return" sub-header is ignored (its rows aren't Total).
    """
    header: list[str] | None = None
    for row in csv.reader(io.StringIO(content.removeprefix("\ufeff"))):
        if not row or row[0] != "Net Asset Value":
            continue
        if row[1] == "Header":
            header = row
        elif row[1] == "Data" and header is not None:
            record = dict(zip(header, row))
            if record.get("Asset Class") == "Total" and record.get("Current Total"):
                return _num(record["Current Total"])
    return None


def parse_open_positions_cost(content: str) -> dict[ContractKey, float]:
    """Map each open position's contract key to IBKR's per-unit `Cost Price`.

    From the `Open Positions` section -- the fallback entry price for holdings
    with no opening fill in the YTD Trades ledger. Cost Price is per share /
    per contract-unit (like a campaign's avg_entry_price), so it plugs into the
    same open-position join. Symbols are the statement format, keyed identically.
    """
    header: list[str] | None = None
    costs: dict[ContractKey, float] = {}
    for row in csv.reader(io.StringIO(content.removeprefix("\ufeff"))):
        if not row or row[0] != "Open Positions":
            continue
        if row[1] == "Header":
            header = row
        elif row[1] == "Data" and header is not None:
            record = dict(zip(header, row))
            costs[statement_symbol_contract_key(record["Symbol"])] = _num(
                record["Cost Price"]
            )
    return costs


def parse_instrument_exchanges(content: str) -> dict[str, str]:
    """Map each stock symbol to its IBKR `Listing Exch` (for Yahoo suffixing).

    Reads the `Financial Instrument Information` section, which mixes stock and
    option header schemas, so we track the most recent Header row and keep only
    `Stocks` rows (options are always US-listed / OCC, no suffix needed).
    """
    header: list[str] | None = None
    exchanges: dict[str, str] = {}
    for row in csv.reader(io.StringIO(content.removeprefix("\ufeff"))):
        if not row or row[0] != "Financial Instrument Information":
            continue
        if row[1] == "Header":
            header = row
        elif row[1] == "Data" and header is not None:
            record = dict(zip(header, row))
            if record.get("Asset Category") == "Stocks":
                exchanges[record["Symbol"]] = record["Listing Exch"]
    return exchanges
