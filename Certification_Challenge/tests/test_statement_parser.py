"""Seam: parse_activity_statement(content) -> list[Trade].

Parses the Trades section of the IBKR YTD Activity Statement into typed Trade
records. Header lines are verbatim from the real export; data rows are
synthetic (real fills live only in gitignored data/private/).
"""

from datetime import datetime
from pathlib import Path

import pytest

from app.trading.ingest.statement import parse_activity_statement
from app.trading.domain import Trade

REAL_FILE = Path(__file__).parent.parent / "data" / "private" / "IBKR YTD Statement.csv"

# Verbatim Trades header from the real statement.
TRADES_HEADER = (
    "Trades,Header,DataDiscriminator,Asset Category,Currency,Symbol,Date/Time,"
    "Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code"
)


def test_parses_a_single_stock_order():
    content = (
        TRADES_HEADER + "\n"
        "Trades,Data,Order,Stocks,CAD,HPS.A,\"2026-05-26, 09:49:54\","
        "9,317.99,314.56,-2861.91,-1,2862.91,0,-30.87,O\n"
    )

    trades = parse_activity_statement(content)

    assert trades == [
        Trade(
            symbol="HPS.A",
            root_ticker="HPS.A",
            asset_category="Stocks",
            currency="CAD",
            timestamp=datetime(2026, 5, 26, 9, 49, 54),
            quantity=9.0,
            price=317.99,
            proceeds=-2861.91,
            commission=-1.0,
            basis=2862.91,
            realized_pl=0.0,
            mtm_pl=-30.87,
            code="O",
        )
    ]


def test_parses_numeric_fields_with_thousands_separators():
    # IBKR writes thousands separators into numeric fields, e.g. "-20,000".
    content = (
        TRADES_HEADER + "\n"
        "Trades,Data,Order,Stocks,USD,FAKE,\"2026-05-26, 09:49:54\","
        "\"-20,000\",1.5,1.4,\"30,000\",-5,\"29,995\",\"1,234.5\",-3,O\n"
    )

    trades = parse_activity_statement(content)

    assert trades[0].quantity == -20000.0
    assert trades[0].proceeds == 30000.0
    assert trades[0].basis == 29995.0
    assert trades[0].realized_pl == 1234.5


def test_empty_numeric_fields_become_zero():
    # The Forex conversion row leaves C. Price / Basis / Realized P/L blank
    # (no cost-basis concept for an FX leg). Treat blank numerics as 0.0.
    content = (
        TRADES_HEADER + "\n"
        "Trades,Data,Order,Forex,ILS,USD.ILS,\"2026-07-03, 08:13:59\","
        "\"-20,000\",2.9973,,59946,-2.668,,,-0.216,\n"
    )

    trades = parse_activity_statement(content)

    assert trades[0].basis == 0.0
    assert trades[0].realized_pl == 0.0
    assert trades[0].proceeds == 59946.0


def test_skips_subtotal_and_total_aggregation_rows():
    # IBKR mixes SubTotal/Total rows into the Trades section: same columns,
    # empty DataDiscriminator, and the SubTotal even repeats the symbol + P/L.
    # Only the Order fill is a real trade; aggregation rows must not double-count.
    content = (
        TRADES_HEADER + "\n"
        "Trades,Data,Order,Stocks,USD,FAKE,\"2026-05-26, 09:49:54\","
        "9,10,10,-90,-1,91,0,-3,O\n"
        "Trades,SubTotal,,Stocks,USD,FAKE,,9,,,-90,-1,91,0,-3,\n"
        "Trades,Total,,Stocks,USD,,,,,,-90,-1,91,0,-3, \n"
    )

    trades = parse_activity_statement(content)

    assert len(trades) == 1
    assert trades[0].symbol == "FAKE"


def test_parses_an_option_order_and_extracts_root_ticker():
    # Statement option symbols are "TICKER DDMMMYY STRIKE RIGHT"; the underlying
    # (first token) drives per-symbol attribution.
    content = (
        TRADES_HEADER + "\n"
        "Trades,Data,Order,Equity and Index Options,USD,FAKE 16JAN26 40 C,"
        "\"2026-01-16, 10:36:27\",-2,1.5,1.4,300,-1.5,688.76,-388.76228,-20,C;P\n"
    )

    trades = parse_activity_statement(content)

    trade = trades[0]
    assert trade.symbol == "FAKE 16JAN26 40 C"
    assert trade.root_ticker == "FAKE"
    assert trade.asset_category == "Equity and Index Options"
    assert trade.quantity == -2.0
    assert trade.realized_pl == -388.76228
    assert trade.commission == -1.5


@pytest.mark.skipif(not REAL_FILE.exists(), reason="real (gitignored) statement not present")
def test_parses_real_statement_only_order_fills():
    import csv

    content = REAL_FILE.read_text(encoding="utf-8")
    rows = list(csv.reader(content.splitlines()))
    order_rows = [
        r for r in rows if r and r[0] == "Trades" and r[1] == "Data" and r[2] == "Order"
    ]

    trades = parse_activity_statement(content)

    # Every Order fill parsed, and no SubTotal/Total row leaked in.
    assert len(trades) == len(order_rows)
    assert {t.asset_category for t in trades} <= {
        "Stocks",
        "Equity and Index Options",
        "Forex",
    }
    # Root ticker is always the first whitespace token of the raw symbol.
    for t in trades:
        assert t.root_ticker == t.symbol.split()[0]
