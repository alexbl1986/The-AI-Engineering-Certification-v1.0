"""Seam: check_exposure(positions, nav) -> ExposureReport | MissingData.

Measures book exposure against the trader's % -of-NAV limits. NAV is the cash-inclusive
statement Total (chosen denominator); with no NAV the tool refuses (MissingData)
rather than divide by a fabricated zero. First metric: options <= 10% NAV.
The "20% offensive exposure" bucket is deferred until 'offensive' is defined.
"""

from pathlib import Path

import pytest

from app.trading.domain import ExposureCheck, ExposureReport, MissingData, Position
from app.trading.exposure import check_exposure
from app.trading.ingest.statement import parse_account_nav
from app.trading.ingest.tactical import parse_tactical_book

DATA = Path(__file__).parent.parent / "data" / "private"
BOOK = DATA / "Tactical_Boot.csv"
STMT = DATA / "IBKR YTD Statement.csv"


def _stock(symbol, value, fx=1.0, currency="USD"):
    return Position(
        symbol=symbol, asset_class="STK", currency=currency, fx_rate_to_base=fx,
        quantity=100.0, mark_price=value / 100.0, position_value=value,
    )


def _option(symbol, value, fx=1.0, currency="USD"):
    return Position(
        symbol=symbol, asset_class="OPT", currency=currency, fx_rate_to_base=fx,
        quantity=10.0, mark_price=value / 10.0, position_value=value,
        strike=50.0, right="C",
    )


def test_reports_options_pct_of_nav():
    # Options bucket = 12,000 of 150,000 NAV = 8% (<= 10% -> within policy).
    # The stock line must NOT count toward the options bucket.
    positions = [_stock("AEHR", 90000.0), _option("AAPL", 12000.0)]

    report = check_exposure(positions, nav=150000.0)

    assert report == ExposureReport(
        nav=150000.0,
        checks=(ExposureCheck(label="options", value_base=12000.0, pct_of_nav=0.08, limit=0.10),),
    )
    assert report.checks[0].within_policy is True


# --- contract guards (pass on arrival; they pin behavior slice 2 already covers) ---

def test_foreign_option_value_normalized_to_base_usd():
    # SIVE-adjacent guard: a EUR option must be converted via fx before the ratio,
    # not compared in its native currency. 10,000 EUR x 1.1 = 11,000 USD.
    report = check_exposure([_option("SAP", 10000.0, fx=1.1, currency="EUR")], nav=150000.0)
    assert report.checks[0].value_base == 11000.0


def test_options_over_limit_breach_flagged():
    # 20,000 of 100,000 NAV = 20% > 10% -> not within policy.
    report = check_exposure([_option("AAPL", 20000.0)], nav=100000.0)
    assert report.checks[0].pct_of_nav == 0.20
    assert report.checks[0].within_policy is False


def test_no_nav_refuses_with_missing_data():
    # Cold start: only a book uploaded, no statement -> refuse, don't imply 0%.
    result = check_exposure([_option("AAPL", 12000.0)], nav=None)
    assert isinstance(result, MissingData)
    assert result.store == "statement NAV"


@pytest.mark.skipif(
    not (BOOK.exists() and STMT.exists()), reason="real files not present"
)
def test_real_book_options_exposure_is_sane():
    positions = parse_tactical_book(BOOK.read_text(encoding="utf-8"))
    nav = parse_account_nav(STMT.read_text(encoding="utf-8"))

    report = check_exposure(positions, nav)

    assert isinstance(report, ExposureReport)
    assert report.nav == nav and nav > 0
    (options,) = report.checks
    # Structural, no hardcoded pct: a real fraction of NAV, within the 10% rule.
    assert 0.0 < options.pct_of_nav < options.limit
    assert options.within_policy is True
