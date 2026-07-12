"""Seam: classify_scaleout(gain) -> ScaleOutSignal.

Pure threshold rule from the trader's rulebook: scale out at +100% gain, moonshot at
+150% (moonshot supersedes). `gain` is a ratio vs entry: +100% == 1.0.
"""

from datetime import date, datetime

import pytest

from app.trading.domain import MissingData, Position, ScaleOutSignal, Trade
from app.trading.scaleout import classify_scaleout, scan_scaleout


def _entry(symbol, qty, price):
    """A statement opening fill (statement symbol format)."""
    return Trade(
        symbol=symbol, root_ticker=symbol.split()[0],
        asset_category="Equity and Index Options", currency="USD",
        timestamp=datetime(2026, 6, 1, 9, 30), quantity=float(qty), price=price,
        proceeds=-qty * price * 100, commission=-1.0, basis=qty * price * 100,
        realized_pl=0.0, mtm_pl=0.0, code="O",
    )


def _opt_position(root, strike, expiry, right, mark):
    return Position(
        symbol=root, asset_class="OPT", currency="USD", fx_rate_to_base=1.0,
        quantity=2.0, mark_price=mark, position_value=mark * 200,
        strike=strike, expiry=expiry, right=right,
    )


@pytest.mark.parametrize(
    "gain, expected",
    [
        (-0.769, ScaleOutSignal.NONE),   # FLEX runner, deep loss
        (0.0, ScaleOutSignal.NONE),
        (0.99, ScaleOutSignal.NONE),     # just under the scale-out line
        (1.0, ScaleOutSignal.SCALE_OUT), # +100% exactly
        (1.2, ScaleOutSignal.SCALE_OUT),
        (1.5, ScaleOutSignal.MOONSHOT),  # +150% exactly, moonshot supersedes
        (2.6, ScaleOutSignal.MOONSHOT),
    ],
)
def test_classify_scaleout_thresholds(gain, expected):
    assert classify_scaleout(gain) == expected


def test_scan_flags_winners_by_gain_vs_campaign_entry():
    expiry = date(2026, 7, 17)
    # Two contracts opened @ 0.40 each; a losing runner opened @ 0.40 too.
    trades = [
        _entry("ADEA 17JUL26 35 C", 2, 0.40),
        _entry("BEAM 17JUL26 50 C", 2, 0.40),
        _entry("DOWN 17JUL26 10 C", 2, 0.40),
    ]
    positions = [
        _opt_position("ADEA", 35.0, expiry, "C", mark=1.00),  # +150% -> moonshot
        _opt_position("BEAM", 50.0, expiry, "C", mark=0.85),  # +112% -> scale out
        _opt_position("DOWN", 10.0, expiry, "C", mark=0.30),  # -25%  -> not flagged
    ]

    candidates = scan_scaleout(positions, trades)

    by_symbol = {c.symbol: c for c in candidates}
    assert set(by_symbol) == {"ADEA", "BEAM"}
    assert by_symbol["ADEA"].signal == ScaleOutSignal.MOONSHOT
    assert by_symbol["BEAM"].signal == ScaleOutSignal.SCALE_OUT
    assert by_symbol["ADEA"].gain == pytest.approx(1.5)


def test_scan_refuses_without_a_ledger():
    # Cost basis lives in the statement (tactical CostBasisPrice is always 0).
    # With no ledger, refuse loudly rather than imply "no candidates".
    positions = [_opt_position("ADEA", 35.0, date(2026, 7, 17), "C", mark=1.00)]

    result = scan_scaleout(positions, [])

    assert isinstance(result, MissingData)
    assert result.store == "ledger"
    assert "statement" in result.remedy.lower()
