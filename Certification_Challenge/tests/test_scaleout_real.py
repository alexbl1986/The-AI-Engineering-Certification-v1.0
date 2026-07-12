"""Characterization: scan_scaleout over the real (gitignored) book + statement.

Structural invariants only — no specific holdings or numbers hardcoded.
"""

from pathlib import Path

import pytest

from app.trading.domain import MissingData, ScaleOutSignal
from app.trading.ingest.statement import parse_activity_statement
from app.trading.ingest.tactical import parse_tactical_book
from app.trading.scaleout import scan_scaleout

DATA = Path(__file__).parent.parent / "data" / "private"
BOOK = DATA / "Tactical_Boot.csv"
STMT = DATA / "IBKR YTD Statement.csv"


@pytest.mark.skipif(not (BOOK.exists() and STMT.exists()), reason="real files not present")
def test_scan_on_real_data_is_consistent_and_ledger_gated():
    positions = parse_tactical_book(BOOK.read_text(encoding="utf-8"))
    trades = parse_activity_statement(STMT.read_text(encoding="utf-8"))

    # No ledger -> refuse, regardless of positions.
    assert isinstance(scan_scaleout(positions, []), MissingData)

    # With a ledger -> a list; every candidate genuinely clears a threshold and
    # its flag matches its gain.
    candidates = scan_scaleout(positions, trades)
    assert isinstance(candidates, list)
    for c in candidates:
        assert c.signal is not ScaleOutSignal.NONE
        assert c.gain >= 1.0
        assert c.mark_price / c.avg_entry_price - 1 == pytest.approx(c.gain)
        expected = (
            ScaleOutSignal.MOONSHOT if c.gain >= 1.5 else ScaleOutSignal.SCALE_OUT
        )
        assert c.signal == expected
