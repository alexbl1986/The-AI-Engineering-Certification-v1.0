"""Scale-out scanning: flag open positions that hit profit-taking rules.

Gain is measured as a ratio vs average entry price (current_mark / entry - 1),
which is multiplier- and currency-free. Thresholds: scale out at +100%,
moonshot at +150% (moonshot supersedes).
"""

from __future__ import annotations

from app.trading.domain import (
    MissingData,
    Position,
    ScaleOutCandidate,
    ScaleOutSignal,
    Trade,
)
from app.trading.ledger import group_campaigns
from app.trading.symbols import position_contract_key, statement_symbol_contract_key

_SCALE_OUT_GAIN = 1.0  # +100%
_MOONSHOT_GAIN = 1.5  # +150%


def classify_scaleout(gain: float) -> ScaleOutSignal:
    if gain >= _MOONSHOT_GAIN:
        return ScaleOutSignal.MOONSHOT
    if gain >= _SCALE_OUT_GAIN:
        return ScaleOutSignal.SCALE_OUT
    return ScaleOutSignal.NONE


def scan_scaleout(
    positions: list[Position], trades: list[Trade]
) -> list[ScaleOutCandidate] | MissingData:
    """Flag open positions whose gain vs campaign entry hits a scale-out rule.

    Cost basis comes from the ledger (tactical CostBasisPrice is always 0), so
    with no ledger the scan refuses rather than implying "no candidates".
    """
    if not trades:
        return MissingData(store="ledger", remedy="Upload a recent activity statement.")

    entry_by_key = {
        statement_symbol_contract_key(c.symbol): c.avg_entry_price
        for c in group_campaigns(trades)
        if c.is_open
    }

    candidates: list[ScaleOutCandidate] = []
    for pos in positions:
        entry = entry_by_key.get(position_contract_key(pos))
        if entry is None:
            continue  # no matching open campaign -> no cost basis to judge
        gain = pos.mark_price / entry - 1
        signal = classify_scaleout(gain)
        if signal is not ScaleOutSignal.NONE:
            candidates.append(
                ScaleOutCandidate(
                    symbol=pos.symbol,
                    signal=signal,
                    gain=gain,
                    avg_entry_price=entry,
                    mark_price=pos.mark_price,
                )
            )
    return candidates
