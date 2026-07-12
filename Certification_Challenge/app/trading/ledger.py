"""Campaign grouping over the parsed trades ledger.

A campaign is a continuously-open run of fills in one contract: it opens at
the first fill, absorbs scale-ins/scale-outs, and closes when net quantity
returns to zero. Contract identity is the raw symbol, so different expiries
(rolls) are separate campaigns.
"""

from __future__ import annotations

from itertools import groupby

from app.trading.domain import Campaign, Trade


def group_campaigns(trades: list[Trade]) -> list[Campaign]:
    campaigns: list[Campaign] = []
    by_symbol = sorted(trades, key=lambda t: (t.symbol, t.timestamp))
    for symbol, group in groupby(by_symbol, key=lambda t: t.symbol):
        run: list[Trade] = []
        net = 0.0
        for fill in group:
            run.append(fill)
            net += fill.quantity
            if net == 0:  # position flat → this campaign is closed
                campaigns.append(_campaign(symbol, run, net))
                run = []
        if run:  # leftover open position
            campaigns.append(_campaign(symbol, run, net))
    return campaigns


def get_trades(
    trades: list[Trade], ticker: str, full_history: bool = False
) -> list[Campaign]:
    """Campaigns for an underlying. Default = currently-open only."""
    campaigns = [c for c in group_campaigns(trades) if c.root_ticker == ticker]
    if full_history:
        return campaigns
    return [c for c in campaigns if c.is_open]


def _campaign(symbol: str, fills: list[Trade], net: float) -> Campaign:
    return Campaign(
        symbol=symbol,
        root_ticker=fills[0].root_ticker,
        fills=tuple(fills),
        net_quantity=net,
    )
