"""Deterministic pre-fetch by route (ADR-0006, step 2).

The calls that *must* happen for a route are graph code, not LLM discretion:
status/rebalance always run exposure + scale-out + P/L over the snapshot;
desk questions always retrieve; trade history reads the ledger. Multi-label
intent fetches the union. Every call's outcome lands in the evidence table, and
a store that was never uploaded surfaces as `MissingData` (cold-start contract)
rather than a silent empty result — so synthesis can answer what it can and name
the missing upload, and the audit node can block any number with no backing.

Live-quote / technical / web enrichment are the tool agent's job (the long tail,
a later slice): exposure, scale-out and P/L are self-contained on the snapshot's
own marks, so they need no network here.
"""

from __future__ import annotations

from typing import Callable, Sequence

from app.graphs.trading_assistant.deps import (
    AgentContext,
    NavLoader,
    PositionsLoader,
    TradesLoader,
)
from app.graphs.trading_assistant.policy_model import DEFAULT_POLICY
from app.graphs.trading_assistant.state import AgentState, EvidenceItem, Scope
from app.graphs.trading_assistant.util import latest_user_text
from app.trading.domain import MissingData, Position, Trade
from app.trading.exposure import check_exposure
from app.trading.ledger import get_trades
from app.trading.pnl import open_position_pnl
from app.trading.scaleout import scan_scaleout

_PORTFOLIO_INTENTS = {"status_check", "rebalance_advice"}

# The daily briefing is a composition: it runs the portfolio checks AND pulls a
# desk summary. Since a briefing has no specific question, the desk retrieval uses
# a standing summary query rather than the trigger phrase ("morning briefing").
BRIEFING_QUERY = "market outlook, key risks, sector positioning, and hedging guidance for today"


def make_prefetch_node(context: AgentContext) -> Callable[[AgentState], dict]:
    """Bind the pre-fetch to its data-access seams; returns the node callable."""

    def prefetch_node(state: AgentState) -> dict:
        scope: Scope = state["scope"]
        user_id = state.get("user_id") or context.default_user_id
        intents = set(scope.intents)
        query = latest_user_text(state["messages"])

        positions = _load(
            context.load_positions,
            user_id,
            MissingData("positions snapshot", "Upload your tactical book export."),
        )
        trades = _load(
            context.load_trades,
            user_id,
            MissingData("ledger", "Upload a recent activity statement."),
        )
        nav = context.load_nav(user_id) if context.load_nav else None
        policy = context.load_policy(user_id) if context.load_policy else DEFAULT_POLICY

        # A briefing needs the whole book picture plus the desk's read, so it runs
        # the union of the portfolio and desk fetches. Flags dedupe multi-label
        # requests so each capability runs at most once.
        briefing = "daily_briefing" in intents
        run_portfolio = bool(intents & _PORTFOLIO_INTENTS) or briefing
        run_desk = "desk_question" in intents or briefing

        evidence: list[EvidenceItem] = []
        if run_portfolio:
            evidence += _portfolio_prefetch(positions, trades, nav, policy.options_limit)
        if "trade_history" in intents:
            evidence += _history_prefetch(trades, scope.tickers)
        if run_desk:
            desk_query = query if "desk_question" in intents else BRIEFING_QUERY
            evidence += _desk_prefetch(context.retriever, desk_query, user_id)

        return {"evidence": evidence, "missing": _dedupe_missing(evidence)}

    return prefetch_node


# -- route pre-fetches ----------------------------------------------------


def _portfolio_prefetch(
    positions: list[Position] | MissingData,
    trades: list[Trade] | MissingData,
    nav: float | None,
    options_limit: float,
) -> list[EvidenceItem]:
    if isinstance(positions, MissingData):
        # Every portfolio tool needs the snapshot; block them all on it.
        return [EvidenceItem("list_positions", ok=False, missing=positions)] + [
            EvidenceItem(tool, ok=False, missing=positions)
            for tool in ("check_exposure", "scan_scaleout", "open_position_pnl")
        ]

    items = [
        EvidenceItem("list_positions", ok=True, result=tuple(positions)),
        _wrap("check_exposure", check_exposure(positions, nav, options_limit)),
    ]
    if isinstance(trades, MissingData):
        items += [
            EvidenceItem("scan_scaleout", ok=False, missing=trades),
            EvidenceItem("open_position_pnl", ok=False, missing=trades),
        ]
    else:
        items += [
            _wrap("scan_scaleout", scan_scaleout(positions, trades)),
            _wrap("open_position_pnl", open_position_pnl(positions, trades)),
        ]
    return items


def _history_prefetch(
    trades: list[Trade] | MissingData, tickers: Sequence[str]
) -> list[EvidenceItem]:
    if isinstance(trades, MissingData):
        return [EvidenceItem("get_trades", ok=False, missing=trades)]
    if not tickers:
        return [
            EvidenceItem(
                "get_trades", ok=False, note="No ticker named; ask which position."
            )
        ]
    # Default = current active campaign per ticker (full history is an intake flag).
    return [
        EvidenceItem(f"get_trades:{ticker}", ok=True, result=tuple(get_trades(trades, ticker)))
        for ticker in tickers
    ]


def _desk_prefetch(retriever, query: str, user_id: str) -> list[EvidenceItem]:
    if retriever is None:
        return [
            EvidenceItem(
                "search_desk_reviews", ok=False, note="Desk-review retriever not configured."
            )
        ]
    docs = retriever.retrieve(query, user_id=user_id, k=5)
    if not docs:
        # The corpus is the two always-relevant reviews, so an empty hit for this
        # user means none were uploaded — a cold-start miss, not "no opinion".
        return [
            EvidenceItem(
                "search_desk_reviews",
                ok=False,
                missing=MissingData(
                    "desk reviews",
                    "Upload your latest daily and weekly desk review PDFs.",
                ),
            )
        ]
    return [EvidenceItem("search_desk_reviews", ok=True, result=tuple(docs))]


# -- helpers --------------------------------------------------------------


def _load(loader, user_id: str, if_unconfigured: MissingData):
    return loader(user_id) if loader is not None else if_unconfigured


def _wrap(tool: str, result) -> EvidenceItem:
    if isinstance(result, MissingData):
        return EvidenceItem(tool=tool, ok=False, missing=result)
    return EvidenceItem(tool=tool, ok=True, result=result)


def _dedupe_missing(evidence: Sequence[EvidenceItem]) -> list[MissingData]:
    seen: dict[str, MissingData] = {}
    for item in evidence:
        if item.missing is not None:
            seen.setdefault(item.missing.store, item.missing)
    return list(seen.values())
