"""Deterministic pre-fetch node: route -> tool calls, threaded through the
evidence table with the cold-start `MissingData` contract.

Offline: data-access seams are plain lambdas over synthetic fixtures and the
retriever is a duck-typed fake, so no network or Qdrant is touched. These tests
verify the WIRING (which tools run for which route, and how a never-uploaded
store propagates) — the tools' own maths are covered by their own suites.
"""

from datetime import date, datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.graph import build_graph
from app.graphs.trading_assistant.prefetch import make_prefetch_node
from app.graphs.trading_assistant.state import Scope
from app.trading.domain import MissingData, Position, Trade


# --- fixtures ------------------------------------------------------------


def _opt_position() -> Position:
    return Position(
        symbol="AAOI 16JAN26 40 C", asset_class="OPT", currency="USD",
        fx_rate_to_base=1.0, quantity=5, mark_price=4.0, position_value=2000.0,
        strike=40.0, expiry=date(2026, 1, 16), right="C",
    )


def _trade() -> Trade:
    return Trade(
        symbol="AAOI 16JAN26 40 C", root_ticker="AAOI",
        asset_category="Equity and Index Options", currency="USD",
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc), quantity=5, price=2.0,
        proceeds=-1000.0, commission=-1.0, basis=1001.0, realized_pl=0.0,
        mtm_pl=0.0, code="O",
    )


class FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(self, query, *, user_id, k):
        self.calls.append((query, user_id, k))
        return list(self._docs)


def _context(*, positions="present", trades="present", nav=100_000.0, retriever=None):
    pos = [_opt_position()] if positions == "present" else positions
    trd = [_trade()] if trades == "present" else trades
    return AgentContext(
        chat_model=object(),  # pre-fetch never calls the model
        retriever=retriever,
        load_positions=(lambda u: pos),
        load_trades=(lambda u: trd),
        load_nav=(lambda u: nav),
        default_user_id="alex",
    )


def _prefetch(scope: Scope, context: AgentContext, *, text="hi", user_id="alex") -> dict:
    node = make_prefetch_node(context)
    return node({"scope": scope, "messages": [HumanMessage(content=text)], "user_id": user_id})


def _by_tool(result: dict) -> dict:
    return {item.tool: item for item in result["evidence"]}


# --- portfolio route -----------------------------------------------------


def test_status_check_runs_all_portfolio_tools():
    out = _prefetch(Scope(intents=["status_check"]), _context())
    tools = _by_tool(out)
    assert {"list_positions", "check_exposure", "scan_scaleout", "open_position_pnl"} <= tools.keys()
    assert all(tools[t].ok for t in ("list_positions", "check_exposure", "scan_scaleout"))
    assert out["missing"] == []


def test_missing_snapshot_blocks_every_portfolio_tool():
    ctx = _context(positions=MissingData("positions snapshot", "Upload your tactical book export."))
    out = _prefetch(Scope(intents=["rebalance_advice"]), ctx)
    tools = _by_tool(out)
    for t in ("list_positions", "check_exposure", "scan_scaleout", "open_position_pnl"):
        assert tools[t].ok is False
        assert tools[t].missing.store == "positions snapshot"
    assert [m.store for m in out["missing"]] == ["positions snapshot"]


def test_missing_ledger_blocks_scaleout_and_pnl_but_not_exposure():
    ctx = _context(trades=MissingData("ledger", "Upload a recent activity statement."))
    out = _prefetch(Scope(intents=["status_check"]), ctx)
    tools = _by_tool(out)
    assert tools["check_exposure"].ok is True  # exposure needs only snapshot + NAV
    assert tools["scan_scaleout"].ok is False and tools["scan_scaleout"].missing.store == "ledger"
    assert tools["open_position_pnl"].ok is False
    assert [m.store for m in out["missing"]] == ["ledger"]


def test_missing_nav_makes_exposure_report_missing_data():
    out = _prefetch(Scope(intents=["status_check"]), _context(nav=None))
    tools = _by_tool(out)
    assert tools["check_exposure"].ok is False
    assert tools["check_exposure"].missing.store == "statement NAV"


# --- desk route ----------------------------------------------------------


def test_desk_question_retrieves_with_user_scope():
    retriever = FakeRetriever(docs=["doc-a", "doc-b"])
    out = _prefetch(
        Scope(intents=["desk_question"], tickers=["TSMC"]),
        _context(retriever=retriever),
        text="what does the desk think of TSMC?",
    )
    hit = _by_tool(out)["search_desk_reviews"]
    assert hit.ok is True and len(hit.result) == 2
    assert retriever.calls == [("what does the desk think of TSMC?", "alex", 5)]


def test_desk_question_empty_corpus_is_cold_start_missing():
    out = _prefetch(
        Scope(intents=["desk_question"]), _context(retriever=FakeRetriever(docs=[]))
    )
    hit = _by_tool(out)["search_desk_reviews"]
    assert hit.ok is False and hit.missing.store == "desk reviews"
    assert [m.store for m in out["missing"]] == ["desk reviews"]


# --- history route -------------------------------------------------------


def test_trade_history_fetches_campaign_per_ticker():
    out = _prefetch(Scope(intents=["trade_history"], tickers=["AAOI"]), _context())
    assert _by_tool(out)["get_trades:AAOI"].ok is True


def test_trade_history_without_ticker_asks_which_position():
    out = _prefetch(Scope(intents=["trade_history"]), _context())
    item = _by_tool(out)["get_trades"]
    assert item.ok is False and "ticker" in item.note.lower()


def test_trade_history_missing_ledger():
    ctx = _context(trades=MissingData("ledger", "Upload a recent activity statement."))
    out = _prefetch(Scope(intents=["trade_history"], tickers=["AAOI"]), ctx)
    assert _by_tool(out)["get_trades"].missing.store == "ledger"


# --- multi-label union & routing ----------------------------------------


def test_multi_label_fetches_the_union():
    retriever = FakeRetriever(docs=["doc-a"])
    out = _prefetch(
        Scope(intents=["status_check", "desk_question"]),
        _context(retriever=retriever),
    )
    tools = _by_tool(out)
    assert "check_exposure" in tools and "search_desk_reviews" in tools


class _StubModel:
    """Answers the scoper's structured call (Scope) and synthesis's free-text call."""

    def __init__(self, scope, answer="Looks fine."):
        self._scope = scope
        self._answer = answer

    def with_structured_output(self, schema):
        return _ScopeInvoker(self._scope)

    def invoke(self, messages):
        return AIMessage(content=self._answer)


class _ScopeInvoker:
    def __init__(self, scope):
        self._scope = scope

    def invoke(self, messages):
        return self._scope


def _graph_context(scope: Scope) -> AgentContext:
    base = _context(retriever=FakeRetriever(docs=["d"]))
    return AgentContext(
        chat_model=_StubModel(scope),
        retriever=base.retriever,
        load_positions=base.load_positions,
        load_trades=base.load_trades,
        load_nav=base.load_nav,
        default_user_id="alex",
    )


def test_graph_runs_prefetch_for_actionable_intent():
    ctx = _graph_context(Scope(intents=["status_check"]))
    result = build_graph(ctx).invoke({"messages": [HumanMessage(content="am I ok?")], "user_id": "alex"})
    assert "evidence" in result and result["evidence"]


def test_graph_skips_prefetch_for_off_topic():
    ctx = _graph_context(Scope(intents=["off_topic"]))
    result = build_graph(ctx).invoke({"messages": [HumanMessage(content="weather?")], "user_id": "alex"})
    assert "evidence" not in result
