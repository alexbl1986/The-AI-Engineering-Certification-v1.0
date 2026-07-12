"""daily_briefing: a composition of the existing fetches (no new tools) plus a
briefing-shaped synthesis prompt. Offline with fakes.
"""

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.prefetch import BRIEFING_QUERY, make_prefetch_node
from app.graphs.trading_assistant.state import EvidenceItem, Scope
from app.graphs.trading_assistant.synthesis import BRIEFING_SYSTEM, make_synthesis_node
from app.trading.domain import ExposureCheck, ExposureReport, MissingData, Position


class _FakeChat:
    def __init__(self, text="Book looks fine."):
        self.text = text
        self.seen = None

    def invoke(self, messages):
        self.seen = messages
        return AIMessage(content=self.text)


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
        self.calls: list[str] = []

    def retrieve(self, query, *, user_id, k):
        self.calls.append(query)
        return list(self._docs)


def _position():
    return Position(
        symbol="AAOI 16JAN26 40 C", asset_class="OPT", currency="USD",
        fx_rate_to_base=1.0, quantity=5, mark_price=4.0, position_value=7300.0,
        strike=40.0, expiry=date(2026, 1, 16), right="C",
    )


def _ctx(retriever, chat=None):
    return AgentContext(
        chat_model=chat or _FakeChat(),
        retriever=retriever,
        load_positions=lambda u: [_position()],
        load_trades=lambda u: MissingData("ledger", "Upload a recent activity statement."),
        load_nav=lambda u: 100_000.0,
        default_user_id="alex",
    )


def _prefetch(scope, ctx, text="morning briefing"):
    return make_prefetch_node(ctx)(
        {"scope": scope, "messages": [HumanMessage(content=text)], "user_id": "alex"}
    )


def test_briefing_composes_portfolio_and_desk_summary():
    retriever = _FakeRetriever(["d1", "d2"])
    out = _prefetch(Scope(intents=["daily_briefing"]), _ctx(retriever))
    tools = {item.tool for item in out["evidence"]}
    assert {"check_exposure", "scan_scaleout", "open_position_pnl", "list_positions",
            "search_desk_reviews"} <= tools
    assert retriever.calls == [BRIEFING_QUERY]  # standing query, not the trigger phrase


def test_briefing_with_explicit_desk_question_uses_that_question():
    retriever = _FakeRetriever(["d1"])
    out = make_prefetch_node(_ctx(retriever))(
        {
            "scope": Scope(intents=["daily_briefing", "desk_question"]),
            "messages": [HumanMessage(content="briefing, and what about TSMC?")],
            "user_id": "alex",
        }
    )
    assert retriever.calls == ["briefing, and what about TSMC?"]
    assert any(i.tool == "check_exposure" for i in out["evidence"])  # portfolio still runs


def test_synthesis_uses_briefing_prompt_for_briefing_intent():
    chat = _FakeChat()
    ctx = _ctx(_FakeRetriever([]), chat=chat)
    evidence = [
        EvidenceItem(
            "check_exposure", ok=True,
            result=ExposureReport(120_000.0, (ExposureCheck("options", 8760.0, 0.073, 0.10),)),
        )
    ]
    make_synthesis_node(ctx)(
        {"messages": [HumanMessage(content="morning briefing")], "evidence": evidence,
         "missing": [], "scope": Scope(intents=["daily_briefing"])}
    )
    assert chat.seen[0].content == BRIEFING_SYSTEM  # briefing prompt, not the general one


def test_non_briefing_still_uses_general_prompt():
    chat = _FakeChat()
    ctx = _ctx(_FakeRetriever([]), chat=chat)
    evidence = [
        EvidenceItem(
            "check_exposure", ok=True,
            result=ExposureReport(120_000.0, (ExposureCheck("options", 8760.0, 0.073, 0.10),)),
        )
    ]
    make_synthesis_node(ctx)(
        {"messages": [HumanMessage(content="am I ok?")], "evidence": evidence,
         "missing": [], "scope": Scope(intents=["status_check"])}
    )
    assert chat.seen[0].content != BRIEFING_SYSTEM
