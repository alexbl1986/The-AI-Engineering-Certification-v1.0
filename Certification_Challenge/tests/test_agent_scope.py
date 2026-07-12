"""Intake/scoper node: routing, entity extraction, and the one clarify round.

Offline: a stub structured model stands in for the gateway chat model, so the
graph's routing wiring is tested deterministically without the network. The
scoper's job is only to turn a message into a `Scope`; it must not answer.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.scope import make_scope_node
from app.graphs.trading_assistant.state import Scope


class StubModel:
    """Minimal chat-model stand-in: returns a canned Scope, records what it saw."""

    def __init__(self, scope: Scope) -> None:
        self._scope = scope
        self.seen: list | None = None

    def with_structured_output(self, schema):  # noqa: ANN001 - duck-typed seam
        assert schema is Scope
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.seen = messages
        return self._scope


def _run(scope: Scope, text: str = "hi") -> dict:
    # Test the scoper node in isolation — the full graph would run on to synthesis.
    node = make_scope_node(AgentContext(chat_model=StubModel(scope)))
    return node({"messages": [HumanMessage(content=text)]})


def test_scope_node_classifies_single_intent():
    result = _run(Scope(intents=["status_check"]))
    assert result["scope"].intents == ["status_check"]


def test_scope_node_is_multi_label_with_entities():
    result = _run(
        Scope(intents=["status_check", "desk_question"], tickers=["TSMC"]),
        text="am I within policy and what does the desk think of TSMC?",
    )
    assert set(result["scope"].intents) == {"status_check", "desk_question"}
    assert result["scope"].tickers == ["TSMC"]


def test_hypothetical_is_analysis_not_a_policy_write():
    result = _run(Scope(intents=["status_check"], hypothetical=True))
    assert result["scope"].hypothetical is True
    assert "policy_change" not in result["scope"].intents


def test_hypothetical_policy_change_is_stripped_to_analysis():
    # The observed real-model failure: hypothetical=True but still tagged a write.
    result = _run(Scope(intents=["status_check", "policy_change"], hypothetical=True))
    assert "policy_change" not in result["scope"].intents
    assert "status_check" in result["scope"].intents


def test_real_policy_change_is_preserved():
    result = _run(Scope(intents=["policy_change"], hypothetical=False))
    assert result["scope"].intents == ["policy_change"]


def test_clarification_emits_one_assistant_question():
    question = "Do you mean your current open AAOI campaign or all of them?"
    result = _run(
        Scope(intents=["trade_history"], needs_clarification=True, clarifying_question=question),
        text="how's AAOI",
    )
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == question


def test_scoper_sees_system_prompt_and_user_message():
    stub = StubModel(Scope(intents=["off_topic"]))
    node = make_scope_node(AgentContext(chat_model=stub))
    node({"messages": [HumanMessage(content="what's the weather")]})
    assert isinstance(stub.seen[0], SystemMessage)
    assert isinstance(stub.seen[-1], HumanMessage)
    assert "intake router" in stub.seen[0].content
