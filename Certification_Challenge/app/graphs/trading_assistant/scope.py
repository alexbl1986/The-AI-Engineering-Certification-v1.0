"""Intake / scoper node (ADR-0006, step 1).

The graph's routing brain: one structured-output call turns the user's message
(plus thread history) into a `Scope` — a multi-label intent, extracted tickers,
the `hypothetical` flag, and at most one clarifying question. Everything the
graph does downstream is deterministic and keys off this read, so the scoper is
the only place free-text becomes routing.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, SystemMessage

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.state import AgentState, Scope

SCOPER_SYSTEM = """You are the intake router for a retail options day-trader's assistant.
The user reconciles his IBKR book against his own exposure rules and his desk's Hebrew
daily/weekly market reviews. Classify his message into one or more of these routes and
extract the entities the downstream tools need. Do not answer the question yourself.

Routes (choose ALL that apply — a single message can need several):
- status_check: "am I within policy / exposure?" — book vs. exposure limits.
- rebalance_advice: "what should I change?" — full reconcile and concrete actions.
- desk_question: what the desk thinks about a name/theme/sector (from the reviews).
- trade_history: how a specific position/campaign is going (entry, scales, basis).
- performance_review: realized P/L, win rate, cost drag over a period.
- policy_change: a request to CHANGE a rule/threshold (e.g. "raise my options cap to 12%").
- market_regime: live macro / index read (needs no uploaded data).
- trade_signal_eval: a pasted trade shorthand like "AAOI 150 NEXT WEEK 3.1".
- daily_briefing: "morning briefing" / "start my day" — the full composed rundown.
- off_topic: anything not about this book, this desk's reviews, or these rules — refuse.

Extraction:
- tickers: uppercase symbols named or clearly implied.
- signal_text: the verbatim shorthand string, ONLY for trade_signal_eval, else null.
- hypothetical: true for "what if I raised my cap…" / "would I still be within policy if…".
  A hypothetical about a rule is ANALYSIS (status/rebalance), never a policy_change write.

Clarification is a LAST RESORT. Almost every message has a reasonable default — take it
and record it in `assumptions` (e.g. "assumed you mean the current open AAOI campaign")
instead of asking. Set needs_clarification true ONLY when the message is genuinely
undecidable AND no assumption could resolve it (e.g. a bare ticker with no action, or a
request that could mean two opposite trades). NEVER clarify a status / exposure / policy /
performance question, a named desk question, or a trade-history question about a named
ticker — just proceed. A missing detail a downstream tool can ask for (e.g. which position)
is not a reason to clarify here. Never ask more than one question."""

OFF_TOPIC_REFUSAL = (
    "I'm your trading-desk assistant — I can help with your book, your exposure and "
    "hedging rules, your desk's daily/weekly reviews, and trade-signal checks, but that "
    "one's outside what I do."
)


def _normalize_scope(scope: Scope) -> Scope:
    """Deterministic safety net around the LLM's routing (ADR-0006).

    A hypothetical about a rule ("if I raised my cap to 12%…") is analysis, never
    a write — but the model sometimes still tags `policy_change` on it. Strip it
    so the interrupt-gated write path can never fire on a hypothetical; fall back
    to `status_check` if that empties the routes.
    """
    if scope.hypothetical and "policy_change" in scope.intents:
        intents = [i for i in scope.intents if i != "policy_change"] or ["status_check"]
        return scope.model_copy(update={"intents": intents})
    return scope


def make_scope_node(context: AgentContext) -> Callable[[AgentState], dict]:
    """Bind the scoper to its chat model; returns the LangGraph node callable."""
    model = context.chat_model.with_structured_output(Scope)

    def scope_node(state: AgentState) -> dict:
        conversation = [SystemMessage(content=SCOPER_SYSTEM), *state["messages"]]
        scope: Scope = _normalize_scope(model.invoke(conversation))
        update: dict = {"scope": scope}
        # One clarify round: surface the question as an assistant turn and stop;
        # the user's reply re-enters the graph as a fresh message.
        if scope.needs_clarification and scope.clarifying_question:
            update["messages"] = [AIMessage(content=scope.clarifying_question)]
        elif set(scope.intents) <= {"off_topic"}:
            update["messages"] = [AIMessage(content=OFF_TOPIC_REFUSAL)]
        return update

    return scope_node
