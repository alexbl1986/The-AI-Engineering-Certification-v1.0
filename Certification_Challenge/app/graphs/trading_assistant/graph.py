"""Assemble the trading-assistant graph from an injected `AgentContext`.

Topology so far: intake/scoper -> deterministic pre-fetch by route -> per-route
synthesis -> deterministic audit (one bounce back to synthesis, then deliver with
a warning banner). Still to come behind the same `build_graph(context)` seam: the
open-ended tool agent and interrupt-gated policy writes. A scoped-out message
(needs a clarifying answer, or purely off_topic) is answered by the scoper and
skips the rest.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graphs.trading_assistant.audit import MAX_ATTEMPTS, make_audit_node
from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.policy import (
    make_policy_confirm_node,
    make_policy_prepare_node,
)
from app.graphs.trading_assistant.prefetch import make_prefetch_node
from app.graphs.trading_assistant.scope import make_scope_node
from app.graphs.trading_assistant.state import AgentState
from app.graphs.trading_assistant.synthesis import make_synthesis_node


def _route_after_scope(state: AgentState) -> str:
    scope = state["scope"]
    if scope.needs_clarification:
        return END  # the clarifying question was already emitted; wait for the reply
    if set(scope.intents) <= {"off_topic"}:
        return END  # the scoper already emitted the refusal
    if "policy_change" in scope.intents:
        return "policy_prepare"  # a write takes precedence; gate it before anything else
    return "prefetch"


def _route_after_policy_prepare(state: AgentState) -> str:
    # Only pause for approval if a concrete change was parsed; otherwise the
    # prepare node already asked which rule to change.
    return "policy_confirm" if state.get("proposed_change") else END


def _route_after_audit(state: AgentState) -> str:
    if state["audit"].ok:
        return END
    if state.get("synthesis_attempts", 1) >= MAX_ATTEMPTS:
        return END  # delivered with a warning banner
    return "synthesis"  # one bounce to fix the unbacked figures


def build_graph(
    context: AgentContext, *, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """Compile the graph against `context`. Tests inject fakes; dev injects real.

    `checkpointer` is left unset for the LangGraph platform (`langgraph dev` /
    deploy manage persistence); tests pass a `MemorySaver` so the policy
    `interrupt()` can pause and resume.
    """
    builder = StateGraph(AgentState)
    builder.add_node("scope", make_scope_node(context))
    builder.add_node("prefetch", make_prefetch_node(context))
    builder.add_node("synthesis", make_synthesis_node(context))
    builder.add_node("audit", make_audit_node(context))
    builder.add_node("policy_prepare", make_policy_prepare_node(context))
    builder.add_node("policy_confirm", make_policy_confirm_node(context))

    builder.add_edge(START, "scope")
    builder.add_conditional_edges(
        "scope",
        _route_after_scope,
        {"prefetch": "prefetch", "policy_prepare": "policy_prepare", END: END},
    )
    builder.add_edge("prefetch", "synthesis")
    builder.add_edge("synthesis", "audit")
    builder.add_conditional_edges(
        "audit", _route_after_audit, {"synthesis": "synthesis", END: END}
    )
    builder.add_conditional_edges(
        "policy_prepare",
        _route_after_policy_prepare,
        {"policy_confirm": "policy_confirm", END: END},
    )
    builder.add_edge("policy_confirm", END)
    return builder.compile(checkpointer=checkpointer)
