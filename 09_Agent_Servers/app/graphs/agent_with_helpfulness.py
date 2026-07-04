from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from app.graphs.simple_agent import graph as simple_agent
from app.models import get_judge_model

# Safety valve: never re-answer more than this many times.
MAX_ATTEMPTS = 3

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. Given a user question and an assistant's answer, "
    "decide whether the answer was related to cat health and genuinely helpful: relevant, correct, and complete "
    "enough to satisfy the question. Did the answer use the tools provided to answer the question? Reply with exactly 'Y' if it is helpful, or 'N' "
    "if it is not. Reply with only that single character."
)


class HelpfulnessState(MessagesState):
    """MessagesState (messages + add_messages reducer) plus loop bookkeeping."""

    attempts: int
    helpful: bool


def call_agent(state: HelpfulnessState) -> dict:
    """Run the existing simple_agent, appending only its new messages."""
    prior = state["messages"]
    result = simple_agent.invoke({"messages": prior})
    new_messages = result["messages"][len(prior):]
    return {
        "messages": new_messages,
        "attempts": state.get("attempts", 0) + 1,
    }


def check_helpfulness(state: HelpfulnessState) -> dict:
    """Judge model rates the latest answer against the latest user question."""
    messages = state["messages"]
    query = next(
        (m.text for m in reversed(messages) if isinstance(m, HumanMessage)),
        "",
    )
    answer = messages[-1].text

    verdict = get_judge_model().with_config(tags=["nostream"]).invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                content=f"User question:\n{query}\n\nAssistant answer:\n{answer}"
            ),
        ]
    )
    helpful = str(verdict.text).strip().upper().startswith("Y")
    return {"helpful": helpful}


def route(state: HelpfulnessState) -> str:
    """Loop back on an unhelpful answer, but respect the attempt limit."""
    if state.get("helpful"):
        return END
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return END
    return "agent"


builder = StateGraph(HelpfulnessState)
builder.add_node("agent", call_agent)
builder.add_node("judge", check_helpfulness)
builder.add_edge(START, "agent")
builder.add_edge("agent", "judge")
builder.add_conditional_edges("judge", route, {"agent": "agent", END: END})

graph = builder.compile()
