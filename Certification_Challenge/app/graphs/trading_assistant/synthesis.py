"""Synthesis node (ADR-0006, step 4).

The numbers are deterministic; the prose is the LLM's. `build_facts` turns the
evidence table into a fact digest with every figure pre-formatted (copied
verbatim from the typed tool outputs — synthesis never recomputes), plus the
quoted desk snippets and the cold-start upload lines. The model then writes a
grounded answer from that digest only; the audit node (step 5) enforces that it
did. The upload lines are appended in code so the cold-start remedy always
reaches the user regardless of what the model wrote.
"""

from __future__ import annotations

from typing import Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.state import AgentState, EvidenceItem, SynthesisResult
from app.graphs.trading_assistant.util import latest_user_text, message_text
from app.trading.domain import MissingData

SYNTHESIS_SYSTEM = """You are a retail options day-trader's assistant. Answer his question
directly and concretely, grounded ONLY in the evidence provided.

Hard rules:
- Every dollar amount and every percentage in your answer MUST appear in the evidence
  digest or the desk snippets. Never compute, estimate, or round to a new number.
- Give a clear verdict, then the specific actions his rules imply (what to change).
- When the daily and weekly desk views conflict, the daily view wins.
- Keep it tight: a short paragraph or a few bullets.
- If audit feedback is present, fix exactly those figures and change nothing else.
Do not restate the missing-data upload lines; they are appended for you."""

BRIEFING_SYSTEM = """You are producing the trader's MORNING BRIEFING — a scannable rundown
of where he stands going into the day, grounded ONLY in the evidence provided.

Same number rules as always: every dollar amount and percentage MUST come from the
evidence; never compute or invent one. When the daily and weekly desk views conflict, the
daily wins. Structure the answer under these headings, each a few short bullets, and omit a
heading if there is no evidence for it:
1. **Book status** — exposure vs limit, unrealized P/L, position count.
2. **Rule flags** — any breach or scale-out signal; say "all clear" if none.
3. **Desk's read** — what the reviews say to watch, in his terms.
4. **What to watch today** — the concrete things to act on or monitor.
If audit feedback is present, fix exactly those figures and change nothing else.
Do not restate the missing-data upload lines; they are appended for you."""


def make_synthesis_node(context: AgentContext) -> Callable[[AgentState], dict]:
    def synthesis_node(state: AgentState) -> dict:
        evidence = state.get("evidence", [])
        missing = state.get("missing", [])
        attempts = state.get("synthesis_attempts", 0) + 1

        digest, snippets, upload_asks = build_facts(evidence, missing)
        grounding = "\n\n".join(part for part in (digest, snippets) if part)

        if not grounding and not upload_asks:
            return {
                "synthesis": SynthesisResult(
                    answer=(
                        "I can help with your book, your exposure rules, and your desk's "
                        "reviews, but I don't have the tools wired for that yet."
                    ),
                    grounding="",
                    upload_asks=(),
                ),
                "synthesis_attempts": attempts,
            }

        scope = state.get("scope")
        system = BRIEFING_SYSTEM if scope and "daily_briefing" in scope.intents else SYNTHESIS_SYSTEM
        answer = _generate(
            context.chat_model,
            system=system,
            question=latest_user_text(state["messages"]),
            digest=digest,
            snippets=snippets,
            upload_asks=upload_asks,
            feedback=state.get("audit_feedback"),
        )
        answer = _append_upload_block(answer, upload_asks)
        return {
            "synthesis": SynthesisResult(
                answer=answer, grounding=grounding, upload_asks=tuple(upload_asks)
            ),
            "synthesis_attempts": attempts,
        }

    return synthesis_node


def _generate(chat_model, *, system, question, digest, snippets, upload_asks, feedback) -> str:
    sections = [f"QUESTION:\n{question}", f"EVIDENCE DIGEST (numbers you may use):\n{digest or '(none)'}"]
    if snippets:
        sections.append(f"DESK SNIPPETS (quote as needed):\n{snippets}")
    if upload_asks:
        sections.append("MISSING STORES:\n" + "\n".join(upload_asks))
    if feedback:
        sections.append(f"AUDIT FEEDBACK (fix exactly this):\n{feedback}")
    messages = [
        SystemMessage(content=system),
        HumanMessage(content="\n\n".join(sections)),
    ]
    return message_text(chat_model.invoke(messages)).strip()


def _append_upload_block(answer: str, upload_asks: Sequence[str]) -> str:
    if not upload_asks:
        return answer
    return answer + "\n\n**To answer the rest, upload:**\n" + "\n".join(upload_asks)


# -- deterministic fact extraction ---------------------------------------


def build_facts(
    evidence: Sequence[EvidenceItem], missing: Sequence[MissingData]
) -> tuple[str, str, tuple[str, ...]]:
    """Evidence table -> (fact digest, quoted desk snippets, upload lines)."""
    sections: list[str] = []
    snippets: list[str] = []
    notes: list[str] = []

    for item in evidence:
        if item.tool == "check_exposure" and item.ok:
            sections.append(_exposure(item.result))
        elif item.tool == "scan_scaleout" and item.ok:
            sections.append(_scaleout(item.result))
        elif item.tool == "open_position_pnl" and item.ok:
            sections.append(_pnl(item.result))
        elif item.tool == "list_positions" and item.ok:
            sections.append(f"### Positions\n- {len(item.result)} open lines in the book")
        elif item.tool.startswith("get_trades:") and item.ok:
            sections.append(_campaigns(item.tool.split(":", 1)[1], item.result))
        elif item.tool == "search_desk_reviews" and item.ok:
            snippets.extend(_snippets(item.result))
        elif item.note:
            notes.append(item.note)

    if notes:
        sections.append("### Open items\n" + "\n".join(f"- {n}" for n in notes))

    upload_asks = tuple(f"- {m.store}: {m.remedy}" for m in missing)
    return "\n\n".join(sections), "\n\n".join(snippets), upload_asks


def _exposure(report) -> str:
    lines = ["### Exposure vs policy", f"- account NAV: ${report.nav:,.0f}"]
    for check in report.checks:
        status = "within policy" if check.within_policy else "BREACH"
        lines.append(
            f"- {check.label}: {check.pct_of_nav:.1%} of NAV "
            f"(limit {check.limit:.0%}) — {status}"
        )
    return "\n".join(lines)


def _scaleout(candidates) -> str:
    if not candidates:
        return "### Scale-out\n- no open position has hit a scale-out threshold"
    lines = ["### Scale-out signals"]
    for c in candidates:
        lines.append(
            f"- {c.symbol}: +{c.gain:.0%} vs entry "
            f"(${c.avg_entry_price:.2f} → ${c.mark_price:.2f}) — {c.signal.value}"
        )
    return "\n".join(lines)


def _pnl(report) -> str:
    return (
        "### Unrealized P/L\n"
        f"- total unrealized: ${report.total_unrealized_pl:,.0f} "
        f"across {len(report.lines)} priced positions"
    )


def _campaigns(ticker: str, campaigns) -> str:
    if not campaigns:
        return f"### {ticker} campaign\n- no open {ticker} campaign"
    lines = [f"### {ticker} campaign"]
    for camp in campaigns:
        lines += [
            f"- net quantity: {camp.net_quantity:g}",
            f"- average entry: ${camp.avg_entry_price:.2f}",
            f"- realized so far: ${camp.realized_pl:,.0f}",
            f"- house money: {'yes' if camp.house_money else 'no'}",
        ]
    return "\n".join(lines)


def _snippets(docs) -> list[str]:
    out = []
    for d in docs[:3]:
        header = f"[{d.doc_type} {d.review_date} · {d.section or '—'}]"
        out.append(f"> {header}\n> {d.text[:600]}")
    return out
