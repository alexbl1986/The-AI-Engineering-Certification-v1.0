"""Synthesis: deterministic fact digest, grounded generation, cold-start block.

Offline: a fake chat model returns a canned answer, so we test that the digest is
built verbatim from the typed evidence, that the upload block is appended
deterministically, and that a route with no evidence refuses instead of inventing.
"""

from langchain_core.messages import AIMessage, HumanMessage

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.state import EvidenceItem
from app.graphs.trading_assistant.synthesis import build_facts, make_synthesis_node
from app.trading.domain import (
    ExposureCheck,
    ExposureReport,
    MissingData,
    ScaleOutCandidate,
    ScaleOutSignal,
)


def _exposure_evidence():
    report = ExposureReport(
        nav=120_000.0,
        checks=(ExposureCheck(label="options", value_base=8760.0, pct_of_nav=0.073, limit=0.10),),
    )
    return EvidenceItem("check_exposure", ok=True, result=report)


# --- build_facts ---------------------------------------------------------


def test_digest_formats_exposure_numbers_verbatim():
    digest, snippets, asks = build_facts([_exposure_evidence()], [])
    assert "7.3% of NAV" in digest
    assert "limit 10%" in digest
    assert "$120,000" in digest
    assert "within policy" in digest
    assert snippets == "" and asks == ()


def test_scaleout_breach_is_flagged_in_digest():
    cand = ScaleOutCandidate("AAOI", ScaleOutSignal.SCALE_OUT, 1.2, 2.0, 4.4)
    digest, _, _ = build_facts([EvidenceItem("scan_scaleout", ok=True, result=[cand])], [])
    assert "AAOI: +120% vs entry" in digest and "scale_out" in digest


def test_missing_becomes_upload_asks():
    _, _, asks = build_facts([], [MissingData("ledger", "Upload a recent activity statement.")])
    assert asks == ("- ledger: Upload a recent activity statement.",)


# --- synthesis node ------------------------------------------------------


class _FakeChat:
    def __init__(self, text):
        self.text = text
        self.seen = None

    def invoke(self, messages):
        self.seen = messages
        return AIMessage(content=self.text)


def _run(evidence, missing, answer="You're at 7.3% of NAV, within your 10% limit."):
    node = make_synthesis_node(AgentContext(chat_model=_FakeChat(answer)))
    return node({"messages": [HumanMessage(content="am I ok?")], "evidence": evidence, "missing": missing})


def test_synthesis_grounds_answer_and_counts_attempt():
    out = _run([_exposure_evidence()], [])
    assert "7.3%" in out["synthesis"].answer
    assert "7.3% of NAV" in out["synthesis"].grounding
    assert out["synthesis_attempts"] == 1


def test_synthesis_appends_upload_block_deterministically():
    out = _run([_exposure_evidence()], [MissingData("ledger", "Upload a recent activity statement.")])
    answer = out["synthesis"].answer
    assert "To answer the rest, upload:" in answer
    assert "ledger: Upload a recent activity statement." in answer


def test_synthesis_refuses_when_no_evidence_or_missing():
    out = _run([], [])
    assert "don't have the tools" in out["synthesis"].answer
    assert out["synthesis"].grounding == ""
