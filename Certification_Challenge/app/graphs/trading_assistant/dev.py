"""Local / Studio wiring: the real gateway-backed graph `langgraph dev` serves.

`langgraph.json` points at the module-level `graph` below. This builds a real
`AgentContext`:
  * chat model  -> the verified gateway slug `openai/gpt-5.4-mini`;
  * retriever   -> real OpenAI `text-embedding-3-large` over the committed desk
                   reviews in `data/reviews/`, in an in-memory Qdrant (no cloud
                   provisioning needed to iterate locally);
  * portfolio   -> the trader's local, gitignored CSVs in `data/private/` parsed
                   into positions / trades / NAV, seeded under one dev user.

Everything degrades: if the embeddings/PDFs or the private CSVs are absent (a
fresh clone, no network), the corresponding loader returns `MissingData` and the
graph still starts — you can drive the scoper and see the cold-start path. The
pytest layer builds its own context with fakes, so importing this module (and its
network calls) is never required to run the tests.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.graph import build_graph
from app.graphs.trading_assistant.policy_model import DEFAULT_POLICY, PolicyRecord
from app.rag.chunk import chunk_document, parent_documents
from app.rag.index import OPENAI_3_LARGE_DIM, CorpusIndex, openai_embedder
from app.rag.retrieve import HybridRetriever
from app.trading.domain import MissingData
from app.trading.ingest.statement import parse_account_nav, parse_activity_statement
from app.trading.ingest.tactical import parse_tactical_book

load_dotenv()

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
GATEWAY_MODEL = "openai/gpt-5.4-mini"  # verified live via the OpenAI-compat endpoint

# Seeded from the trader's real (gitignored) data, so the demo is coherent with
# the committed reviews. "alex-demo" stays reserved for synthetic isolation tests.
DEV_USER = "real-user"

_ROOT = Path(__file__).resolve().parents[3]  # .../Certification_Challenge
_REVIEWS = _ROOT / "data" / "reviews"
_PRIVATE = _ROOT / "data" / "private"
_TACTICAL_CSV = _PRIVATE / "Tactical_Boot.csv"
_STATEMENT_CSV = _PRIVATE / "IBKR YTD Statement.csv"


def gateway_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=GATEWAY_MODEL,
        base_url=GATEWAY_BASE_URL,
        api_key=os.environ["AI_GATEWAY_API_KEY"],
        temperature=0,  # a router wants stable routing (gateway accepts it for this slug)
    )


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else None


def _build_retriever(user_id: str) -> HybridRetriever | None:
    """Index the committed reviews with real embeddings; None if unavailable."""
    pdfs = sorted(_REVIEWS.glob("*.pdf"))
    if not pdfs:
        return None
    try:
        index = CorpusIndex(
            QdrantClient(location=":memory:"),
            openai_embedder(),
            vector_size=OPENAI_3_LARGE_DIM,
        )
        for pdf in pdfs:
            index.replace_document(
                chunk_document(str(pdf)), parent_documents(str(pdf)), user_id=user_id
            )
        return HybridRetriever(index)
    except Exception as exc:  # noqa: BLE001 - dev convenience: start even offline
        print(f"[dev] retriever unavailable ({exc!r}); desk questions will cold-start")
        return None


def _positions_loader():
    content = _read(_TACTICAL_CSV)
    positions = parse_tactical_book(content) if content else None
    missing = MissingData("positions snapshot", "Upload your tactical book export.")
    return lambda user_id: positions if positions is not None else missing


def _trades_loader():
    content = _read(_STATEMENT_CSV)
    trades = parse_activity_statement(content) if content else None
    missing = MissingData("ledger", "Upload a recent activity statement.")
    return lambda user_id: trades if trades is not None else missing


def _nav_loader():
    content = _read(_STATEMENT_CSV)
    nav = parse_account_nav(content) if content else None
    return lambda user_id: nav


# In-process policy store: persists across turns within one `langgraph dev` run.
# The deploy build swaps this for the LangGraph Store / Postgres, same seam.
_POLICY: dict[str, PolicyRecord] = {}


def dev_context() -> AgentContext:
    return AgentContext(
        chat_model=gateway_chat_model(),
        retriever=_build_retriever(DEV_USER),
        load_positions=_positions_loader(),
        load_trades=_trades_loader(),
        load_nav=_nav_loader(),
        load_policy=lambda user_id: _POLICY.get(user_id, DEFAULT_POLICY),
        save_policy=lambda user_id, policy: _POLICY.__setitem__(user_id, policy),
        default_user_id=DEV_USER,
    )


graph = build_graph(dev_context())
