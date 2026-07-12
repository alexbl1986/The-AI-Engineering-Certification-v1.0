"""The injected seam the whole graph runs on (house style, cf. `quotes.fetch`).

One graph, three wirings of the SAME `AgentContext`:
  * tests  -> a stub chat model, no network (deterministic, the graded layer);
  * `langgraph dev` / Studio -> the real gateway `ChatOpenAI` + seeded local data;
  * deploy -> gateway model + Postgres/Qdrant-backed loaders.

Per-user DATA (snapshot, ledger, policy) is loaded inside nodes keyed by
`user_id`, not baked into this frozen context — the server compiles the graph
once and serves every user. Those data-access callables land with the pre-fetch
node (Slice 2); the scoper needs only the chat model, so that is all Slice 1
requires here. `retriever` is optional until pre-fetch consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models import BaseChatModel

from app.graphs.trading_assistant.policy_model import PolicyRecord
from app.rag.retrieve import HybridRetriever
from app.trading.domain import MissingData, Position, Trade

# Per-user data-access seams. Each returns MissingData when the backing store was
# never uploaded, so an empty book can never be mistaken for a flat one (ADR-0006
# cold-start contract). `user_id` is passed at call time — the server compiles the
# graph once and serves every user; data is never baked into this frozen context.
PositionsLoader = Callable[[str], "list[Position] | MissingData"]
TradesLoader = Callable[[str], "list[Trade] | MissingData"]
NavLoader = Callable[[str], "float | None"]
# Policy is read-mostly and edited only through the interrupt-gated write path.
PolicyLoader = Callable[[str], PolicyRecord]
PolicySaver = Callable[[str, PolicyRecord], None]


@dataclass(frozen=True)
class AgentContext:
    """Capabilities the graph is built against; data arrives per-request."""

    chat_model: BaseChatModel
    retriever: HybridRetriever | None = None
    load_positions: PositionsLoader | None = None
    load_trades: TradesLoader | None = None
    load_nav: NavLoader | None = None
    load_policy: PolicyLoader | None = None
    save_policy: PolicySaver | None = None
    default_user_id: str = "alex-demo"
