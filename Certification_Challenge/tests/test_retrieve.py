"""Hybrid retrieval pipeline: tokenizer, RRF, and end-to-end parent recovery.

Offline throughout: in-memory Qdrant + a deterministic token-hashing embedder
whose tokenizer (\\w+) covers Hebrew, so dense retrieval works on the real
reviews without the network.
"""

import math
import re
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.rag.chunk import Chunk, Parent, chunk_document, parent_documents
from app.rag.index import CorpusIndex
from app.rag.retrieve import (
    HybridRetriever,
    RetrievedDoc,
    reciprocal_rank_fusion,
    tokenize,
)

DIM = 256
REVIEWS = Path(__file__).resolve().parent.parent / "data" / "reviews"
DAILY = REVIEWS / "סקירת דסק יומית מידעפנים 08_07_26.pdf"


class FakeEmbedder:
    """Unicode-aware token-hashing bag-of-words -> L2-normalized vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for tok in tokenize(text):
            vec[hash(tok) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def _index() -> CorpusIndex:
    return CorpusIndex(QdrantClient(location=":memory:"), FakeEmbedder(), vector_size=DIM)


# --- tokenizer -----------------------------------------------------------


def test_tokenizer_covers_hebrew():
    # The whole point: the course's [a-z0-9]+ would return [] for Hebrew.
    assert tokenize("זיכרון HBM ו-SMR 25%") == ["זיכרון", "hbm", "ו", "smr", "25"]


# --- RRF -----------------------------------------------------------------


def _doc(doc_id, parent_id="p"):
    return RetrievedDoc(
        id=doc_id, text="x", score=None, doc_type="daily", review_date=None,
        source="s", section=None, parent_id=parent_id, pages=(1,), tickers=(),
    )


def test_rrf_rewards_agreement_across_lists():
    list_a = [_doc("1"), _doc("2"), _doc("3")]
    list_b = [_doc("2"), _doc("4"), _doc("1")]
    fused = reciprocal_rank_fusion([list_a, list_b], limit=4)
    # id2 (ranks 2,1) edges id1 (ranks 1,3); singletons follow.
    assert [d.id for d in fused] == ["2", "1", "4", "3"]
    assert [d.score for d in fused] == sorted((d.score for d in fused), reverse=True)


# --- end-to-end pipeline -------------------------------------------------


def _synthetic_corpus():
    parents = [
        Parent("daily-d-s00", "Memory HBM DRAM shortage Micron capex pricing power full section",
               "daily", "2026-07-08", "d.pdf", "Memory", (1,), ("MU",)),
        Parent("daily-d-s01", "Iran oil Oman risk premium macro energy full section",
               "daily", "2026-07-08", "d.pdf", "Oil", (2,), ()),
    ]
    children = [
        _child("daily-d-s00-c00", "daily-d-s00", "Memory HBM DRAM shortage Micron", "Memory", ("MU",)),
        _child("daily-d-s00-c01", "daily-d-s00", "capex memory pricing power", "Memory", ()),
        _child("daily-d-s01-c00", "daily-d-s01", "Iran oil Oman risk premium", "Oil", ()),
    ]
    return children, parents


def _child(chunk_id, parent_id, text, section, tickers):
    return Chunk(
        text=text, doc_type="daily", review_date="2026-07-08", source="d.pdf",
        chunk_id=chunk_id, parent_id=parent_id, section=section, pages=(1,),
        tickers=tickers, start_index=0,
    )


def test_retrieve_returns_deduped_full_section_parents():
    idx = _index()
    children, parents = _synthetic_corpus()
    idx.replace_document(children, parents, user_id="alex")

    results = HybridRetriever(idx).retrieve("DRAM shortage Micron", user_id="alex", k=2)

    assert results[0].id == "daily-d-s00"  # parent id, not a child id
    assert results[0].text == parents[0].text  # full section, not the child snippet
    assert results[0].parent_id == "daily-d-s00"
    assert len({r.id for r in results}) == len(results)  # each parent once


def test_bm25_finds_exact_lexical_token():
    idx = _index()
    children, parents = _synthetic_corpus()
    idx.replace_document(children, parents, user_id="alex")

    hits = HybridRetriever(idx).bm25("Oman", user_id="alex", k=1)
    assert hits and hits[0].parent_id == "daily-d-s01"


def test_retrieval_respects_user_isolation():
    idx = _index()
    children, parents = _synthetic_corpus()
    idx.replace_document(children, parents, user_id="alex")
    idx.replace_document(
        [_child("daily-d-s00-c00", "daily-d-s09", "Palladium squeeze demo only", "PD", ("PALL",))],
        [Parent("daily-d-s09", "Palladium squeeze synthetic demo section", "daily",
                "2026-07-08", "d.pdf", "PD", (1,), ("PALL",))],
        user_id="demo",
    )

    alex = HybridRetriever(idx).retrieve("palladium squeeze", user_id="alex", k=5)
    assert all("Palladium" not in r.text for r in alex)


@pytest.mark.parametrize("query,needle", [("TSMC PIC CPO אופטית צוואר בקבוק", "TSMC")])
def test_end_to_end_on_real_daily_review(query, needle):
    idx = _index()
    idx.replace_document(
        chunk_document(str(DAILY)), parent_documents(str(DAILY)), user_id="alex"
    )
    results = HybridRetriever(idx).retrieve(query, user_id="alex", k=3)

    assert results
    assert all("-c" not in r.id for r in results)  # parents, not children
    assert any(needle in r.text for r in results)
