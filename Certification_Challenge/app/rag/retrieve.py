"""Hybrid retrieval pipeline: dense + BM25 (RRF) + parent-child recovery.

This is the Task-4 base pipeline (build_plan): first-stage dense and lexical
retrieval over the child chunks are fused with reciprocal rank fusion, then each
surviving child is resolved to its full section (parent) so the model reads
whole-section context. Cohere reranking (Task 6.1) layers on top later.

BM25 is rebuilt per query from the user's child chunks in Qdrant — the corpus is
tiny (two reviews) and replace-on-upload keeps it fresh, so there is no separate
lexical index to fall out of sync. The tokenizer is Unicode-aware: the course's
``[a-z0-9]+`` would drop every Hebrew token, which is most of this corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi

from app.rag.index import CorpusIndex, SearchHit

_TOKEN = re.compile(r"\w+", re.UNICODE)  # \w matches Hebrew, Latin, and digits


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class RetrievedDoc:
    id: str  # chunk_id for children, parent_id for recovered parents
    text: str
    score: float | None
    doc_type: str
    review_date: str | None
    source: str
    section: str | None
    parent_id: str
    pages: tuple[int, ...]
    tickers: tuple[str, ...]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[RetrievedDoc]],
    *,
    limit: int,
    rrf_constant: int = 60,
) -> list[RetrievedDoc]:
    """Fuse ranked lists by summed reciprocal rank, keyed on document id."""
    scores: dict[str, float] = {}
    docs: dict[str, RetrievedDoc] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            docs.setdefault(doc.id, doc)
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_constant + rank)
    ordered = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[:limit]
    return [replace(docs[doc_id], score=scores[doc_id]) for doc_id in ordered]


class HybridRetriever:
    """Dense + BM25 (RRF) over children, resolved to full-section parents."""

    def __init__(
        self,
        index: CorpusIndex,
        *,
        first_stage_k: int = 8,
        rrf_constant: int = 60,
    ) -> None:
        self._index = index
        self._first_stage_k = first_stage_k
        self._rrf_constant = rrf_constant

    def dense(self, query: str, *, user_id: str, k: int) -> list[RetrievedDoc]:
        return [_from_hit(hit) for hit in self._index.search(query, user_id=user_id, k=k)]

    def bm25(self, query: str, *, user_id: str, k: int) -> list[RetrievedDoc]:
        docs = self._index.all_children(user_id=user_id)
        if not docs:
            return []
        bm25 = BM25Okapi([tokenize(doc.text) for doc in docs])
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
        return [replace(_from_hit(docs[i]), score=float(scores[i])) for i in ranked]

    def hybrid_children(self, query: str, *, user_id: str, k: int) -> list[RetrievedDoc]:
        return reciprocal_rank_fusion(
            [
                self.dense(query, user_id=user_id, k=self._first_stage_k),
                self.bm25(query, user_id=user_id, k=self._first_stage_k),
            ],
            limit=k,
            rrf_constant=self._rrf_constant,
        )

    def retrieve(self, query: str, *, user_id: str, k: int = 5) -> list[RetrievedDoc]:
        """Full base pipeline: hybrid children resolved to unique parents."""
        children = self.hybrid_children(query, user_id=user_id, k=self._first_stage_k)
        return self._recover_parents(children, user_id=user_id, k=k)

    def _recover_parents(
        self, children: Sequence[RetrievedDoc], *, user_id: str, k: int
    ) -> list[RetrievedDoc]:
        order: list[tuple[str, float | None]] = []
        seen: set[str] = set()
        for child in children:
            if child.parent_id and child.parent_id not in seen:
                seen.add(child.parent_id)
                order.append((child.parent_id, child.score))

        parents = self._index.get_parents([pid for pid, _ in order], user_id=user_id)
        recovered: list[RetrievedDoc] = []
        for parent_id, score in order:
            hit = parents.get(parent_id)
            if hit is None:
                continue
            recovered.append(replace(_from_hit(hit, use_parent_id=True), score=score))
            if len(recovered) == k:
                break
        return recovered


def _from_hit(hit: SearchHit, *, use_parent_id: bool = False) -> RetrievedDoc:
    return RetrievedDoc(
        id=hit.parent_id if use_parent_id else hit.chunk_id,
        text=hit.text,
        score=hit.score or None,
        doc_type=hit.doc_type,
        review_date=hit.review_date,
        source=hit.source,
        section=hit.section,
        parent_id=hit.parent_id,
        pages=hit.pages,
        tickers=hit.tickers,
    )
