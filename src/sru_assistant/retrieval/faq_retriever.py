"""Nearest-neighbour FAQ lookup over the QA table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sru_assistant.config import get_settings
from sru_assistant.embeddings.model import EmbeddingModel, get_embedding_model
from sru_assistant.vectorstore.lancedb_store import LanceDBStore


@dataclass
class FAQHit:
    question: str
    answer: str
    similarity: float
    distance: float
    raw: dict[str, Any]


class FAQRetriever:
    def __init__(
        self,
        store: LanceDBStore | None = None,
        embedder: EmbeddingModel | None = None,
        table_name: str | None = None,
    ):
        settings = get_settings()
        self.store = store or LanceDBStore()
        self.embedder = embedder or get_embedding_model()
        self.table_name = table_name or settings.faq_table
        self.k = settings.faq_k

    def retrieve(self, question: str, k: int | None = None) -> list[FAQHit]:
        k = k if k is not None else self.k
        query_vec = self.embedder.encode_one(question).tolist()
        rows = self.store.search(self.table_name, query_vec, k=k, metric="cosine")

        hits: list[FAQHit] = []
        for row in rows:
            distance = float(row.get("_distance", 1.0))
            # Cosine distance in LanceDB ≈ 1 - cosine_similarity
            similarity = max(0.0, 1.0 - distance)
            hits.append(
                FAQHit(
                    question=row.get("question") or row.get("سوال", ""),
                    answer=row.get("answer") or row.get("پاسخ", ""),
                    similarity=similarity,
                    distance=distance,
                    raw=row,
                )
            )
        return hits

    def best(self, question: str) -> FAQHit | None:
        hits = self.retrieve(question, k=1)
        return hits[0] if hits else None
