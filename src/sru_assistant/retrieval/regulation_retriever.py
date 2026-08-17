"""Semantic retrieval over regulation chunks (page-aware)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sru_assistant.config import get_settings
from sru_assistant.embeddings.model import EmbeddingModel, get_embedding_model
from sru_assistant.vectorstore.lancedb_store import LanceDBStore


@dataclass
class RegulationChunk:
    chunk_id: str
    page_number: int
    text: str
    char_length: int
    source_file: str
    distance: float
    raw: dict[str, Any]


class RegulationRetriever:
    def __init__(
        self,
        store: LanceDBStore | None = None,
        embedder: EmbeddingModel | None = None,
        table_name: str | None = None,
    ):
        settings = get_settings()
        self.store = store or LanceDBStore()
        self.embedder = embedder or get_embedding_model()
        self.table_name = table_name or settings.regulation_table
        self.k = settings.regulation_k

    def retrieve(self, question: str, k: int | None = None) -> list[RegulationChunk]:
        k = k if k is not None else self.k
        query_vec = self.embedder.encode_one(question).tolist()
        rows = self.store.search(self.table_name, query_vec, k=k, metric="cosine")

        chunks: list[RegulationChunk] = []
        for row in rows:
            chunks.append(
                RegulationChunk(
                    chunk_id=str(row.get("chunk_id", "")),
                    page_number=int(row.get("page_number", 0)),
                    text=row.get("text", ""),
                    char_length=int(row.get("char_length", len(row.get("text", "")))),
                    source_file=row.get("source_file", ""),
                    distance=float(row.get("_distance", 1.0)),
                    raw=row,
                )
            )
        return chunks
