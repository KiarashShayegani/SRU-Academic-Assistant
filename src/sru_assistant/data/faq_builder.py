"""Build the FAQ LanceDB table from a CSV of سوال / پاسخ pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sru_assistant.config import get_settings
from sru_assistant.embeddings.model import EmbeddingModel, get_embedding_model
from sru_assistant.vectorstore.lancedb_store import LanceDBStore
from sru_assistant.vectorstore.schemas import faq_record


def build_faq_table(
    csv_path: str | Path,
    *,
    table_name: str | None = None,
    question_col: str = "سوال",
    answer_col: str = "پاسخ",
    embedder: EmbeddingModel | None = None,
    store: LanceDBStore | None = None,
    batch_size: int = 32,
) -> int:
    settings = get_settings()
    table_name = table_name or settings.faq_table
    embedder = embedder or get_embedding_model()
    store = store or LanceDBStore()

    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    if question_col not in df.columns or answer_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{question_col}' and '{answer_col}'. "
            f"Found: {list(df.columns)}"
        )

    questions = df[question_col].astype(str).tolist()
    answers = df[answer_col].astype(str).tolist()

    vectors = embedder.encode(questions, batch_size=batch_size, show_progress=True)

    records = [
        faq_record(
            id_=i,
            question=questions[i],
            answer=answers[i],
            vector=vectors[i].tolist(),
        )
        for i in range(len(df))
    ]

    store.create_table(table_name, records, replace=True)
    return len(records)
