"""Arrow schemas and record builders for LanceDB tables."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

EMBEDDING_DIM = 384

REGULATION_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("char_length", pa.int32()),
        pa.field("source_file", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=EMBEDDING_DIM)),
    ]
)

FAQ_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int32()),
        pa.field("question", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=EMBEDDING_DIM)),
    ]
)


def regulation_record(
    *,
    chunk_id: str,
    page_number: int,
    text: str,
    source_file: str,
    vector: list[float],
) -> dict[str, Any]:
    return {
        "chunk_id": str(chunk_id),
        "page_number": int(page_number),
        "text": text,
        "char_length": len(text),
        "source_file": source_file,
        "vector": vector,
    }


def faq_record(
    *,
    id_: int,
    question: str,
    answer: str,
    vector: list[float],
) -> dict[str, Any]:
    return {
        "id": int(id_),
        "question": question,
        "answer": answer,
        "vector": vector,
    }
