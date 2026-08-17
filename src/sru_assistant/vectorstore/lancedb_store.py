"""Unified LanceDB access for regulation chunks and FAQ entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from sru_assistant.config import get_settings


REGULATION_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("char_length", pa.int32()),
        pa.field("source_file", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=384)),
    ]
)

FAQ_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int32()),
        pa.field("question", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=384)),
    ]
)


class LanceDBStore:
    def __init__(self, db_path: str | Path | None = None):
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else settings.db_path_abs
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))

    def table_names(self) -> list[str]:
        return list(self.db.table_names())

    def open_table(self, name: str):
        return self.db.open_table(name)

    def drop_table_if_exists(self, name: str) -> None:
        if name in self.db.table_names():
            self.db.drop_table(name)

    def create_table(self, name: str, data: list[dict[str, Any]], *, replace: bool = True):
        if replace:
            self.drop_table_if_exists(name)
        return self.db.create_table(name, data=data)

    def search(
        self,
        table_name: str,
        query_vector: list[float] | Any,
        *,
        k: int = 5,
        metric: str = "cosine",
    ) -> list[dict[str, Any]]:
        table = self.open_table(table_name)
        query = table.search(query_vector).limit(k)
        try:
            query = query.metric(metric)
        except Exception:
            # Older LanceDB without .metric()
            pass
        return query.to_list()
