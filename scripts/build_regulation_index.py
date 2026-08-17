#!/usr/bin/env python3
"""Build the regulation vector index from a cleaned .txt file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sru_assistant.config import get_settings
from sru_assistant.data.chunking import process_text_file
from sru_assistant.embeddings.model import EmbeddingModel
from sru_assistant.vectorstore.lancedb_store import LanceDBStore
from sru_assistant.vectorstore.schemas import regulation_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regulation LanceDB table")
    parser.add_argument(
        "--text",
        type=Path,
        required=True,
        help="Path to cleaned regulation .txt (with (صفحه N) markers)",
    )
    parser.add_argument("--table", default=None, help="Override table name")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    settings = get_settings()
    table_name = args.table or settings.regulation_table

    print(f"Loading text: {args.text}")
    chunks = process_text_file(args.text)
    print(f"Chunks: {len(chunks)}")
    sizes = [c["char_length"] for c in chunks]
    print(f"  min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.0f}")

    print("Loading embedding model...")
    embedder = EmbeddingModel()
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, batch_size=args.batch_size, show_progress=True)

    records = [
        regulation_record(
            chunk_id=c["chunk_id"],
            page_number=c["page_number"],
            text=c["text"],
            source_file=c["source_file"],
            vector=vectors[i].tolist(),
        )
        for i, c in enumerate(chunks)
    ]

    store = LanceDBStore()
    store.create_table(table_name, records, replace=True)
    print(f"✅ Table '{table_name}' created with {len(records)} records at {store.db_path}")


if __name__ == "__main__":
    main()
