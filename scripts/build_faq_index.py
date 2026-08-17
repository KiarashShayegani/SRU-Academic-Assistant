#!/usr/bin/env python3
"""Build the FAQ vector index from a CSV of سوال / پاسخ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sru_assistant.config import get_settings
from sru_assistant.data.faq_builder import build_faq_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAQ LanceDB table")
    parser.add_argument("--csv", type=Path, required=True, help="Path to FAQ CSV")
    parser.add_argument("--table", default=None)
    parser.add_argument("--question-col", default="سوال")
    parser.add_argument("--answer-col", default="پاسخ")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    settings = get_settings()
    table = args.table or settings.faq_table

    n = build_faq_table(
        args.csv,
        table_name=table,
        question_col=args.question_col,
        answer_col=args.answer_col,
        batch_size=args.batch_size,
    )
    print(f"✅ Table '{table}' created with {n} FAQ entries")


if __name__ == "__main__":
    main()
