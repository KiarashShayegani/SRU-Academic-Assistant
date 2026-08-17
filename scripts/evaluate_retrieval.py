#!/usr/bin/env python3
"""Offline retrieval evaluation against a golden question set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sru_assistant.retrieval.faq_retriever import FAQRetriever
from sru_assistant.retrieval.regulation_retriever import RegulationRetriever


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return data


def keyword_hit(text: str, keywords: list[str]) -> bool:
    t = text.replace("\u200c", "")
    return any(kw.replace("\u200c", "") in t for kw in keywords if kw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "evaluation" / "sample_questions.json",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["regulation", "faq", "both"], default="both")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} evaluation questions from {args.questions}\n")

    if args.mode in ("regulation", "both"):
        reg = RegulationRetriever()
        hits = 0
        page_hits = 0
        total = 0
        print("=" * 60)
        print("REGULATION RETRIEVAL")
        print("=" * 60)
        for q in questions:
            if "expected_keywords" not in q and "expected_pages" not in q:
                continue
            total += 1
            chunks = reg.retrieve(q["question"], k=args.k)
            joined = " ".join(c.text for c in chunks)
            kw_ok = keyword_hit(joined, q.get("expected_keywords", []))
            pages = {c.page_number for c in chunks}
            expected_pages = set(q.get("expected_pages", []))
            page_ok = bool(pages & expected_pages) if expected_pages else False
            if kw_ok:
                hits += 1
            if page_ok:
                page_hits += 1
            status = "✓" if (kw_ok or page_ok) else "✗"
            print(f"  {status} {q['question'][:60]}...")
            print(f"      pages={sorted(pages)}  kw_hit={kw_ok}  page_hit={page_ok}")
        if total:
            print(f"\nKeyword hit rate@{args.k}: {hits}/{total} = {hits/total:.0%}")
            print(f"Page hit rate@{args.k}:     {page_hits}/{total} = {page_hits/total:.0%}")

    if args.mode in ("faq", "both"):
        faq = FAQRetriever()
        print("\n" + "=" * 60)
        print("FAQ RETRIEVAL (nearest neighbour similarity)")
        print("=" * 60)
        for q in questions:
            hit = faq.best(q["question"])
            if hit is None:
                print(f"  ✗ {q['question'][:50]}... → no hit")
                continue
            print(
                f"  sim={hit.similarity:.2f}  Q: {q['question'][:40]}...\n"
                f"           matched: {hit.question[:50]}..."
            )


if __name__ == "__main__":
    main()
