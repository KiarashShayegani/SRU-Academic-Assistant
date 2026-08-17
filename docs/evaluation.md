# Evaluation

## Golden set

`data/evaluation/sample_questions.json` holds representative student questions with:

- `expected_keywords` — strings that should appear in retrieved regulation text
- `expected_pages` — page numbers that should appear among top-k hits

Expand this set as you collect real user questions.

## Running offline eval

```bash
python scripts/evaluate_retrieval.py --mode both --k 5
```

Reports:

- **Keyword hit rate@k** — at least one expected keyword appears in the joined top-k texts
- **Page hit rate@k** — at least one expected page is among retrieved pages
- FAQ nearest-neighbour similarity distribution (qualitative)

## Future metrics (roadmap)

- nDCG / MRR with graded relevance
- Faithfulness / relevance LLM-as-judge on RAG answers
- Human preference study (A/B FAQ vs RAG)

Store quantitative reports under `evaluations/` (gitignored for large artifacts; commit small summaries).
