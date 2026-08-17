# Data pipeline

## Regulation text

1. **Source**: Official PDF (`Ayeen_Name_1402.pdf`).
2. **Raw extract**: `PyPDF2` (noisy OCR / layout issues). Kept for reference only.
3. **Production text**: LLM-assisted cleaned `.txt` with explicit markers `(صفحه N)`.
   - This is the **source of truth** for indexing.
4. **Chunking** (`sru_assistant.data.chunking`):
   - Split on page markers → one candidate per page.
   - Pages longer than `max_chars_per_page` (default 850) are split on sentence boundaries with overlap.
   - Page marker is preserved on every sub-chunk so the LLM and UI can cite pages.

## FAQ

1. Curated CSV with columns `سوال` / `پاسخ`.
2. Embed **questions only** (answers are returned by nearest-neighbour lookup).
3. Stored in LanceDB table `QA_v2` (configurable).

## Rebuild commands

```bash
python scripts/build_regulation_index.py --text data/processed/Ayeen_Name_1402.txt
python scripts/build_faq_index.py --csv data/processed/Ayeen_Name_FAQ_v2.csv
```

Indexes are written under `db/lancedb/` (gitignored).
