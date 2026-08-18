# sru-academic-assistant

**دستیار هوشمند آیین‌نامه دانشگاه تربیت دبیر شهید رجایی**

A **FAQ + RAG** academic assistant that answers junior students’ questions about educational regulations at *Shahid Rajaee Teacher Training University* (SRU).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[web-app screenshot](assets/app_screenshot.png)

---

## Features

- **Dual pipelines**
  - **FAQ** — fast nearest-neighbour lookup over curated Q&A (no LLM required)
  - **RAG** — page-aware retrieval over the full regulation text + streaming LLM answers
- **Persian-first UI** (RTL, Vazirmatn, categorised quick questions)
- **Modular Python package** with typed config, clean retrieval/generation separation
- **Offline evaluation** script + golden question set
- **Reproducible index builds** via CLI scripts

## Architecture (short)

```
Question → [FAQ mode] → LanceDB QA table → answer
         → [RAG mode] → LanceDB regulation chunks → prompt → LLM stream
```

See [docs/architecture.md](docs/architecture.md) for details.

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USER/sru-academic-assistant.git
cd sru-academic-assistant

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set AGNES_API_KEY (or leave empty for FAQ-only)
```

Place your MiniLM model under `models/MiniLM` (or set `SRU_MODEL_PATH`).

### 3. Build indexes

```bash
# Regulation (cleaned text with (صفحه N) markers)
python scripts/build_regulation_index.py --text path/to/Ayeen_Name_1402.txt

# FAQ
python scripts/build_faq_index.py --csv path/to/Ayeen_Name_FAQ_v2.csv
```

### 4. Run the app

```bash
streamlit run src/sru_assistant/ui/streamlit_app.py
```

## Project structure

```
sru-academic-assistant/
├── config/default.yaml          # Non-secret defaults
├── src/sru_assistant/           # Installable package
│   ├── embeddings/
│   ├── vectorstore/
│   ├── retrieval/
│   ├── generation/
│   ├── pipeline/
│   ├── data/
│   └── ui/
├── assets/css/                  # Streamlit theme (separated from app logic)
├── scripts/                     # Index build + evaluation CLIs
├── tests/
├── data/evaluation/             # Golden questions
├── docs/
└── evaluations/
```

## Evaluation

```bash
python scripts/evaluate_retrieval.py --mode both --k 5
```

Edit `data/evaluation/sample_questions.json` to expand the golden set.

## Configuration

| Variable | Purpose |
|----------|---------|
| `AGNES_API_KEY` | OpenAI-compatible LLM key (optional for FAQ-only) |
| `AGNES_BASE_URL` | Provider base URL |
| `AGNES_MODEL` | Model id |
| `SRU_MODEL_PATH` | Local SentenceTransformer path |
| `SRU_DB_PATH` | LanceDB directory |
| `SRU_REGULATION_TABLE` / `SRU_FAQ_TABLE` | Table names |

Non-secret defaults live in `config/default.yaml`.

## Development

```bash
pytest
ruff check src tests
```

## Limitations & honesty

- Embedding model is MiniLM (384-d, 128-token context). Chunk size is tuned accordingly.
- Original PDF extraction is noisy; production indexes use **LLM-assisted cleaned text** with explicit page markers.
- FAQ quality depends on the curated CSV; RAG quality depends on retrieval + prompt + LLM.
- Not a substitute for official university advice — always verify critical decisions with the education office.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built for students of *دانشگاه تربیت دبیر شهید رجایی*. Regulations content is property of the university / Ministry sources cited in the source document.
