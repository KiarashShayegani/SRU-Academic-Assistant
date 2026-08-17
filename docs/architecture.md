# Architecture

## Overview

`sru-academic-assistant` answers student questions about Shahid Rajaee Teacher Training University educational regulations using two complementary pipelines:

| Mode | Data source | LLM? | Best for |
|------|-------------|------|----------|
| **FAQ** | Pre-authored Q&A pairs in LanceDB | No | Common, well-phrased questions |
| **RAG** | Page-aware regulation chunks + LLM | Yes | Novel / complex questions |

```
User question
     │
     ├─ mode=faq ──► FAQRetriever (cosine NN) ──► answer + similarity
     │
     └─ mode=rag ──► RegulationRetriever (top-k) ──► prompt ──► LLM stream
```

## Package layout

```
src/sru_assistant/
├── config.py              # Pydantic Settings + YAML
├── embeddings/            # SentenceTransformer wrapper
├── vectorstore/           # LanceDB + schemas
├── retrieval/             # FAQ + regulation retrievers
├── generation/            # LLM client + prompts
├── pipeline/              # answer_question() entry point
├── data/                  # chunking, FAQ builder, PDF helpers
└── ui/                    # Streamlit app (CSS externalised)
```

## Design choices

- **Cosine distance** for FAQ so `similarity = 1 - distance` is meaningful for display.
- **Page markers** `(صفحه N)` are kept inside chunk text so the LLM can cite pages and the UI can show source pages.
- **No automatic FAQ↔RAG fallback** by default — the user explicitly chooses the pipeline (product clarity). Optional degradation to FAQ only when the LLM key is missing.
- **Config** lives in `config/default.yaml`; secrets only via environment / `.env`.

## Data flow (index build)

1. Clean regulation text (LLM-assisted or manual) with page markers → `process_text_file`
2. Embed chunks → LanceDB table `university_regulation_v2`
3. FAQ CSV (`سوال`, `پاسخ`) → embed questions → LanceDB table `QA_v2`
