"""Lightweight PDF text extraction helpers (optional dependency: PyPDF2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def clean_whitespace(text: str) -> str:
    return " ".join(text.split())


def extract_pdf(path: str | Path) -> dict[str, Any]:
    """
    Extract text page-by-page from a PDF.
    Requires PyPDF2. Prefer the manually cleaned .txt for production indexes.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError as e:
        raise ImportError(
            "PyPDF2 is required for PDF extraction. Install with: pip install 'sru-academic-assistant[pdf]'"
        ) from e

    path = Path(path)
    reader = PdfReader(path)
    pages: list[dict[str, Any]] = []
    full_parts: list[str] = []

    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = clean_whitespace(raw)
        pages.append({"page_number": i + 1, "text": cleaned})
        full_parts.append(cleaned)

    return {
        "file_name": path.stem,
        "source_file": path.name,
        "full_text": "\n".join(full_parts),
        "pages": pages,
        "total_pages": len(pages),
    }
