"""Page-aware chunking for Persian regulation text with (صفحه N) markers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sru_assistant.config import get_settings

PAGE_MARKER_RE = re.compile(r"\(\s*صفحه\s+(\d+)\s*\)")
SENTENCE_SPLIT_RE = re.compile(r"([.!؟]\s+|\n\s*\n)")


def has_page_markers(text: str) -> bool:
    return bool(PAGE_MARKER_RE.search(text))


def extract_pages(text: str, source_file: str = "unknown") -> list[dict[str, Any]]:
    """
    Split text on (صفحه N) markers. Each page becomes a candidate chunk.
    """
    matches = list(PAGE_MARKER_RE.finditer(text))
    pages: list[dict[str, Any]] = []

    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        settings = get_settings()
        if len(content) < settings.min_chunk_chars:
            continue

        marker = f"(صفحه {page_num})\n"
        pages.append(
            {
                "page_number": page_num,
                "text": marker + content,
                "char_length": len(marker) + len(content),
                "source_file": source_file,
                "chunk_id": str(page_num),
            }
        )
    return pages


def split_large_page(
    page: dict[str, Any],
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict[str, Any]]:
    """
    Split an oversized page into smaller chunks on sentence boundaries.
    Preserves the page marker on every resulting chunk.
    """
    settings = get_settings()
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = overlap if overlap is not None else settings.overlap
    max_chars = settings.max_chars_per_page

    if page["char_length"] <= max_chars:
        return [page]

    text = page["text"]
    marker_match = re.match(r"(\(\s*صفحه\s+\d+\s*\)\s*)", text)
    page_marker = marker_match.group(1) if marker_match else f"(صفحه {page['page_number']})\n"
    content = text[len(page_marker) :].strip()

    sentences = SENTENCE_SPLIT_RE.split(content)
    chunks: list[dict[str, Any]] = []
    current = ""
    counter = 0

    for sent in sentences:
        if not sent or not sent.strip():
            continue
        if len(current) + len(sent) > chunk_size and current:
            chunks.append(
                {
                    "chunk_id": f"{page['page_number']}_{counter}",
                    "page_number": page["page_number"],
                    "text": page_marker + current.strip(),
                    "char_length": len(page_marker) + len(current),
                    "source_file": page.get("source_file", "unknown"),
                }
            )
            counter += 1
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + sent
            else:
                current = sent
        else:
            current += sent

    if current.strip():
        chunks.append(
            {
                "chunk_id": f"{page['page_number']}_{counter}",
                "page_number": page["page_number"],
                "text": page_marker + current.strip(),
                "char_length": len(page_marker) + len(current),
                "source_file": page.get("source_file", "unknown"),
            }
        )
    return chunks


def process_text_file(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a cleaned regulation .txt and produce final chunks ready for embedding.
    Expects the LLM-cleaned format with (صفحه N) markers.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    source = path.name

    if not has_page_markers(text):
        raise ValueError(
            f"{path} has no (صفحه N) markers. "
            "Use the cleaned regulation text or implement raw fallback chunking."
        )

    pages = extract_pages(text, source_file=source)
    final: list[dict[str, Any]] = []
    for page in pages:
        final.extend(split_large_page(page))
    return final
