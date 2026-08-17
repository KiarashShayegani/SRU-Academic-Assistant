"""Unit tests for page-aware chunking."""

from __future__ import annotations

from sru_assistant.data.chunking import extract_pages, has_page_markers, split_large_page


SAMPLE = """
(صفحه 1)
متن کوتاه صفحه یک.

(صفحه 2)
""" + ("جمله طولانی. " * 80) + """

(صفحه 3)
متن صفحه سه.
"""


def test_has_page_markers():
    assert has_page_markers(SAMPLE) is True
    assert has_page_markers("بدون نشانگر صفحه") is False


def test_extract_pages():
    pages = extract_pages(SAMPLE, source_file="test.txt")
    assert len(pages) >= 2
    assert pages[0]["page_number"] == 1
    assert "(صفحه 1)" in pages[0]["text"]


def test_split_large_page_preserves_marker():
    pages = extract_pages(SAMPLE, source_file="test.txt")
    large = [p for p in pages if p["page_number"] == 2][0]
    parts = split_large_page(large, chunk_size=200, overlap=20)
    assert len(parts) >= 2
    for p in parts:
        assert p["text"].startswith("(صفحه 2)")
        assert p["page_number"] == 2
