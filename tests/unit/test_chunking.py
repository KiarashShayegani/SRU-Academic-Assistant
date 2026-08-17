"""Unit tests for page-aware chunking."""

from __future__ import annotations

from sru_assistant.data.chunking import extract_pages, has_page_markers, split_large_page


# Each page body must be >= min_chunk_chars (default 50) or extract_pages drops it.
_SHORT_BUT_VALID = "این یک متن آزمایشی با طول کافی برای عبور از حداقل کاراکتر است. "
_LONG = "جمله طولانی. " * 80

SAMPLE = f"""
(صفحه 1)
{_SHORT_BUT_VALID}

(صفحه 2)
{_LONG}

(صفحه 3)
{_SHORT_BUT_VALID} صفحه سه هم محتوای کافی دارد.
"""

# Page 1 intentionally below min_chunk_chars so it is filtered out
SAMPLE_WITH_TINY_PAGE = """
(صفحه 1)
کوتاه

(صفحه 2)
""" + ("جمله طولانی. " * 80)


def test_has_page_markers():
    assert has_page_markers(SAMPLE) is True
    assert has_page_markers("بدون نشانگر صفحه") is False


def test_extract_pages():
    pages = extract_pages(SAMPLE, source_file="test.txt")
    assert len(pages) >= 2
    assert pages[0]["page_number"] == 1
    assert "(صفحه 1)" in pages[0]["text"]
    page_nums = {p["page_number"] for p in pages}
    assert 1 in page_nums
    assert 2 in page_nums


def test_extract_pages_skips_too_short():
    pages = extract_pages(SAMPLE_WITH_TINY_PAGE, source_file="test.txt")
    # Page 1 ("کوتاه") is under min_chunk_chars and must be dropped
    assert all(p["page_number"] != 1 for p in pages)
    assert any(p["page_number"] == 2 for p in pages)


def test_split_large_page_preserves_marker():
    pages = extract_pages(SAMPLE, source_file="test.txt")
    large = [p for p in pages if p["page_number"] == 2][0]
    parts = split_large_page(large, chunk_size=200, overlap=20)
    assert len(parts) >= 2
    for p in parts:
        assert p["text"].startswith("(صفحه 2)")
        assert p["page_number"] == 2