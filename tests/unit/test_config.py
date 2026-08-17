"""Basic settings load test."""

from sru_assistant.config import get_settings


def test_settings_load():
    s = get_settings()
    assert s.embedding_dim == 384
    assert s.regulation_table
    assert s.faq_table
    assert s.chunk_size > 0
    assert s.overlap >= 0
