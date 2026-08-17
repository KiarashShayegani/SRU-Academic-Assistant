"""Unified entry point: FAQ or RAG depending on mode."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from sru_assistant.generation.llm_client import LLMClient, get_llm_client
from sru_assistant.generation.prompts import build_rag_prompt
from sru_assistant.retrieval.faq_retriever import FAQHit, FAQRetriever
from sru_assistant.retrieval.regulation_retriever import RegulationChunk, RegulationRetriever


@dataclass
class AnswerResult:
    mode: Literal["faq", "rag"]
    answer: str | None = None
    stream: Iterator[str] | None = None
    similarity: float | None = None
    chunks: list[RegulationChunk] = field(default_factory=list)
    faq_hit: FAQHit | None = None
    page_numbers: list[int] = field(default_factory=list)
    source_label: str = ""

    def is_streaming(self) -> bool:
        return self.stream is not None


def answer_question(
    question: str,
    *,
    mode: Literal["faq", "rag"] = "faq",
    faq_retriever: FAQRetriever | None = None,
    regulation_retriever: RegulationRetriever | None = None,
    llm_client: LLMClient | None = None,
    k: int | None = None,
) -> AnswerResult:
    """
    Answer a student question.

    - mode="faq"  → nearest FAQ entry (no LLM required)
    - mode="rag"  → retrieve regulation chunks + stream LLM answer
    """
    question = (question or "").strip()
    if not question:
        return AnswerResult(
            mode=mode,
            answer="لطفاً سوال خود را وارد کنید.",
            source_label="",
        )

    if mode == "faq":
        return _answer_faq(question, faq_retriever=faq_retriever)

    return _answer_rag(
        question,
        regulation_retriever=regulation_retriever,
        llm_client=llm_client,
        k=k,
    )


def _answer_faq(
    question: str,
    *,
    faq_retriever: FAQRetriever | None = None,
) -> AnswerResult:
    retriever = faq_retriever or FAQRetriever()
    hit = retriever.best(question)
    if hit is None:
        return AnswerResult(
            mode="faq",
            answer="پاسخ دقیقی برای این سوال در پایگاه سوالات متداول یافت نشد. "
            "لطفاً سوال خود را با عبارت دیگری مطرح کنید یا حالت «هوش مصنوعی» را امتحان کنید.",
            source_label="بدون تطابق",
        )
    return AnswerResult(
        mode="faq",
        answer=hit.answer,
        similarity=hit.similarity,
        faq_hit=hit,
        source_label=f"📋 پایگاه سوالات متداول (شباهت: {hit.similarity:.0%})",
    )


def _answer_rag(
    question: str,
    *,
    regulation_retriever: RegulationRetriever | None = None,
    llm_client: LLMClient | None = None,
    k: int | None = None,
) -> AnswerResult:
    retriever = regulation_retriever or RegulationRetriever()
    client = llm_client if llm_client is not None else get_llm_client()

    if client is None:
        # Graceful degradation: fall back to FAQ if LLM is unavailable
        return _answer_faq(question)

    chunks = retriever.retrieve(question, k=k)
    prompt = build_rag_prompt(question, chunks)
    page_numbers = sorted({c.page_number for c in chunks if c.page_number > 0})

    if page_numbers:
        source_label = f"📖 استناد به صفحات آیین‌نامه: {', '.join(map(str, page_numbers))}"
    else:
        source_label = "📖 مستندات دانشگاه"

    return AnswerResult(
        mode="rag",
        stream=client.stream(prompt),
        chunks=chunks,
        page_numbers=page_numbers,
        source_label=source_label,
    )
