"""OpenAI-compatible LLM client (Agnes AI by default)."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from openai import OpenAI

from sru_assistant.config import get_settings


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.agnes_api_key
        self.base_url = base_url or settings.agnes_base_url
        self.model = model or settings.agnes_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

        if not self.api_key:
            raise RuntimeError(
                "LLM API key not found. Set AGNES_API_KEY in the environment or .env file."
            )

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=False,
        )
        return completion.choices[0].message.content or ""

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Yield tokens. Skips empty trailing usage chunks from some providers."""
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content
            if token:
                yield token


@lru_cache
def get_llm_client() -> LLMClient | None:
    """Return client if API key is present, else None (FAQ-only mode still works)."""
    settings = get_settings()
    if not settings.agnes_api_key:
        return None
    try:
        return LLMClient()
    except Exception:
        return None
