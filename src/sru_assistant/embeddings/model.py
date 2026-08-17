"""Thin wrapper around SentenceTransformer for consistent encoding."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from sru_assistant.config import Settings, get_settings


class EmbeddingModel:
    """Load once, encode many. Always returns float32 numpy arrays."""

    def __init__(self, model_path: str | Path | None = None, normalize: bool | None = None):
        settings = get_settings()
        self.model_path = Path(model_path) if model_path else settings.model_path_abs
        self.normalize = settings.embedding_normalize if normalize is None else normalize
        self._model = SentenceTransformer(str(self.model_path))
        self.dimension = self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        settings = get_settings()
        bs = batch_size or settings.embedding_batch_size
        vectors = self._model.encode(
            list(texts),
            batch_size=bs,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    """Process-wide singleton (safe for Streamlit + scripts)."""
    return EmbeddingModel()
