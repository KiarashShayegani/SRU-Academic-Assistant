"""Thin wrapper around SentenceTransformer for consistent encoding."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

# Reduce native-thread contention BEFORE importing torch / sentence_transformers.
# On Windows + Streamlit this often prevents:
#   Fatal Python error: take_gil: PyCOND_WAIT(gil->cond) failed
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from sru_assistant.config import get_settings


def _configure_torch() -> None:
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # May already be set if torch was imported earlier
            pass
    except Exception:
        pass


class EmbeddingModel:
    """Load once, encode many. Always returns float32 numpy arrays."""

    def __init__(self, model_path: str | Path | None = None, normalize: bool | None = None):
        _configure_torch()
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self.model_path = Path(model_path) if model_path else settings.model_path_abs
        self.normalize = settings.embedding_normalize if normalize is None else normalize

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found at: {self.model_path}\n"
                "Put MiniLM under models/MiniLM or set SRU_MODEL_PATH in .env"
            )

        # local_files_only avoids Hub network calls that can hang/crash on some Windows setups
        try:
            self._model = SentenceTransformer(
                str(self.model_path),
                local_files_only=True,
            )
        except TypeError:
            # Older sentence-transformers without local_files_only
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
    """Process-wide singleton (scripts / non-Streamlit)."""
    return EmbeddingModel()
