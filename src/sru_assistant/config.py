"""Application configuration loaded from YAML + environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Resolve project root (directory that contains config/ and src/)."""
    # When installed editable, this file lives under src/sru_assistant/
    here = Path(__file__).resolve()
    # src/sru_assistant/config.py → project root is two levels up from package
    candidates = [
        here.parents[2],  # .../sru-academic-assistant
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "config" / "default.yaml").exists():
            return c
    return Path.cwd()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """Typed settings. Values come from (in order of precedence):
    1. Environment variables (SRU_*, AGNES_*, etc.)
    2. config/local.yaml (gitignored overrides)
    3. config/default.yaml
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Paths ---
    model_path: str = Field(default="models/MiniLM", validation_alias="SRU_MODEL_PATH")
    db_path: str = Field(default="db/lancedb", validation_alias="SRU_DB_PATH")
    regulation_table: str = Field(
        default="university_regulation_v2", validation_alias="SRU_REGULATION_TABLE"
    )
    faq_table: str = Field(default="QA_v2", validation_alias="SRU_FAQ_TABLE")

    # --- Embedding ---
    embedding_dim: int = 384
    embedding_normalize: bool = True
    embedding_batch_size: int = 32

    # --- Chunking ---
    max_chars_per_page: int = 850
    chunk_size: int = 750
    overlap: int = 100
    min_chunk_chars: int = 50

    # --- Retrieval ---
    regulation_k: int = 5
    faq_k: int = 1
    faq_similarity_threshold: float = 0.0

    # --- LLM (OpenAI-compatible) ---
    agnes_api_key: str | None = Field(default=None, validation_alias="AGNES_API_KEY")
    agnes_base_url: str = Field(
        default="https://apihub.agnes-ai.com/v1", validation_alias="AGNES_BASE_URL"
    )
    agnes_model: str = Field(default="agnes-2.0-flash", validation_alias="AGNES_MODEL")
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    llm_stream: bool = True

    # --- UI ---
    page_title: str = "دستیار آیین‌نامه دانشگاه شهید رجایی"
    page_icon: str = "🎓"
    default_mode: Literal["faq", "rag"] = "faq"
    show_sources: bool = True
    show_timing: bool = True

    # --- Logging ---
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path relative to project root (or return absolute as-is)."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return _project_root() / p

    @property
    def model_path_abs(self) -> Path:
        return self.resolve_path(self.model_path)

    @property
    def db_path_abs(self) -> Path:
        return self.resolve_path(self.db_path)


def _merge_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache
def get_settings() -> Settings:
    """Load and cache application settings."""
    root = _project_root()
    default = _load_yaml(root / "config" / "default.yaml")
    local = _load_yaml(root / "config" / "local.yaml")
    merged = _merge_dict(default, local)

    # Flatten nested YAML into the flat Settings fields where useful
    flat: dict[str, Any] = {}
    if "paths" in merged:
        flat["model_path"] = merged["paths"].get("model_path", "models/MiniLM")
        flat["db_path"] = merged["paths"].get("db_path", "db/lancedb")
    if "tables" in merged:
        flat["regulation_table"] = merged["tables"].get("regulation", "university_regulation_v2")
        flat["faq_table"] = merged["tables"].get("faq", "QA_v2")
    if "embedding" in merged:
        flat["embedding_dim"] = merged["embedding"].get("dimension", 384)
        flat["embedding_normalize"] = merged["embedding"].get("normalize", True)
        flat["embedding_batch_size"] = merged["embedding"].get("batch_size", 32)
    if "chunking" in merged:
        flat["max_chars_per_page"] = merged["chunking"].get("max_chars_per_page", 850)
        flat["chunk_size"] = merged["chunking"].get("chunk_size", 750)
        flat["overlap"] = merged["chunking"].get("overlap", 100)
        flat["min_chunk_chars"] = merged["chunking"].get("min_chunk_chars", 50)
    if "retrieval" in merged:
        flat["regulation_k"] = merged["retrieval"].get("regulation_k", 5)
        flat["faq_k"] = merged["retrieval"].get("faq_k", 1)
        flat["faq_similarity_threshold"] = merged["retrieval"].get(
            "faq_similarity_threshold", 0.0
        )
    if "llm" in merged:
        flat["llm_temperature"] = merged["llm"].get("temperature", 0.7)
        flat["llm_max_tokens"] = merged["llm"].get("max_tokens", 1024)
        flat["llm_stream"] = merged["llm"].get("stream", True)
    if "ui" in merged:
        flat["page_title"] = merged["ui"].get("page_title", flat.get("page_title"))
        flat["page_icon"] = merged["ui"].get("page_icon", "🎓")
        flat["default_mode"] = merged["ui"].get("default_mode", "faq")
        flat["show_sources"] = merged["ui"].get("show_sources", True)
        flat["show_timing"] = merged["ui"].get("show_timing", True)
    if "logging" in merged:
        flat["log_level"] = merged["logging"].get("level", "INFO")

    # Environment variables still win via pydantic-settings
    return Settings(**{k: v for k, v in flat.items() if v is not None})
