from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _in_docker() -> bool:
    return os.path.exists("/.dockerenv")


@dataclass(frozen=True)
class Settings:
    # LLM / provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    llm_model: str = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "gemma3:1b"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    ollama_base_url: str | None = os.getenv("OLLAMA_BASE_URL")

    # API KEY
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Claude (Anthropic)
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Vector store
    chroma_persist_dir: str = os.getenv(
        "CHROMA_PERSIST_DIR",
        "/data/chroma" if _in_docker() else ".chroma-risk",
    )

    # Classifier / inference knobs
    sent_batch_size: int = int(os.getenv("SENT_BATCH_SIZE", "16"))
    sent_max_len: int = int(os.getenv("SENT_MAX_LEN", "512"))
    zsl_model: str = os.getenv("ZSL_MODEL", "facebook/bart-large-mnli")
    zsl_max_len: int = int(os.getenv("ZSL_MAX_LEN", "512"))
    zsl_label_batch: int = int(os.getenv("ZSL_LABEL_BATCH", "16"))

    tokenizers_parallelism: bool = _as_bool(os.getenv("TOKENIZERS_PARALLELISM"), False)
    torch_num_threads: int = int(os.getenv("TORCH_NUM_THREADS", "4"))
