from __future__ import annotations

import httpx
from langchain_ollama import ChatOllama

from .setting import Settings


def _resolve_ollama_url(cfg: Settings) -> str:
    """
    Determine the base URL for the Ollama service.

    Priority:
    1. Use the user-provided override if available.
    2. If not set, use the Docker service name if `chroma_persist_dir` starts with `/data/`.
    3. Otherwise, default to localhost.

    Args:
        cfg (Settings): The settings object containing configuration.

    Returns:
        str: The resolved Ollama base URL.
    """
    # 1) User override
    if cfg.ollama_base_url:
        return cfg.ollama_base_url
    # 2) Default: Docker uses service name, local uses localhost
    return "http://ollama:11434" if cfg.chroma_persist_dir.startswith("/data/") else "http://127.0.0.1:11434"


def _assert_up(url: str) -> None:
    r"""
    Determine the base URL for the Ollama service.

    Priority:
    1\. Use the user-provided override if available.
    2\. If not set, use the Docker service name if `chroma_persist_dir` starts with `/data/`.
    3\. Otherwise, default to localhost.

    Args:
        cfg (Settings): The settings object containing configuration.

    Returns:
        str: The resolved Ollama base URL.
    """
    with httpx.Client(timeout=2.0) as c:
        r = c.get(url.rstrip("/") + "/api/version")
        r.raise_for_status()


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
) -> ChatOllama:
    """
    Initialize and return a ChatOllama LLM client.

    Args:
        model (str | None): The model name to use. Defaults to value from settings.
        temperature (float | None): Sampling temperature. Defaults to value from settings.
        base_url (str | None): Override for the Ollama base URL.

    Returns:
        ChatOllama: An instance of the ChatOllama client.

    Raises:
        ValueError: If the LLM provider is not 'ollama'.
        RuntimeError: If the Ollama service is not reachable.
    """
    cfg = Settings()
    if cfg.llm_provider != "ollama":
        raise ValueError(f"Only LLM_PROVIDER=ollama supported for now, got {cfg.llm_provider}")

    model = model or cfg.llm_model
    temperature = cfg.llm_temperature if temperature is None else temperature
    url = base_url or _resolve_ollama_url(cfg)

    try:
        _assert_up(url)
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable at {url}. Start it locally (`ollama serve`) " f"and `ollama pull {model}`, or run docker compose.") from e

    return ChatOllama(model=model, temperature=temperature, base_url=url)
