from __future__ import annotations

from typing import Any

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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
    provider: str | None = None, model: str | None = None, temperature: float | None = None, openai_api_key: str | None = None, anthropic_api_key: str | None = None
) -> Any:
    """
    Initialize and return an LLM client (Ollama, OpenAI, or Claude/Anthropic).

    Args:
        provider (str | None): The LLM provider to use.
        model (str | None): The model name to use.
        temperature (float | None): Sampling temperature.
        openai_api_key (str | None): API key for OpenAI.
        anthropic_api_key (str | None): API key for Anthropic.

    Returns:
        LLM instance appropriate for the provider.

    Raises:
        ValueError: If the LLM provider is not supported or misconfigured.
    """
    cfg = Settings()
    provider = cfg.llm_provider if provider is None else provider
    if provider == "ollama":
        model = cfg.llm_model if model is None else model
        temperature = cfg.llm_temperature if temperature is None else temperature
        url = _resolve_ollama_url(cfg)
        _assert_up(url)
        return ChatOllama(model=model, temperature=temperature, base_url=url)

    elif provider == "openai":
        api_key = cfg.openai_api_key if openai_api_key is None else openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set for LLM_PROVIDER=openai")
        model = model or cfg.openai_model
        temperature = cfg.llm_temperature if temperature is None else temperature
        return ChatOpenAI(openai_api_key=api_key, model_name=model, temperature=temperature)

    elif provider in ("claude", "anthropic"):
        api_key = cfg.anthropic_api_key if anthropic_api_key is None else anthropic_api_key
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set for LLM_PROVIDER=claude")
        model = model or cfg.anthropic_model
        temperature = cfg.llm_temperature if temperature is None else temperature
        return ChatAnthropic(anthropic_api_key=api_key, model=model, temperature=temperature)

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
