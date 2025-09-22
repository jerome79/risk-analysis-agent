# risk_analysis_agent/llm.py

from __future__ import annotations

import os

from langchain_ollama import ChatOllama


def get_llm(temperature: float = 0.2, base_url: str | None = None, model: str | None = None) -> ChatOllama:
    """
    Return a ChatOllama (LangChain 0.3) instance.
    Note: invoke() returns a BaseMessage; use .content to read the text.
    """
    # Resolve base_url deterministically and avoid any scope ambiguity
    resolved_base_url = base_url if base_url is not None else os.getenv("OLLAMA_BASE_URL")  # e.g. "http://localhost:11434"
    model = model if model is not None else os.getenv("OLLAMA_MODEL", "mistral")
    if resolved_base_url:
        return ChatOllama(model=model, temperature=temperature, base_url=resolved_base_url)
    else:
        return ChatOllama(model=model, temperature=temperature)
