# test/test_llm.py

import os
import sys
from pathlib import Path

import httpx
import pytest
from langchain_ollama import ChatOllama

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from typing import Any
from unittest.mock import MagicMock, patch

from risk_analysis_agent.llm import Settings, _assert_up, _resolve_ollama_url, get_llm


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Ollama not available in CI")
def test_llm_main() -> None:
    """
    Test the main LLM function for basic output or side effects.

    Replace 'main' with the actual function name if different.
    """
    result = get_llm()
    assert isinstance(result, ChatOllama)  # Check if result is an instance of ChatOllama


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Ollama not available in CI")
def test_llm_model() -> None:
    """
    Test that the LLM model name matches the expected environment variable or default.
    """
    output = get_llm()
    assert output.model == os.getenv("OLLAMA_MODEL", "mistral")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Ollama not available in CI")
def test_llm_temperature() -> None:
    """
    Test that the LLM temperature value matches the expected environment variable or default.
    """
    output = get_llm()
    assert output.temperature == float(os.getenv("LLM_TEMPERATURE", "0.2"))


# test/test_llm_extra.py


def test_resolve_ollama_url_user_override() -> None:
    cfg = Settings(ollama_base_url="http://custom:1234")
    assert _resolve_ollama_url(cfg) == "http://custom:1234"


def test_resolve_ollama_url_docker() -> None:
    cfg = Settings(ollama_base_url=None, chroma_persist_dir="/data/test")
    assert _resolve_ollama_url(cfg) == "http://ollama:11434"


def test_resolve_ollama_url_localhost() -> None:
    cfg = Settings(ollama_base_url=None, chroma_persist_dir="localdir")
    assert _resolve_ollama_url(cfg) == "http://127.0.0.1:11434"


def test_assert_up_success() -> None:
    with patch("httpx.Client.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        _assert_up("http://localhost:1234")  # Should not raise


def test_assert_up_failure() -> None:
    with patch("httpx.Client.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("fail", request=None, response=None)
        mock_get.return_value = mock_response
        with pytest.raises(httpx.HTTPStatusError):
            _assert_up("http://localhost:1234")


def test_get_llm_unsupported_provider(monkeypatch: Any) -> None:
    monkeypatch.setattr("risk_analysis_agent.llm.Settings", lambda: Settings(llm_provider="foo"))
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER: foo"):
        get_llm(provider="foo")


def test_get_llm_openai_missing_key(monkeypatch: Any) -> None:
    monkeypatch.setattr("risk_analysis_agent.llm.Settings", lambda: Settings(llm_provider="openai", openai_api_key=""))
    with pytest.raises(ValueError, match="OPENAI_API_KEY must be set"):
        get_llm(provider="openai")


def test_get_llm_anthropic_missing_key(monkeypatch: Any) -> None:
    monkeypatch.setattr("risk_analysis_agent.llm.Settings", lambda: Settings(llm_provider="claude", anthropic_api_key=""))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY must be set"):
        get_llm(provider="claude")
