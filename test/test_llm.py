# test/test_llm.py

import os
import sys
from pathlib import Path

from langchain_community.llms import Ollama

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.llm import get_llm


def test_llm_main() -> None:
    """
    Test the main LLM function for basic output or side effects.

    Replace 'main' with the actual function name if different.
    """
    result = get_llm()
    assert isinstance(result, Ollama)  # Check if result is an instance of Ollama


def test_llm_model() -> None:
    """
    Test that the LLM model name matches the expected environment variable or default.
    """
    output = get_llm()
    assert output.model == os.getenv("OLLAMA_MODEL", "mistral")


def test_llm_temperature() -> None:
    """
    Test that the LLM temperature value matches the expected environment variable or default.
    """
    output = get_llm()
    assert output.temperature == float(os.getenv("LLM_TEMPERATURE", "0.2"))
