# test/test_ui_streamlit.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
# test/test_ui_streamlit_analyze.py
import pandas as pd

from risk_analysis_agent.ui_streamlit import _get_llm, _get_zsl


def test_get_llm_returns_runnable() -> None:
    """
    Test that _get_llm returns the mocked runnable object from get_llm.
    """
    with patch("risk_analysis_agent.ui_streamlit.get_llm") as mock_get_llm:
        mock_runnable = MagicMock()
        mock_get_llm.return_value = mock_runnable
        print(mock_get_llm)
        result = _get_llm("", "", 0.5, "", "")
        assert result == mock_runnable


def test_cache_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that the cache functions in risk_analysis_agent.ui_streamlit return the expected mocked values.
    """

    # Mock dependencies for cache functions
    monkeypatch.setattr("risk_analysis_agent.ui_streamlit.ZeroShotRisk", lambda: "ZSL")
    monkeypatch.setattr("risk_analysis_agent.ui_streamlit.get_llm", lambda **kwargs: "LLM")

    # Test the cache_resource functions
    zsl = _get_zsl()
    llm = _get_llm("ollama", "", 0.5, "", "")
    print(llm)
    assert zsl == "ZSL"
    assert llm == "LLM"


def test_ingest_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test the ingest_tab function in risk_analysis_agent.ui_streamlit by mocking Streamlit UI and risk_analysis_agent functions.
    """
    import risk_analysis_agent.ui_streamlit

    # Mock Streamlit UI calls
    monkeypatch.setattr("streamlit.subheader", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.text_input", lambda *a, **k: "data/samples")
    monkeypatch.setattr("streamlit.button", lambda *a, **k: True)
    monkeypatch.setattr("streamlit.warning", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.success", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.dataframe", lambda *a, **k: None)

    # Mock risk_analysis_agent functions
    monkeypatch.setattr(
        risk_analysis_agent.ui_streamlit,
        "ingest_folder",
        lambda folder: pd.DataFrame([{"issuer": "ACME_CORP", "fiscal_year": "2024", "section": "Item 1A", "filepath": "somepath", "text": "txt", "chunk_id": "id"}]),
    )
    monkeypatch.setattr(risk_analysis_agent.ui_streamlit, "save_parquet", lambda df, path: None)
    monkeypatch.setattr(risk_analysis_agent.ui_streamlit, "index_dataframe", lambda df: None)

    # Run tab logic
    risk_analysis_agent.ui_streamlit.ingest_tab()


def test_analyze_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test the analyze_tab function in risk_analysis_agent.ui_streamlit by mocking Streamlit UI, retriever, classifier, and LLM.
    """
    import risk_analysis_agent.ui_streamlit

    # Patch Streamlit UI elements
    monkeypatch.setattr("streamlit.subheader", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.text_input", lambda label, val=None: "ACME_CORP" if "Issuer" in label else "2024" if "Fiscal year" in label else "key risks")
    monkeypatch.setattr("streamlit.slider", lambda *a, **k: 8)
    monkeypatch.setattr("streamlit.button", lambda *a, **k: True)
    monkeypatch.setattr("streamlit.warning", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.write", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.dataframe", lambda *a, **k: None)

    # Mock retriever
    class DummyDoc:
        def __init__(self) -> None:
            """
            Initialize a DummyDoc with preset metadata and page content.
            """
            self.metadata = {"chunk_id": "id123", "issuer": "ACME_CORP", "fiscal_year": "2024", "filepath": "data/samples/ACME_CORP/2024/item_1a.txt"}
            self.page_content = "Risk content"

    class DummyRetriever:
        def invoke(self, query: str) -> list:
            """
            Retrieve a list of DummyDoc instances based on the provided query.

            Args:
                query (str): The search query.

            Returns:
                list: A list of DummyDoc objects.
            """
            return [DummyDoc() for _ in range(8)]

    monkeypatch.setattr(risk_analysis_agent.ui_streamlit, "get_retriever", lambda **k: DummyRetriever())

    # Mock classifier and LLM
    class DummyZSL:
        def classify(self, texts: list[str], top_k: int = 3) -> list[list[tuple[str, float]]]:
            """
            Classify the given texts into risk categories.

            Args:
                texts (list[str]): List of text strings to classify.
                top_k (int, optional): Number of top categories to return for each text. Defaults to 3.

            Returns:
                list[list[tuple[str, float]]]: A list where each element is a list of tuples containing
                    the risk category and its score for each input text.
            """
            return [[("Market", 0.9), ("Credit", 0.8), ("Operational", 0.7)] for _ in texts]
            return [[("Market", 0.9), ("Credit", 0.8), ("Operational", 0.7)] for _ in texts]

    monkeypatch.setattr(risk_analysis_agent.ui_streamlit, "_get_zsl", lambda: DummyZSL())

    class DummyLLM:
        def __init__(self, content: str = "Summary") -> None:
            """
            Initialize the DummyLLM instance.
            """
            self.content = content
            self.temperature = 0.5
            self.provider = "openai"
            self.model = "gpt-3.5-turbo"
            self.openai_api_key = None
            self.anthropic_api_key = None

        def invoke(self, prompt: str) -> "DummyLLM":
            """
            Generate a summary based on the provided prompt.

            Args:
                prompt (str): The input prompt for the LLM.

            Returns:
                dict: A dictionary with a 'content' key containing the generated summary.
            """
            self.content = "Summary"
            return self

    monkeypatch.setattr(risk_analysis_agent.ui_streamlit, "_get_llm", lambda a, b, c, d, e: DummyLLM())
    monkeypatch.setattr(risk_analysis_agent.ui_streamlit, "RISK_SUMMARY_PROMPT", "Issuer: {issuer}\nYear: {year}\nContext: {context}")

    # Run tab logic
    risk_analysis_agent.ui_streamlit.analyze_tab()
