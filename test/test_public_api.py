import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from risk_analysis_agent.public_api import summarize_risk


@pytest.fixture
def mock_docs() -> list:
    """
    Creates a list of mocked document objects for testing.

    Returns:
        list: A list of MagicMock objects simulating document instances.
    """
    doc: MagicMock = MagicMock()
    doc.page_content = "Risk content"
    doc.metadata = {"source": "file1.txt", "id": "chunk1"}
    return [doc] * 8
    doc = MagicMock()
    doc.page_content = "Risk content"
    doc.metadata = {"source": "file1.txt", "id": "chunk1"}
    return [doc] * 8


def test_summarize_risk_basic(mock_docs: list) -> None:
    """
    Test the summarize_risk function with basic mocked dependencies.

    Args:
        mock_docs (list): List of mocked document objects.

    Returns:
        None
    """
    with (
        patch("risk_analysis_agent.public_api.get_retriever") as mock_retriever,
        patch("risk_analysis_agent.public_api._get_zsl") as mock_zsl,
        patch("risk_analysis_agent.public_api.get_llm") as mock_llm,
    ):
        mock_retriever.return_value.invoke.return_value = mock_docs
        mock_zsl.return_value.classify.return_value = [{"label": "Risk", "confidence": 0.9}]
        mock_llm.return_value.invoke.return_value = "Summary text"
        year = 2023
        result = summarize_risk("IssuerA", year)
        assert result["issuer"] == "IssuerA"
        assert result["year"] == year
        assert result["summary"] == "Summary text"
        assert isinstance(result["categories"], list)
