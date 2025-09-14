# test/test_llm.py

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from risk_analysis_agent.embeddings import get_embedder


def test_embedding_model() -> None:
    """
    Test the embedding model name matches the expected environment variable or default.
    """
    result = get_embedder()
    assert result.model_name == os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
