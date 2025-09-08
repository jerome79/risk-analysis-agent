import sys
from pathlib import Path

import pandas as pd
import pytest

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from app.retriever import get_retriever, index_dataframe

SCHEMA = ["issuer", "fiscal_year", "section", "filepath", "text", "chunk_id"]


class FakeEmbedder:
    # Minimal stand-in to avoid network/model downloads in CI
    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim


def test_retriever_filters_and_returns_docs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Patch embedding to avoid downloading sentence-transformers
    import app.retriever as retr

    monkeypatch.setattr(retr, "get_embedder", lambda: FakeEmbedder(8))

    # Build tiny in-memory corpus
    rows = [
        {
            "issuer": "ACME_CORP",
            "fiscal_year": "2024",
            "section": "Item 1A",
            "filepath": "data/samples/ACME_CORP/2024/item_1a.txt",
            "text": "Interest rate volatility and liquidity risks may impact funding costs.",
            "chunk_id": "item_1a.txt:::0",
        },
        {
            "issuer": "ACME_CORP",
            "fiscal_year": "2023",
            "section": "Item 1A",
            "filepath": "data/samples/ACME_CORP/2023/item_1a.txt",
            "text": "Cybersecurity incidents could disrupt operations and harm reputation.",
            "chunk_id": "item_1a.txt:::1",
        },
    ]
    df = pd.DataFrame(rows, columns=SCHEMA)

    # Use a temp Chroma persist dir for isolation
    persist_dir = tmp_path / ".chroma-test"
    index_dataframe(df, persist_dir=str(persist_dir))

    # Filter to only 2024
    retriever = get_retriever(
        k=2,
        persist_dir=str(persist_dir),
        where={"$and": [{"issuer": "ACME_CORP"}, {"fiscal_year": "2024"}]},
    )
    docs = retriever.invoke("funding costs and liquidity risk")
    assert len(docs) >= 1
    # Verify metadata propagated and filter took effect
    for d in docs:
        assert d.metadata.get("issuer") == "ACME_CORP"
        assert d.metadata.get("fiscal_year") == "2024"
        assert "chunk_id" in d.metadata
