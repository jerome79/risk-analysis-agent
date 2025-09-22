import os
from typing import Any

import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from risk_analysis_agent.embeddings import get_embedder


def get_vectorstore(persist_dir: str | None = None) -> Chroma:
    """
    Returns a Chroma vector store instance, using the provided persist directory or the default from environment.

    Args:
        persist_dir (str | None): Directory to persist the vector store. If None, uses CHROMA_PERSIST_DIR env var or '.chroma-risk'.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    persist = persist_dir or os.getenv("CHROMA_PERSIST_DIR", ".chroma-risk")
    return Chroma(
        client=chromadb.PersistentClient(path=str(persist)),
        collection_name="risk_docs",
        embedding_function=get_embedder(),
    )


def index_dataframe(df: pd.DataFrame, persist_dir: str | None = None) -> Chroma:
    """
    Indexes a pandas DataFrame into the Chroma vector store.

    Args:
        df (pandas.DataFrame): DataFrame with a 'text' column and optional metadata columns.
        persist_dir (str | None): Directory to persist the vector store. If None, uses CHROMA_PERSIST_DIR env var or '.chroma-risk'.

    Returns:
        Chroma: The updated Chroma vector store instance.
    """
    vs = get_vectorstore(persist_dir)
    texts = df["text"].astype(str).tolist()
    metas = df.drop(columns=["text"], errors="ignore").to_dict(orient="records")
    ids = [f"doc-{i}" for i in range(len(texts))]
    vs.add_texts(texts=texts, metadatas=metas, ids=ids)
    return vs


def get_retriever(k: int = 8, persist_dir: str | None = None, where: Any | None = None) -> VectorStoreRetriever:
    """
    Returns a retriever object from the Chroma vector store.

    Args:
        k (int): Number of results to retrieve.
        persist_dir (str | None): Directory to persist the vector store. If None, uses CHROMA_PERSIST_DIR env var or '.chroma-risk'.
        where (dict | None): Optional metadata filter for retrieval.

    Returns:
        VectorStoreRetriever: A retriever object from the Chroma vector store.
    """
    vs = get_vectorstore(persist_dir)
    search_kwargs = {"k": k, "fetch_k": max(20, k * 2), "lambda_mult": 0.2}
    if where:
        search_kwargs["filter"] = where

    return vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
