from __future__ import annotations

import os
from typing import Any

import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from .setting import Settings


def get_embedder(model: str | None = None) -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFaceEmbeddings instance for the specified model.

    Args:
        model (str | None): The name of the embedding model to use. If None, uses the default from settings.

    Returns:
        HuggingFaceEmbeddings: The embedding function instance.
    """
    cfg = Settings()
    return HuggingFaceEmbeddings(model_name=model or cfg.embedding_model)


def get_vectorstore(collection: str = "risk_docs") -> Chroma:
    """
    Initializes and returns a Chroma vector store client for the specified collection.

    Args:
        collection (str): The name of the Chroma collection to use. Defaults to "risk_docs".

    Returns:
        Chroma: An instance of the Chroma vector store for the given collection.
    """
    cfg = Settings()
    os.environ["CHROMADB_TELEMETRY_IMPLEMENTATION"] = "none"
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    os.makedirs(cfg.chroma_persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
    return Chroma(client=client, collection_name=collection, embedding_function=get_embedder())


def get_retriever(k: int = 5, where: Any | None = None) -> VectorStoreRetriever:
    """
    Returns a retriever object for querying the Chroma vector store.

    Args:
        k (int): Number of results to return. Defaults to 5.
        where (dict | None): Optional filter for metadata fields.

    Returns:
        Chroma: A retriever configured for MMR search.
    """
    vs = get_vectorstore()
    kwargs = {"k": k}
    if where:  # OMIT empty filters; Chroma 1.x rejects {}
        kwargs["filter"] = where
    return vs.as_retriever(search_type="mmr", search_kwargs=kwargs)


def index_dataframe(df: pd.DataFrame) -> None:
    """
    Indexes a pandas DataFrame into the Chroma vector store.

    Args:
        df (pandas.DataFrame): DataFrame containing a 'text' column and optional metadata columns.
    """
    vs = get_vectorstore()
    texts = df["text"].astype(str).tolist()
    metas = df.drop(columns=["text"], errors="ignore").to_dict(orient="records")
    ids = [f"doc-{i}" for i in range(len(texts))]
    vs.add_texts(texts=texts, metadatas=metas, ids=ids)
