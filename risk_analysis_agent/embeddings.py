import os

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedder() -> HuggingFaceEmbeddings:
    """
    Returns an instance of HuggingFaceEmbeddings using the model specified
    by the EMBEDDING_MODEL environment variable, or a default model if not set.
    """
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model)
