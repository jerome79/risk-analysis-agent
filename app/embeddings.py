import os
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedder():
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model)
