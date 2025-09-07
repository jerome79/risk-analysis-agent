import os
from langchain_community.vectorstores import Chroma
from app.embeddings import get_embedder

def get_vectorstore(persist_dir: str | None = None):
    persist = persist_dir or os.getenv("CHROMA_PERSIST_DIR", ".chroma-risk")
    return Chroma(persist_directory=persist, embedding_function=get_embedder())

def index_dataframe(df, persist_dir: str | None = None):
    vs = get_vectorstore(persist_dir)
    texts = df["text"].tolist()
    metas = df.drop(columns=["text"]).to_dict(orient="records")
    vs.add_texts(texts=texts, metadatas=metas)
    vs.persist()
    return vs

def get_retriever(k: int = 8, persist_dir: str | None = None, where: dict | None = None):
    vs = get_vectorstore(persist_dir)
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(20, k * 2),
            "lambda_mult": 0.2,
            "filter": where or {},   # optional metadata filter
        },
    )
