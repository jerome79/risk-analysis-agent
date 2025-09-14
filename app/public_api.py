import sys
from pathlib import Path

from app.ui_streamlit import _get_zsl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.llm import get_llm  # if your summary uses LLM
from app.retriever import get_retriever  # whatever you use to open Chroma/FAISS


def summarize_risk(issuer: str, year: int, question: str = "top risks", k: int = 8) -> dict:
    """
    Return:
      {
        "issuer":..., "year":...,
        "summary": str,
        "categories": [{"label":..., "confidence":...}, ...],
        "sources": [{"path":..., "chunk_id":...}, ...]
      }
    """
    retriever = get_retriever(k=k, where={"$and": [{"issuer": issuer}, {"fiscal_year": year}]})
    docs = retriever.invoke(question)  # or get_relevant_documents()
    # classify categories from the retrieved text
    texts = [d.page_content for d in docs]
    zsl = _get_zsl()
    categories = zsl.classify(texts, top_k=3)

    # (Optional) LLM summary over top-k docs
    llm = get_llm()
    context = "\n\n".join(t.page_content for t in docs[:4])
    prompt = f"Summarize the {issuer} {year} {question} based on:\n{context}\n\nSummary:"
    summary = llm.invoke(prompt) if llm else "LLM not configured"

    sources = []
    for d in docs[:8]:
        src = d.metadata.get("source") or d.metadata.get("path") or "unknown"
        chunk_id = d.metadata.get("id") or d.metadata.get("chunk_id") or ""
        sources.append({"path": src, "chunk_id": chunk_id})

    return {
        "issuer": issuer,
        "year": year,
        "summary": str(summary),
        "categories": categories,
        "sources": sources,
    }
