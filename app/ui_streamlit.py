import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables import Runnable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.classifier import ZeroShotRisk
from app.ingest import ingest_folder, save_parquet
from app.llm import get_llm
from app.prompts import QA_PROMPT, RISK_SUMMARY_PROMPT
from app.retriever import get_retriever, index_dataframe

load_dotenv()
st.set_page_config(page_title="Risk Analysis Agent", layout="wide")
st.title("🛡️ Risk Analysis Agent")


# ---------- Caches ----------
@st.cache_resource
def _get_zsl() -> ZeroShotRisk:
    """
    Returns a cached instance of ZeroShotRisk classifier.

    :return: ZeroShotRisk instance
    """
    return ZeroShotRisk()


@st.cache_resource
def _get_llm() -> Runnable[Any, Any]:
    """
    Returns a cached instance of the LLM (Large Language Model) used for text generation and analysis.

    :return: LLM instance
    """
    return cast(Runnable[Any, Any], get_llm())


# ---------- Tabs ----------
tab_ingest, tab_analyze, tab_qa = st.tabs(["Ingest", "Analyze", "Q&A"])

# ---------- Ingest ----------
with tab_ingest:
    st.subheader("1) Ingest filings and index")
    folder = st.text_input("Folder with TXT filings (issuer/year/*.txt)", "data/samples")
    if st.button("Index folder", use_container_width=True):
        df = ingest_folder(folder)
        if df.empty:
            st.warning("No .txt files found. Expected structure: data/samples/<ISSUER>/<YEAR>/*.txt")
        else:
            save_parquet(df, "data/filings.parquet")
            index_dataframe(df)
            st.success(f"Indexed {len(df)} chunks → Chroma")
            st.dataframe(df.head(10))

# ---------- Analyze ----------
with tab_analyze:
    st.subheader("2) Classify & summarize top risks")
    issuer = st.text_input("Issuer (folder name)", "ACME_CORP")
    year = st.text_input("Fiscal year", "2024")
    focus = st.text_input("Focus (optional, e.g., 'key risks', 'changes vs prior year')", "key risks")

    k = st.slider("Top-k chunks to retrieve", 4, 24, 12, 1)
    if st.button("Run analysis", use_container_width=True):
        retriever = get_retriever(k=k, where={"$and": [{"issuer": issuer}, {"fiscal_year": year}]})
        query = f"{issuer} {year} {focus}"
        docs = retriever.invoke(query)

        if not docs:
            st.warning("No documents returned. Did you index the right issuer/year?")
        else:
            # Build context with citations
            context = "\n\n".join([f"[{d.metadata.get('chunk_id','?')}] {d.page_content}" for d in docs])
            # Zero-shot classify top chunks
            zsl = _get_zsl()
            top_texts = [d.page_content for d in docs[: min(8, len(docs))]]
            tags = zsl.classify(top_texts, top_k=3)

            rows = []
            for d, ts in zip(docs[: len(top_texts)], tags):
                rows.append(
                    {
                        "chunk_id": d.metadata.get("chunk_id", "?"),
                        "issuer": d.metadata.get("issuer"),
                        "year": d.metadata.get("fiscal_year"),
                        "tags": ", ".join([f"{risk}:{score:.2f}" for risk, score in ts]),
                    }
                )
            st.write("**Tagged chunks (top-8):**")
            st.dataframe(pd.DataFrame(rows))

            # Structured summary with citations
            llm = _get_llm()
            prompt = RISK_SUMMARY_PROMPT.format(issuer=issuer, year=year, context=context)
            st.write("### Executive Summary")
            st.write(llm.invoke(prompt))

# ---------- Q&A ----------
with tab_qa:
    st.subheader("3) Ask questions (RAG with citations)")
    q = st.text_input("Question", "What new cybersecurity risks are disclosed?")
    kq = st.slider("Top-k chunks to retrieve", 4, 16, 8, 1, key="qa_k")
    if st.button("Ask", use_container_width=True):
        retriever = get_retriever(k=kq)
        docs = retriever.get_relevant_documents(q)
        if not docs:
            st.warning("No documents returned.")
        else:
            context = "\n\n".join([f"[{d.metadata.get('chunk_id','?')}] {d.page_content}" for d in docs])
            ans = _get_llm().invoke(QA_PROMPT.format(question=q, context=context))
            st.write(ans)
            with st.expander("Sources"):
                st.write(
                    pd.DataFrame(
                        [
                            {
                                "chunk_id": d.metadata.get("chunk_id", "?"),
                                "issuer": d.metadata.get("issuer"),
                                "year": d.metadata.get("fiscal_year"),
                                "file": d.metadata.get("filepath"),
                            }
                            for d in docs
                        ]
                    )
                )
