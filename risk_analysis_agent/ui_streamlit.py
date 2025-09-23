import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_analysis_agent.classifier import ZeroShotRisk
from risk_analysis_agent.ingest import ingest_folder, save_parquet
from risk_analysis_agent.llm import get_llm
from risk_analysis_agent.prompts import QA_PROMPT, RISK_SUMMARY_PROMPT
from risk_analysis_agent.retriever import get_retriever, index_dataframe

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
def _get_llm(provider: str, model: str, temperature: float, openai_api_key: str, anthropic_api_key: str) -> Any:
    """
    Returns a cached LLM instance based on the selected provider, model, and temperature.

    Args:
        provider (str): The LLM provider identifier (e.g., "openai", "claude", "ollama").
        model (str): The model name to use.
        temperature (float): Sampling temperature for the LLM.
        openai_api_key (str, optional): API key for OpenAI, if required.
        anthropic_api_key (str, optional): API key for Anthropic, if required.


    Returns:
        An LLM instance as returned by get_llm.
    """
    return get_llm(provider=provider, model=model, temperature=temperature, openai_api_key=openai_api_key, anthropic_api_key=anthropic_api_key)


# -----------Helper ----------
# Add this to risk_analysis_agent/ui_streamlit.py
import os

LLM_PROVIDERS = {
    "Ollama (Mistral, Gemma, etc.)": {"id": "ollama", "models": ["gemma3:1b"]},
    "OpenAI (GPT-3.5, GPT-4, etc.)": {"id": "openai", "models": ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]},
    "Claude (Anthropic)": {"id": "claude", "models": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]},
}

st.sidebar.header("🔧 LLM Settings")
provider_name = st.sidebar.selectbox("LLM Provider", list(LLM_PROVIDERS.keys()), index=0)
provider = LLM_PROVIDERS[provider_name]["id"]
model = st.sidebar.selectbox("Model", LLM_PROVIDERS[provider_name]["models"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

# Optionally, set API keys in sidebar when needed
if provider == "openai":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
if provider == "claude":
    anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))


def ingest_tab() -> None:
    """
    Displays the UI for ingesting filings and indexing them.

    Allows the user to specify a folder containing TXT filings,
    triggers ingestion and indexing, and displays a preview of the ingested data.
    """
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


def analyze_tab() -> None:
    """
    Displays the UI for analyzing and classifying top risks.

    Allows the user to specify issuer, fiscal year, and focus, retrieves relevant document chunks,
    classifies them, displays tags, and generates an executive summary using an LLM.
    """
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
            context = "\n\n".join([f"[{d.metadata.get('chunk_id','?')}] {d.page_content}" for d in docs])
            zsl = _get_zsl()
            top_texts = [d.page_content for d in docs[: min(8, len(docs))]]
            tags = zsl.classify(top_texts, top_k=3)

            rows = []
            for d, ts in zip(docs[: len(top_texts)], tags, strict=False):
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
            llm = _get_llm(provider, model, temperature, openai_api_key if provider == "openai" else None, anthropic_api_key if provider == "claude" else None)
            prompt = RISK_SUMMARY_PROMPT.format(issuer=issuer, year=year, context=context)
            st.write("### Executive Summary")
            st.write(llm.invoke(prompt).content)


# ---------- Tabs ----------
tab_ingest, tab_analyze, tab_qa = st.tabs(["Ingest", "Analyze", "Q&A"])

# ---------- Ingest ----------
with tab_ingest:
    ingest_tab()
# ---------- Analyze ----------
with tab_analyze:
    analyze_tab()
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
            llm = _get_llm(provider, model, temperature, openai_api_key if provider == "openai" else None, anthropic_api_key if provider == "claude" else None)
            ans = llm.invoke(QA_PROMPT.format(question=q, context=context))
            st.write(ans.content)
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
