# Risk Analysis Agent

A domain-focused extension of the Finance RAG Agent: adds a risk classifier layer on top of retrieval so outputs are structured, explainable, and auditable for Portfolio Managers and Risk teams.

## Quick Demo (60–90s)
```bash
git clone https://github.com/jerome79/risk-analysis-agent.git
cd risk-analysis-agent
make demo
# open http://localhost:8501
```

![CI](https://github.com/jerome79/risk-analysis-agent/actions/workflows/ci.yml/badge.svg)


## Run with Docker
```bash
docker compose up --build
# App → http://localhost:8501
# (Optional) Ollama API → http://localhost:11434
```

## One-liner CLI
```bash
pip install -e .
msa demo   # opens http://localhost:8502 with a tiny sample
```
## Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for pipeline + trade-offs.

## CI & Quality
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-70%25+-blue)


## 🎯 Business Goal

* Turn dense disclosures (10-K Item 1A, KIIDs, prospectuses) into actionable, comparable risk insights:

* Classify chunks into a risk taxonomy

* Summarize top risks with citations

* Enable Q&A with source documents

* Compare year-over-year changes (optional)

* Personas: Portfolio Managers, Traders, Chief Risk Officers, Product Managers.

## 🧱 Architecture
TXT filings (issuer/year/*.txt)
│
▼
[Ingest] ──► chunked docs (text + metadata)
│ │
│ └──► [Chroma Vector Store] (embeddings + metadata)
│ ▲
│ │
└──► [Zero-Shot Risk Classifier] ◄─┘ (taxonomy tags per chunk)

UI (Streamlit)
├─ Ingest: index corpus
├─ Analyze: retrieve top chunks → classify → summarize (citations)
└─ Q&A: RAG answers + sources

LLM (Ollama/Mistral) — local, free
Embeddings (MiniLM) — local, free
Zero-shot (BART-MNLI) — local, free

## ✨ Features

* Ingest .txt filings and chunk with provenance (issuer, year, file, chunk_id)

* Local embeddings → Chroma persistent vector DB

* Zero-shot risk classification (BART-MNLI) across a risk taxonomy

* Executive summary grouped by categories with citations

* Q&A over the corpus with sources

* Fully local & free (Ollama + Hugging Face models)

## 🚀 Quick Start
git clone <repo> risk-analysis-agent
cd risk-analysis-agent
python -m venv .venv && . .venv/Scripts/activate   # (Windows)
pip install -r requirements.txt
cp .env.example .env

### Add data:
data/samples/ACME_CORP/2024/item_1a.txt

### Run the UI:
streamlit run risk_analysis_agent/ui_streamlit.py

## 🗂️ Risk Taxonomy

Default labels (see risk_analysis_agent/taxonomy.py):

* Market

* Liquidity

* Credit

* Operational

* Cybersecurity

* Regulatory / Legal

* Supply Chain

* ESG / Climate

* Reputational

* Model Risk

You can edit risk_analysis_agent/taxonomy.py to customize.

## 🔍 Zero-Shot Classification (How It Works)

* Pretrained NLI model (facebook/bart-large-mnli)

For each chunk and label, we score entailment of the hypothesis:

* “This text is about <label>.”

The entailment probability = confidence score

Top-k labels (or all above threshold) are displayed per chunk

## 🔮 Classifier Roadmap (All Free & Local)

* Phase 1 – Zero-Shot Baseline (current)

facebook/bart-large-mnli

No training needed, robust baseline

* Phase 2 – Few-Shot Prompting

Use local LLM (Mistral via Ollama)

Add labeled examples in prompt for domain adaptation

* Phase 3 – Fine-Tuned Local Classifier

Train RoBERTa/FinBERT on a small labeled set (200–500 chunks)

* Phase 4 – Hybrid / Rules

Add keyword/regex rules for transparency

Combine with ML scores for robustness

## 📊 Example Output

Sample input: ACME 2024, Item 1A

Analyze tab output:

Chunk ID	Risks	Confidence
item1a.txt::0	Operational (0.86), Market (0.61)	High
item1a.txt::1	Liquidity (0.72), Credit (0.55)	Medium
item1a.txt::2	Cybersecurity (0.81)	High



# 🧪 Repo Layout
risk_analysis_agent/
  ├─ taxonomy.py      # labels & helpers
  ├─ ingest.py        # read .txt, chunk, normalize
  ├─ embeddings.py    # MiniLM embeddings (HF)
  ├─ retriever.py     # Chroma index + retriever (MMR, filters)
  ├─ classifier.py    # zero-shot NLI tags
  ├─ prompts.py       # summary + QA prompts (citations)
  ├─ llm.py           # Ollama (Mistral/Gemma)
  └─ ui_streamlit.py  # Streamlit UI

scripts/
  ├─ ingest_cli.py
  └─ demo_classifier.py

data/
  └─ samples/         # place issuer/year/*.txt here

## 🧰 Notes

For PDFs, convert to .txt with tools like pdfminer.six or pypdf before ingestion

Set CHROMA_PERSIST_DIR in .env to control DB location

Keep LLM_TEMPERATURE low (0.0–0.2) for consistent summaries

## 📜 License

MIT

👉 This new README now has:

Clearer flow (goal → features → quick start → taxonomy → roadmap → output)

Example risk output table (you can paste your screenshot too)

Classifier roadmap staged for interviews

Do you want me to also add a “Comparison” section (Project 1 vs Project 2 vs Project 3) in the README so you can show progression to interviewers?
