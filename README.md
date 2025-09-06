# 🛡️ Risk Analysis Agent

A domain-focused extension of your Finance RAG agent: it adds a **risk classifier layer** on top of retrieval so outputs are **structured, explainable, and auditable** for PMs and Risk teams.

## 🎯 Business goal
Turn dense disclosures (10-K Item 1A, KIIDs, prospectuses) into **actionable, comparable risk insights**:
- Classify chunks into a **risk taxonomy**
- Summarize top risks with **citations**
- Enable **Q&A** with sources
- (Optional) Compare **YoY changes**

**Personas:** Portfolio Managers, Traders, CRO/Risk, Product.

---

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


---

## ✨ Features
- Ingest `.txt` filings and **chunk** with provenance (issuer, year, file, chunk_id)
- **Local embeddings** → **Chroma** persistent vector DB
- **Zero-shot risk classification** (BART-MNLI) across a **risk taxonomy**
- **Executive summary** grouped by categories with **citations**
- **Q&A** over the corpus with sources
- Fully **local & free** (Ollama + HF models)

---

## 🚀 Quick start

```bash
git clone <repo> risk-analysis-agent
cd risk-analysis-agent
python -m venv .venv && . .venv/Scripts/activate   # (Windows)
pip install -r requirements.txt
cp .env.example .env
```

# Add data:
data/samples/ACME_CORP/2024/item_1a.txt  (plain text)

streamlit run app/ui_streamlit.py

# Risk taxonomy

Default labels:

Market, Liquidity, Credit, Operational, Cybersecurity, Regulatory/Legal,
Supply Chain, ESG/Climate, Reputational, Model Risk

Edit in app/taxonomy.py.

## 🔍 Zero-shot classification (how it works)

We use a pretrained NLI model (BART-MNLI).
For each chunk and label, we score entailment of the hypothesis:

“This text is about <label>.”
The entailment probability becomes the label score.
Top-k labels are shown per chunk with confidence.

## 🔮 Classifier roadmap (free, API-less)

Phase 1 – Zero-Shot Baseline (current)

facebook/bart-large-mnli (free, local)

No training needed, good baseline

Phase 2 – Few-Shot Prompting (upgrade)

Use local LLM (Mistral via Ollama)

Add labeled examples in prompt for domain adaptation

Phase 3 – Fine-Tuned Local Classifier

Train RoBERTa/FinBERT locally on small labeled set (200–500 chunks)

Phase 4 – Hybrid / Rules

Add keyword/regex rules for transparency + combine with ML scores

## 🧪 Repo layout
app/
  ├─ taxonomy.py      # labels & helpers
  ├─ ingest.py        # read .txt, chunk, normalize
  ├─ embeddings.py    # MiniLM embeddings (HF)
  ├─ retriever.py     # Chroma index + retriever
  ├─ classifier.py    # zero-shot NLI tags
  ├─ prompts.py       # summary + QA prompts (citations)
  ├─ llm.py           # Ollama (Mistral)
  └─ ui_streamlit.py  # Streamlit UI

scripts/
  └─ ingest_cli.py

data/
  └─ samples/         # place issuer/year/*.txt here

## 🧰 Notes

For PDFs, convert to .txt with an external tool (e.g., pdfminer.six, pypdf) before ingestion.

Set CHROMA_PERSIST_DIR in .env to control the DB location.

Keep LLM_TEMPERATURE low (0.0–0.2) for consistent summaries.

## 📜 License

MIT
