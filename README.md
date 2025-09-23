# Risk Analysis Agent

**Local, explainable RAG that turns 10-K, KIID, or fund-prospectus text into structured risk insights—with citations—no cloud required.**

[![CI](https://github.com/jerome79/risk-analysis-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/jerome79/risk-analysis-agent/actions)
[![Coverage](https://img.shields.io/codecov/c/github/jerome79/risk-analysis-agent)](https://app.codecov.io/gh/jerome79/risk-analysis-agent)

---

## Overview
Risk Analysis Agent ingests large financial or regulatory documents and surfaces **auditable risk summaries** across seven categories:

- **Market**
- **Liquidity**
- **Credit**
- **Operational**
- **Cyber**
- **Regulatory / ESG**
- **Model**

Key features:

* **Zero-shot NLI classifier (BART-MNLI)** for risk tagging, with confidence scores.
* **Retrieval-augmented generation (RAG)** using **Chroma** and **MiniLM** embeddings.
* **Streamlit web UI** and **CLI** for ingestion, analysis, and Q&A.
* Runs fully **local and free** via [Ollama](https://ollama.com/) and open-source LLMs (e.g., Mistral, Gemma).
* Can connect to openai and claude models if API keys are provided.

---

## Quick Start

### One-command demo
```bash
make demo
# or with Docker
docker compose up --build
docker compose exec ollama ollama pull gemma3:1b
```
Then open [http://localhost:8501](http://localhost:8501).
(Optional) Ollama API → http://localhost:11434

This seeds sample filings, builds the index, and launches the Streamlit app for interactive Q&A.

---

## Installation (local dev)

```bash
git clone https://github.com/jerome79/risk-analysis-agent.git
cd risk-analysis-agent
pip install -e .
# optional: ollama pull mistral
```

---

## Configuration

| Variable            | Default              | Description                           |
|---------------------|----------------------|---------------------------------------|
| `ZSL_MODEL`         | `facebook/bart-large-mnli` | Zero-shot classifier model |
| `LLM_MODEL`         | `mistral`           | Local Ollama model for summaries      |
| `CHROMA_PERSIST_DIR`| `.chroma`           | Vector DB path                        |

Create a `.env` file or export env vars to override.

---

## Architecture

The Risk Analysis Agent processes documents through several modular components:

```
[TXT/PDF] → [Chunking + MiniLM Embeddings] → [Chroma DB]
       → [Zero-shot NLI Risk Classifier] → [Summary + Q&A with citations]
```

- **Input:** Accepts TXT or PDF financial/regulatory documents.
- **Chunking & Embeddings:** Splits text and generates MiniLM embeddings for semantic search.
- **Chroma DB:** Stores and retrieves document chunks efficiently.
- **Risk Classifier:** Tags risks using a zero-shot NLI model.
- **RAG Engine:** Answers questions and summarizes, citing source text.
- **Output:** Provides structured risk summaries and Q\&A with traceable citations.

---

## Security

- **Data never leaves your machine.**
- All models run locally; ideal for regulated or sensitive documents.

---

## Roadmap

- Few-shot / fine-tuned classifiers (FinBERT, RoBERTa)
- Hybrid keyword–rules engine for higher precision/recall
- PDF → TXT converter for fully automated ingestion

---

## Contributing

Issues and PRs welcome!
Run tests with:

```bash
pytest
```

---

## License
MIT License © 2025 Jérôme **(jerome79)**

---

### GitHub Topics
Add these under *Settings ▸ General ▸ Topics* for discoverability:

```
risk-analysis rag nlp zero-shot-classification streamlit ollama
retrieval-augmented-generation financial-documents
```
