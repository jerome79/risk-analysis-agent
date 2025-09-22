# Market Sentiment Analyzer – Architecture

CSV/Folder Upload ─▶ Preprocess (clean, dedupe)
                  └▶ Label (VADER | FinBERT | RoBERTa, batch & max_seq_len)
                        └▶ Aggregate (market / sector / ticker)
                              └▶ Dashboard (Streamlit: trends, filters, export)
## Key choices
- **Dedup** before inference to avoid overcounting syndicated news.
- **Model trade-offs**: VADER (fast baseline) vs FinBERT (finance-tuned) vs RoBERTa (social/news).
- **Performance knobs**: `BATCH_SIZE`, `MAX_SEQ_LEN`, tokenizer threads.
- **Outputs**: labeled CSV + time-series aggregates for dashboard & analysis.

## Next steps
Entity-level sentiment, alerts (webhooks), accuracy eval (Financial PhraseBank/FiQA), and cost/latency notes.
