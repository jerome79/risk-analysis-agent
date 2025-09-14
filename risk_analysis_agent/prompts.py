RISK_SUMMARY_PROMPT = """You are a financial risk analyst.
Summarize the top risks for {issuer} (FY {year}) from the context.
Group by the risk taxonomy and CITE chunk_ids used. Be concise and actionable.

Format:
- Risk Category: <name>
  - Why it matters: <1-2 sentences>
  - Evidence (chunk_ids): <ids>
  - Suggested Mitigation: <1 sentence>

Context:
{context}
"""

QA_PROMPT = """You are an assistant for Q&A over risk disclosures.
Use the retrieved context to answer briefly. Cite chunk_ids in your answer.

Question: {question}

Context:
{context}

Answer:
"""
