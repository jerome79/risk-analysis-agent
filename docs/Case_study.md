# PM Case Study

## Problem
Risk insights are buried in disparate docs. Manual triage is slow and error-prone.

## Constraints
- Sensitive content (no cloud dependency required)
- Review-friendly outputs
- Minutes to first value for a reviewer

## Solution
Local-first agent with a small taxonomy, explainable retrieval, and short summaries.

## Success Metrics
- Time-to-triage (minutes per doc set)
- Precision@top-5 for retrieval
- Stakeholder satisfaction score (1–5)

## Rollout
- V0: Tiny demo with cached outputs
- V1: Pluggable models and CI metrics
- V2: Workspace integration (alerts, dashboards)

## Risks & Mitigations
- Drift → add monitoring hooks
- Privacy → keep processing local, redact PII in ingest
- Adoption → fast demo path and simple CLI
