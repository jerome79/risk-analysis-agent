# Risk Analysis Agent — One Pager (Interview Edition)

**What it is:** Local-first pipeline that ingests small document sets, classifies risk themes, retrieves relevant context, and outputs concise summaries.

**Who it's for:** PMs, risk teams, and quants who need fast triage and explainable snippets.

**Why now:** Teams are evaluating AI copilots but need privacy, determinism for review, and simple deployments.

## Capabilities
- Classify into a small, editable taxonomy (Operational, Compliance, Market, Model, Data)
- Retrieve supporting passages
- Summarize findings with short, decision-ready blurbs


## Architecture
See README (high level: ingest → embed → index → classify → retrieve → summarize).
