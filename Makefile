.PHONY: fmt lint type test cov audit all

fmt:
	black .
	ruff check --fix .

lint:
	ruff check .

type:
	mypy .

test:
	pytest

cov:
	coverage run -m pytest
	coverage report -m
	coverage html

audit:
	bandit -r . -x tests || true
	pip-audit -s || true

all: fmt lint type test

run:
	streamlit run risk_analysis_agent/ui_streamlit.py --server.port 8502

demo:
	python -m risk_analysis_agent.cli demo
serve:
	python -m risk_analysis_agent.cli serve
bench:
	python -m risk_analysis_agent.cli benchmark --csv data/news_perf_test_10k.csv --model vader
