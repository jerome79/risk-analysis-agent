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
	coverage run --source=risk_analysis_agent -m pytest test/
	coverage report -m
	coverage html

audit:
	bandit -r . -x tests || true
	pip-audit -s || true

all: fmt lint type test

run:
	streamlit run risk_analysis_agent/ui_streamlit.py --server.port 8501

demo:
	python -m risk_analysis_agent.cli demo --port 8501
serve:
	python -m risk_analysis_agent.cli serve --port 8501
bench:
	python -m risk_analysis_agent.cli benchmark --csv data/news_perf_test_10k.csv --model vader
