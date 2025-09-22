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
