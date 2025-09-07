.PHONY: fmt lint type test cov audit all

fmt:
\tblack .
\truff check --fix .

lint:
\truff check .

type:
\tmypy .

test:
\tpytest

cov:
\tcoverage run -m pytest
\tcoverage report -m
\tcoverage html

audit:
\tbandit -r . -x tests || true
\tpip-audit -s || true

all: fmt lint type test
