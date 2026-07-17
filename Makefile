# CHARKHA local pipeline. CPU-only targets — safe to run while a GPU job is live.
PY := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

.PHONY: check proof test

check:            ## fast: syntax check + fast tests
	$(PY) scripts/proof_bundle.py --syntax-only
	$(PY) tests/run_tests.py

proof:            ## full CPU proof bundle (every selftest; pre-push gate)
	$(PY) scripts/proof_bundle.py

test:             ## run the fast test suite
	$(PY) tests/run_tests.py
