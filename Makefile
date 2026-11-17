.PHONY: verify reproduce metrics test check-numbers clean

PYTHON ?= python3

verify: metrics test check-numbers
	@echo "===================================================================="
	@echo "MAKE VERIFY SUCCEEDED: All metrics, coverage, and doc numbers green."
	@echo "===================================================================="

reproduce:
	PYTHONPATH=src $(PYTHON) -m churn_flip_audit.cli

metrics:
	PYTHONPATH=src $(PYTHON) tools/collect_metrics.py

test:
	PYTHONPATH=src $(PYTHON) tools/run_tests_with_coverage.py

check-numbers:
	PYTHONPATH=src $(PYTHON) tools/check_numbers.py --prove-failure

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
