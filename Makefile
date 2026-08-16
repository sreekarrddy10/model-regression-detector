.PHONY: help venv install seed test lint fmt record eval clean
.DEFAULT_GOAL := help

PY      := .venv/bin/python
PYTEST  := .venv/bin/pytest
TIER    ?= smoke

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## Create the 3.11 virtualenv
	uv venv --python 3.11 .venv

install: venv ## Install the package and dev tooling
	uv pip install --python $(PY) -e '.[dev]'

install-providers: ## Add the live provider SDKs (needed for smoke/full/record)
	uv pip install --python $(PY) -e '.[providers]'

seed: ## Regenerate offline cassettes from deterministic stubs
	$(PY) scripts/seed_cassettes.py

test: ## Offline tier: no network, no API keys
	$(PYTEST) -m unit --cov=src/mrd --cov-report=term-missing

test-integration: ## Live tier: requires API keys
	$(PYTEST) -m integration

lint: ## ruff + black + isort + mypy + bandit
	.venv/bin/ruff check src tests scripts
	.venv/bin/black --check src tests scripts
	.venv/bin/isort --check-only src tests scripts
	.venv/bin/mypy
	.venv/bin/bandit -q -r src

fmt: ## Apply formatting
	.venv/bin/black src tests scripts
	.venv/bin/isort src tests scripts
	.venv/bin/ruff check --fix src tests scripts

record: ## Re-record cassettes from live providers (needs keys)
	MRD_TIER=record $(PY) -c "raise SystemExit('Phase 3: wire the runner, then record via the runner')"

eval: ## Run the eval suite (Phase 3)
	@echo "Not implemented until Phase 3. Current phase: 1 (feature + providers)."
	@exit 1

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
