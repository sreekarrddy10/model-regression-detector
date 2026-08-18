.PHONY: help venv install install-e2e seed test test-e2e lint fmt record eval clean demo-report docker-build \
        dataset-validate dataset-report dataset-lock dataset-verify dataset-new
.DEFAULT_GOAL := help

PY      := .venv/bin/python
PYTEST  := .venv/bin/pytest
TIER    ?= smoke
# v001 explicitly, not "latest": v002 is the deliberately degraded demo variant,
# and the default invocation must not silently run a prompt the gate blocks.
PROMPT  ?= v001
VERSION ?= v1
ID      ?= tc_0001

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## Create the 3.11 virtualenv
	uv venv --python 3.11 .venv

install: venv ## Install the package and dev tooling
	uv pip install --python $(PY) -e '.[dev]'

install-providers: ## Add the live provider SDKs (needed for smoke/full/record)
	uv pip install --python $(PY) -e '.[providers]'

install-e2e: ## Add Playwright and a headless chromium for the report tests
	uv pip install --python $(PY) -e '.[e2e]'
	.venv/bin/playwright install chromium

seed: ## Regenerate offline cassettes from deterministic stubs
	$(PY) scripts/seed_cassettes.py

test: ## Offline tier: no network, no API keys
	$(PYTEST) -m unit --cov=src/mrd --cov-report=term-missing

test-e2e: ## Render the HTML report in a real browser (needs make install-e2e)
	$(PYTEST) -m e2e

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

dataset-validate: ## Validate the golden dataset; every error with a line number
	$(PY) -m mrd.dataset validate

dataset-report: ## Coverage report and remaining gaps
	$(PY) -m mrd.dataset report

dataset-lock: ## Freeze ground truth: make dataset-lock VERSION=v1
	$(PY) -m mrd.dataset lock --version $(VERSION)

dataset-verify: ## Fail if the dataset drifted from its lock
	$(PY) -m mrd.dataset verify

dataset-new: ## Append a blank case row: make dataset-new ID=tc_0007
	$(PY) -m mrd.dataset new --id $(ID) >> data/golden/emails.jsonl
	@echo "Appended blank $(ID) to data/golden/emails.jsonl - now fill it in."

demo-report: ## Regenerate docs/sample-report.html from a scripted regression
	$(PY) scripts/demo_report.py || true

record: ## Re-record cassettes from live providers (needs keys)
	$(PY) -m mrd.cli eval --tier smoke --no-slack

eval: ## Run the gate: make eval TIER=smoke [PROMPT=v002 for the degraded demo]
	$(PY) -m mrd.cli eval --tier $(TIER) --prompt $(PROMPT) \
		--git-sha $$(git rev-parse HEAD 2>/dev/null || echo unknown)

docker-build: ## Build the runtime image
	docker build -t mrd:local .

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
