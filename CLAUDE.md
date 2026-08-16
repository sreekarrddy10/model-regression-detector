# CLAUDE.md — Model Regression Detection System

Spec: [docs/SPEC.md](docs/SPEC.md) · Plan: [tasks/todo.md](tasks/todo.md) · Lessons: [tasks/lessons.md](tasks/lessons.md)

## Project Overview

CI/CD for model behavior. Continuously evaluates a prompt-driven LLM feature against a
human-labeled golden dataset, detects statistically significant quality regressions, and blocks
the merge before bad outputs ship. BASWE Guide §Project 1.

## Workflow

1. **Plan first.** Any change touching 3+ files or a threshold goes to `tasks/todo.md` before code.
2. **Verify before done.** A phase is complete only when its proof artifact in `docs/SPEC.md` §6
   exists. "It runs" is not proof; a named regression caught by the engine is.
3. **Capture lessons.** After any correction, append the pattern to `tasks/lessons.md`.
4. **Demand elegance.** If a threshold needs a special case, the threshold model is wrong — fix the
   model, not the case.
5. **Autonomous bug fixing.** Failing CI is a task, not a question. Fix it, then add the regression
   case that would have caught it.

## Non-negotiables

- **Never generate golden dataset cases with an LLM.** Human ground truth is the entire premise.
- **Deterministic graders gate; model graders advise.** See SPEC §3.2.
- **A judge is untrusted until calibrated.** κ ≥ 0.60 against the human holdout or the run aborts.
- **Dataset hash is part of the comparison key.** Runs with mismatched hashes are incomparable.
- **`temperature=0`, `N=3`** on every eval run. Single-repeat flips are `flaky`, not `regressed`.
- **Every production failure becomes a golden case** with `source: from_failure`.

## Conventions (ECC)

- Python 3.11+, PEP 8, type annotations on every signature, Pydantic v2 `frozen=True`
- black · isort · ruff · mypy · bandit — all clean before commit
- pytest, `@pytest.mark.{unit,integration}`, `--cov=src --cov-report=term-missing`, ≥80%
- Files 200–400 LOC typical, 800 max; functions <50 LOC
- Secrets via `os.environ["KEY"]` (KeyError on missing, never `.get`). Nothing hardcoded.
- `logging`, never `print()`
- Commits: `<type>: <description>` — feat, fix, refactor, docs, test, chore, perf, ci
- Markdown assets: lowercase-with-hyphens

## Commands

```bash
make test          # pytest -m unit (cassettes, zero network)
make eval TIER=smoke   # 20 cases, code graders only
make eval TIER=full    # all cases × 3 repeats + judge
make report        # regenerate HTML from latest run
make lint          # ruff + black --check + isort --check + mypy + bandit
```

## Applicable ECC skills

`eval-harness` · `ai-regression-testing` · `python-testing` · `python-patterns` ·
`cost-aware-llm-pipeline` · `docker-patterns` · `deployment-patterns`
