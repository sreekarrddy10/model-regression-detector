# Model Regression Detection System — Build Specification

> BASWE Project 1. CI/CD for model behavior: catch LLM quality regressions before they reach users.
> Source: `BASWE_15_AI_Engineering_Projects_Guide.md` §Project 1
> Conventions: `~/Desktop/everything-claude-code` (ECC) rules, skills, hooks
> Workflow: Plan-first / verify-before-done (CLAUDE.md)

---

## 1. Requirements

### Functional

| ID | Requirement |
|---|---|
| F1 | An LLM feature (support-email classifier) whose prompt is a versioned, external artifact |
| F2 | A human-labeled golden dataset with stable IDs, difficulty tags, and provenance |
| F3 | An eval runner that executes every case and records multi-dimensional per-case results |
| F4 | A comparison engine that diffs any run against a baseline and separates signal from noise |
| F5 | An HTML diff report + Slack alert with pass/warn/fail status |
| F6 | A GitHub Action that gates merge on `/prompts/**` changes |
| F7 | Slow-drift detection independent of per-run diffs |
| F8 | Full containerization; reproducible from config alone |

### Non-functional

| ID | Requirement | Target |
|---|---|---|
| N1 | Smoke gate cost per PR | < $0.05 |
| N2 | Smoke gate wall-clock | < 90s |
| N3 | Full gate wall-clock | < 8 min |
| N4 | Harness unit tests make zero live API calls | 100% cassette-backed |
| N5 | Test coverage (ECC `rules/common/testing.md`) | ≥ 80% |
| N6 | No secrets in source (ECC `rules/python/security.md`) | bandit clean |

### Domain assumptions

- Single-tenant, single feature under test in V1. Multi-feature is a V2 extension, not a V1 requirement.
- Ground truth is human-authored and treated as immutable once merged; corrections require a dataset version bump.
- Budget: hobby-scale. No paid observability vendor, no managed DB.

---

## 2. Tech Stack

Guide baseline, with three deliberate substitutions (justified in §7).

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | ECC `rules/python/*` applies: PEP 8, type annotations, frozen dataclasses |
| LLM provider | **Provider-agnostic layer**; OpenAI `gpt-4o` / `gpt-4o-mini` + Anthropic `claude-sonnet-5` | Guide says "easy to swap later" — build the swap on day 1, it costs ~40 LOC |
| Eval framework | Custom scorers + judge through the provider layer | RAGAS is retrieval-specific; DeepEval was dropped in Phase 3 (see D5) |
| Structured output | Pydantic v2 + `instructor` | Type-safe parsing; ECC immutability rule |
| Storage | SQLite (`runs`, `case_results`, `judge_calibration`) + JSONL dataset | Zero infra, git-friendly, queryable |
| Statistics | `scipy.stats` (exact binomial / McNemar, Spearman) | Replaces flat %-thresholds with a real significance test |
| Reporting | Jinja2 → single-file HTML | No build step, artifact-uploadable from CI |
| Dashboard | Streamlit (`dashboard/app.py`) | Optional local trend view |
| Alerting | Slack incoming webhook | What real teams use |
| CI | GitHub Actions | Free tier sufficient |
| Container | Docker + docker-compose | ECC skill: `docker-patterns` |
| Tooling | pytest, ruff, black, isort, mypy, bandit | ECC `rules/python/*` |

---

## 3. Architecture

```
prompts/classifier/v00N.yaml ──┐
data/golden/emails.jsonl ──────┼──► runner.py ──► providers/{openai,anthropic}
                               │        │
                               │        ├──► graders/code.py    (deterministic)
                               │        └──► graders/judge.py   (LLM-as-judge)
                               │                    │
                               │                    └──► graders/calibration.py
                               ▼                              (judge vs human κ)
                        store/sqlite.py
                               │
                               ▼
                        compare.py ──► McNemar · flip detection · EWMA drift
                               │
                 ┌─────────────┼──────────────┐
                 ▼             ▼              ▼
          report/html.py  alerts/slack.py  CI exit code
```

### 3.1 Data contracts

All models are Pydantic v2, `frozen=True` (ECC immutability rule).

```python
class PromptConfig:      # prompts/classifier/v003.yaml
    version_id: str      # "v003"
    created_at: datetime
    model: str
    temperature: float   # pinned to 0.0 for eval runs
    max_tokens: int
    system_prompt: str
    few_shot: list[Example]
    commit_message: str  # why this version exists

class GoldenCase:          # data/golden/emails.jsonl (one object per line)
    id: str              # "tc_0042" — stable forever
    input_email: str
    expected_category: Literal["billing", "technical", "account", "general"]
    expected_summary: str
    difficulty: Literal["easy", "medium", "hard"]     # how demanding
    strata: tuple[Literal["ambiguous", "adversarial", "critical"], ...]  # overlapping tags
    source: Literal["handwritten", "from_failure"]
    notes: str           # why this case matters
    added_at: datetime

class CaseResult:
    run_id: str; case_id: str; repeat_idx: int
    raw_output: str
    category: str | None; summary: str | None; parse_error: str | None
    category_match: bool
    judge_score: int | None; judge_rationale: str | None
    latency_ms: int; input_tokens: int; output_tokens: int; cost_usd: float

class EvalRun:
    run_id: str; git_sha: str
    prompt_version: str
    dataset_version: str; dataset_hash: str   # sha256 of emails.jsonl
    model: str; repeats: int; started_at: datetime
    tier: Literal["smoke", "full"]
```

`dataset_hash` is load-bearing: a run whose hash differs from the baseline's is **not comparable**, and `compare.py` refuses to diff it. This is the ECC `content-hash-cache-pattern` skill applied to eval integrity — it makes silent ground-truth drift impossible.

### 3.2 Grader separation (core design decision)

ECC's `eval-harness` skill lists *"allowing flaky graders in release gates"* as an anti-pattern. Therefore:

| Grader | Type | Measures | May block merge? |
|---|---|---|---|
| `code.py::category_match` | Deterministic | exact label match | **Yes** |
| `code.py::schema_valid` | Deterministic | output parses to schema | **Yes** |
| `code.py::latency`, `cost` | Deterministic | p95 ms, $/case | Warn only |
| `judge.py::summary_quality` | Model (G-Eval, 1–5) | relevance & faithfulness | Only when confirmed across 3 seeds |
| `calibration.py` | Meta | judge κ vs. human labels | **Yes** — an uncalibrated judge invalidates its own signal |

### 3.3 Signal vs. noise

The guide proposes flat thresholds (warn >3%, critical >8%). Refined to a two-part test, because 2 flips out of 80 is not the same event at n=80 as at n=800:

1. **Repeat-based flip detection.** Every case runs `N=3` at `temperature=0`. A case counts as a **hard regression** only if it passes on baseline and fails in ≥2 of 3 repeats. Single-repeat flips are logged as `flaky`, not `regressed`.
2. **Paired significance.** Run McNemar's exact test on the (baseline, candidate) pass/fail pairs. Report `p`, plus the raw accuracy delta as effect size. Gate on *both* significance and effect size — never on p-value alone.

Reported alongside, using ECC `eval-harness` vocabulary: `pass@1`, `pass@3`, `pass^3`. `pass^3 = 1.00` is required on every `critical: true` case.

### 3.4 Gate matrix

| Signal | Threshold | Action |
|---|---|---|
| Any `critical: true` case loses `pass^3` | ≥ 1 | **BLOCK** |
| Schema parse failures | > 0 | **BLOCK** |
| Category accuracy delta | ≤ −8% | **BLOCK** |
| Category accuracy delta, McNemar p < 0.05, net negative | any | **BLOCK** |
| Judge κ vs. human holdout | < 0.60 | **BLOCK** (judge untrusted) |
| Category accuracy delta | −3% to −8% | WARN |
| Judge mean delta, confirmed 3 seeds | ≤ −0.5 | **BLOCK** |
| Judge mean delta | −0.3 to −0.5 | WARN |
| p95 latency delta | > +25% | WARN |
| Cost per case delta | > +30% | WARN |
| 7-run EWMA accuracy | < 0.90 floor | WARN (slow drift) |
| Dataset hash ≠ baseline hash | any | **BLOCK** (incomparable) |

### 3.5 Tiered CI (cost control)

| Tier | Cases | Repeats | Judge | Trigger | Est. cost |
|---|---|---|---|---|---|
| `unit` | 0 (cassettes) | — | — | every push | $0.00 |
| `smoke` | 20 stratified | 1 | no | every push to `/prompts/**` or `/src/**` | ~$0.02 |
| `full` | all (80–100) | 3 | yes | PR → `main`, nightly cron | ~$0.60 |

The `unit` tier replays recorded provider responses from `tests/cassettes/`. This is ECC's `ai-regression-testing` sandbox-mode pattern applied to LLM calls: the harness's own logic is tested with zero API spend and zero flake, and only the *model behavior* tiers touch the network.

---

## 4. Repository Layout

```
model-regression-detector/
├── .claude/
│   ├── evals/                     # eval-harness skill layout
│   │   ├── classifier.md          # eval definition
│   │   ├── classifier.log         # run history
│   │   └── baseline.json          # regression baseline pointer
│   ├── commands/eval.md
│   └── settings.json              # PostToolUse: ruff · black · mypy on **/*.py
├── .github/workflows/eval.yml
├── prompts/classifier/v001.yaml … v00N.yaml
├── data/golden/
│   ├── cases.yaml                 # authoring surface (what you edit)
│   ├── holdout.yaml               # authoring surface for the judge holdout
│   ├── emails.jsonl               # generated, canonical, hashed
│   ├── dataset.lock.json          # {version, sha256, count, by_difficulty}
│   └── judge_holdout.jsonl        # 20 human-scored summaries for calibration
├── src/mrd/
│   ├── feature/classifier.py      # the system under test
│   ├── providers/{base,openai,anthropic}.py
│   ├── dataset/{loader,hash}.py
│   ├── graders/{code,judge,calibration}.py
│   ├── runner.py
│   ├── compare.py
│   ├── report/{html.py,templates/}
│   ├── alerts/slack.py
│   └── store/{sqlite.py,schema.sql}
├── dashboard/app.py               # Streamlit trend view
├── tests/
│   ├── cassettes/                 # recorded provider responses
│   └── test_*.py
├── tasks/{todo.md,lessons.md}
├── docs/{SPEC.md,DECISIONS.md}
├── Dockerfile · docker-compose.yml
├── .env.example · pyproject.toml · Makefile · README.md
```

---

## 5. ECC Integration (concrete)

| ECC asset | Path | Applied to |
|---|---|---|
| `rules/common/*` | coding-style, testing, security, git-workflow | 200–400 LOC files, ≥80% coverage, no hardcoded secrets, conventional commits |
| `rules/python/coding-style.md` | — | PEP 8, type annotations on every signature, `@dataclass(frozen=True)` / Pydantic frozen, black + isort + ruff |
| `rules/python/testing.md` | — | pytest, `--cov=src --cov-report=term-missing`, `@pytest.mark.{unit,integration}` |
| `rules/python/security.md` | — | `os.environ["OPENAI_API_KEY"]` (KeyError-on-missing, never `.get`), `bandit -r src/` in CI |
| `rules/python/hooks.md` | — | PostToolUse: black/ruff format + mypy on `.py` edit; warn on `print()` → use `logging` |
| `skills/eval-harness` | SKILL.md | `.claude/evals/` layout, pass@k / pass^k, 4-grader taxonomy, anti-pattern list drives §3.2 and §3.4 |
| `skills/ai-regression-testing` | SKILL.md | Cassette/sandbox tier; "test where bugs were found"; every failure case is named after the bug it prevents |
| `skills/python-testing`, `python-patterns` | — | pytest fixtures, idiomatic patterns |
| `skills/cost-aware-llm-pipeline` | — | Model routing (judge on `gpt-4o`, feature on `gpt-4o-mini`), budget tracking, prompt caching |
| `skills/docker-patterns`, `deployment-patterns` | — | Multi-stage Dockerfile, healthcheck, non-root user |
| `agents/*` | planner, tdd-guide, code-reviewer, security-reviewer, build-error-resolver | One agent per phase concern (see §6) |
| `hooks/hooks.json` | format reference | `{matcher, hooks: [{type, command}], description}` |
| `.env.example` | format reference | Canonical var list, sectioned comments, `cp .env.example .env`, never commit `.env` |
| Naming | CONTRIBUTING.md | lowercase-with-hyphens for all `.md` assets |

**Env vars** (`.env.example`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SLACK_WEBHOOK_URL`, `MRD_MODEL`, `MRD_JUDGE_MODEL`, `MRD_REPEATS`, `MRD_TIER`, `MRD_WARN_THRESHOLD`, `MRD_BLOCK_THRESHOLD`, `MRD_DB_PATH`.

---

## 6. Build Plan — 12 Days

Mapped 1:1 to the guide's six phases. Each phase ends with a **proof artifact** — no phase is "done" without it (CLAUDE.md §4 Verification Before Done).

### Phase 1 — Feature under test · Days 1–2
- [ ] `feature/classifier.py`: email → `{category, summary}`, prompt injected as `PromptConfig`
- [ ] `prompts/classifier/v001.yaml` with `version_id`, `commit_message`, few-shot block
- [ ] `providers/base.py` protocol + OpenAI and Anthropic implementations returning a normalized `Response`
- [ ] Pydantic output schema enforced via `instructor`
- **Proof:** `pytest -m unit` green against cassettes; same email classified identically through both providers.
- *Agent:* `tdd-guide`

### Phase 2 — Golden dataset · Days 2–4
- [ ] 80–100 hand-written cases across 4 categories. **Do not generate with an LLM.**
- [ ] Deliberate edge cases: two-category ambiguity, 5-word emails, heavy typos, mixed-language, sarcasm, empty body → tag `ambiguous` / `adversarial`
- [ ] Mark ~12 cases `critical: true` (the ones that must never break)
- [ ] `dataset.lock.json` with sha256 + per-difficulty counts; `judge_holdout.jsonl` with 20 human 1–5 summary scores
- **Proof:** loader validates every line against `GoldenCase`; hash reproduces on re-run; difficulty distribution printed.

### Phase 3 — Evaluation engine · Days 4–7
- [ ] Async batched runner with bounded concurrency + retry/backoff; `temperature=0`, `N=3`
- [ ] Code graders: category match, schema validity, latency, cost
- [ ] Judge grader: DeepEval G-Eval, 1–5 with rationale
- [ ] **Judge calibration**: score the 20-case holdout, compute Cohen's κ / Spearman ρ vs. human; abort the run if κ < 0.60
- [ ] `compare.py`: flip detection (≥2/3), McNemar exact test, per-category deltas, pass@1 / pass@3 / pass^3, EWMA drift
- [ ] SQLite persistence for runs + case results
- **Proof:** deliberately degrade `v001` → `v002` (drop the few-shot block); the engine reports the drop with `p < 0.05` and names the exact regressed case IDs.
- *Agent:* `tdd-guide`, then `code-reviewer`

### Phase 4 — Alerting & reporting · Days 7–9
- [ ] Jinja2 single-file HTML: run metadata, scorecard vs. baseline, side-by-side old/new output for every regressed case, last-N trend chart, judge-calibration panel
- [ ] Slack webhook: status block, headline numbers, top 3 regressed case IDs, report link
- [ ] EWMA slow-drift warning independent of per-run diffs
- **Proof:** screenshots of a PASS alert, a WARN alert, and a BLOCK alert; report opens standalone with no network.

### Phase 5 — CI/CD · Days 9–11
- [ ] `.github/workflows/eval.yml`: `unit` on every push; `smoke` on `/prompts/**` + `/src/**`; `full` on PR→`main` and nightly cron
- [ ] Sticky PR comment with the gate table; non-zero exit on BLOCK; HTML report uploaded as workflow artifact
- [ ] Multi-stage Dockerfile (non-root, healthcheck); all thresholds via env
- [ ] `bandit -r src/` + `mypy src/` in the pipeline
- **Proof:** a PR that worsens the prompt is blocked by CI; a PR that improves it merges — both linked in the README.
- *Agent:* `security-reviewer`, `build-error-resolver`

### Phase 6 — Portfolio polish · Days 11–12
- [ ] README written as onboarding docs, not a tutorial: what it does, setup, how to add a case, how to tune thresholds, architecture decisions with rationale
- [ ] `docs/DECISIONS.md`: the four decisions in §7, each with the alternative rejected and why
- [ ] 3-minute Loom: edit prompt → CI runs → Slack alert → diff report → merge blocked
- [ ] Blog post / README section on slow-drift vs. per-run regression
- **Proof:** a stranger can `docker compose up` and get a green run from the README alone.

---

## 7. Design Decisions (interview material)

**D1 — The judge is evaluated before it is trusted.**
LLM-as-judge is itself an unvalidated model. 20 human-scored summaries are held out; if judge–human agreement falls below κ = 0.60 the run aborts rather than reporting a number nobody should believe. Rejected alternative: use the judge score unconditionally — which is how eval suites quietly become theater.

**D2 — Deterministic graders gate; probabilistic graders advise.**
Category match blocks merge. Summary quality only blocks when the degradation reproduces across three seeds. Directly follows ECC `eval-harness`'s "no flaky graders in release gates." Rejected alternative: one blended score — it makes a blocked merge unexplainable.

**D3 — Significance test, not a flat percentage.**
`N=3` repeats separate flaky from regressed; McNemar's exact test on paired outcomes answers "is 2-of-80 signal?" properly. Effect size *and* p-value must both trip. Rejected alternative: the guide's flat 3%/8% thresholds — kept as the effect-size half, discarded as the whole test.

**D4 — Dataset hash is part of the comparison key.**
A run against a mutated golden set is marked incomparable instead of silently diffed. Rejected alternative: compare by timestamp — which lets a quiet ground-truth edit masquerade as a model improvement.

**D5 - The judge runs through the provider layer, not a framework.**
DeepEval brings its own client stack and would bypass the normalized `Response`,
dropping the token, latency and cost accounting the gate depends on - the same
objection that removed `instructor` in Phase 1. The judge is a prompt plus a
structured-output schema, which keeps it cassette-replayable offline and keeps
its spend on the same ledger as everything else. Rejected alternative: adopt the
framework and reconcile two accounting paths.

**Headline for the portfolio:** *"CI/CD for model behavior — a merge gate that blocks prompt changes on statistically significant quality regressions, with a calibrated judge and slow-drift detection, for under $0.05 per pull request."*

---

## 7b. Deliberate cuts

Two items from the original layout in §4 were dropped on purpose.

**`dashboard/app.py` (Streamlit).** The HTML report already carries the scorecard, the
per-case diff and the trend chart, and the Slack alert renders from the same `ReportData`
structure. A third rendering path would be a third thing to keep in sync — exactly the
failure D5 and the shared-`ReportData` design exist to avoid. The trend view that
justified it is now a panel in the report.

**`.claude/evals/`.** This would have been a second bookkeeping layer over a repo that
already has 288 tests plus a per-phase proof artifact in `tasks/todo.md` written as
pass/fail criteria. The `eval-harness` skill's substance — pass@k / pass^k, the grader
taxonomy, the anti-pattern list — is implemented in `stats.py` and `compare.py` rather
than transcribed into a parallel checklist.

Both are recoverable if the need appears. Neither is blocking anything.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Judge non-determinism swamps the signal | Calibration gate (D1) + 3-seed confirmation (D2) |
| CI cost creep | Tiered gates (§3.5); cassettes for harness tests; `gpt-4o-mini` for the feature |
| Golden set overfitting | Every production failure enters as `source: from_failure`; growth tracked in the report |
| Provider API drift breaks the harness | Provider protocol + cassette tests catch shape changes without spend |
| Flaky graders in the gate | ECC anti-pattern list encoded directly in §3.4 |
| Secrets leaked to CI logs | GitHub Secrets only; `bandit` in pipeline; redact prompts containing PII before storage |

---

## 9. Definition of Done

- [ ] `make eval TIER=full` reproduces a run from config alone
- [ ] Coverage ≥ 80%; `ruff`, `black --check`, `mypy`, `bandit` all clean
- [ ] Judge calibration κ ≥ 0.60 recorded in the repo
- [ ] One merged PR blocked by the gate, one passed — both linked from the README
- [ ] `docker compose up` works from a clean clone with only `.env` filled
- [ ] Loom recorded; `DECISIONS.md` written
- [ ] `tasks/lessons.md` non-empty
