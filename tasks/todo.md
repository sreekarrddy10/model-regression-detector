# Model Regression Detection System — Todo

Spec: [docs/SPEC.md](../docs/SPEC.md) · Source: BASWE Guide §Project 1

**Status:** Phase 1 complete. 47 offline tests pass, 88% coverage, all lint gates clean.

---

## Phase 1 — Feature under test (Days 1–2) ✅

- [x] `src/mrd/providers/base.py` — `Provider` protocol + normalized `Response` + content-addressed `Request.fingerprint()`
- [x] `src/mrd/providers/openai.py`, `anthropic.py` — lazy SDK import, native structured output
- [x] `src/mrd/providers/registry.py` — prefix routing + tier selection
- [x] `src/mrd/providers/cassette.py` — record/replay, zero-network offline tier
- [x] `src/mrd/providers/pricing.py` + `config/pricing.yaml` — prices as config; unknown model → cost unavailable, never $0
- [x] `src/mrd/feature/classifier.py` — email → `{category, summary}`, prompt injected, never raises on bad output
- [x] Pydantic v2 frozen output schema — **native structured output, not `instructor`** (see deviation below)
- [x] `src/mrd/prompts.py` + `prompts/classifier/v001.yaml` — versioned, `temperature=0` enforced at load
- [x] `tests/cassettes/` seeded; `pytest -m unit` green with zero network and zero keys
- [x] **Proof (offline):** both providers normalize to an identical `Response`; editing a prompt misses its old cassette
- [ ] **Proof (live, deferred):** identical classification through both providers — integration test written, skips without keys

### Deviation from spec

`instructor` was dropped in favour of each provider's native structured output. `instructor`
patches vendor clients individually and returns the model object directly, which bypasses the
normalized `Response` and drops the token/latency/cost accounting the gate depends on. Native
`json_schema` (OpenAI) and forced tool use (Anthropic) both yield a JSON string, giving one parse
path and one schema across vendors with the accounting intact.

## Phase 2 — Golden dataset (Days 2–4)

- [ ] 80–100 hand-written cases, 4 categories (NOT LLM-generated)
- [ ] Edge cases tagged `ambiguous` / `adversarial`: two-category overlap, ultra-short, typo-heavy, mixed-language, sarcastic, empty body
- [ ] ~12 cases marked `critical: true`
- [ ] `data/golden/emails.jsonl` + `dataset.lock.json` (sha256, counts by difficulty)
- [ ] `data/golden/judge_holdout.jsonl` — 20 human-scored summaries
- [ ] **Proof:** loader validates every line; hash reproduces

## Phase 3 — Evaluation engine (Days 4–7)

- [ ] Async batched runner, bounded concurrency, retry/backoff, `temperature=0`, `N=3`
- [ ] `graders/code.py` — category match, schema validity, latency, cost
- [ ] `graders/judge.py` — DeepEval G-Eval 1–5 + rationale
- [ ] `graders/calibration.py` — Cohen's κ / Spearman vs. human holdout; abort if κ < 0.60
- [ ] `compare.py` — flip detection (≥2/3), McNemar exact, per-category delta, pass@1/@3/^3, EWMA
- [ ] `store/sqlite.py` + `schema.sql` (`runs`, `case_results`, `judge_calibration`)
- [ ] **Proof:** degrade v001→v002, engine reports drop at p<0.05 and names regressed case IDs

## Phase 4 — Alerting & reporting (Days 7–9)

- [ ] `report/html.py` — Jinja2 single file: metadata, scorecard, side-by-side regressions, trend chart, calibration panel
- [ ] `alerts/slack.py` — status, headline numbers, top 3 regressed IDs, report link
- [ ] EWMA slow-drift warning, independent of per-run diff
- [ ] **Proof:** PASS / WARN / BLOCK alert screenshots; report opens offline

## Phase 5 — CI/CD (Days 9–11)

- [ ] `.github/workflows/eval.yml` — unit (every push) / smoke (`prompts/**`,`src/**`) / full (PR→main + nightly)
- [ ] Sticky PR comment with gate table; non-zero exit on BLOCK; HTML uploaded as artifact
- [ ] Multi-stage Dockerfile (non-root, healthcheck) + docker-compose
- [ ] `bandit -r src/` and `mypy src/` in pipeline
- [ ] **Proof:** one PR blocked, one PR passed

## Phase 6 — Portfolio polish (Days 11–12)

- [ ] README as onboarding docs (not a tutorial)
- [ ] `docs/DECISIONS.md` — D1–D4 with rejected alternatives
- [ ] 3-min Loom: prompt edit → CI → Slack → diff report → merge blocked
- [ ] Write-up: slow drift vs. per-run regression
- [ ] **Proof:** clean clone → `docker compose up` → green run from README alone

---

## Definition of Done

- [ ] `make eval TIER=full` reproducible from config alone
- [ ] Coverage ≥ 80%; ruff / black --check / mypy / bandit clean
- [ ] Judge κ ≥ 0.60 recorded in repo
- [ ] Blocked PR + passed PR linked from README
- [ ] `docker compose up` works from clean clone
- [ ] Loom recorded, DECISIONS.md written, lessons.md non-empty

---

## Review

_(Fill in after each phase — summary of what changed and why.)_
