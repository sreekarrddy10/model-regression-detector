# Model Regression Detection System — Todo

Spec: [docs/SPEC.md](../docs/SPEC.md) · Source: BASWE Guide §Project 1

**Status:** Phases 1, 3, 4 complete and Phase 2 plumbing built. 216 offline tests, 97% coverage,
all lint gates clean. Remaining blocker: the golden cases are hand-authored (Phase 2, below).

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

### Plumbing ✅ (built)

- [x] `dataset/schema.py` — `GoldenCase`, `HoldoutSample`; frozen, `extra="forbid"`, `notes` required
- [x] `dataset/loader.py` — JSONL parse, all errors aggregated with line numbers, duplicate id/email detection, **few-shot leakage guard**
- [x] `dataset/hashing.py` — content hash + `dataset.lock.json`, `verify` refuses drifted ground truth
- [x] `dataset/report.py` — coverage report; errors vs. warnings separated; holdout score-variance check
- [x] `dataset/__main__.py` — CLI: `validate` · `report` · `lock` · `verify` · `new`
- [x] `data/golden/AUTHORING.md` — how to write a case, edge cases worth covering, the two enforced rules
- [x] **Proof:** leakage guard rejects a case copied from `v001` few-shot, live against the real prompt; every label change moves the hash

### Authoring ⬜ (human, not automatable)

- [ ] 80 hand-written cases, ≥12 per category — `make dataset-new ID=tc_00NN`
- [ ] ≥12 `ambiguous`, ≥8 `adversarial`: two-category overlap, ultra-short, typo-heavy, mixed-language, sarcastic, empty body, buried request
- [ ] ≥10 marked `critical: true`
- [ ] 20 judge-holdout summaries scored 1–5 **with deliberate spread** — a flat holdout yields an undefined κ
- [ ] `make dataset-report` shows READY
- [ ] `make dataset-lock VERSION=v1`

Run `make dataset-report` for live progress against every target.

## Phase 3 — Evaluation engine (Days 4–7) ✅

- [x] `runner.py` — async, bounded concurrency, retry/backoff, `temperature=0`, `N=3`; a provider outage is recorded per attempt, never fatal
- [x] `graders/code.py` — schema validity, category match, latency, cost
- [x] `graders/judge.py` — 1–5 rubric + rationale **through the provider layer, not DeepEval** (D5)
- [x] `graders/calibration.py` — quadratic-weighted κ / Spearman vs. human holdout; run aborts below κ 0.60
- [x] `stats.py` — McNemar exact, weighted κ, Spearman, majority/flaky, pass@1/@k/^k, EWMA
- [x] `compare.py` — flip detection (≥2/3), significance + effect size, full gate matrix
- [x] `store/sqlite.py` + `schema.sql` — baseline selection filters on `dataset_hash`
- [x] **Proof:** 25 gate tests encode each threshold's operational question; statistics verified against hand-computed values

### Remaining for Phase 3

- [ ] `MRD_TIER` wiring + `make eval` CLI entry point (needs a locked dataset)
- [ ] **Proof (deferred):** degrade v001→v002, engine names the regressed case IDs at p<0.05 — needs real cases

## Phase 4 — Alerting & reporting (Days 7–9) ✅

- [x] `report/model.py` — one verified structure both views render from, so HTML and Slack cannot disagree
- [x] `report/html.py` + template — single self-contained file: metadata, scorecard, side-by-side attempts, hand-rolled SVG trend, calibration panel
- [x] `alerts/slack.py` — Block Kit payload built separately from sending; https-only; webhook redacted from errors
- [x] EWMA slow-drift warning, independent of the per-run diff
- [x] `scripts/demo_report.py` + `make demo-report`
- [x] **Proof:** [docs/sample-report.html](../docs/sample-report.html) — a real BLOCK report generated offline, well-formed, zero external references, zero script tags

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
