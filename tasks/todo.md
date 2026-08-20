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

## Phase 5 — CI/CD (Days 9–11) ✅ (except the live PR proof)

- [x] `cli.py` — the eval entry point CI calls; exit code is the gate verdict, nothing else
- [x] `sampling.py` — deterministic stratified smoke selection; every critical case always included
- [x] `report/markdown.py` — sticky PR comment with the scorecard, the statistics and the regressed-case table
- [x] `.github/workflows/eval.yml` — quality (every push) / dataset integrity / smoke (`prompts/**`,`src/mrd/**`) / full (PR→main + nightly cron)
- [x] Gate job caches `runs.sqlite` so a baseline exists; uploads the report artifact; updates its own PR comment in place
- [x] `gate-skipped` job explains itself when no dataset is locked, instead of silently not running
- [x] `bandit`, `mypy`, `ruff`, `black`, `isort`, coverage floor 80% all in the pipeline
- [x] Multi-stage `Dockerfile` (non-root uid 10001, healthcheck, no dev deps in runtime) + `docker-compose.yml` + `.dockerignore`
- [x] **Proof (local, verified):** locked an 8-case demo dataset, ran the CLI end to end offline — first run recorded a baseline and exited 0; a seeded regression exited **1** naming `tc_0000` as a critical regression; editing a label without re-locking exited **1** with "not comparable"
- [ ] **Proof (deferred):** one PR blocked and one passed on GitHub — needs a remote and real cases
- [x] **Docker verified:** `mrd:local` builds (576 MB); runs as uid 10001; no dev tooling in the runtime layer; provider SDKs present; healthcheck asserts and fails correctly when config is unreachable; `docker compose config` valid

### Bug the container build caught

Three modules resolved project paths from `__file__` alone. Correct in a source checkout,
silently wrong in an installed wheel where the code sits in `site-packages` and the data is
mounted beside the working directory. Pricing and prompts both failed to load in the image, and
the original healthcheck **printed `ok` anyway** because it called `lookup()` without checking
the result — precisely the failure it existed to catch. Fixed by `src/mrd/paths.py`: one
resolver, one precedence order (env override → working directory → source tree), used by
pricing, prompts and both CLIs.

## Phase 6 — Portfolio polish (Days 11–12)

- [x] README as onboarding docs (not a tutorial)
- [x] `docs/DECISIONS.md` — D1–D8, each with the alternative it rejected and the test that proves it
- [x] `docs/writeup-slow-drift.md` — slow drift vs. per-run regression, and why an uncalibrated judge hides it
- [ ] 3-min Loom: prompt edit → CI → Slack → diff report → merge blocked (needs real cases)
- [x] **Proof:** image builds; eval runs inside the container against mounted data, exiting 0 clean and 1 on a seeded critical regression; Docker's own HEALTHCHECK reaches `healthy`

## Report rendering (browser)

- [x] 31 Playwright tests over the HTML report: self-containment, verdict styling, case detail,
      SVG geometry, scorecard, WCAG AA contrast in both schemes, three viewports
- [x] Hostile content: injected markup is escaped and shown, never executed; no off-disk fetch
- [x] Pathological content: unbroken 600-char tokens, long URLs and 60-line threads do not
      scroll the document sideways
- [x] Cross-engine: chromium + firefox + webkit
- [x] Stability: 10 consecutive full runs, 0 failures
- [x] Failure artifacts verified — screenshot, trace and video are all captured
- [x] **Found and fixed a live XSS-class bug**: autoescape was off because the template is
      `report.html.j2` and `select_autoescape(["html"])` matches the final extension. The unit
      test that was supposed to cover it passed vacuously, on a page rendering no case detail.

## Developer loop

- [x] `.claude/settings.json` + `.claude/hooks/format-python.sh` — PostToolUse black · isort ·
      ruff --fix on any `.py` edit, then a package type-check. Non-blocking by design: a formatter
      that blocks an edit is worse than an unformatted file, and `make lint` plus CI are the
      enforcement points. Exits early for non-Python files, and survives malformed payloads.

## Authoring ergonomics (added while setting up Phase 2 work)

- [x] `dataset new` picks the next free `tc_NNNN` — no manual id bookkeeping across 80 cases
- [x] Options may follow the subcommand (`mrd.dataset report --cases X`), the conventional order
- [x] `dataset report` suggests the three scarcest strata, hardest first — shortfalls normalized
      against their own targets so adversarial cases are not left until last

---

## Definition of Done

- [x] Coverage ≥ 80% (96%); ruff / black --check / isort / mypy --strict / bandit clean
- [x] `docker compose` works from a clean build — image verified, gate exits 0 clean and 1 on a seeded regression
- [x] `docs/DECISIONS.md` written (D1–D8, each with its rejected alternative)
- [x] `tasks/lessons.md` non-empty
- [x] The degraded `v002` prompt exists, so the gate demo is one command
- [ ] `make eval TIER=full` reproducible from config alone — **blocked on the golden cases**
- [ ] Judge κ ≥ 0.60 recorded in repo — **blocked on the 20 holdout summaries**
- [ ] Blocked PR + passed PR linked from README — **blocked on cases + a git remote**
- [ ] Loom recorded — **blocked on the above**

Everything unchecked traces to one dependency: the hand-written golden cases.

---

## Review

_(Fill in after each phase — summary of what changed and why.)_
