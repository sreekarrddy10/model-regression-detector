# Model Regression Detection System

CI/CD for model behavior. Every prompt change is evaluated against a human-labeled golden dataset;
statistically significant quality regressions block the merge before bad outputs reach users.

**Status: Phase 5 of 6** — the whole pipeline is built and runs end to end offline. Golden cases
are being authored by hand; the gate activates the moment a dataset is locked. See [tasks/todo.md](tasks/todo.md) and [docs/SPEC.md](docs/SPEC.md).

## Quick start

```bash
make install          # uv venv @ 3.11 + dev tooling
make test             # offline tier: 245 tests, no network, no API keys
make lint             # ruff + black + isort + mypy --strict + bandit
make dataset-report   # golden dataset coverage and remaining gaps
make demo-report      # regenerate docs/sample-report.html from a scripted regression
make eval TIER=smoke  # run the gate (needs a locked dataset)
```

No API key is needed to run the test suite. That is deliberate — see *Cassettes* below.

## What exists today

| Module | Responsibility |
|---|---|
| [src/mrd/prompts.py](src/mrd/prompts.py) | Versioned prompt artifacts loaded from `prompts/classifier/vNNN.yaml` |
| [src/mrd/feature/](src/mrd/feature/) | The system under test: support email → `{category, summary}` |
| [src/mrd/providers/](src/mrd/providers/) | One normalized contract over OpenAI and Anthropic |
| [src/mrd/dataset/](src/mrd/dataset/) | Golden dataset validation, content hashing, coverage reporting |
| [src/mrd/runner.py](src/mrd/runner.py) | Async runner: N repeats at temperature 0, bounded concurrency, retry |
| [src/mrd/graders/](src/mrd/graders/) | Deterministic graders, LLM-as-judge, judge calibration |
| [src/mrd/stats.py](src/mrd/stats.py) | McNemar exact, weighted κ, Spearman, EWMA |
| [src/mrd/compare.py](src/mrd/compare.py) | Run diffing and the merge gate |
| [src/mrd/store/](src/mrd/store/) | SQLite run history and baseline selection |
| [src/mrd/report/](src/mrd/report/) | Single-file HTML diff report |
| [src/mrd/alerts/](src/mrd/alerts/) | Slack Block Kit alerting |
| [src/mrd/sampling.py](src/mrd/sampling.py) | Deterministic stratified tier selection |
| [src/mrd/cli.py](src/mrd/cli.py) | The entry point CI calls |
| [config/pricing.yaml](config/pricing.yaml) | Token prices as reviewable config, not hardcoded constants |

## Golden dataset

Ground truth is written by hand — see [data/golden/AUTHORING.md](data/golden/AUTHORING.md). Nothing
in the codebase generates cases; a model-generated golden set only measures whether the model agrees
with itself.

```bash
make dataset-new ID=tc_0007   # append a blank row to fill in
make dataset-validate         # every error at once, with line numbers
make dataset-report           # coverage against each target
make dataset-lock VERSION=v1  # freeze ground truth
```

Two guards worth knowing about:

**Leakage.** A case may not duplicate a few-shot example from any prompt version. The model was
shown those answers, so such a case measures recall of the prompt rather than capability. The check
is whitespace- and case-insensitive.

**Drift.** `dataset.lock.json` records a SHA-256 over the semantic content of every case. `make
dataset-verify` fails if ground truth changed, because runs scored against different ground truth
are not comparable — otherwise a quiet label edit can masquerade as a model improvement. Reordering
lines or reflowing JSON does not invalidate a baseline; changing any label, email or tag does.

The judge holdout carries one non-obvious requirement: score a deliberate **spread** of quality.
Agreement statistics need disagreement to measure, so a holdout scored 5/5 across the board yields
an undefined κ and the judge would pass calibration without ever being tested. The report warns when
the spread is too narrow.

## Three decisions worth knowing

**The prompt is data, not code.** It lives in `prompts/classifier/v001.yaml` with a `version_id` and
a `commit_message`. `temperature` is validated to be exactly `0.0` at load time — sampling would make
flip detection meaningless, so non-zero is refused rather than warned about.

**A bad model output is data, not an exception.** `classify()` never raises on malformed output; it
returns `parse_error` alongside the raw response. Schema validity is a blocking gate signal, so a
crash would discard exactly the evidence the eval engine needs.

**Cassettes make the harness testable without spending anything.** Every request is content-addressed
by `Request.fingerprint()` — a SHA-256 over model, system prompt, user text, sampling params and
output schema. Recorded responses replay off that key, so:

- the offline tier runs with zero network calls and zero keys (`make test`)
- editing a prompt invalidates its recording *by construction* — a stale response can never be
  replayed against a changed prompt, it simply misses

This is ECC's sandbox-mode pattern (`skills/ai-regression-testing`) applied to LLM calls.

## Provider parity

`resolve()` routes by model-name prefix; callers name a model, never a vendor. Both providers
normalize to the same `Response` carrying text, token counts, latency and cost — because cost and
latency drift are treated as regressions, not as trivia.

Structured output uses each vendor's native mechanism (OpenAI `json_schema` strict mode, Anthropic
forced tool use) and both hand the feature layer a JSON string, so there is one parse path and one
schema regardless of provider.

Cross-provider agreement on real traffic is verified by an integration test that skips without live
keys:

```bash
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... make test-integration
```

## Cassette workflow

```bash
make seed    # regenerate offline cassettes from deterministic stubs
```

The committed cassettes are stub responses — plumbing fixtures that let a fresh clone run green.
They are not model behavior and are never used to measure quality. Real recorded responses arrive
with the first live run against a locked dataset.

## The gate

A merge is blocked only by signals that are reproducible and explainable.

| Signal | Action |
|---|---|
| A `critical` case loses `pass^3` | **BLOCK** |
| Output stops matching the response schema | **BLOCK** |
| Accuracy falls ≥ 8% | **BLOCK** |
| Regressions significantly outnumber improvements (McNemar exact, p < 0.05) | **BLOCK** |
| Judge fails calibration against human scores | **BLOCK** |
| Summary quality falls ≥ 0.5, confirmed across 3 seeds | **BLOCK** |
| Accuracy falls 3–8% | WARN |
| Summary quality falls 0.3–0.5, or unconfirmed | WARN |
| p95 latency +25%, or cost per case +30% | WARN |
| Cases became non-deterministic across repeats | WARN |
| 7-run EWMA accuracy below 90% | WARN |
| Baseline scored against different ground truth | refuses to diff |

Three properties are worth calling out.

**A flip is not a regression.** Every case runs three times at temperature 0. A case that fails once
of three is recorded as *flaky* and warns; it must fail at least twice to count as regressed. This is
what stops sampling noise from blocking merges.

**Significance and effect size both have to trip.** The guide this project follows proposes flat 3%
and 8% thresholds, which cannot distinguish 2-of-80 from noise at all — and a p-value alone would
block on trivia once the dataset is large. The gate keeps the percentage as an effect-size floor and
adds McNemar's exact test on the discordant pairs. Six regressions and zero improvements out of a
hundred blocks on significance even though the headline drop is only 6%.

**Cost and latency are regressions too.** Chasing pass rates while the bill doubles is a failure
mode, so both are measured and reported — but neither can block, because neither is a correctness
signal.

## The report

[docs/sample-report.html](docs/sample-report.html) is a real one, generated offline by
`make demo-report` from a scripted regression. It is a single file: inline CSS, a hand-generated SVG
trend line, no JavaScript, no external references at all — it is opened from a CI artifact on a
laptop with no network, so one CDN link would break it exactly when someone needs to read it.

It leads with the verdict and every rule that fired, then shows each regressed case with both runs'
attempts side by side, the expected answer, and the `notes` line explaining why that case exists.
The McNemar p-value and discordant-pair count are printed in full, because a reader should be able
to audit the gate rather than just obey it.

The Slack alert is built from the same `ReportData` structure, so the two views cannot disagree
about what happened. It never truncates silently — a reader who cannot tell 5 regressions from 40
will under-react to the larger failure.

## CI

[.github/workflows/eval.yml](.github/workflows/eval.yml) runs four jobs.

| Job | When | Cost |
|---|---|---|
| `quality` | every push — lint, types, bandit, offline tests, 80% coverage floor | free |
| `dataset` | every push — validate the golden set, verify the lock hasn't drifted | free |
| `gate` (smoke) | `prompts/**` or `src/mrd/**` changed — 20 stratified cases, 1 repeat, no judge | cents |
| `gate` (full) | PR → `main`, and nightly — every case × 3 repeats with the judge | ~$0.60 |

Running `full` on every push would cost more than it catches.

Three details that matter operationally:

**No API key is present in the `quality` job.** If a test reaches the network it fails there, rather
than quietly costing money later.

**The gate job caches `runs.sqlite`.** Without a restored baseline every run is a first run, and a
first run can never detect a regression.

**Smoke selection is deterministic**, stratified across category and difficulty, and always includes
every `critical` case. Random sampling would report sampling churn as model change.

The PR comment is updated in place via a hidden marker — a PR with eleven stacked eval comments is
one nobody reads. The exit code is the gate verdict and nothing else; that is the whole contract
with CI.

## Reading

- **[docs/DECISIONS.md](docs/DECISIONS.md)** — the eight decisions this system turns on, each with
  the alternative it rejected and the test that proves it.
- **[docs/writeup-slow-drift.md](docs/writeup-slow-drift.md)** — why per-run diffing is structurally
  blind to the most common way a production LLM feature degrades, and why an uncalibrated judge
  hides it.
- **[docs/sample-report.html](docs/sample-report.html)** — a real diff report.
- **[docs/SPEC.md](docs/SPEC.md)** — the full build specification.

## Next

Write the golden cases. `make dataset-report` shows progress against every target and names the
three strata to write next. Everything downstream activates the moment `make dataset-lock` runs.

Two proofs remain deferred: a blocked PR on GitHub (needs a remote and real cases) and
`docker compose up` from a clean clone (the image has not yet been built).
