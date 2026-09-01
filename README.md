# Model Regression Detection System

CI/CD for model behavior. Every prompt change is evaluated against a human-labeled golden dataset;
statistically significant quality regressions block the merge before bad outputs reach users.

**Status: Phase 5 of 6** — the whole pipeline is built and runs end to end offline. Golden cases
are being authored by hand; the gate activates the moment a dataset is locked. See [tasks/todo.md](tasks/todo.md) and [docs/SPEC.md](docs/SPEC.md).

## Quick start

```bash
make install          # uv venv @ 3.11 + dev tooling
make test             # offline tier: 253 tests, no network, no API keys
make lint             # ruff + black + isort + mypy --strict + bandit
make dataset-report   # golden dataset coverage and remaining gaps
make install-e2e      # Playwright + chromium, then: make test-e2e
make test-e2e-all     # the report in chromium + firefox + webkit
make demo-report      # regenerate docs/sample-report.html from a scripted regression
make eval TIER=smoke  # run the gate (needs a locked dataset)
make eval-demo        # run the degraded fixture prompt — should BLOCK
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
| [prompts/classifier/](prompts/classifier/) | The shipping prompt lineage |
| [prompts/demo/](prompts/demo/) | The deliberately degraded fixture, kept out of the lineage |
| [config/pricing.yaml](config/pricing.yaml) | Token prices as reviewable config, not hardcoded constants |

## Golden dataset

Ground truth is written by hand — see [data/golden/AUTHORING.md](data/golden/AUTHORING.md). Nothing
in the codebase generates cases; a model-generated golden set only measures whether the model agrees
with itself.

```bash
$EDITOR data/golden/cases.yaml   # write cases as YAML — real line breaks, no escaping
make dataset-build               # compile to emails.jsonl, validate, report gaps
make dataset-lock VERSION=v1     # freeze ground truth
```

`cases.yaml` is the authoring surface; `emails.jsonl` is generated and is what the lock
hashes. Rebuilding is idempotent — `added_at` is carried over by id, so a no-op rebuild
never churns the hash and never invalidates the lock.

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

Because the report is the primary human-facing artifact, it is tested in a real browser
([tests/e2e/](tests/e2e/)) as well as by string assertions. `assert "McNemar exact" in page` proves
the text exists; it proves nothing about whether the SVG trend line has non-zero height, whether the
BLOCK banner actually renders red, whether dark mode clears WCAG AA contrast, or whether the
side-by-side case columns stack on a phone. Those are the regressions a string check cannot see, so
31 Playwright tests cover them, across chromium, firefox and webkit — readers open CI artifacts in
Safari as often as Chrome, and the CSS variables, `prefers-color-scheme` and grid stacking are
exactly the sort of thing that renders differently per engine. Screenshots upload at three viewports
and both colour schemes; failures also upload a trace and a video. Playwright is an optional extra —
`pytest -m unit` runs unchanged without it.

Those tests earned their keep immediately by finding a live XSS-class bug: the template is named
`report.html.j2`, and `select_autoescape(["html"])` matches the *final* extension, so escaping was
silently off and case content was injected raw. Golden cases carry real customer email, and
`source: from_failure` cases come straight from production traffic, so that was a genuine path from
untrusted text to script execution in whoever opened the artifact. `autoescape` is now
unconditional — there is no reason to make escaping depend on a filename.

## CI

[.github/workflows/eval.yml](.github/workflows/eval.yml) runs four jobs.

| Job | When | Cost |
|---|---|---|
| `quality` | every push — lint, types, bandit, offline tests, 80% coverage floor | free |
| `dataset` | every push — validate the golden set, verify the lock hasn't drifted | free |
| `report-render` | every push — render the HTML report in headless chromium | free |
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

## Measured results

Every number below is from a real run against the locked golden dataset. Nothing here
is estimated or projected; where a result has not been reproduced in CI, it says so.

### Baseline — `v001` on dataset `v2`

| Metric | Value | Where measured |
|---|---:|---|
| Accuracy | 98.8% | local and CI, agreeing |
| pass^3 | 97.5% | local and CI |
| Summary quality (judge) | 4.91 / 5 | CI |
| Judge calibration κ | 0.96 | CI, 20/20 holdout scored |
| p95 latency | 1128 ms | CI |
| Cost per case | $0.0003 | CI, classifier only |

κ has been measured five times across local probes and CI runs and has landed
between 0.96 and 0.99 every time. The floor is 0.60; below it the run aborts
rather than reporting a quality number.

One case fails deliberately. `acct-013` — *"how do I reduce our seat count, and
will our bill drop accordingly?"* — is classified `billing` on every attempt,
because v001 lets any mention of a bill outrank the action the customer asked
for. Tie-break 3 says otherwise. It is a real blind spot and it stays red; a
golden set where everything passes measures nothing.

### The gate demo — degraded `v002`

Verified locally against a clean `v001` baseline:

```
BLOCK: 3 regressions and 1 improvement: accuracy 98.8% -> 96.2%
  BLOCK  1 critical case(s) regressed: gen-017
  warn   accuracy fell 3.8%; not significant (McNemar p=0.250)
```

Read the third line. **The accuracy drop is not statistically significant** —
McNemar puts it at p=0.250, so a gate keyed on accuracy alone waves this through
as noise. It blocks because a case tagged `critical` regressed, and critical
cases block regardless of significance. That is the argument for the critical
stratum, and it is the whole point of the demo.

`v002` is written to be a *realistic* bad commit: it keeps the task statement and
the category definitions and drops only the tie-break rules and the few-shot
anchor. Spot-checking easy emails shows nothing wrong, because easy emails do not
depend on what was removed.

**Reproduced in CI.** [PR #2](https://github.com/sreekarrddy10/model-regression-detector/pull/2)
carries this prompt and is blocked by the gate:

```
Cache restored from key: mrd-runs-main-33529454696
BLOCK: 5 regressions and 1 improvement: accuracy 98.8% -> 93.8%
  BLOCK  1 critical case(s) regressed: gen-017
  warn   accuracy fell 5.0%; not significant (McNemar p=0.219)
```

[PR #1](https://github.com/sreekarrddy10/model-regression-detector/pull/1) is the
passing counterpart: same dataset, same tier, merged green at 98.8%. The
[baseline run on main](https://github.com/sreekarrddy10/model-regression-detector/actions/runs/33529454696)
is what #2 compares against.

## Known gaps

Stated here rather than left to be discovered.

- **68 of 80 labels have not been read case-by-case.** They have been audited
  mechanically against the field spec, which is a weaker claim than reading them.
  What that audit covers, across all 80 cases:

  | check | result |
  |---|---|
  | Summary under 25 words, single sentence | all pass |
  | Summary cites a number or name absent from the email | none |
  | Category name leaked into the summary | none |
  | Near-duplicate emails (≥0.45 token overlap) | none |
  | `expected_summary` reused across cases | none |
  | Few-shot leakage against every prompt version | all ≤0.22 (threshold 0.45) |
  | Cross-case label consistency by definitional term | no violations |

  Difficulty labels are calibrated against live results — `easy` 100%, `medium`
  97%, `hard` 95% — so the difficulty axis predicts failure rate rather than
  being assigned by feel.

  What the audit cannot catch: a case where the label and the prompt are both
  wrong, or a summary that is accurate but emphasises the wrong thing. Both of
  those were found by reading — `tech-011`, `gen-014`, `tech-018`, `tech-020` and
  `acct-007` were all relabelled that way. So the defensible claim is **audited
  three ways — rule consistency, live baseline, few-shot leakage** — not
  "hand-verified every label".
- **`technical` has thinned to 18 cases** as seat and lockout cases moved to
  `account` during review. Every coverage floor still passes.
- **`latest_baseline` filters on `dataset_hash` but not `prompt_version`**, so
  `make eval-demo` records itself and becomes the next baseline. Run it once per
  baseline or the second run passes.

## Container

```bash
make docker-build
docker compose run --rm eval          # reads ./data, writes ./reports
```

The runtime image carries no build tooling and no dev dependencies, runs as uid 10001, and
healthchecks by asserting that its pricing config and prompt versions actually resolved — not
merely that Python starts.

That healthcheck earned its place immediately. The first build revealed that pricing, prompts and
the CLI defaults all resolved paths from `__file__`, which is correct in a source checkout and
silently wrong in an installed wheel. The original healthcheck reported `ok` through it, because it
called `lookup()` without checking the result. [`src/mrd/paths.py`](src/mrd/paths.py) now provides
one resolver with one precedence order — env override, then working directory, then source tree —
and the healthcheck asserts.
