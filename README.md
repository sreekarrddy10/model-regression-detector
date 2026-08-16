# Model Regression Detection System

CI/CD for model behavior. Every prompt change is evaluated against a human-labeled golden dataset;
statistically significant quality regressions block the merge before bad outputs reach users.

**Status: Phase 2 of 6** — feature, providers and offline tier done; golden-dataset tooling done,
cases being authored by hand. See [tasks/todo.md](tasks/todo.md) and [docs/SPEC.md](docs/SPEC.md).

## Quick start

```bash
make install          # uv venv @ 3.11 + dev tooling
make test             # offline tier: 94 tests, no network, no API keys
make lint             # ruff + black + isort + mypy --strict + bandit
make dataset-report   # golden dataset coverage and remaining gaps
```

No API key is needed to run the test suite. That is deliberate — see *Cassettes* below.

## What exists today

| Module | Responsibility |
|---|---|
| [src/mrd/prompts.py](src/mrd/prompts.py) | Versioned prompt artifacts loaded from `prompts/classifier/vNNN.yaml` |
| [src/mrd/feature/](src/mrd/feature/) | The system under test: support email → `{category, summary}` |
| [src/mrd/providers/](src/mrd/providers/) | One normalized contract over OpenAI and Anthropic |
| [src/mrd/dataset/](src/mrd/dataset/) | Golden dataset validation, content hashing, coverage reporting |
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
the flip detection in Phase 3 meaningless, so non-zero is refused rather than warned about.

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
in Phase 3 alongside the runner.

## Next

Phase 3 builds the evaluation engine: async batched runner at `temperature=0` with `N=3` repeats,
deterministic and model graders, judge calibration against the holdout, and the comparison layer
(flip detection, McNemar's exact test, EWMA drift). It needs a locked dataset to run against.
