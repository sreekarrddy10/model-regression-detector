# Model Regression Detection System

CI/CD for model behavior. Every prompt change is evaluated against a human-labeled golden dataset;
statistically significant quality regressions block the merge before bad outputs reach users.

**Status: Phase 1 of 6 complete** — feature under test, provider abstraction, offline test tier.
See [tasks/todo.md](tasks/todo.md) for the plan and [docs/SPEC.md](docs/SPEC.md) for the design.

## Quick start

```bash
make install    # uv venv @ 3.11 + dev tooling
make test       # offline tier: 47 tests, no network, no API keys
make lint       # ruff + black + isort + mypy --strict + bandit
```

No API key is needed to run the test suite. That is deliberate — see *Cassettes* below.

## What exists today

| Module | Responsibility |
|---|---|
| [src/mrd/prompts.py](src/mrd/prompts.py) | Versioned prompt artifacts loaded from `prompts/classifier/vNNN.yaml` |
| [src/mrd/feature/](src/mrd/feature/) | The system under test: support email → `{category, summary}` |
| [src/mrd/providers/](src/mrd/providers/) | One normalized contract over OpenAI and Anthropic |
| [config/pricing.yaml](config/pricing.yaml) | Token prices as reviewable config, not hardcoded constants |

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

Phase 2 builds the golden dataset: 80–100 hand-written, human-labeled cases with difficulty tags
and a content hash. Ground truth is written by hand, never generated — that constraint is the whole
premise of the project.
