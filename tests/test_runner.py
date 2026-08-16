from __future__ import annotations

import pytest

from mrd.compare import measure
from mrd.dataset.schema import HoldoutSample
from mrd.feature.classifier import classify
from mrd.graders import judge as judge_grader
from mrd.graders.calibration import Calibration, calibrate
from mrd.graders.code import grade
from mrd.prompts import PromptConfig
from mrd.providers.base import ProviderError, Response, Usage
from mrd.runner import RunConfig, run

from .conftest import run as drive
from .engine_fixtures import NOW, ScriptedProvider, make_case, make_dataset

pytestmark = pytest.mark.unit


@pytest.fixture
def prompt(prompt_v001: PromptConfig) -> PromptConfig:
    return prompt_v001


def config(**overrides: object) -> RunConfig:
    base = {"run_id": "run-1", "repeats": 3, "backoff_base": 0.0}
    base.update(overrides)
    return RunConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Code grader
# --------------------------------------------------------------------------- #


def test_code_grader_marks_a_correct_answer(prompt: PromptConfig) -> None:
    case = make_case(0)
    provider = ScriptedProvider((case,), {})
    outcome = drive(classify(case.input_email, prompt, provider))

    scores = grade(case, outcome)
    assert scores.schema_valid and scores.category_match and scores.passed


def test_code_grader_separates_schema_failure_from_wrong_answer(prompt: PromptConfig) -> None:
    """Both fail, but the report must be able to tell them apart."""
    case = make_case(0)

    wrong = grade(
        case,
        drive(classify(case.input_email, prompt, ScriptedProvider((case,), {case.id: "wrong"}))),
    )
    assert wrong.schema_valid and not wrong.category_match

    broken = grade(
        case,
        drive(
            classify(case.input_email, prompt, ScriptedProvider((case,), {case.id: "malformed"}))
        ),
    )
    assert not broken.schema_valid and not broken.category_match


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def test_runs_every_case_the_configured_number_of_times(prompt: PromptConfig) -> None:
    dataset = make_dataset(5)
    provider = ScriptedProvider(dataset.cases, {})

    outcome = drive(run(dataset, prompt, provider, config(), dataset_hash="h", now=NOW))

    assert len(outcome.results) == 15
    assert all(len(f) == 3 for f in outcome.case_flags().values())
    assert measure(outcome).accuracy == pytest.approx(1.0)


def test_records_metadata_needed_for_comparison(prompt: PromptConfig) -> None:
    dataset = make_dataset(2)
    outcome = drive(
        run(
            dataset,
            prompt,
            ScriptedProvider(dataset.cases, {}),
            config(git_sha="deadbeef", dataset_version="v1", tier="smoke"),
            dataset_hash="hash-a",
            now=NOW,
        )
    )

    assert outcome.run.dataset_hash == "hash-a"
    assert outcome.run.prompt_version == "v001"
    assert outcome.run.git_sha == "deadbeef"
    assert outcome.run.tier == "smoke"


def test_a_provider_outage_is_recorded_not_fatal(prompt: PromptConfig) -> None:
    """A run that dies on case 3 of 80 tells you nothing."""
    dataset = make_dataset(4)
    provider = ScriptedProvider(dataset.cases, {"tc_0002": "error"})

    outcome = drive(run(dataset, prompt, provider, config(), dataset_hash="h", now=NOW))

    failed = [r for r in outcome.results if r.error is not None]
    assert len(failed) == 3
    assert all(r.case_id == "tc_0002" for r in failed)
    assert not any(r.passed for r in failed)
    assert measure(outcome).accuracy == pytest.approx(0.75)
    assert measure(outcome).provider_errors == 3


def test_transient_failure_is_retried(prompt: PromptConfig) -> None:
    dataset = make_dataset(1)
    provider = ScriptedProvider(dataset.cases, {"tc_0000": ["error", "correct", "correct"]})

    outcome = drive(
        run(dataset, prompt, provider, config(repeats=1, max_attempts=2), dataset_hash="h", now=NOW)
    )

    assert outcome.results[0].error is None
    assert outcome.results[0].passed


def test_retries_are_bounded(prompt: PromptConfig) -> None:
    dataset = make_dataset(1)
    provider = ScriptedProvider(dataset.cases, {"tc_0000": "error"})

    outcome = drive(
        run(dataset, prompt, provider, config(repeats=1, max_attempts=2), dataset_hash="h", now=NOW)
    )
    assert outcome.results[0].error is not None


def test_flaky_case_is_visible_across_repeats(prompt: PromptConfig) -> None:
    dataset = make_dataset(2)
    provider = ScriptedProvider(dataset.cases, {"tc_0001": ["correct", "wrong", "correct"]})

    outcome = drive(run(dataset, prompt, provider, config(), dataset_hash="h", now=NOW))

    assert measure(outcome).flaky_cases == ("tc_0001",)
    assert measure(outcome).accuracy == pytest.approx(1.0), "majority still passes"


def test_subset_selection_powers_the_smoke_tier(prompt: PromptConfig) -> None:
    dataset = make_dataset(10)
    provider = ScriptedProvider(dataset.cases, {})
    subset = dataset.cases[:3]

    outcome = drive(
        run(dataset, prompt, provider, config(repeats=1), dataset_hash="h", now=NOW, cases=subset)
    )

    assert set(outcome.case_flags()) == {c.id for c in subset}


def test_concurrency_is_bounded(prompt: PromptConfig) -> None:
    dataset = make_dataset(6)
    provider = ScriptedProvider(dataset.cases, {})

    outcome = drive(
        run(dataset, prompt, provider, config(concurrency=2), dataset_hash="h", now=NOW)
    )
    assert len(outcome.results) == 18


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #


def test_judge_scores_are_attached_when_a_judge_is_configured(prompt: PromptConfig) -> None:
    dataset = make_dataset(2)
    provider = ScriptedProvider(dataset.cases, {})
    provider.judge_score = 4  # type: ignore[attr-defined]

    outcome = drive(
        run(
            dataset,
            prompt,
            provider,
            config(judge_model="gpt-4o"),
            dataset_hash="h",
            judge_provider=provider,
            now=NOW,
        )
    )

    assert all(r.judge_score == 4 for r in outcome.results)
    assert measure(outcome).judge_mean == pytest.approx(4.0)


def test_malformed_output_is_not_sent_to_the_judge(prompt: PromptConfig) -> None:
    """Judging a parse failure would score the parser, not the model."""
    dataset = make_dataset(2)
    provider = ScriptedProvider(dataset.cases, {"tc_0001": "malformed"})
    provider.judge_score = 5  # type: ignore[attr-defined]

    outcome = drive(
        run(
            dataset,
            prompt,
            provider,
            config(judge_model="gpt-4o"),
            dataset_hash="h",
            judge_provider=provider,
            now=NOW,
        )
    )

    scored = {r.case_id for r in outcome.results if r.judge_score is not None}
    assert scored == {"tc_0000"}


def test_judge_parses_a_verdict() -> None:
    verdict, error = judge_grader.parse('{"score": 4, "rationale": "close enough"}')
    assert error is None and verdict is not None and verdict.score == 4


@pytest.mark.parametrize("payload", ['{"score": 9, "rationale": "x"}', "not json", '{"score": 3}'])
def test_bad_judge_output_is_reported_not_raised(payload: str) -> None:
    verdict, error = judge_grader.parse(payload)
    assert verdict is None and error is not None


def test_seed_note_changes_the_request_fingerprint() -> None:
    """Confirmation runs must not just replay the first verdict."""
    plain = judge_grader.build_request("e", "r", "c", model="gpt-4o")
    seeded = judge_grader.build_request("e", "r", "c", model="gpt-4o", seed_note="Seed 2.")
    assert plain.fingerprint() != seeded.fingerprint()


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


class FixedJudgeProvider:
    """Returns a judge score derived from the candidate summary text."""

    name = "fixed-judge"

    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = mapping

    async def complete(self, request):  # type: ignore[no-untyped-def]
        import json as _json

        candidate = request.user.rsplit("Candidate summary:\n", 1)[-1].strip()
        score = self._mapping.get(candidate, 3)
        return Response(
            text=_json.dumps({"rationale": "fixed", "score": score}),
            model=request.model,
            provider=self.name,
            usage=Usage(10, 5),
            latency_ms=1,
            cost_usd=None,
            fingerprint=request.fingerprint(),
        )


def _holdout(pairs: list[tuple[str, int]]) -> tuple[HoldoutSample, ...]:
    return tuple(
        HoldoutSample(
            id=f"ho_{i:04d}",
            case_id="tc_0000",
            summary=summary,
            human_score=score,
            scorer="human",
            scored_at=NOW,
        )
        for i, (summary, score) in enumerate(pairs)
    )


def _calibrate(mapping: dict[str, int], pairs: list[tuple[str, int]]) -> Calibration:
    return drive(
        calibrate(
            _holdout(pairs),
            emails={"tc_0000": "email body"},
            references={"tc_0000": "reference summary"},
            provider=FixedJudgeProvider(mapping),
            model="gpt-4o",
        )
    )


def test_agreeing_judge_passes_calibration() -> None:
    pairs = [(f"s{i}", (i % 5) + 1) for i in range(20)]
    mapping = {s: score for s, score in pairs}
    result = _calibrate(mapping, pairs)

    assert result.kappa == pytest.approx(1.0)
    assert result.passed
    assert "agrees with human scores" in result.reason


def test_disagreeing_judge_fails_calibration() -> None:
    """The judge is confidently wrong; the run must not report its numbers."""
    pairs = [(f"s{i}", (i % 5) + 1) for i in range(20)]
    mapping = {s: 6 - score for s, score in pairs}
    result = _calibrate(mapping, pairs)

    assert not result.passed
    assert "not calibrated" in result.reason


def test_judge_that_answers_nothing_fails_calibration() -> None:
    class BrokenJudge:
        name = "broken"

        async def complete(self, request):  # type: ignore[no-untyped-def]
            return Response(
                text="no thanks",
                model=request.model,
                provider="broken",
                usage=Usage(1, 1),
                latency_ms=1,
                cost_usd=None,
                fingerprint=request.fingerprint(),
            )

    result = drive(
        calibrate(
            _holdout([("s", 3)]),
            emails={"tc_0000": "e"},
            references={"tc_0000": "r"},
            provider=BrokenJudge(),
            model="gpt-4o",
        )
    )
    assert not result.passed
    assert "no parseable verdicts" in result.reason


def test_judge_outage_degrades_quality_signal_without_failing_the_run(
    prompt: PromptConfig,
) -> None:
    """Deterministic graders succeeded; losing the advisory dimension must not lose the run."""

    class BrokenJudgeProvider:
        name = "broken-judge"

        async def complete(self, request):  # type: ignore[no-untyped-def]
            raise ProviderError("judge is down")

    dataset = make_dataset(2)
    outcome = drive(
        run(
            dataset,
            prompt,
            ScriptedProvider(dataset.cases, {}),
            config(judge_model="gpt-4o", max_attempts=1),
            dataset_hash="h",
            judge_provider=BrokenJudgeProvider(),
            now=NOW,
        )
    )

    assert all(r.error is None for r in outcome.results), "the run itself survived"
    assert all(r.passed for r in outcome.results)
    assert all(r.judge_score is None for r in outcome.results)
    assert measure(outcome).judge_mean is None
