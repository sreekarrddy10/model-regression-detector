"""Synthetic fixtures for engine tests.

A scripted provider stands in for a model so the engine can be tested against
outcomes chosen in advance: this case regresses, that one flakes, this one times
out. Nothing here is a golden case - the real dataset stays hand-authored.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import count

from mrd.dataset.loader import Dataset
from mrd.dataset.schema import GoldenCase
from mrd.providers.base import Provider, ProviderError, Request, Response, Usage
from mrd.results import CaseResult, EvalRun, RunOutcome

NOW = datetime(2026, 8, 16, tzinfo=UTC)
CATEGORIES = ("billing", "technical", "account", "general")


def make_case(
    index: int,
    *,
    critical: bool = False,
    difficulty: str = "easy",
    strata: tuple[str, ...] = (),
) -> GoldenCase:
    tags = list(strata)
    if critical and "critical" not in tags:
        tags.append("critical")
    return GoldenCase(
        id=f"tc_{index:04d}",
        input_email=f"Synthetic email body {index}.",
        expected_category=CATEGORIES[index % 4],
        expected_summary=f"Reference summary {index}.",
        difficulty=difficulty,  # type: ignore[arg-type]
        strata=tuple(tags),  # type: ignore[arg-type]
        source="handwritten",
        notes=f"Engine fixture {index}.",
        added_at=NOW,
    )


def make_dataset(count: int = 8, *, critical_ids: frozenset[str] = frozenset()) -> Dataset:
    cases = tuple(make_case(i, critical=f"tc_{i:04d}" in critical_ids) for i in range(count))
    return Dataset(path=None, cases=cases)  # type: ignore[arg-type]


class ScriptedProvider(Provider):
    """Returns a pre-decided answer per (case email, repeat).

    `behaviours` maps a case id to one of:
      "correct"    - matches the expected category
      "wrong"      - a different, valid category
      "malformed"  - prose instead of JSON
      "error"      - raises ProviderError on every attempt
      list[str]    - one behaviour per repeat, for flakiness
    """

    name = "scripted"

    def __init__(self, cases: tuple[GoldenCase, ...], behaviours: dict[str, object]) -> None:
        self._by_email = {c.input_email: c for c in cases}
        self._behaviours = behaviours
        self._calls: dict[str, int] = {}
        self.latency_ms = 100
        self.cost_usd: float | None = 0.001

    def _behaviour_for(self, case_id: str) -> str:
        spec = self._behaviours.get(case_id, "correct")
        seen = self._calls.get(case_id, 0)
        self._calls[case_id] = seen + 1
        if isinstance(spec, list):
            return str(spec[seen % len(spec)])
        return str(spec)

    async def complete(self, request: Request) -> Response:
        # Judge requests carry a different system prompt; answer them separately.
        if "You grade one-sentence summaries" in request.system:
            return self._judge_response(request)

        case = self._by_email[request.user]
        behaviour = self._behaviour_for(case.id)

        if behaviour == "error":
            raise ProviderError(f"scripted outage for {case.id}")
        if behaviour == "malformed":
            text = "I am not going to answer in JSON today."
        else:
            category = (
                case.expected_category
                if behaviour == "correct"
                else next(c for c in CATEGORIES if c != case.expected_category)
            )
            text = json.dumps(
                {"category": category, "summary": f"Produced summary for {case.id}."},
                sort_keys=True,
            )
        return self._respond(request, text)

    def _judge_response(self, request: Request) -> Response:
        score = getattr(self, "judge_score", 4)
        text = json.dumps({"rationale": "Scripted verdict.", "score": score}, sort_keys=True)
        return self._respond(request, text)

    def _respond(self, request: Request, text: str) -> Response:
        return Response(
            text=text,
            model=request.model,
            provider=self.name,
            usage=Usage(input_tokens=100, output_tokens=20),
            latency_ms=self.latency_ms,
            cost_usd=self.cost_usd,
            fingerprint=request.fingerprint(),
        )


def make_result(
    case_id: str,
    repeat_idx: int,
    *,
    run_id: str = "run-1",
    passed: bool = True,
    schema_valid: bool = True,
    judge_score: int | None = None,
    latency_ms: int = 100,
    cost_usd: float | None = 0.001,
    error: str | None = None,
) -> CaseResult:
    return CaseResult(
        run_id=run_id,
        case_id=case_id,
        repeat_idx=repeat_idx,
        raw_output="{}",
        category="billing" if passed else "general",
        summary="s",
        parse_error=None if schema_valid else "invalid JSON",
        schema_valid=schema_valid,
        category_match=passed and schema_valid,
        judge_score=judge_score,
        judge_rationale=None if judge_score is None else "because",
        latency_ms=latency_ms,
        input_tokens=100,
        output_tokens=20,
        cost_usd=cost_usd,
        error=error,
    )


def make_outcome(
    flags_by_case: dict[str, tuple[bool, ...]],
    *,
    run_id: str = "run-1",
    dataset_hash: str = "hash-a",
    judge_score: int | Sequence[int] | None = None,
    latency_ms: int = 100,
    cost_usd: float | None = 0.001,
    schema_valid: bool = True,
) -> RunOutcome:
    """`judge_score` may be a sequence, cycled across attempts, to produce a
    fractional mean - the warn band between -0.3 and -0.5 is unreachable with a
    single constant score."""
    scores = (
        None
        if judge_score is None
        else ([judge_score] if isinstance(judge_score, int) else list(judge_score))
    )
    counter = count()
    results = tuple(
        make_result(
            case_id,
            idx,
            run_id=run_id,
            passed=flag,
            schema_valid=schema_valid or flag,
            judge_score=None if scores is None else scores[next(counter) % len(scores)],
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
        for case_id, flags in flags_by_case.items()
        for idx, flag in enumerate(flags)
    )
    return RunOutcome(
        run=EvalRun(
            run_id=run_id,
            git_sha="abc123",
            prompt_version="v001",
            dataset_version="v1",
            dataset_hash=dataset_hash,
            model="gpt-4o-mini",
            judge_model="gpt-4o" if judge_score is not None else None,
            repeats=max((len(f) for f in flags_by_case.values()), default=0),
            tier="full",
            started_at=NOW,
            finished_at=NOW,
        ),
        results=results,
    )
