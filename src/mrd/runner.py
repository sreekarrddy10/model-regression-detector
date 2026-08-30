"""The eval runner.

Runs every case N times at temperature 0 with bounded concurrency, grades each
attempt, and returns a complete run record. A provider failure on one attempt is
recorded on that attempt and does not abort the run - a run that dies on case 7
of 80 tells you nothing, while a run with 3 recorded errors tells you exactly
where the provider struggled.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from . import retry
from .dataset.loader import Dataset
from .dataset.schema import GoldenCase
from .feature.classifier import classify
from .graders.code import grade
from .graders.judge import score_summary
from .prompts import PromptConfig
from .providers.base import Provider, ProviderError
from .results import CaseResult, EvalRun, RunOutcome
from .retry import with_retry

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 8
DEFAULT_REPEATS = 3
MAX_ATTEMPTS = retry.MAX_ATTEMPTS
BACKOFF_BASE_SECONDS = retry.BACKOFF_BASE_SECONDS


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_id: str
    git_sha: str = "unknown"
    dataset_version: str = "unlocked"
    tier: str = "full"
    repeats: int = DEFAULT_REPEATS
    concurrency: int = DEFAULT_CONCURRENCY
    judge_model: str | None = None
    max_attempts: int = MAX_ATTEMPTS
    backoff_base: float = BACKOFF_BASE_SECONDS


async def _run_one(
    case: GoldenCase,
    repeat_idx: int,
    prompt: PromptConfig,
    provider: Provider,
    judge_provider: Provider | None,
    config: RunConfig,
) -> CaseResult:
    label = f"{case.id} r{repeat_idx}"

    try:
        outcome = await with_retry(
            lambda: classify(case.input_email, prompt, provider),
            attempts=config.max_attempts,
            backoff_base=config.backoff_base,
            label=label,
        )
    except ProviderError as exc:
        return CaseResult(
            run_id=config.run_id,
            case_id=case.id,
            repeat_idx=repeat_idx,
            raw_output="",
            category=None,
            summary=None,
            parse_error=None,
            schema_valid=False,
            category_match=False,
            judge_score=None,
            judge_rationale=None,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=None,
            error=str(exc),
        )

    scores = grade(case, outcome)

    judge_score: int | None = None
    judge_rationale: str | None = None
    # The judge only sees summaries the feature actually produced. Judging a
    # malformed output would score the parser, not the model.
    if judge_provider is not None and config.judge_model and outcome.classification is not None:
        summary = outcome.classification.summary
        try:
            verdict = await with_retry(
                lambda: score_summary(
                    case.input_email,
                    case.expected_summary,
                    summary,
                    judge_provider,
                    model=config.judge_model or "",
                ),
                attempts=config.max_attempts,
                backoff_base=config.backoff_base,
                label=f"judge {label}",
            )
            if verdict.verdict is not None:
                judge_score = verdict.verdict.score
                judge_rationale = verdict.verdict.rationale
        except ProviderError as exc:
            # A judge outage degrades the advisory dimension; it must not fail a
            # run whose deterministic graders succeeded.
            logger.warning("judge unavailable for %s: %s", label, exc)

    return CaseResult(
        run_id=config.run_id,
        case_id=case.id,
        repeat_idx=repeat_idx,
        raw_output=outcome.response.text,
        category=outcome.classification.category if outcome.classification else None,
        summary=outcome.classification.summary if outcome.classification else None,
        parse_error=outcome.parse_error,
        schema_valid=scores.schema_valid,
        category_match=scores.category_match,
        judge_score=judge_score,
        judge_rationale=judge_rationale,
        latency_ms=outcome.response.latency_ms,
        input_tokens=outcome.response.usage.input_tokens,
        output_tokens=outcome.response.usage.output_tokens,
        cost_usd=outcome.response.cost_usd,
    )


async def run(
    dataset: Dataset,
    prompt: PromptConfig,
    provider: Provider,
    config: RunConfig,
    *,
    dataset_hash: str,
    judge_provider: Provider | None = None,
    judge_kappa: float | None = None,
    now: datetime | None = None,
    cases: Sequence[GoldenCase] | None = None,
) -> RunOutcome:
    """Execute an eval run.

    `cases` narrows the run to a subset - this is how the smoke tier evaluates 20
    stratified cases while the full tier evaluates all of them.
    """
    started = now or datetime.now(UTC)
    selected = tuple(cases) if cases is not None else dataset.cases
    semaphore = asyncio.Semaphore(config.concurrency)

    async def guarded(case: GoldenCase, repeat_idx: int) -> CaseResult:
        async with semaphore:
            return await _run_one(case, repeat_idx, prompt, provider, judge_provider, config)

    tasks = [guarded(case, repeat_idx) for case in selected for repeat_idx in range(config.repeats)]
    results = await asyncio.gather(*tasks)

    return RunOutcome(
        run=EvalRun(
            run_id=config.run_id,
            git_sha=config.git_sha,
            prompt_version=prompt.version_id,
            dataset_version=config.dataset_version,
            dataset_hash=dataset_hash,
            model=prompt.model,
            judge_model=config.judge_model,
            repeats=config.repeats,
            tier=config.tier,
            started_at=started,
            finished_at=datetime.now(UTC) if now is None else started,
            judge_kappa=judge_kappa,
        ),
        results=tuple(sorted(results, key=lambda r: (r.case_id, r.repeat_idx))),
    )
