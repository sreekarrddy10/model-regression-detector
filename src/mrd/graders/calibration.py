"""Judge calibration.

An LLM-as-judge is an unvalidated model until it is measured against a human.
This module scores the holdout with the judge, compares the results to the human
scores, and reports whether the judge may be trusted at all.

If calibration fails, the eval run aborts rather than reporting a quality number
nobody should believe. This is the single design decision most eval suites skip,
and skipping it is how a suite becomes theater: it reports a confident 4.2 that
correlates with nothing a person would recognise as quality.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ..dataset.schema import HoldoutSample
from ..providers.base import Provider, ProviderError
from ..retry import BACKOFF_BASE_SECONDS, MAX_ATTEMPTS, with_retry
from ..stats import quadratic_weighted_kappa, spearman
from .judge import score_summary

logger = logging.getLogger(__name__)

DEFAULT_KAPPA_FLOOR = 0.60

# A provider that rate-limits half the holdout would otherwise yield a confident
# kappa computed on whatever survived. Retrying is not enough on its own: it
# turns a loud abort into a quiet one. Calibration must also refuse to speak for
# a holdout it mostly failed to score.
DEFAULT_MIN_SCORED_RATIO = 0.80


@dataclass(frozen=True, slots=True)
class Calibration:
    kappa: float
    spearman: float
    sample_count: int
    scored_count: int
    floor: float
    min_scored_ratio: float = DEFAULT_MIN_SCORED_RATIO

    @property
    def enough_scored(self) -> bool:
        """Did enough of the holdout survive for the number to mean anything?"""
        if self.sample_count == 0:
            return False
        return self.scored_count / self.sample_count >= self.min_scored_ratio

    @property
    def passed(self) -> bool:
        return self.scored_count > 0 and self.enough_scored and self.kappa >= self.floor

    @property
    def reason(self) -> str:
        if self.scored_count == 0:
            return "judge produced no parseable verdicts on the holdout"
        if not self.enough_scored:
            return (
                f"only {self.scored_count} of {self.sample_count} holdout summaries could be "
                f"scored ({self.min_scored_ratio:.0%} required). A kappa computed on the "
                "remainder would describe whichever samples the provider happened to serve, "
                "so the run is aborted rather than reported."
            )
        if self.passed:
            return (
                f"judge agrees with human scores "
                f"(kappa={self.kappa:.2f} >= {self.floor:.2f}, rho={self.spearman:.2f}, "
                f"n={self.scored_count})"
            )
        return (
            f"judge is not calibrated: kappa={self.kappa:.2f} < {self.floor:.2f} "
            f"on {self.scored_count} human-scored summaries. Its quality scores "
            f"cannot be interpreted, so the run is aborted rather than reported."
        )


class CalibrationFailed(RuntimeError):
    def __init__(self, calibration: Calibration) -> None:
        super().__init__(calibration.reason)
        self.calibration = calibration


async def calibrate(
    holdout: Sequence[HoldoutSample],
    emails: dict[str, str],
    references: dict[str, str],
    provider: Provider,
    *,
    model: str,
    floor: float = DEFAULT_KAPPA_FLOOR,
    concurrency: int = 4,
    attempts: int = MAX_ATTEMPTS,
    backoff_base: float = BACKOFF_BASE_SECONDS,
    min_scored_ratio: float = DEFAULT_MIN_SCORED_RATIO,
) -> Calibration:
    """Score every holdout summary with the judge and measure agreement.

    `emails` and `references` map case_id to the source email and the human
    reference summary, so the judge sees exactly what it sees during a real run.
    """
    semaphore = asyncio.Semaphore(concurrency)

    def resolve(sample: HoldoutSample) -> tuple[str, str] | None:
        """Source text and reference, from the sample itself or its linked case."""
        if sample.email and sample.reference_summary:
            return sample.email, sample.reference_summary
        if sample.case_id and sample.case_id in emails:
            return emails[sample.case_id], references[sample.case_id]
        return None

    async def score(sample: HoldoutSample) -> tuple[int, int] | None:
        resolved = resolve(sample)
        if resolved is None:
            logger.warning("holdout %s cannot be resolved to an email; skipping", sample.id)
            return None
        email, reference = resolved
        async with semaphore:
            try:
                result = await with_retry(
                    lambda: score_summary(
                        email,
                        reference,
                        sample.candidate_summary,
                        provider,
                        model=model,
                    ),
                    attempts=attempts,
                    backoff_base=backoff_base,
                    label=f"judge calibration {sample.id}",
                )
            except ProviderError as exc:
                # Survivable: one unscored sample lowers scored_count, and the
                # coverage guard above decides whether what is left still counts.
                # Aborting here would let a rate limit masquerade as a bad judge.
                logger.warning("holdout %s could not be scored: %s", sample.id, exc)
                return None
        if result.verdict is None:
            return None
        return sample.human_score, result.verdict.score

    pairs = [p for p in await asyncio.gather(*(score(s) for s in holdout)) if p is not None]

    if not pairs:
        return Calibration(
            kappa=0.0,
            spearman=0.0,
            sample_count=len(holdout),
            scored_count=0,
            floor=floor,
            min_scored_ratio=min_scored_ratio,
        )

    human = [p[0] for p in pairs]
    machine = [p[1] for p in pairs]
    return Calibration(
        kappa=quadratic_weighted_kappa(human, machine),
        spearman=spearman(human, machine),
        sample_count=len(holdout),
        scored_count=len(pairs),
        floor=floor,
        min_scored_ratio=min_scored_ratio,
    )
