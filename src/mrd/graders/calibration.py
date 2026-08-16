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
from collections.abc import Sequence
from dataclasses import dataclass

from ..dataset.schema import HoldoutSample
from ..providers.base import Provider
from ..stats import quadratic_weighted_kappa, spearman
from .judge import score_summary

DEFAULT_KAPPA_FLOOR = 0.60


@dataclass(frozen=True, slots=True)
class Calibration:
    kappa: float
    spearman: float
    sample_count: int
    scored_count: int
    floor: float

    @property
    def passed(self) -> bool:
        return self.scored_count > 0 and self.kappa >= self.floor

    @property
    def reason(self) -> str:
        if self.scored_count == 0:
            return "judge produced no parseable verdicts on the holdout"
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
) -> Calibration:
    """Score every holdout summary with the judge and measure agreement.

    `emails` and `references` map case_id to the source email and the human
    reference summary, so the judge sees exactly what it sees during a real run.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def score(sample: HoldoutSample) -> tuple[int, int] | None:
        async with semaphore:
            result = await score_summary(
                emails[sample.case_id],
                references[sample.case_id],
                sample.summary,
                provider,
                model=model,
            )
        if result.verdict is None:
            return None
        return sample.human_score, result.verdict.score

    pairs = [p for p in await asyncio.gather(*(score(s) for s in holdout)) if p is not None]

    if not pairs:
        return Calibration(
            kappa=0.0, spearman=0.0, sample_count=len(holdout), scored_count=0, floor=floor
        )

    human = [p[0] for p in pairs]
    machine = [p[1] for p in pairs]
    return Calibration(
        kappa=quadratic_weighted_kappa(human, machine),
        spearman=spearman(human, machine),
        sample_count=len(holdout),
        scored_count=len(pairs),
        floor=floor,
    )
