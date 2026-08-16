"""Run comparison and the merge gate.

Two principles run through this module.

First, deterministic signals gate and probabilistic ones advise. Category match
and schema validity can block a merge on their own; judge scores can only block
when the same degradation reproduces across independent seeds, and only when the
judge has passed calibration.

Second, a difference must be both real and large enough to matter. A flip is only
a regression when it reproduces across repeats, and an aggregate drop must clear
an effect-size floor *and* an exact significance test. A p-value alone would
block on trivia at large n; a flat percentage alone cannot tell 2-of-80 from
noise at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .dataset.schema import GoldenCase
from .results import RunOutcome
from .stats import ewma, is_flaky, majority_pass, mcnemar_exact, pass_at_k, pass_hat_k


class Verdict(StrEnum):
    # A gate verdict, not a credential.
    PASS = "PASS"  # nosec B105
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Every number the gate depends on, in one reviewable place."""

    accuracy_warn: float = -0.03
    accuracy_block: float = -0.08
    significance_alpha: float = 0.05
    judge_warn: float = -0.3
    judge_block: float = -0.5
    judge_confirmations_required: int = 3
    latency_warn_ratio: float = 0.25
    cost_warn_ratio: float = 0.30
    ewma_floor: float = 0.90
    ewma_alpha: float = 0.3


class IncomparableRuns(ValueError):
    """The two runs cannot be meaningfully diffed."""


def percentile(values: Sequence[int | float], q: float) -> float:
    """Nearest-rank percentile. No numpy dependency for one formula."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(q * len(ordered) + 0.5))))
    return float(ordered[rank - 1])


@dataclass(frozen=True, slots=True)
class Metrics:
    """Aggregate view of a single run."""

    case_count: int
    accuracy: float
    schema_failures: int
    provider_errors: int
    pass_at_1: float
    pass_at_k_rate: float
    pass_hat_k_rate: float
    flaky_cases: tuple[str, ...]
    judge_mean: float | None
    p95_latency_ms: float
    cost_per_case: float | None


def measure(outcome: RunOutcome) -> Metrics:
    flags = outcome.case_flags()
    if not flags:
        return Metrics(0, 0.0, 0, 0, 0.0, 0.0, 0.0, (), None, 0.0, None)

    passed = [majority_pass(f) for f in flags.values()]
    first_attempt = [f[0] for f in flags.values() if f]
    judge_scores = [s for scores in outcome.judge_scores().values() for s in scores]
    total_cost = outcome.total_cost

    return Metrics(
        case_count=len(flags),
        accuracy=sum(passed) / len(passed),
        schema_failures=sum(1 for r in outcome.results if not r.schema_valid and r.error is None),
        provider_errors=sum(1 for r in outcome.results if r.error is not None),
        pass_at_1=sum(first_attempt) / len(first_attempt) if first_attempt else 0.0,
        pass_at_k_rate=sum(pass_at_k(f) for f in flags.values()) / len(flags),
        pass_hat_k_rate=sum(pass_hat_k(f) for f in flags.values()) / len(flags),
        flaky_cases=tuple(sorted(cid for cid, f in flags.items() if is_flaky(f))),
        judge_mean=sum(judge_scores) / len(judge_scores) if judge_scores else None,
        p95_latency_ms=percentile(outcome.latencies(), 0.95),
        cost_per_case=None if total_cost is None else total_cost / len(flags),
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    """Candidate against baseline."""

    baseline: Metrics
    candidate: Metrics
    regressed: tuple[str, ...]
    improved: tuple[str, ...]
    critical_regressed: tuple[str, ...]
    newly_flaky: tuple[str, ...]
    accuracy_delta: float
    judge_delta: float | None
    latency_ratio: float
    cost_ratio: float | None
    mcnemar_p: float

    @property
    def net_negative(self) -> bool:
        return len(self.regressed) > len(self.improved)


def compare(
    baseline: RunOutcome,
    candidate: RunOutcome,
    cases: Sequence[GoldenCase],
) -> Comparison:
    """Diff two runs.

    Refuses runs scored against different ground truth. Without this check a
    quiet edit to a label can present itself as a model improvement.
    """
    if baseline.run.dataset_hash != candidate.run.dataset_hash:
        raise IncomparableRuns(
            "runs were scored against different ground truth and cannot be diffed.\n"
            f"  baseline dataset {baseline.run.dataset_version} "
            f"({baseline.run.dataset_hash[:12]})\n"
            f"  candidate dataset {candidate.run.dataset_version} "
            f"({candidate.run.dataset_hash[:12]})"
        )

    base_flags = baseline.case_flags()
    cand_flags = candidate.case_flags()
    shared = sorted(set(base_flags) & set(cand_flags))
    critical = {c.id for c in cases if c.critical}

    regressed = tuple(
        cid
        for cid in shared
        if majority_pass(base_flags[cid]) and not majority_pass(cand_flags[cid])
    )
    improved = tuple(
        cid
        for cid in shared
        if not majority_pass(base_flags[cid]) and majority_pass(cand_flags[cid])
    )
    newly_flaky = tuple(
        cid for cid in shared if is_flaky(cand_flags[cid]) and not is_flaky(base_flags[cid])
    )

    base = measure(baseline)
    cand = measure(candidate)

    judge_delta = (
        None
        if base.judge_mean is None or cand.judge_mean is None
        else cand.judge_mean - base.judge_mean
    )
    latency_ratio = (
        0.0
        if base.p95_latency_ms == 0
        else (cand.p95_latency_ms - base.p95_latency_ms) / base.p95_latency_ms
    )
    cost_ratio = (
        None
        if not base.cost_per_case or cand.cost_per_case is None
        else (cand.cost_per_case - base.cost_per_case) / base.cost_per_case
    )

    return Comparison(
        baseline=base,
        candidate=cand,
        regressed=regressed,
        improved=improved,
        critical_regressed=tuple(cid for cid in regressed if cid in critical),
        newly_flaky=newly_flaky,
        accuracy_delta=cand.accuracy - base.accuracy,
        judge_delta=judge_delta,
        latency_ratio=latency_ratio,
        cost_ratio=cost_ratio,
        mcnemar_p=mcnemar_exact(len(regressed), len(improved)),
    )


@dataclass(frozen=True, slots=True)
class GateReport:
    verdict: Verdict
    blocking: tuple[str, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())

    @property
    def exit_code(self) -> int:
        return 1 if self.verdict is Verdict.BLOCK else 0


def evaluate(
    comparison: Comparison,
    *,
    thresholds: Thresholds | None = None,
    judge_calibrated: bool = True,
    judge_confirmations: int = 0,
    recent_accuracy: Sequence[float] = (),
) -> GateReport:
    """Apply the gate matrix.

    `judge_confirmations` is how many independent seeds reproduced a judge
    degradation. Below the required count a judge drop can only warn, never
    block - one model's opinion is not evidence.
    """
    t = thresholds or Thresholds()
    blocking: list[str] = []
    warnings: list[str] = []
    c = comparison

    # --- deterministic, blocking ------------------------------------------- #
    if c.critical_regressed:
        blocking.append(
            f"{len(c.critical_regressed)} critical case(s) regressed: "
            f"{', '.join(c.critical_regressed)}"
        )

    if c.candidate.schema_failures > 0:
        blocking.append(
            f"{c.candidate.schema_failures} attempt(s) produced output that does not "
            "match the response schema"
        )

    if c.accuracy_delta <= t.accuracy_block:
        blocking.append(
            f"accuracy fell {abs(c.accuracy_delta):.1%} "
            f"({c.baseline.accuracy:.1%} -> {c.candidate.accuracy:.1%}), "
            f"past the {abs(t.accuracy_block):.0%} block threshold"
        )
    elif c.mcnemar_p < t.significance_alpha and c.net_negative:
        blocking.append(
            f"{len(c.regressed)} regressed vs {len(c.improved)} improved is statistically "
            f"significant (McNemar exact p={c.mcnemar_p:.4f} < {t.significance_alpha})"
        )
    elif c.accuracy_delta <= t.accuracy_warn:
        warnings.append(
            f"accuracy fell {abs(c.accuracy_delta):.1%} "
            f"({c.baseline.accuracy:.1%} -> {c.candidate.accuracy:.1%}); "
            f"not significant (McNemar p={c.mcnemar_p:.3f})"
        )

    # --- judge, advisory unless calibrated and confirmed -------------------- #
    if not judge_calibrated:
        blocking.append(
            "judge failed calibration against human scores; its quality numbers "
            "cannot be interpreted"
        )
    elif c.judge_delta is not None:
        if c.judge_delta <= t.judge_block:
            if judge_confirmations >= t.judge_confirmations_required:
                blocking.append(
                    f"summary quality fell {abs(c.judge_delta):.2f} points, confirmed across "
                    f"{judge_confirmations} independent seeds"
                )
            else:
                warnings.append(
                    f"summary quality fell {abs(c.judge_delta):.2f} points but was confirmed "
                    f"by only {judge_confirmations}/{t.judge_confirmations_required} seeds; "
                    "advisory only"
                )
        elif c.judge_delta <= t.judge_warn:
            warnings.append(f"summary quality fell {abs(c.judge_delta):.2f} points")

    # --- cost and latency: regressions too, but never blocking -------------- #
    if c.latency_ratio > t.latency_warn_ratio:
        warnings.append(
            f"p95 latency rose {c.latency_ratio:.0%} "
            f"({c.baseline.p95_latency_ms:.0f}ms -> {c.candidate.p95_latency_ms:.0f}ms)"
        )
    if c.cost_ratio is not None and c.cost_ratio > t.cost_warn_ratio:
        warnings.append(f"cost per case rose {c.cost_ratio:.0%}")

    if c.candidate.provider_errors:
        warnings.append(f"{c.candidate.provider_errors} attempt(s) failed at the provider")

    if c.newly_flaky:
        warnings.append(
            f"{len(c.newly_flaky)} case(s) became non-deterministic across repeats: "
            f"{', '.join(c.newly_flaky[:5])}"
        )

    # --- slow drift, invisible to any single diff --------------------------- #
    if recent_accuracy:
        trend = ewma(list(recent_accuracy) + [c.candidate.accuracy], alpha=t.ewma_alpha)
        if trend < t.ewma_floor:
            warnings.append(
                f"7-run trend accuracy is {trend:.1%}, below the {t.ewma_floor:.0%} floor; "
                "gradual drift that no single run triggered"
            )

    if blocking:
        verdict = Verdict.BLOCK
    elif warnings:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.PASS

    return GateReport(verdict=verdict, blocking=tuple(blocking), warnings=tuple(warnings))
