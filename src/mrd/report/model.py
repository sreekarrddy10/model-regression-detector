"""Report assembly.

Pure data construction, separate from rendering, so the HTML report and the
Slack alert are two views of one verified structure rather than two independent
formatting paths that can disagree about what happened.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ..compare import Comparison, GateReport, Metrics, measure
from ..dataset.schema import GoldenCase
from ..graders.calibration import Calibration
from ..results import EvalRun, RunOutcome
from ..stats import is_flaky, majority_pass

DiffKind = Literal["regressed", "improved", "flaky"]


@dataclass(frozen=True, slots=True)
class Attempt:
    """One repeat, as the report displays it."""

    repeat_idx: int
    passed: bool
    category: str | None
    summary: str | None
    parse_error: str | None
    error: str | None
    judge_score: int | None
    judge_rationale: str | None


@dataclass(frozen=True, slots=True)
class CaseDiff:
    """A case worth showing, with both runs' output side by side."""

    case_id: str
    kind: DiffKind
    email: str
    expected_category: str
    expected_summary: str
    difficulty: str
    critical: bool
    notes: str
    baseline: tuple[Attempt, ...]
    candidate: tuple[Attempt, ...]

    @property
    def baseline_verdict(self) -> str:
        return "pass" if majority_pass([a.passed for a in self.baseline]) else "fail"

    @property
    def candidate_verdict(self) -> str:
        return "pass" if majority_pass([a.passed for a in self.candidate]) else "fail"

    @property
    def candidate_categories(self) -> str:
        seen = {a.category or "—" for a in self.candidate}
        return ", ".join(sorted(seen))


@dataclass(frozen=True, slots=True)
class ReportData:
    """Everything both the HTML report and the Slack alert render from."""

    run: EvalRun
    gate: GateReport
    metrics: Metrics
    comparison: Comparison | None
    diffs: tuple[CaseDiff, ...]
    trend: tuple[float, ...]
    calibration: Calibration | None
    generated_at: datetime
    report_url: str | None = None

    @property
    def has_baseline(self) -> bool:
        return self.comparison is not None

    @property
    def regressed(self) -> tuple[CaseDiff, ...]:
        return tuple(d for d in self.diffs if d.kind == "regressed")

    @property
    def improved(self) -> tuple[CaseDiff, ...]:
        return tuple(d for d in self.diffs if d.kind == "improved")

    @property
    def flaky(self) -> tuple[CaseDiff, ...]:
        return tuple(d for d in self.diffs if d.kind == "flaky")

    @property
    def headline(self) -> str:
        """One sentence a person can act on without opening the report."""
        c = self.comparison
        if c is None:
            return (
                f"First run against dataset {self.run.dataset_version}: "
                f"{self.metrics.accuracy:.1%} accuracy over {self.metrics.case_count} cases. "
                "No baseline to compare against yet."
            )
        if not c.regressed and not c.improved:
            return f"No cases changed. Accuracy holding at {c.candidate.accuracy:.1%}."
        parts = []
        if c.regressed:
            parts.append(f"{len(c.regressed)} regression{'s' if len(c.regressed) != 1 else ''}")
        if c.improved:
            parts.append(f"{len(c.improved)} improvement{'s' if len(c.improved) != 1 else ''}")
        return (
            f"{' and '.join(parts)}: accuracy "
            f"{c.baseline.accuracy:.1%} → {c.candidate.accuracy:.1%}."
        )


def _attempts(outcome: RunOutcome, case_id: str) -> tuple[Attempt, ...]:
    return tuple(
        Attempt(
            repeat_idx=r.repeat_idx,
            passed=r.passed,
            category=r.category,
            summary=r.summary,
            parse_error=r.parse_error,
            error=r.error,
            judge_score=r.judge_score,
            judge_rationale=r.judge_rationale,
        )
        for r in outcome.by_case().get(case_id, ())
    )


def build(
    candidate: RunOutcome,
    cases: Sequence[GoldenCase],
    *,
    gate: GateReport,
    generated_at: datetime,
    baseline: RunOutcome | None = None,
    comparison: Comparison | None = None,
    trend: Sequence[float] = (),
    calibration: Calibration | None = None,
    report_url: str | None = None,
) -> ReportData:
    by_id = {c.id: c for c in cases}
    cand_flags = candidate.case_flags()

    kinds: dict[str, DiffKind] = {}
    if comparison is not None:
        for case_id in comparison.regressed:
            kinds[case_id] = "regressed"
        for case_id in comparison.improved:
            kinds.setdefault(case_id, "improved")
    # Flaky cases are shown too: a case that stopped being deterministic is worth
    # a human's attention even though it still passes by majority.
    for case_id, flags in cand_flags.items():
        if is_flaky(flags):
            kinds.setdefault(case_id, "flaky")

    order: dict[DiffKind, int] = {"regressed": 0, "flaky": 1, "improved": 2}
    diffs = tuple(
        sorted(
            (
                CaseDiff(
                    case_id=case_id,
                    kind=kind,
                    email=by_id[case_id].input_email,
                    expected_category=by_id[case_id].expected_category,
                    expected_summary=by_id[case_id].expected_summary,
                    difficulty=by_id[case_id].difficulty,
                    critical=by_id[case_id].critical,
                    notes=by_id[case_id].notes,
                    baseline=_attempts(baseline, case_id) if baseline else (),
                    candidate=_attempts(candidate, case_id),
                )
                for case_id, kind in kinds.items()
                if case_id in by_id
            ),
            # Critical cases first within each group: they are why the gate fired.
            key=lambda d: (order[d.kind], not d.critical, d.case_id),
        )
    )

    return ReportData(
        run=candidate.run,
        gate=gate,
        metrics=measure(candidate),
        comparison=comparison,
        diffs=diffs,
        trend=tuple(trend),
        calibration=calibration,
        generated_at=generated_at,
        report_url=report_url,
    )
