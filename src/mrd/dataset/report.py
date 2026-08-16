"""Coverage report for the golden dataset.

Distinguishes errors (raised by the loader - the dataset is unusable) from
warnings (raised here - the dataset is usable but under-built). Authoring
80-100 cases by hand takes days, so the report is a progress instrument as much
as a quality gate.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..feature.schema import CATEGORIES
from .loader import Dataset
from .schema import DIFFICULTIES, HoldoutSample

TARGET_CASES = 80
MIN_PER_CATEGORY = 12
MIN_ADVERSARIAL = 8
MIN_AMBIGUOUS = 12
MIN_CRITICAL = 10
TARGET_HOLDOUT = 20
MIN_HOLDOUT_DISTINCT_SCORES = 3


@dataclass(frozen=True, slots=True)
class Report:
    count: int
    by_category: dict[str, int]
    by_difficulty: dict[str, int]
    by_source: dict[str, int]
    critical_count: int
    holdout_count: int
    holdout_scores: tuple[int, ...]
    warnings: tuple[str, ...] = field(default=())

    @property
    def ready(self) -> bool:
        return not self.warnings


def _dataset_warnings(dataset: Dataset) -> list[str]:
    warnings: list[str] = []

    if len(dataset) < TARGET_CASES:
        warnings.append(f"{len(dataset)}/{TARGET_CASES} cases written")

    for category in CATEGORIES:
        found = len(dataset.by_category(category))
        if found < MIN_PER_CATEGORY:
            warnings.append(f"category {category!r}: {found}/{MIN_PER_CATEGORY}")

    ambiguous = len(dataset.by_difficulty("ambiguous"))
    if ambiguous < MIN_AMBIGUOUS:
        warnings.append(
            f"only {ambiguous}/{MIN_AMBIGUOUS} ambiguous cases; "
            "an all-easy set cannot detect a real regression"
        )

    adversarial = len(dataset.by_difficulty("adversarial"))
    if adversarial < MIN_ADVERSARIAL:
        warnings.append(f"only {adversarial}/{MIN_ADVERSARIAL} adversarial cases")

    critical = len(dataset.critical)
    if critical < MIN_CRITICAL:
        warnings.append(f"only {critical}/{MIN_CRITICAL} cases marked critical")

    return warnings


def _holdout_warnings(samples: Sequence[HoldoutSample]) -> list[str]:
    warnings: list[str] = []

    if len(samples) < TARGET_HOLDOUT:
        warnings.append(f"judge holdout: {len(samples)}/{TARGET_HOLDOUT} scored summaries")

    if not samples:
        return warnings

    scores = [s.human_score for s in samples]
    distinct = len(set(scores))
    if distinct < MIN_HOLDOUT_DISTINCT_SCORES:
        # Agreement statistics need disagreement to measure. A holdout scored 5/5
        # across the board yields an undefined or meaningless kappa, so the judge
        # would pass calibration without ever being tested.
        warnings.append(
            f"judge holdout spans only {distinct} distinct score(s); "
            f"need at least {MIN_HOLDOUT_DISTINCT_SCORES} - include deliberately "
            "weak summaries, not just good ones, or kappa is meaningless"
        )
    elif len(scores) > 1 and statistics.pstdev(scores) < 0.5:
        warnings.append(
            f"judge holdout score spread is narrow (sd={statistics.pstdev(scores):.2f}); "
            "calibration will be weak"
        )

    return warnings


def build(dataset: Dataset, holdout: Sequence[HoldoutSample] = ()) -> Report:
    return Report(
        count=len(dataset),
        by_category={c: len(dataset.by_category(c)) for c in CATEGORIES},
        by_difficulty={d: len(dataset.by_difficulty(d)) for d in DIFFICULTIES},
        by_source={
            s: sum(1 for c in dataset if c.source == s) for s in ("handwritten", "from_failure")
        },
        critical_count=len(dataset.critical),
        holdout_count=len(holdout),
        holdout_scores=tuple(s.human_score for s in holdout),
        warnings=tuple(_dataset_warnings(dataset) + _holdout_warnings(holdout)),
    )


def render(report: Report) -> str:
    """Plain-text rendering for the CLI."""
    lines = [
        "Golden dataset",
        "=" * 46,
        f"  cases          {report.count}",
        f"  critical       {report.critical_count}",
        f"  judge holdout  {report.holdout_count}",
        "",
        "  by category",
    ]
    lines += [f"    {name:<12} {count:>3}" for name, count in report.by_category.items()]
    lines += ["", "  by difficulty"]
    lines += [f"    {name:<12} {count:>3}" for name, count in report.by_difficulty.items()]
    lines += ["", "  by source"]
    lines += [f"    {name:<12} {count:>3}" for name, count in report.by_source.items()]

    lines.append("")
    if report.warnings:
        lines.append(f"NOT READY - {len(report.warnings)} gap(s):")
        lines += [f"  - {w}" for w in report.warnings]
    else:
        lines.append("READY - all coverage targets met.")

    return "\n".join(lines) + "\n"
