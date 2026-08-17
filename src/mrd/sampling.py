"""Tier case selection.

The smoke tier evaluates a subset, and that subset must be **deterministic**.
A randomly sampled smoke run is not comparable to the one before it: cases would
enter and leave the sample between commits, and the diff would report sampling
churn as model change. Selection is therefore a pure function of the dataset.

Stratification matters for the same reason accuracy alone is a weak signal - a
sample that happens to contain only easy billing cases cannot detect a
regression in adversarial account handling.
"""

from __future__ import annotations

from collections.abc import Sequence

from .dataset.loader import Dataset
from .dataset.schema import GoldenCase

DEFAULT_SMOKE_SIZE = 20


def stratified(dataset: Dataset, size: int = DEFAULT_SMOKE_SIZE) -> tuple[GoldenCase, ...]:
    """Pick `size` cases spread across category and difficulty, deterministically.

    Every case marked `critical` is included first regardless of size: the smoke
    tier exists to catch the failures that matter most, and dropping a critical
    case to hit a round number defeats the point. The remainder is filled by
    round-robin over (category, difficulty) strata, each ordered by case id.
    """
    if size <= 0:
        return ()

    selected: list[GoldenCase] = sorted(dataset.critical, key=lambda c: c.id)
    chosen = {c.id for c in selected}

    strata: dict[tuple[str, str], list[GoldenCase]] = {}
    for case in sorted(dataset.cases, key=lambda c: c.id):
        if case.id in chosen:
            continue
        strata.setdefault((case.expected_category, case.difficulty), []).append(case)

    # Sorted keys keep the round-robin order stable across runs and machines.
    queues = [strata[key] for key in sorted(strata)]
    while len(selected) < size and any(queues):
        for queue in queues:
            if not queue:
                continue
            selected.append(queue.pop(0))
            if len(selected) >= size:
                break

    return tuple(sorted(selected, key=lambda c: c.id))


def select(
    dataset: Dataset, tier: str, *, smoke_size: int = DEFAULT_SMOKE_SIZE
) -> Sequence[GoldenCase]:
    """Cases for a tier. `full` and `unit` use everything; `smoke` samples."""
    if tier == "smoke":
        return stratified(dataset, smoke_size)
    return dataset.cases
