"""Run and per-attempt result records.

`EvalRun` carries `dataset_hash` because it is part of the comparison key: two
runs are only comparable when they were scored against byte-identical ground
truth. `compare.py` refuses to diff runs whose hashes differ.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One attempt at one case."""

    run_id: str
    case_id: str
    repeat_idx: int
    raw_output: str
    category: str | None
    summary: str | None
    parse_error: str | None
    schema_valid: bool
    category_match: bool
    judge_score: int | None
    judge_rationale: str | None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.schema_valid and self.category_match


@dataclass(frozen=True, slots=True)
class EvalRun:
    """Metadata for one evaluation run."""

    run_id: str
    git_sha: str
    prompt_version: str
    dataset_version: str
    dataset_hash: str
    model: str
    judge_model: str | None
    repeats: int
    tier: str
    started_at: datetime
    finished_at: datetime | None = None
    judge_kappa: float | None = None


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """A run plus every attempt it produced."""

    run: EvalRun
    results: tuple[CaseResult, ...] = field(default=())

    def by_case(self) -> dict[str, tuple[CaseResult, ...]]:
        grouped: dict[str, list[CaseResult]] = {}
        for result in self.results:
            grouped.setdefault(result.case_id, []).append(result)
        return {
            case_id: tuple(sorted(items, key=lambda r: r.repeat_idx))
            for case_id, items in grouped.items()
        }

    def case_flags(self) -> dict[str, tuple[bool, ...]]:
        """Per case, the pass/fail flag for each repeat, in repeat order."""
        return {
            case_id: tuple(r.passed for r in results) for case_id, results in self.by_case().items()
        }

    def judge_scores(self) -> dict[str, tuple[int, ...]]:
        return {
            case_id: tuple(r.judge_score for r in results if r.judge_score is not None)
            for case_id, results in self.by_case().items()
        }

    @property
    def total_cost(self) -> float | None:
        costs = [r.cost_usd for r in self.results]
        return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)

    def latencies(self) -> Sequence[int]:
        return [r.latency_ms for r in self.results]
