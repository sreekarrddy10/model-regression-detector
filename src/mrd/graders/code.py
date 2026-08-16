"""Deterministic graders.

These are the only graders permitted to block a merge on their own. They are
reproducible, cheap, and explainable - when CI blocks, a human can see exactly
which case flipped and why, without asking a model to justify itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..dataset.schema import GoldenCase
from ..feature.classifier import Outcome


@dataclass(frozen=True, slots=True)
class CodeScores:
    """Deterministic signals for one attempt at one case."""

    schema_valid: bool
    category_match: bool
    latency_ms: int
    cost_usd: float | None

    @property
    def passed(self) -> bool:
        """The binary outcome the gate's flip detection operates on."""
        return self.schema_valid and self.category_match


def grade(case: GoldenCase, outcome: Outcome) -> CodeScores:
    schema_valid = outcome.classification is not None
    category_match = (
        schema_valid
        and outcome.classification is not None
        and outcome.classification.category == case.expected_category
    )
    return CodeScores(
        schema_valid=schema_valid,
        category_match=category_match,
        latency_ms=outcome.response.latency_ms,
        cost_usd=outcome.response.cost_usd,
    )
