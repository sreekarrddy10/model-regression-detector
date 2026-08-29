"""Golden dataset contracts.

The golden dataset is human-authored ground truth. Nothing in this package
generates cases; it only validates, hashes and reports on what a human wrote.
Evaluation quality is bounded by data quality, so the constraints here are
deliberately strict - a case that cannot explain why it exists is rejected.

Difficulty and strata are separate axes on purpose. An earlier version of this
schema collapsed them into a single field of easy/ambiguous/adversarial, which
forced a false choice: how *demanding* a case is and *what kind* of demanding it
is are independent. An adversarial case can be easy to label once you see the
trick, and a plain case can be genuinely hard. Keeping the axes apart lets
coverage and sampling reason about each one.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..feature.schema import Category

Difficulty = Literal["easy", "medium", "hard"]
Stratum = Literal["ambiguous", "adversarial", "critical"]
Source = Literal["handwritten", "from_failure"]

DIFFICULTIES: tuple[str, ...] = get_args(Difficulty)
STRATA: tuple[str, ...] = get_args(Stratum)
SOURCES: tuple[str, ...] = get_args(Source)

# Accepts both `tc_0001` and category-prefixed `bill-001`. Ids must be stable
# forever - run history references them - but the format is the author's choice.
# One caveat if you prefix by category: re-labelling a case makes its id lie.
CASE_ID_RE = re.compile(r"^[a-z][a-z0-9]*[-_]\d{3,4}$")
HOLDOUT_ID_RE = re.compile(r"^[a-z][a-z0-9]*[-_]\d{3,4}$")


class GoldenCase(BaseModel):
    """One human-labeled evaluation case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=CASE_ID_RE.pattern)
    input_email: str = Field(min_length=1)
    expected_category: Category
    expected_summary: str = Field(min_length=1, max_length=400)
    difficulty: Difficulty
    # Overlapping tags, not a partition: a case can be ambiguous, adversarial
    # and critical at once. `critical` here means a regression blocks the merge
    # on its own, regardless of aggregate accuracy.
    strata: tuple[Stratum, ...] = ()
    source: Source = "handwritten"
    # Required, non-empty, on purpose: a case whose author cannot say why it
    # matters is a case nobody can interpret when it fails two months from now.
    notes: str = Field(min_length=1)
    added_at: datetime

    @property
    def critical(self) -> bool:
        return "critical" in self.strata

    @model_validator(mode="after")
    def _no_duplicate_strata(self) -> GoldenCase:
        if len(set(self.strata)) != len(self.strata):
            raise ValueError(f"duplicate strata tags: {list(self.strata)}")
        return self


class HoldoutSample(BaseModel):
    """One human-scored summary, used to calibrate the LLM-as-judge.

    The judge is an unvalidated model until it is measured against these. If
    judge-human agreement falls below the configured kappa floor, the eval run
    aborts rather than reporting a quality number nobody should believe.

    A sample either points at a golden case (`case_id`, from which the email and
    reference summary are taken) or carries its own `email` and
    `reference_summary`. Self-contained samples let the judge be calibrated on
    text that is deliberately not in the golden set, which keeps calibration
    independent of the thing being measured.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=HOLDOUT_ID_RE.pattern)
    candidate_summary: str = Field(min_length=1)
    human_score: int = Field(ge=1, le=5)
    scorer: str = Field(min_length=1)
    scored_at: datetime

    case_id: str | None = Field(default=None, pattern=CASE_ID_RE.pattern)
    email: str | None = None
    reference_summary: str | None = None

    # Author-facing analysis fields; never shown to the judge.
    failure_mode: str = ""
    rationale: str = ""

    @model_validator(mode="after")
    def _resolvable(self) -> HoldoutSample:
        if self.case_id is None and not (self.email and self.reference_summary):
            raise ValueError(
                "needs either case_id, or both email and reference_summary - the "
                "judge cannot score a candidate without the source text and "
                "something to compare it against"
            )
        return self
