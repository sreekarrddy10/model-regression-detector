"""Golden dataset contracts.

The golden dataset is human-authored ground truth. Nothing in this package
generates cases; it only validates, hashes and reports on what a human wrote.
Evaluation quality is bounded by data quality, so the constraints here are
deliberately strict - a case that cannot explain why it exists is rejected.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from ..feature.schema import Category

Difficulty = Literal["easy", "ambiguous", "adversarial"]
Source = Literal["handwritten", "from_failure"]

DIFFICULTIES: tuple[str, ...] = get_args(Difficulty)
SOURCES: tuple[str, ...] = get_args(Source)

CASE_ID_RE = re.compile(r"^tc_\d{4}$")
HOLDOUT_ID_RE = re.compile(r"^ho_\d{4}$")


class GoldenCase(BaseModel):
    """One human-labeled evaluation case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=CASE_ID_RE.pattern)
    input_email: str = Field(min_length=1)
    expected_category: Category
    expected_summary: str = Field(min_length=1, max_length=400)
    difficulty: Difficulty
    # A regression on a critical case blocks the merge outright, regardless of
    # aggregate accuracy. Reserve it for behavior that must never break.
    critical: bool = False
    source: Source = "handwritten"
    # Required, non-empty, on purpose: a case whose author cannot say why it
    # matters is a case nobody can interpret when it fails two months from now.
    notes: str = Field(min_length=1)
    added_at: datetime


class HoldoutSample(BaseModel):
    """One human-scored summary, used to calibrate the LLM-as-judge.

    The judge is an unvalidated model until it is measured against these. If
    judge-human agreement falls below the configured kappa floor, the eval run
    aborts rather than reporting a quality number nobody should believe.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=HOLDOUT_ID_RE.pattern)
    case_id: str = Field(pattern=CASE_ID_RE.pattern)
    summary: str = Field(min_length=1)
    human_score: int = Field(ge=1, le=5)
    scorer: str = Field(min_length=1)
    scored_at: datetime
    notes: str = ""
