"""JSONL loading and validation.

Errors are aggregated with line numbers rather than raised on the first bad row.
Someone hand-authoring a hundred cases wants every problem at once, not a
hundred sequential runs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ..prompts import PromptConfig
from .schema import GoldenCase, HoldoutSample


@dataclass(frozen=True, slots=True)
class DatasetError:
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


class DatasetValidationError(ValueError):
    def __init__(self, path: Path, errors: Sequence[DatasetError]) -> None:
        listing = "\n  ".join(str(e) for e in errors)
        super().__init__(f"{len(errors)} problem(s) in {path}:\n  {listing}")
        self.path = path
        self.errors = tuple(errors)


@dataclass(frozen=True, slots=True)
class Dataset:
    path: Path
    cases: tuple[GoldenCase, ...]

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[GoldenCase]:
        return iter(self.cases)

    @property
    def critical(self) -> tuple[GoldenCase, ...]:
        return tuple(c for c in self.cases if c.critical)

    def by_category(self, category: str) -> tuple[GoldenCase, ...]:
        return tuple(c for c in self.cases if c.expected_category == category)

    def by_difficulty(self, difficulty: str) -> tuple[GoldenCase, ...]:
        return tuple(c for c in self.cases if c.difficulty == difficulty)


def _iter_rows(path: Path) -> Iterator[tuple[int, str]]:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if raw.strip():
            yield lineno, raw


def _normalize(text: str) -> str:
    """Whitespace- and case-insensitive form, for leakage comparison."""
    return " ".join(text.split()).casefold()


def few_shot_emails(prompts_root: Path) -> dict[str, str]:
    """Every few-shot email across every prompt version, mapped to its version."""
    found: dict[str, str] = {}
    for path in sorted(prompts_root.rglob("v*.yaml")):
        config = PromptConfig.from_file(path)
        for example in config.few_shot:
            found.setdefault(_normalize(example.email), config.version_id)
    return found


def load_cases(path: Path, *, prompts_root: Path | None = None) -> Dataset:
    """Parse and validate a golden dataset.

    When `prompts_root` is given, cases are additionally checked against every
    prompt version's few-shot examples. A case that also appears in a prompt is
    train/test leakage: the model was shown the answer, so the case measures
    recall of the prompt rather than capability.
    """
    errors: list[DatasetError] = []
    cases: list[GoldenCase] = []
    seen_ids: dict[str, int] = {}
    seen_emails: dict[str, int] = {}

    for lineno, raw in _iter_rows(path):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(DatasetError(lineno, f"invalid JSON: {exc.msg}"))
            continue

        try:
            case = GoldenCase.model_validate(payload)
        except ValidationError as exc:
            for detail in exc.errors(include_url=False):
                field = ".".join(str(p) for p in detail["loc"]) or "<root>"
                errors.append(DatasetError(lineno, f"{field}: {detail['msg']}"))
            continue

        if (first := seen_ids.get(case.id)) is not None:
            errors.append(
                DatasetError(lineno, f"duplicate id {case.id!r}, first seen on line {first}")
            )
            continue
        seen_ids[case.id] = lineno

        key = _normalize(case.input_email)
        if (first := seen_emails.get(key)) is not None:
            errors.append(DatasetError(lineno, f"duplicate email text, first seen on line {first}"))
            continue
        seen_emails[key] = lineno

        cases.append(case)

    if prompts_root is not None:
        leaked = few_shot_emails(prompts_root)
        for case in cases:
            version = leaked.get(_normalize(case.input_email))
            if version is not None:
                errors.append(
                    DatasetError(
                        seen_ids[case.id],
                        f"{case.id} duplicates a few-shot example in prompt {version}; "
                        "a case the model was shown measures recall, not capability",
                    )
                )

    if errors:
        raise DatasetValidationError(path, errors)

    return Dataset(path=path, cases=tuple(sorted(cases, key=lambda c: c.id)))


def load_holdout(path: Path, *, dataset: Dataset | None = None) -> tuple[HoldoutSample, ...]:
    """Parse and validate the judge-calibration holdout."""
    errors: list[DatasetError] = []
    samples: list[HoldoutSample] = []
    seen_ids: dict[str, int] = {}
    known = {c.id for c in dataset} if dataset is not None else None

    for lineno, raw in _iter_rows(path):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(DatasetError(lineno, f"invalid JSON: {exc.msg}"))
            continue

        try:
            sample = HoldoutSample.model_validate(payload)
        except ValidationError as exc:
            for detail in exc.errors(include_url=False):
                field = ".".join(str(p) for p in detail["loc"]) or "<root>"
                errors.append(DatasetError(lineno, f"{field}: {detail['msg']}"))
            continue

        if (first := seen_ids.get(sample.id)) is not None:
            errors.append(
                DatasetError(lineno, f"duplicate id {sample.id!r}, first on line {first}")
            )
            continue
        seen_ids[sample.id] = lineno

        if known is not None and sample.case_id not in known:
            errors.append(DatasetError(lineno, f"case_id {sample.case_id!r} is not in the dataset"))
            continue

        samples.append(sample)

    if errors:
        raise DatasetValidationError(path, errors)

    return tuple(sorted(samples, key=lambda s: s.id))
