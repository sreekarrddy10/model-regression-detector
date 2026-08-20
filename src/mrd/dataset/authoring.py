"""YAML authoring surface, compiled to the canonical JSONL.

JSONL stays the hashed artifact - it is what the lock covers and what runs are
compared against. But nobody should hand-write it: a five-line customer email
becomes a 458-character single line with six escaped newlines, and every escape
is a chance to introduce a typo the validator only catches after you have written
twenty more cases.

So authoring happens in YAML block scalars, which are readable, and `build`
compiles them down.

The one non-obvious requirement is that compilation be **idempotent**. If
`added_at` were stamped fresh on every build, the content hash would change every
time, and the dataset lock - the thing that makes two runs comparable at all -
would be invalidated by a no-op rebuild. Timestamps are therefore carried over
from the existing JSONL by id, and only genuinely new cases are stamped.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .loader import DatasetError, DatasetValidationError
from .schema import CASE_ID_RE, GoldenCase, HoldoutSample

AUTO_FIELDS = ("id", "added_at")
DEFAULTS: dict[str, Any] = {"critical": False, "source": "handwritten"}
REQUIRED = ("input_email", "expected_category", "expected_summary", "difficulty", "notes")


@dataclass(frozen=True, slots=True)
class BuildResult:
    cases: tuple[GoldenCase, ...]
    added: tuple[str, ...]
    carried: tuple[str, ...]

    @property
    def summary(self) -> str:
        return (
            f"{len(self.cases)} case(s): {len(self.added)} new, "
            f"{len(self.carried)} with timestamps carried over"
        )


def _read_yaml(path: Path, key: str) -> list[Any]:
    """Entries are `Any` on purpose.

    yaml.safe_load returns whatever is in the file. Annotating this as
    list[dict[...]] would be a claim the parser cannot make, and it would render
    the isinstance guards at the call sites unreachable - mypy --strict says so.
    """
    if not path.exists():
        raise FileNotFoundError(f"authoring file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or key not in raw:
        raise DatasetValidationError(path, [DatasetError(0, f"expected a top-level {key!r} list")])
    entries = raw[key] or []
    if not isinstance(entries, list):
        raise DatasetValidationError(path, [DatasetError(0, f"{key!r} must be a list")])
    return entries


def existing_timestamps(jsonl: Path, field: str) -> dict[str, str]:
    """Map row id to its recorded timestamp, so rebuilds stay idempotent.

    The field differs by artifact - cases carry `added_at`, holdout samples
    carry `scored_at` - and reading the wrong one silently re-stamps every row,
    which churns the hash on a no-op rebuild.
    """
    if not jsonl.exists():
        return {}
    stamps: dict[str, str] = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("id") and row.get(field):
            stamps[row["id"]] = row[field]
    return stamps


def next_ids(taken: Sequence[str], count: int) -> list[str]:
    used = set(taken)
    assigned: list[str] = []
    index = 1
    while len(assigned) < count:
        candidate = f"tc_{index:04d}"
        if candidate not in used:
            assigned.append(candidate)
            used.add(candidate)
        index += 1
    return assigned


def build_cases(source: Path, jsonl: Path, *, now: datetime) -> BuildResult:
    """Compile the authoring YAML into validated cases.

    Explicit ids are honoured; omitted ones are assigned from the lowest free
    slot, so a case can be written without also bookkeeping its number.
    """
    entries = _read_yaml(source, "cases")
    stamps = existing_timestamps(jsonl, "added_at")

    errors: list[DatasetError] = []
    prepared: list[dict[str, Any]] = []
    explicit = [str(e["id"]) for e in entries if isinstance(e, dict) and e.get("id") is not None]

    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(DatasetError(position, "each case must be a mapping"))
            continue
        missing = [f for f in REQUIRED if not str(entry.get(f, "")).strip()]
        if missing:
            errors.append(DatasetError(position, f"missing or empty: {', '.join(missing)}"))
            continue
        prepared.append({**DEFAULTS, **entry})

    needs_id = [p for p in prepared if p.get("id") is None]
    for slot, payload in zip(next_ids(explicit, len(needs_id)), needs_id, strict=True):
        payload["id"] = slot

    cases: list[GoldenCase] = []
    added: list[str] = []
    carried: list[str] = []

    for position, payload in enumerate(prepared, start=1):
        case_id = str(payload["id"])
        if not CASE_ID_RE.match(case_id):
            errors.append(DatasetError(position, f"id {case_id!r} must match tc_NNNN"))
            continue

        if payload.get("added_at") is None:
            if case_id in stamps:
                payload["added_at"] = stamps[case_id]
                carried.append(case_id)
            else:
                payload["added_at"] = now.isoformat()
                added.append(case_id)

        try:
            cases.append(GoldenCase.model_validate(payload))
        except ValidationError as exc:
            for detail in exc.errors(include_url=False):
                field = ".".join(str(p) for p in detail["loc"]) or "<root>"
                errors.append(DatasetError(position, f"{case_id}: {field}: {detail['msg']}"))

    if errors:
        raise DatasetValidationError(source, errors)

    return BuildResult(
        cases=tuple(sorted(cases, key=lambda c: c.id)),
        added=tuple(added),
        carried=tuple(carried),
    )


def build_holdout(source: Path, jsonl: Path, *, now: datetime) -> tuple[HoldoutSample, ...]:
    entries = _read_yaml(source, "samples")
    stamps = existing_timestamps(jsonl, "scored_at")
    errors: list[DatasetError] = []
    samples: list[HoldoutSample] = []

    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(DatasetError(position, "each sample must be a mapping"))
            continue
        payload = dict(entry)
        payload.setdefault("id", f"ho_{position:04d}")
        if payload.get("scored_at") is None:
            payload["scored_at"] = stamps.get(str(payload["id"]), now.isoformat())
        try:
            samples.append(HoldoutSample.model_validate(payload))
        except ValidationError as exc:
            for detail in exc.errors(include_url=False):
                field = ".".join(str(p) for p in detail["loc"]) or "<root>"
                errors.append(DatasetError(position, f"{field}: {detail['msg']}"))

    if errors:
        raise DatasetValidationError(source, errors)
    return tuple(sorted(samples, key=lambda s: s.id))


def write_jsonl(rows: Sequence[GoldenCase] | Sequence[HoldoutSample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        json.dumps(row.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for row in rows
    )
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")
    return path
