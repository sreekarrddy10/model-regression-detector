"""Dataset content hashing and lock file.

The hash is part of the run comparison key. Two runs are only comparable when
they were scored against byte-identical ground truth; otherwise a quiet edit to
the golden set can masquerade as a model improvement. `compare.py` (Phase 3)
refuses to diff runs whose dataset hashes differ.

The hash covers the semantic content of the cases, not file formatting, so
reordering lines or reflowing JSON does not invalidate a baseline - but changing
any label, email or difficulty tag does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..feature.schema import CATEGORIES
from .loader import Dataset
from .schema import DIFFICULTIES, SOURCES


def content_hash(dataset: Dataset) -> str:
    """SHA-256 over the canonical form of every case, ordered by id."""
    canonical = json.dumps(
        [case.model_dump(mode="json") for case in sorted(dataset.cases, key=lambda c: c.id)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Lock:
    version: str
    sha256: str
    count: int
    critical_count: int
    by_category: dict[str, int]
    by_difficulty: dict[str, int]
    by_source: dict[str, int]
    generated_at: str

    def to_json(self) -> str:
        payload = {
            "version": self.version,
            "sha256": self.sha256,
            "count": self.count,
            "critical_count": self.critical_count,
            "by_category": self.by_category,
            "by_difficulty": self.by_difficulty,
            "by_source": self.by_source,
            "generated_at": self.generated_at,
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Lock:
        data = json.loads(text)
        return cls(
            version=data["version"],
            sha256=data["sha256"],
            count=data["count"],
            critical_count=data["critical_count"],
            by_category=data["by_category"],
            by_difficulty=data["by_difficulty"],
            by_source=data["by_source"],
            generated_at=data["generated_at"],
        )


def build_lock(dataset: Dataset, *, version: str, now: datetime) -> Lock:
    return Lock(
        version=version,
        sha256=content_hash(dataset),
        count=len(dataset),
        critical_count=len(dataset.critical),
        by_category={c: len(dataset.by_category(c)) for c in CATEGORIES},
        by_difficulty={d: len(dataset.by_difficulty(d)) for d in DIFFICULTIES},
        by_source={s: sum(1 for c in dataset if c.source == s) for s in SOURCES},
        generated_at=now.isoformat(),
    )


class LockMismatch(RuntimeError):
    """The dataset on disk does not match its lock file."""


def verify(dataset: Dataset, lock_path: Path) -> Lock:
    """Raise unless the dataset matches its recorded lock."""
    if not lock_path.exists():
        raise LockMismatch(f"no lock file at {lock_path}; create one with: make dataset-lock")

    lock = Lock.from_json(lock_path.read_text(encoding="utf-8"))
    actual = content_hash(dataset)
    if actual != lock.sha256:
        raise LockMismatch(
            f"dataset content changed since it was locked.\n"
            f"  locked: {lock.sha256} ({lock.count} cases, version {lock.version})\n"
            f"  actual: {actual} ({len(dataset)} cases)\n"
            f"Ground truth changed, so runs before and after are not comparable. "
            f"Bump the version and re-lock: make dataset-lock VERSION=<next>"
        )
    return lock
