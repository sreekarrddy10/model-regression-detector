"""Golden dataset: validation, hashing, coverage reporting.

Nothing here generates cases. Ground truth is human-authored by design.
"""

from .hashing import Lock, LockMismatch, build_lock, content_hash, verify
from .loader import Dataset, DatasetError, DatasetValidationError, load_cases, load_holdout
from .schema import DIFFICULTIES, SOURCES, Difficulty, GoldenCase, HoldoutSample, Source

__all__ = [
    "DIFFICULTIES",
    "SOURCES",
    "Dataset",
    "DatasetError",
    "DatasetValidationError",
    "Difficulty",
    "HoldoutSample",
    "Lock",
    "LockMismatch",
    "Source",
    "GoldenCase",
    "build_lock",
    "content_hash",
    "load_cases",
    "load_holdout",
    "verify",
]
