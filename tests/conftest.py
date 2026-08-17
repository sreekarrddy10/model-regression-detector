from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest

from mrd.prompts import PromptConfig
from mrd.providers.cassette import CassetteProvider

ROOT = Path(__file__).resolve().parents[1]

T = TypeVar("T")


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Drive a coroutine without a pytest-asyncio dependency."""
    return asyncio.run(coro)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip the e2e tree when Playwright is absent.

    The browser tests are an optional extra. Without this guard a fresh
    `make install && make test` fails during collection, which reads as a broken
    repo rather than a missing optional dependency.
    """
    if "e2e" in collection_path.parts:
        try:
            import playwright  # noqa: F401
        except ImportError:
            return True
    return None


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def prompt_v001() -> PromptConfig:
    return PromptConfig.load("v001", root=ROOT / "prompts")


@pytest.fixture(scope="session")
def cassette_dir() -> Path:
    return ROOT / "tests" / "cassettes"


@pytest.fixture
def replay(cassette_dir: Path) -> CassetteProvider:
    """Replay-only provider. A cassette miss raises rather than hitting the network."""
    return CassetteProvider(cassette_dir)
