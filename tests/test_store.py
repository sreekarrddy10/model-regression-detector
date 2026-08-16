from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from mrd.results import EvalRun, RunOutcome
from mrd.store import sqlite

from .engine_fixtures import NOW, make_outcome

pytestmark = pytest.mark.unit

ALL_PASS = {f"tc_{i:04d}": (True, True, True) for i in range(4)}


@pytest.fixture
def conn(tmp_path: Path):  # type: ignore[no-untyped-def]
    with sqlite.connect(tmp_path / "runs.sqlite") as connection:
        sqlite.initialize(connection)
        yield connection


def at(
    outcome: RunOutcome, *, run_id: str, minutes: int, dataset_hash: str | None = None
) -> RunOutcome:
    """Re-stamp a fixture run with a new id, timestamp and optionally dataset hash."""
    stamped = NOW + timedelta(minutes=minutes)
    run: EvalRun = replace(
        outcome.run,
        run_id=run_id,
        started_at=stamped,
        finished_at=stamped,
        dataset_hash=dataset_hash or outcome.run.dataset_hash,
    )
    return RunOutcome(
        run=run,
        results=tuple(replace(r, run_id=run_id) for r in outcome.results),
    )


def test_round_trips_a_run(conn) -> None:  # type: ignore[no-untyped-def]
    outcome = make_outcome(ALL_PASS, judge_score=4)
    sqlite.save(conn, outcome)

    loaded = sqlite.load(conn, "run-1")
    assert loaded is not None
    assert loaded.run == outcome.run
    assert loaded.results == outcome.results


def test_missing_run_returns_none(conn) -> None:  # type: ignore[no-untyped-def]
    assert sqlite.load(conn, "nope") is None


def test_save_is_idempotent(conn) -> None:  # type: ignore[no-untyped-def]
    outcome = make_outcome(ALL_PASS)
    sqlite.save(conn, outcome)
    sqlite.save(conn, outcome)

    loaded = sqlite.load(conn, "run-1")
    assert loaded is not None
    assert len(loaded.results) == len(outcome.results)


def test_nullable_columns_survive(conn) -> None:  # type: ignore[no-untyped-def]
    """cost_usd None must come back as None, never 0.0."""
    sqlite.save(conn, make_outcome(ALL_PASS, cost_usd=None))

    loaded = sqlite.load(conn, "run-1")
    assert loaded is not None
    assert all(r.cost_usd is None for r in loaded.results)
    assert all(r.judge_score is None for r in loaded.results)


def test_baseline_is_the_most_recent_comparable_run(conn) -> None:  # type: ignore[no-untyped-def]
    sqlite.save(conn, at(make_outcome(ALL_PASS), run_id="old", minutes=0))
    sqlite.save(conn, at(make_outcome(ALL_PASS), run_id="recent", minutes=10))

    baseline = sqlite.latest_baseline(conn, dataset_hash="hash-a")
    assert baseline is not None
    assert baseline.run.run_id == "recent"


def test_baseline_skips_runs_against_other_ground_truth(conn) -> None:  # type: ignore[no-untyped-def]
    """A run scored against a different dataset is not a baseline at all."""
    sqlite.save(conn, at(make_outcome(ALL_PASS), run_id="same-truth", minutes=0))
    sqlite.save(
        conn, at(make_outcome(ALL_PASS), run_id="other-truth", minutes=10, dataset_hash="hash-b")
    )

    baseline = sqlite.latest_baseline(conn, dataset_hash="hash-a")
    assert baseline is not None
    assert baseline.run.run_id == "same-truth"


def test_baseline_excludes_the_current_run(conn) -> None:  # type: ignore[no-untyped-def]
    sqlite.save(conn, at(make_outcome(ALL_PASS), run_id="previous", minutes=0))
    sqlite.save(conn, at(make_outcome(ALL_PASS), run_id="current", minutes=10))

    baseline = sqlite.latest_baseline(conn, dataset_hash="hash-a", exclude_run_id="current")
    assert baseline is not None
    assert baseline.run.run_id == "previous"


def test_no_baseline_on_a_first_run(conn) -> None:  # type: ignore[no-untyped-def]
    assert sqlite.latest_baseline(conn, dataset_hash="hash-a") is None


def test_recent_accuracy_is_oldest_first(conn) -> None:  # type: ignore[no-untyped-def]
    degrading = [
        {f"tc_{i:04d}": (True,) * 3 for i in range(4)},
        {**{f"tc_{i:04d}": (True,) * 3 for i in range(4)}, "tc_0003": (False,) * 3},
        {
            **{f"tc_{i:04d}": (True,) * 3 for i in range(4)},
            "tc_0002": (False,) * 3,
            "tc_0003": (False,) * 3,
        },
    ]
    for idx, flags in enumerate(degrading):
        sqlite.save(conn, at(make_outcome(flags), run_id=f"run-{idx}", minutes=idx * 10))

    assert sqlite.recent_accuracy(conn, dataset_hash="hash-a") == pytest.approx((1.0, 0.75, 0.5))


def test_recent_accuracy_respects_the_limit(conn) -> None:  # type: ignore[no-untyped-def]
    for idx in range(5):
        sqlite.save(conn, at(make_outcome(ALL_PASS), run_id=f"run-{idx}", minutes=idx))

    assert len(sqlite.recent_accuracy(conn, dataset_hash="hash-a", limit=3)) == 3
