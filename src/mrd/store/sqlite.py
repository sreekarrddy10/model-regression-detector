"""Run persistence.

Baseline selection is the only subtle part: a baseline must have been scored
against the same ground truth, or the comparison is meaningless. `latest_baseline`
filters on `dataset_hash` for exactly that reason.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from ..results import CaseResult, EvalRun, RunOutcome

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def save(conn: sqlite3.Connection, outcome: RunOutcome) -> None:
    run = outcome.run
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (
            run_id, git_sha, prompt_version, dataset_version, dataset_hash,
            model, judge_model, repeats, tier, started_at, finished_at, judge_kappa
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run.run_id,
            run.git_sha,
            run.prompt_version,
            run.dataset_version,
            run.dataset_hash,
            run.model,
            run.judge_model,
            run.repeats,
            run.tier,
            _iso(run.started_at),
            _iso(run.finished_at),
            run.judge_kappa,
        ),
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO case_results (
            run_id, case_id, repeat_idx, raw_output, category, summary, parse_error,
            schema_valid, category_match, judge_score, judge_rationale,
            latency_ms, input_tokens, output_tokens, cost_usd, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.run_id,
                r.case_id,
                r.repeat_idx,
                r.raw_output,
                r.category,
                r.summary,
                r.parse_error,
                int(r.schema_valid),
                int(r.category_match),
                r.judge_score,
                r.judge_rationale,
                r.latency_ms,
                r.input_tokens,
                r.output_tokens,
                r.cost_usd,
                r.error,
            )
            for r in outcome.results
        ],
    )


def _row_to_run(row: sqlite3.Row) -> EvalRun:
    return EvalRun(
        run_id=row["run_id"],
        git_sha=row["git_sha"],
        prompt_version=row["prompt_version"],
        dataset_version=row["dataset_version"],
        dataset_hash=row["dataset_hash"],
        model=row["model"],
        judge_model=row["judge_model"],
        repeats=row["repeats"],
        tier=row["tier"],
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=(datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None),
        judge_kappa=row["judge_kappa"],
    )


def _row_to_result(row: sqlite3.Row) -> CaseResult:
    return CaseResult(
        run_id=row["run_id"],
        case_id=row["case_id"],
        repeat_idx=row["repeat_idx"],
        raw_output=row["raw_output"],
        category=row["category"],
        summary=row["summary"],
        parse_error=row["parse_error"],
        schema_valid=bool(row["schema_valid"]),
        category_match=bool(row["category_match"]),
        judge_score=row["judge_score"],
        judge_rationale=row["judge_rationale"],
        latency_ms=row["latency_ms"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cost_usd=row["cost_usd"],
        error=row["error"],
    )


def load(conn: sqlite3.Connection, run_id: str) -> RunOutcome | None:
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    results = conn.execute(
        "SELECT * FROM case_results WHERE run_id = ? ORDER BY case_id, repeat_idx",
        (run_id,),
    ).fetchall()
    return RunOutcome(run=_row_to_run(row), results=tuple(_row_to_result(r) for r in results))


def latest_baseline(
    conn: sqlite3.Connection, *, dataset_hash: str, exclude_run_id: str | None = None
) -> RunOutcome | None:
    """Most recent run scored against the same ground truth.

    Runs against a different dataset hash are skipped rather than used, because
    diffing across a ground-truth change reports the edit as a model change.
    """
    row = conn.execute(
        """
        SELECT run_id FROM runs
        WHERE dataset_hash = ? AND run_id != COALESCE(?, '')
        ORDER BY started_at DESC, run_id DESC
        LIMIT 1
        """,
        (dataset_hash, exclude_run_id),
    ).fetchone()
    return None if row is None else load(conn, row["run_id"])


def recent_accuracy(
    conn: sqlite3.Connection, *, dataset_hash: str, limit: int = 7
) -> tuple[float, ...]:
    """Per-run accuracy for the last N comparable runs, oldest first.

    Feeds the EWMA drift check: gradual degradation that never trips a
    single run-to-run diff still shows up in the trend.
    """
    rows = conn.execute(
        """
        SELECT run_id FROM runs
        WHERE dataset_hash = ?
        ORDER BY started_at DESC, run_id DESC
        LIMIT ?
        """,
        (dataset_hash, limit),
    ).fetchall()

    accuracies: list[float] = []
    for row in reversed(rows):
        outcome = load(conn, row["run_id"])
        if outcome is None:
            continue
        from ..compare import measure

        accuracies.append(measure(outcome).accuracy)
    return tuple(accuracies)
